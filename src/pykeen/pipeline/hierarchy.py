"""Hierarchical KG benchmarking tasks (e.g. ancestor-descendant prediction).

The public surface mirrors PyKEEN's function-style pipeline API:

* :func:`ancestor_descendant_split` is pure data preparation — it returns standard
  ``(train, val, test)`` :class:`~pykeen.triples.CoreTriplesFactory` instances, so it composes with
  :func:`pykeen.pipeline.pipeline`, :func:`pykeen.hpo.hpo_pipeline`, or a hand-rolled training loop
  ("beyond the pipeline").
* :func:`ancestor_descendant_pipeline` is the one-call convenience that trains and additionally
  reports hierarchical precision/recall/F1.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import torch

from .api import PipelineResult, pipeline
from ..datasets.base import Dataset
from ..evaluation.hierarchical_classification_evaluator import (
    HierarchicalClassificationEvaluator,
    HierarchicalMetricResults,
)
from ..models.nbase import ERModel
from ..models.unimodal import PoincareE
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_TAIL, MappedTriples

if TYPE_CHECKING:
    # imported lazily inside hpo_ancestor_descendant_pipeline to avoid a circular import
    # (pykeen.hpo imports pykeen.pipeline, which imports this module)
    from ..hpo import HpoPipelineResult

__all__ = [
    "build_ancestor_paths",
    "ancestor_descendant_split",
    "ancestor_descendant_pipeline",
    "hpo_ancestor_descendant_pipeline",
    "HierarchicalPipelineResult",
    "HpoHierarchicalResult",
]

#: Multiplier bounding sampling attempts per hop, so shallow/small graphs that cannot
#: supply the requested number of distinct k-hop pairs still terminate.
_MAX_ATTEMPTS_FACTOR = 100

#: default model used when the caller passes ``model=None``
_DEFAULT_MODEL: type[ERModel] = PoincareE


def build_ancestor_paths(
    mapped_triples: MappedTriples,
    num_entities: int,
    hierarchy_relation: int | None = None,
) -> dict[int, frozenset[int]]:
    """Build the inclusive ancestor set (root-path) for every entity from direct edges.

    :param mapped_triples: The direct edges defining the hierarchy.
    :param num_entities: The total number of entities; all of them become nodes, so the returned
        mapping is total (isolated/root nodes map to ``{n}``).
    :param hierarchy_relation: If given, only edges with this relation id define the hierarchy;
        otherwise all edges are used.

    :returns: A mapping from entity id to its inclusive ancestor set.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_entities))
    graph.add_edges_from(
        (h, t) for h, r, t in mapped_triples.tolist() if hierarchy_relation is None or r == hierarchy_relation
    )
    return {n: frozenset(nx.ancestors(graph, n)) | {n} for n in graph.nodes}


@dataclass
class HierarchicalPipelineResult(PipelineResult):
    """A :class:`PipelineResult` that additionally carries hierarchical classification metrics."""

    #: The hierarchical precision/recall/F1 metrics, or ``None`` if not computed.
    hierarchical_metric_results: HierarchicalMetricResults | None = None

    def _get_results(self) -> Mapping[str, Any]:
        """Extend the serialized results with the hierarchical metrics."""
        results = dict(super()._get_results())
        if self.hierarchical_metric_results is not None:
            results["hierarchical_metrics"] = self.hierarchical_metric_results.to_dict()
        return results


@dataclass
class HpoHierarchicalResult:
    """The result of :func:`hpo_ancestor_descendant_pipeline`.

    Bundles the standard HPO study with the best trial re-fitted on the same split, so the full HPO
    surface (``hpo_result.study``, ``hpo_result.save_to_directory(...)``) stays available alongside the
    best configuration's rank-based and hierarchical metrics on ``result``.
    """

    #: The :class:`~pykeen.hpo.HpoPipelineResult` from the search (optuna study, serialization, ...).
    hpo_result: HpoPipelineResult
    #: The best trial re-trained on the split, or ``None`` when ``hierarchical=False``.
    result: HierarchicalPipelineResult | None = None


def ancestor_descendant_split(
    dataset: Dataset,
    *,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
    hierarchy_relation: int | None = None,
) -> tuple[CoreTriplesFactory, CoreTriplesFactory, CoreTriplesFactory]:
    """Build ``(train, val, test)`` triple factories for ancestor-descendant prediction.

    Training is the regular graph (direct edges only, restricted to ``hierarchy_relation`` when set).
    Held-out ancestor-descendant pairs are obtained by hop-balanced random-walk sampling rather than
    enumerating the transitive closure: a fixed budget ``num_pairs`` is split equally across the
    requested ``hops``, and each *k*-hop pair is produced by a length-*k* downward random walk. The
    sampled pairs are split 50/50 into validation and test, keeping the per-hop balance.

    The returned factories share the dataset's entity/relation vocabulary, so they plug directly into
    :func:`pykeen.pipeline.pipeline`, :func:`pykeen.hpo.hpo_pipeline`, or a custom training loop.

    Method based on *Poincaré Embeddings for Learning Hierarchical Representations* (Nickel & Kiela,
    NeurIPS 2017) and *Hyperbolic Entailment Cones for Learning Hierarchical Embeddings* (Ganea et
    al., ICML 2018).

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param hops: Hop distances to sample ancestor-descendant pairs at.
    :param num_pairs: Total budget of held-out pairs, split equally across ``hops``.
    :param seed: Random seed for reproducible sampling and val/test splits.
    :param hierarchy_relation: If given, only edges with this relation id define the hierarchy;
        otherwise all training edges are used.

    :returns: A ``(train, val, test)`` tuple of :class:`~pykeen.triples.CoreTriplesFactory` instances.
    """
    if hierarchy_relation is None and dataset.num_relations > 1:
        warnings.warn(
            f"hierarchy_relation is None but the dataset has {dataset.num_relations} relations; all edges "
            "(regardless of relation) will be flattened into a single hierarchy relation. Pass "
            "hierarchy_relation to restrict the hierarchy to one relation.",
            stacklevel=2,
        )
    edge_list = [
        (h, t)
        for h, r, t in dataset.training.mapped_triples.tolist()
        if hierarchy_relation is None or r == hierarchy_relation
    ]
    direct_edges = set(edge_list)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(dataset.num_entities))
    graph.add_edges_from(edge_list)

    rng = np.random.default_rng(seed)

    # Split the budget equally across the requested hops; first ``rem`` hops get one extra.
    base, rem = divmod(num_pairs, len(hops))
    targets = {k: base + (1 if i < rem else 0) for i, k in enumerate(hops)}

    starts = [n for n in graph.nodes if graph.out_degree(n) > 0]
    successors = {n: list(graph.successors(n)) for n in starts}

    val_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for k, target in targets.items():
        if target <= 0 or not starts:
            continue
        sampled: list[tuple[int, int]] = []
        for _ in range(target * _MAX_ATTEMPTS_FACTOR):
            if len(sampled) >= target:
                break
            current = starts[rng.integers(len(starts))]
            start = current
            # ponytail: a length-k walk yields a pair reachable in <=k hops (a DAG may also reach it
            # by a shorter path); `seen` dedups across hops, so per-hop counts can drift slightly.
            for _step in range(k):
                children = successors.get(current)
                if not children:  # hit a leaf before completing k steps
                    current = start
                    break
                current = children[rng.integers(len(children))]
            pair = (start, current)
            if start == current or pair in direct_edges or pair in seen:
                continue
            sampled.append(pair)
            seen.add(pair)
        # Split this hop's pairs 50/50 so both val and test stay hop-balanced.
        rng.shuffle(sampled)
        n_val = len(sampled) // 2
        val_pairs.extend(sampled[:n_val])
        test_pairs.extend(sampled[n_val:])

    # num_pairs == 0 is a legitimate "train-only" request; for a positive budget, an empty holdout means
    # the hierarchy is too shallow (or its only transitive pairs are also direct edges). Warn rather than
    # raise, since the latter is a valid graph shape — but a silent empty val/test is a debugging trap.
    if num_pairs > 0 and not val_pairs and not test_pairs:
        warnings.warn(
            f"No ancestor-descendant pairs could be sampled for hops={tuple(hops)}; validation/test will be "
            "empty. The hierarchy is likely too shallow/small for these hops, or its transitive pairs are all "
            "direct edges. Lower `hops`, raise `num_pairs`, or check `hierarchy_relation`.",
            stacklevel=2,
        )

    def _to_factory(pairs: list[tuple[int, int]], base_rows: list[list[int]] | None = None) -> CoreTriplesFactory:
        rows = (base_rows or []) + [[h, 0, t] for h, t in pairs]
        mapped = torch.tensor(rows, dtype=torch.long) if rows else torch.empty((0, 3), dtype=torch.long)
        return CoreTriplesFactory(
            mapped_triples=mapped,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
        )

    direct_rows = [[h, 0, t] for h, t in sorted(direct_edges)]
    train_factory = _to_factory([], base_rows=direct_rows)
    val_factory = _to_factory(val_pairs)
    test_factory = _to_factory(test_pairs)

    return train_factory, val_factory, test_factory


def ancestor_descendant_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    embedding_dim: int = 64,
    epochs: int = 100,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
    hierarchical: bool = True,
    hierarchy_relation: int | None = None,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Train on direct edges; evaluate on sampled ancestor-descendant pairs.

    Convenience wrapper that mirrors :func:`pykeen.pipeline.pipeline`: it builds the split with
    :func:`ancestor_descendant_split`, trains the model, and (unless ``hierarchical`` is ``False``)
    additionally reports hierarchical precision/recall/F1 (over ancestor paths) alongside the regular
    rank-based metrics.

    :param dataset: A hierarchical dataset.
    :param model: Model class, string alias, or instance forwarded to
        :func:`pykeen.pipeline.pipeline`. When ``None``, uses
        :class:`~pykeen.models.unimodal.PoincareE` with ``embedding_dim``.
    :param embedding_dim: Dimensionality of entity embeddings. Only used when ``model`` is ``None``.
    :param epochs: Number of training epochs.
    :param hops: Hop distances to sample ancestor-descendant pairs at. Default ``(2, 3, 4, 5)``.
    :param num_pairs: Total budget of held-out pairs, split equally across ``hops``. Default 400.
    :param seed: Random seed for reproducible sampling and val/test splits.
    :param hierarchical: Whether to compute the hierarchical metrics (an extra evaluation pass).
    :param hierarchy_relation: If given, only edges with this relation id define the hierarchy.
    :param pipeline_kwargs: Additional kwargs forwarded to :func:`pykeen.pipeline.pipeline`.

    :returns: A :class:`HierarchicalPipelineResult`; its ``hierarchical_metric_results`` is ``None``
        when ``hierarchical`` is ``False``.
    """
    train_factory, val_factory, test_factory = ancestor_descendant_split(
        dataset, hops=hops, num_pairs=num_pairs, seed=seed, hierarchy_relation=hierarchy_relation
    )
    resolved_model = model if model is not None else _DEFAULT_MODEL
    # copy so we never mutate a dict the caller still holds a reference to
    model_kwargs = dict(pipeline_kwargs.pop("model_kwargs", None) or {})
    if model is None:
        model_kwargs.setdefault("embedding_dim", embedding_dim)
    result = pipeline(
        training=train_factory,
        testing=test_factory,
        validation=val_factory,
        model=resolved_model,
        model_kwargs=model_kwargs,
        epochs=epochs,
        **pipeline_kwargs,
    )

    hierarchical_metric_results = None
    if hierarchical:
        # direct instantiation → the big ancestors dict never enters the logged config
        ancestors = build_ancestor_paths(dataset.training.mapped_triples, dataset.num_entities, hierarchy_relation)
        # mirror the pipeline's evaluation settings on the hierarchical pass (evaluation_kwargs are the
        # .evaluate()-time params; evaluator_kwargs is the constructor, used as a fallback for batch_size)
        eval_source = {
            **pipeline_kwargs.get("evaluator_kwargs", {}),
            **pipeline_kwargs.get("evaluation_kwargs", {}),
        }
        eval_kwargs = {
            key: eval_source[key]
            for key in ("batch_size", "slice_size", "device", "use_tqdm", "tqdm_kwargs")
            if key in eval_source
        }
        hierarchical_metric_results = cast(
            HierarchicalMetricResults,
            HierarchicalClassificationEvaluator(ancestors=ancestors).evaluate(
                model=result.model,
                mapped_triples=test_factory.mapped_triples,
                targets=(LABEL_HEAD, LABEL_TAIL),
                **eval_kwargs,
                # no additional_filter_triples → Y reflects the held-out targets only
            ),
        )

    return HierarchicalPipelineResult(
        **{field.name: getattr(result, field.name) for field in fields(result)},
        hierarchical_metric_results=hierarchical_metric_results,
    )


#: keys in the best-trial config that are data/dataset references, not forwardable pipeline kwargs
_NON_PIPELINE_CONFIG_KEYS = ("training", "testing", "validation", "dataset", "dataset_kwargs")


def hpo_ancestor_descendant_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
    hierarchy_relation: int | None = None,
    hierarchical: bool = True,
    **hpo_kwargs,
) -> HpoHierarchicalResult:
    """Run HPO on the ancestor-descendant task, then re-fit and score the best trial.

    Mirrors :func:`ancestor_descendant_pipeline` for the HPO case: it builds the split with
    :func:`ancestor_descendant_split`, runs :func:`pykeen.hpo.hpo_pipeline` over it, and (unless
    ``hierarchical`` is ``False``) re-trains the winning configuration on the same split and reports
    its hierarchical precision/recall/F1. HPO optimizes the rank-based objective on validation and
    keeps no fitted model, so the best trial is retrained via :func:`ancestor_descendant_pipeline`.

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param model: Model class, string alias, or ``None`` (defaults to
        :class:`~pykeen.models.unimodal.PoincareE`), forwarded to :func:`pykeen.hpo.hpo_pipeline`.
    :param hops: Hop distances to sample ancestor-descendant pairs at. Default ``(2, 3, 4, 5)``.
    :param num_pairs: Total budget of held-out pairs, split equally across ``hops``. Default 400.
    :param seed: Random seed for reproducible sampling and val/test splits.
    :param hierarchy_relation: If given, only edges with this relation id define the hierarchy.
    :param hierarchical: Whether to re-fit the best trial and compute the hierarchical metrics.
    :param hpo_kwargs: Additional kwargs forwarded to :func:`pykeen.hpo.hpo_pipeline` (e.g.
        ``n_trials``, ``optimizer``, ``model_kwargs_ranges``, ``epochs``).

    :returns: A :class:`HpoHierarchicalResult`; its ``result`` is ``None`` when ``hierarchical`` is
        ``False``.
    """
    train_factory, val_factory, test_factory = ancestor_descendant_split(
        dataset, hops=hops, num_pairs=num_pairs, seed=seed, hierarchy_relation=hierarchy_relation
    )
    resolved_model = model if model is not None else _DEFAULT_MODEL

    # lazy import → breaks the pykeen.hpo <-> pykeen.pipeline import cycle
    from ..hpo import hpo_pipeline

    hpo_result = hpo_pipeline(
        training=train_factory,
        validation=val_factory,
        testing=test_factory,
        model=resolved_model,
        **hpo_kwargs,
    )

    result = None
    if hierarchical:
        # reconstruct the winning configuration (preset + optimized kwargs, early-stopped epochs)
        # ponytail: rides the private `_get_best_study_config()` -> {"pipeline": {...}}; no public API
        # returns the best config inline. If pykeen renames it, update here (see pykeen.hpo.hpo).
        config = dict(hpo_result._get_best_study_config()["pipeline"])
        for key in _NON_PIPELINE_CONFIG_KEYS:
            config.pop(key, None)
        best_model = config.pop("model", resolved_model)
        # an explicit `epochs` overrides training_kwargs["num_epochs"], so route it through `epochs`
        training_kwargs = dict(config.pop("training_kwargs", {}))
        best_epochs = training_kwargs.pop("num_epochs", None)
        if training_kwargs:
            config["training_kwargs"] = training_kwargs
        epoch_kwargs = {"epochs": best_epochs} if best_epochs is not None else {}
        result = ancestor_descendant_pipeline(
            dataset,
            model=best_model,
            hops=hops,
            num_pairs=num_pairs,
            seed=seed,
            hierarchy_relation=hierarchy_relation,
            hierarchical=True,
            **epoch_kwargs,
            **config,
        )

    return HpoHierarchicalResult(hpo_result=hpo_result, result=result)
