"""Hierarchical KG benchmarking tasks (e.g. hierarchy completion).

The public surface mirrors PyKEEN's function-style pipeline API:

* :func:`hierarchy_completion_split` is pure data preparation — it returns standard
  ``(train, val, test)`` :class:`~pykeen.triples.CoreTriplesFactory` instances, so it composes with
  :func:`pykeen.pipeline.pipeline`, :func:`pykeen.hpo.hpo_pipeline`, or a hand-rolled training loop
  ("beyond the pipeline").
* :func:`hierarchy_completion_pipeline` is the one-call convenience that trains and additionally
  reports hierarchical precision/recall/F1.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import torch

from .api import PipelineResult, pipeline
from ..datasets.base import Dataset
from ..datasets.metadata import HierarchicalGraph
from ..evaluation.hierarchical_classification_evaluator import (
    HierarchicalClassificationEvaluator,
    HierarchicalMetricResults,
)
from ..models.nbase import ERModel
from ..models.unimodal import PoincareE
from ..sampling import HierarchyNegativeSampler
from ..training import SLCWATrainingLoop, training_loop_resolver
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_TAIL, MappedTriples

if TYPE_CHECKING:
    # imported lazily inside hpo_hierarchy_completion_pipeline to avoid a circular import
    # (pykeen.hpo imports pykeen.pipeline, which imports this module)
    from ..hpo import HpoPipelineResult

__all__ = [
    "build_ancestor_paths",
    "hierarchy_completion_split",
    "hierarchy_completion_pipeline",
    "hpo_hierarchy_completion_pipeline",
    "HierarchicalPipelineResult",
    "HpoHierarchicalResult",
]

#: default model used when the caller passes ``model=None``
_DEFAULT_MODEL: type[ERModel] = PoincareE


def _resolve_hierarchy_relation(dataset: Dataset, hierarchy_relation: int | str | None) -> int | None:
    """Resolve the hierarchy relation to a relation id.

    Precedence: an explicit ``hierarchy_relation`` (id or label) wins; otherwise, if ``dataset`` is a
    :class:`~pykeen.datasets.metadata.HierarchicalGraph`, its :attr:`hierarchical_relation` label is
    used. Returns ``None`` when neither is available (all edges are treated as hierarchy edges).
    """
    if hierarchy_relation is None and isinstance(dataset, HierarchicalGraph):
        hierarchy_relation = dataset.hierarchical_relation
    if isinstance(hierarchy_relation, str):
        try:
            return dataset.training.relations_to_ids([hierarchy_relation])[0]
        except (AttributeError, KeyError) as exc:
            raise KeyError(f"hierarchy relation {hierarchy_relation!r} not found in dataset relations") from exc
    return hierarchy_relation


def _factory_from_rows(rows: list[list[int]], dataset: Dataset) -> CoreTriplesFactory:
    """Build a :class:`CoreTriplesFactory` over the dataset's vocabulary from ``[h, r, t]`` rows."""
    mapped = torch.tensor(rows, dtype=torch.long) if rows else torch.empty((0, 3), dtype=torch.long)
    return CoreTriplesFactory(
        mapped_triples=mapped,
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
    )


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
    """The result of :func:`hpo_hierarchy_completion_pipeline`.

    Bundles the standard HPO study with the best trial re-fitted on the same split, so the full HPO
    surface (``hpo_result.study``, ``hpo_result.save_to_directory(...)``) stays available alongside the
    best configuration's rank-based and hierarchical metrics on ``result``.
    """

    #: The :class:`~pykeen.hpo.HpoPipelineResult` from the search (optuna study, serialization, ...).
    hpo_result: HpoPipelineResult
    #: The best trial re-trained on the split, or ``None`` when ``hierarchical=False``.
    result: HierarchicalPipelineResult | None = None


def _train_and_score_hierarchical(
    dataset: Dataset,
    train_factory: CoreTriplesFactory,
    val_factory: CoreTriplesFactory,
    test_factory: CoreTriplesFactory,
    *,
    model: type[ERModel] | str | None,
    embedding_dim: int,
    epochs: int,
    hierarchical: bool,
    hierarchy_relation: int | None,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Train on ``train_factory`` and score ``test_factory``, optionally with hierarchical metrics.

    Shared body of the task-specific pipelines (e.g. :func:`hierarchy_completion_pipeline`), which
    differ only in how they build the split. Ground-truth ancestor paths for the hierarchical pass
    come from the full, pre-split hierarchy (``dataset.training``).
    """
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


def _hpo_and_refit(
    dataset: Dataset,
    train_factory: CoreTriplesFactory,
    val_factory: CoreTriplesFactory,
    test_factory: CoreTriplesFactory,
    *,
    refit: Any,
    split_kwargs: Mapping[str, Any],
    model: type[ERModel] | str | None,
    hierarchy_relation: int | None,
    hierarchical: bool,
    **hpo_kwargs,
) -> HpoHierarchicalResult:
    """Run HPO over a fixed split, then re-fit and score the best trial via ``refit``.

    Shared body of the task-specific HPO pipelines (e.g. :func:`hpo_hierarchy_completion_pipeline`);
    ``refit`` is the task's own pipeline function and ``split_kwargs`` are the task-specific split
    parameters forwarded to it so the re-fit uses the same split.
    """
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
        result = refit(
            dataset,
            model=best_model,
            hierarchy_relation=hierarchy_relation,
            hierarchical=True,
            **split_kwargs,
            **epoch_kwargs,
            **config,
        )

    return HpoHierarchicalResult(hpo_result=hpo_result, result=result)


def _default_hierarchy_sampler(kwargs: dict[str, Any], hierarchy_relation: int | None) -> dict[str, Any]:
    """Default hierarchy-completion training to same-depth negatives + filtering (sLCWA only).

    Sets :class:`~pykeen.sampling.HierarchyNegativeSampler` as the negative sampler unless the caller
    chose one explicitly, so ``negative_sampler="pseudotyped"`` / ``"basic"`` (or any registered
    sampler) still wins. A no-op for non-sLCWA loops, where negatives do not apply.
    """
    if kwargs.get("negative_sampler") is not None:
        return kwargs  # respect an explicit override
    if not issubclass(training_loop_resolver.lookup(kwargs.get("training_loop")), SLCWATrainingLoop):
        return kwargs  # lookup(None) -> default SLCWA
    kwargs["negative_sampler"] = HierarchyNegativeSampler
    sampler_kwargs = dict(kwargs.get("negative_sampler_kwargs") or {})
    sampler_kwargs.setdefault("hierarchy_relation", hierarchy_relation)  # which edges define depth
    sampler_kwargs.setdefault("filtered", True)
    kwargs["negative_sampler_kwargs"] = sampler_kwargs
    return kwargs


def hierarchy_completion_split(
    dataset: Dataset,
    *,
    test_ratio: float = 0.1,
    seed: int = 42,
    hierarchy_relation: int | str | None = None,
) -> tuple[CoreTriplesFactory, CoreTriplesFactory, CoreTriplesFactory]:
    """Build ``(train, val, test)`` triple factories by removing direct hierarchy edges.

    Held-out positives are direct hierarchy edges *removed* from training (rather than sampled
    transitive pairs). Edges are removed connectivity-preserving: an edge
    is removable only if both endpoints keep at least one other incident hierarchy edge, so no node is
    ever cut off from the hierarchy (leaves are never orphaned). The removed edges keep their original
    relation id and are split 50/50 into validation and test; ``train`` is the remaining triples
    (non-hierarchy edges, if any, stay in training as context).

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param test_ratio: Fraction of removable hierarchy edges to hold out (``round(test_ratio·|H|)``).
    :param seed: Random seed for reproducible removal and val/test splits.
    :param hierarchy_relation: Relation id or label whose edges define the hierarchy. Defaults to the
        dataset's :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when it is a
        :class:`~pykeen.datasets.metadata.HierarchicalGraph`; otherwise all training edges are used.

    :returns: A ``(train, val, test)`` tuple of :class:`~pykeen.triples.CoreTriplesFactory` instances.
    """
    hierarchy_relation = _resolve_hierarchy_relation(dataset, hierarchy_relation)
    if hierarchy_relation is None and dataset.num_relations > 1:
        warnings.warn(
            f"hierarchy_relation is None but the dataset has {dataset.num_relations} relations; all edges "
            "(regardless of relation) are treated as hierarchy edges. Pass hierarchy_relation to restrict "
            "the hierarchy to one relation.",
            stacklevel=2,
        )

    all_rows = dataset.training.mapped_triples.tolist()
    # indices of candidate hierarchy edges (self-loops are not real hierarchy edges)
    hierarchy_idx = [
        i for i, (h, r, t) in enumerate(all_rows) if (hierarchy_relation is None or r == hierarchy_relation) and h != t
    ]

    # live undirected incidence degree over the hierarchy edges
    deg: dict[int, int] = {}
    for i in hierarchy_idx:
        h, _r, t = all_rows[i]
        deg[h] = deg.get(h, 0) + 1
        deg[t] = deg.get(t, 0) + 1

    rng = np.random.default_rng(seed)
    order = list(hierarchy_idx)
    rng.shuffle(order)

    budget = round(test_ratio * len(hierarchy_idx))
    removed: list[int] = []
    for i in order:
        if len(removed) >= budget:
            break
        h, _r, t = all_rows[i]
        # removable only if both endpoints retain >=1 incident hierarchy edge afterwards
        if deg[h] >= 2 and deg[t] >= 2:
            removed.append(i)
            deg[h] -= 1
            deg[t] -= 1

    if test_ratio > 0 and not removed:
        warnings.warn(
            "No hierarchy edges could be removed without isolating a node; validation/test will be empty. "
            "The hierarchy is likely a sparse tree with no removable edges. Raise `test_ratio` or check "
            "`hierarchy_relation`.",
            stacklevel=2,
        )

    rng.shuffle(removed)
    n_val = len(removed) // 2
    val_idx = set(removed[:n_val])
    test_idx = set(removed[n_val:])
    held_out = val_idx | test_idx

    train_rows = [row for i, row in enumerate(all_rows) if i not in held_out]
    val_rows = [all_rows[i] for i in sorted(val_idx)]
    test_rows = [all_rows[i] for i in sorted(test_idx)]

    return (
        _factory_from_rows(train_rows, dataset),
        _factory_from_rows(val_rows, dataset),
        _factory_from_rows(test_rows, dataset),
    )


def hierarchy_completion_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    embedding_dim: int = 64,
    epochs: int = 100,
    test_ratio: float = 0.1,
    seed: int = 42,
    hierarchical: bool = True,
    hierarchy_relation: int | str | None = None,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Remove direct hierarchy edges, train on the rest, and predict the removed edges.

    Convenience wrapper that mirrors :func:`pykeen.pipeline.pipeline`: it builds the split with
    :func:`hierarchy_completion_split`, trains the model, and (unless ``hierarchical`` is ``False``)
    additionally reports hierarchical precision/recall/F1 alongside the rank-based metrics.

    :param dataset: A hierarchical dataset.
    :param model: Model class, string alias, or instance forwarded to
        :func:`pykeen.pipeline.pipeline`. When ``None``, uses
        :class:`~pykeen.models.unimodal.PoincareE` with ``embedding_dim``.
    :param embedding_dim: Dimensionality of entity embeddings. Only used when ``model`` is ``None``.
    :param epochs: Number of training epochs.
    :param test_ratio: Fraction of removable hierarchy edges to hold out. Default 0.1.
    :param seed: Random seed for reproducible removal and val/test splits.
    :param hierarchical: Whether to compute the hierarchical metrics (an extra evaluation pass).
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param pipeline_kwargs: Additional kwargs forwarded to :func:`pykeen.pipeline.pipeline`. Under
        sLCWA the negative sampler defaults to :class:`~pykeen.sampling.HierarchyNegativeSampler`
        (same-depth hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` to override.

    :returns: A :class:`HierarchicalPipelineResult`; its ``hierarchical_metric_results`` is ``None``
        when ``hierarchical`` is ``False``.
    """
    hierarchy_relation = _resolve_hierarchy_relation(dataset, hierarchy_relation)
    train_factory, val_factory, test_factory = hierarchy_completion_split(
        dataset, test_ratio=test_ratio, seed=seed, hierarchy_relation=hierarchy_relation
    )
    pipeline_kwargs = _default_hierarchy_sampler(pipeline_kwargs, hierarchy_relation)
    return _train_and_score_hierarchical(
        dataset,
        train_factory,
        val_factory,
        test_factory,
        model=model,
        embedding_dim=embedding_dim,
        epochs=epochs,
        hierarchical=hierarchical,
        hierarchy_relation=hierarchy_relation,
        **pipeline_kwargs,
    )


def hpo_hierarchy_completion_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    test_ratio: float = 0.1,
    seed: int = 42,
    hierarchy_relation: int | str | None = None,
    hierarchical: bool = True,
    **hpo_kwargs,
) -> HpoHierarchicalResult:
    """Run HPO on the hierarchy-completion task, then re-fit and score the best trial.

    Builds the split with :func:`hierarchy_completion_split`, runs :func:`pykeen.hpo.hpo_pipeline`
    over it, and (unless ``hierarchical`` is ``False``) re-trains the winning configuration on the
    same split and reports its hierarchical precision/recall/F1.

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param model: Model class, string alias, or ``None`` (defaults to
        :class:`~pykeen.models.unimodal.PoincareE`), forwarded to :func:`pykeen.hpo.hpo_pipeline`.
    :param test_ratio: Fraction of removable hierarchy edges to hold out. Default 0.1.
    :param seed: Random seed for reproducible removal and val/test splits.
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param hierarchical: Whether to re-fit the best trial and compute the hierarchical metrics.
    :param hpo_kwargs: Additional kwargs forwarded to :func:`pykeen.hpo.hpo_pipeline`. Under sLCWA the
        negative sampler defaults to :class:`~pykeen.sampling.HierarchyNegativeSampler` (same-depth
        hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` to override.

    :returns: A :class:`HpoHierarchicalResult`; its ``result`` is ``None`` when ``hierarchical`` is
        ``False``.
    """
    hierarchy_relation = _resolve_hierarchy_relation(dataset, hierarchy_relation)
    train_factory, val_factory, test_factory = hierarchy_completion_split(
        dataset, test_ratio=test_ratio, seed=seed, hierarchy_relation=hierarchy_relation
    )
    hpo_kwargs = _default_hierarchy_sampler(hpo_kwargs, hierarchy_relation)
    return _hpo_and_refit(
        dataset,
        train_factory,
        val_factory,
        test_factory,
        refit=hierarchy_completion_pipeline,
        split_kwargs={"test_ratio": test_ratio, "seed": seed},
        model=model,
        hierarchy_relation=hierarchy_relation,
        hierarchical=hierarchical,
        **hpo_kwargs,
    )
