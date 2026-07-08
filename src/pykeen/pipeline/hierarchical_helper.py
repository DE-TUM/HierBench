"""Shared internals for the hierarchical KG benchmarking tasks.

This module holds everything used by more than one of the task-specific pipelines — the
:class:`HierarchicalPipelineResult` container, the ancestor-path builder, the shared train/score and
HPO-refit bodies, the default hierarchy negative sampler, and the closure/negative sampling helpers.
The task pipelines themselves live in the sibling modules:

* :mod:`pykeen.pipeline.hierarchy` — hierarchy completion.
* :mod:`pykeen.pipeline.subsumption` — multi-hop subsumption prediction (Ganea et al. 2018; He et al. 2024).
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
from ..evaluation.pair_classification_evaluator import PairClassificationMetricResults
from ..models.nbase import ERModel
from ..models.unimodal import PoincareE
from ..sampling import HierarchyNegativeSampler
from ..training import SLCWATrainingLoop, training_loop_resolver
from ..triples import CoreTriplesFactory
from ..typing import LABEL_HEAD, LABEL_TAIL, MappedTriples

if TYPE_CHECKING:
    # imported lazily inside _hpo_and_refit to avoid a circular import
    # (pykeen.hpo imports pykeen.pipeline, which imports this module)
    from ..hpo import HpoPipelineResult

__all__ = [
    "build_ancestor_paths",
    "HierarchicalPipelineResult",
    "HpoHierarchicalResult",
]

#: default model used when the caller passes ``model=None``
_DEFAULT_MODEL: type[ERModel] = PoincareE

#: bound on rejection-sampling attempts per negative before falling back to an exact scan
_MAX_ATTEMPTS_FACTOR = 100

#: keys in the best-trial config that are data/dataset references, not forwardable pipeline kwargs
_NON_PIPELINE_CONFIG_KEYS = ("training", "testing", "validation", "dataset", "dataset_kwargs")


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

    Roots are kept in the sets, following the set augmentation of Kosmopoulos et al.
    (2015, https://doi.org/10.1007/s10618-014-0382-x).

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
    #: mAP/AUROC (and, for subsumption, thresholded precision/recall/F1) over transitive-closure
    #: positives vs. corrupted-descendant negatives (Bai et al. 2021), or ``None`` if not computed.
    ancestor_descendant_metric_results: PairClassificationMetricResults | None = None

    def _get_results(self) -> Mapping[str, Any]:
        """Extend the serialized results with the hierarchical metrics."""
        results = dict(super()._get_results())
        if self.hierarchical_metric_results is not None:
            results["hierarchical_metrics"] = self.hierarchical_metric_results.to_dict()
        if self.ancestor_descendant_metric_results is not None:
            results["ancestor_descendant_metrics"] = self.ancestor_descendant_metric_results.to_dict()
        return results


@dataclass
class HpoHierarchicalResult:
    """The result of the hierarchical HPO pipelines.

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
    ancestors: dict[int, frozenset[int]] | None = None,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Train on ``train_factory`` and score ``test_factory``, optionally with hierarchical metrics.

    Shared body of the task-specific pipelines (e.g. :func:`~pykeen.pipeline.hierarchy.hierarchy_completion_pipeline`),
    which differ only in how they build the split. Ground-truth ancestor paths for the hierarchical
    pass come from the full, pre-split hierarchy (``dataset.training``); pass ``ancestors`` when the
    caller already computed them to avoid a second pass.
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
        ancestors = ancestors or build_ancestor_paths(
            dataset.training.mapped_triples, dataset.num_entities, hierarchy_relation
        )
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
                # [] (not None) -> Y reflects the held-out targets only, no training triples added,
                # while skipping prepare_filter_triples()'s "did you forget training triples?" warning
                additional_filter_triples=[],
            ),
        )

    return HierarchicalPipelineResult(
        **{field.name: getattr(result, field.name) for field in fields(result)},
        hierarchical_metric_results=hierarchical_metric_results,
    )


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

    Shared body of the task-specific HPO pipelines (e.g.
    :func:`~pykeen.pipeline.hierarchy.hpo_hierarchy_completion_pipeline`); ``refit`` is the task's own
    pipeline function and ``split_kwargs`` are the task-specific split parameters forwarded to it so
    the re-fit uses the same split.
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


def _closure_pool(
    dataset: Dataset,
    hierarchy_relation: int | None,
) -> tuple[dict[int, frozenset[int]], list[list[int]], set[tuple[int, int]], list[tuple[int, int]]]:
    """Compute ``(ancestor paths, all training rows, direct edge set, non-direct closure pool)``.

    Shared by the closure-based splits. Warns when the dataset is multi-relational but no hierarchy
    relation was resolved (all edges are then treated as hierarchy edges).
    """
    if hierarchy_relation is None and dataset.num_relations > 1:
        warnings.warn(
            f"hierarchy_relation is None but the dataset has {dataset.num_relations} relations; all edges "
            "(regardless of relation) are treated as hierarchy edges. Pass hierarchy_relation to restrict "
            "the hierarchy to one relation.",
            stacklevel=4,
        )
    paths = build_ancestor_paths(dataset.training.mapped_triples, dataset.num_entities, hierarchy_relation)
    all_rows = dataset.training.mapped_triples.tolist()
    direct = {(h, t) for h, r, t in all_rows if hierarchy_relation is None or r == hierarchy_relation}
    # ponytail: enumerates the full transitive closure; fine up to MeSH scale (~200k pairs). Switch
    # to per-node descendant sampling without enumeration if WordNet-scale closures ever hurt.
    pool = sorted({(a, n) for n, ancestors in paths.items() for a in ancestors if a != n} - direct)
    return paths, all_rows, direct, pool


def _sample_negatives(
    positives: Sequence[Sequence[int]],
    paths: Mapping[int, frozenset[int]],
    num_entities: int,
    rng: np.random.Generator,
    *,
    num_negatives: int = 1,
    siblings: Mapping[int, Sequence[int]] | None = None,
) -> list[list[int]]:
    """Sample ``num_negatives`` corrupted-descendant negatives ``[h, r, t']`` per positive ``[h, r, t]``.

    The ancestor is kept fixed (Bai et al. 2021), which blocks models from scoring high-level nodes
    uniformly high. A candidate ``t'`` is valid iff ``h`` is not among its inclusive ancestors —
    this covers both non-closure pairs and ``t' == h``. Positives whose head is an ancestor-or-self
    of every entity are skipped with a warning.

    When ``siblings`` is given (hard negatives, He et al. 2024 §4.1), a hard negative pairs an
    entity with its own sibling — the pairs hardest to tell apart. Hierarchy relations may point
    either way, so either end may be replaced by a sibling of the other end, keeping whichever
    corruption falls outside the closure; positives lacking enough valid siblings are topped up
    with random corrupted-descendant negatives to keep the positive:negative ratio.
    """
    negatives: list[list[int]] = []
    for head, relation, tail in positives:
        rows: list[list[int]] = []
        if siblings is not None:
            hard = [[head, relation, s] for s in siblings.get(head, ()) if head not in paths[s]]
            hard += [[s, relation, tail] for s in siblings.get(tail, ()) if s not in paths[tail]]
            rng.shuffle(hard)
            rows.extend(hard[:num_negatives])
        while len(rows) < num_negatives:
            tail_new = None
            for _ in range(_MAX_ATTEMPTS_FACTOR):
                candidate = int(rng.integers(num_entities))
                if head not in paths[candidate]:
                    tail_new = candidate
                    break
            if tail_new is None:
                valid = [e for e in range(num_entities) if head not in paths[e]]
                if not valid:
                    warnings.warn(
                        f"entity {head} is an ancestor of every entity; skipping its negative pair", stacklevel=2
                    )
                    break
                tail_new = int(rng.choice(valid))
            rows.append([head, relation, tail_new])
        negatives.extend(rows)
    return negatives
