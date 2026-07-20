"""Shared internals for the hierarchical KG benchmarking tasks.

This module holds everything used by more than one of the task-specific pipelines — the
:class:`HierarchicalPipelineResult` container, the ancestor-path builder, the shared train/score and
HPO-refit bodies, the default hierarchy negative sampler, and the closure/negative sampling helpers.
The task pipelines themselves live in the sibling modules:

* :mod:`pykeen.pipeline.hierarchy` — hierarchy completion.
* :mod:`pykeen.pipeline.transitive_ancestor_descendant` — multi-hop transitive ancestor-descendant prediction
  (Ganea et al. 2018; He et al. 2024).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import torch
from scipy import sparse

from .api import PipelineResult, pipeline
from ..datasets.base import Dataset
from ..evaluation.lca_classification_evaluator import LCAMetricResults
from ..evaluation.pair_classification_evaluator import PairClassificationMetricResults
from ..models.nbase import ERModel
from ..models.unimodal import PoincareE
from ..sampling import HierarchyNegativeSampler
from ..training import SLCWATrainingLoop, training_loop_resolver
from ..triples import CoreTriplesFactory
from ..typing import MappedTriples

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
    if hierarchy_relation is not None:
        mapped_triples = mapped_triples[mapped_triples[:, 1] == hierarchy_relation]
    # Reachability via repeated squaring of the sparse adjacency matrix: R <- R | R@R covers paths
    # of length 2^k after k rounds, so a hierarchy of depth d converges in ceil(log2(d)) C-level
    # sparse matmuls — orders of magnitude faster than per-node networkx BFS at WordNet scale.
    heads = mapped_triples[:, 0].numpy()
    tails = mapped_triples[:, 2].numpy()
    reach = sparse.csr_matrix((np.ones(len(heads), dtype=np.int64), (heads, tails)), shape=(num_entities, num_entities))
    reach.data[:] = 1  # collapse duplicate edges
    while True:
        nxt = reach @ reach + reach
        nxt.data[:] = 1  # path counts -> booleans, also prevents overflow across rounds
        if nxt.nnz == reach.nnz:
            break
        reach = nxt
    # ancestors of n = rows that reach column n -> slice columns via CSC
    reach = reach.tocsc()
    indptr, indices = reach.indptr, reach.indices
    return {n: frozenset(indices[indptr[n] : indptr[n + 1]].tolist()) | {n} for n in range(num_entities)}


@dataclass
class HierarchicalPipelineResult(PipelineResult):
    """A :class:`PipelineResult` that additionally carries hierarchical-task metrics."""

    #: mAP/AUROC (and, for transitive ancestor-descendant prediction, thresholded precision/recall/F1) over
    #: transitive-closure positives vs. corrupted-descendant negatives (Bai et al. 2021), or ``None`` if not computed.
    ancestor_descendant_metric_results: PairClassificationMetricResults | None = None
    #: Raw per-pair test ``rows``/``labels``/``scores``/``predictions``/``threshold`` behind the
    #: ancestor-descendant metrics, populated only when the pipeline is called with ``return_raw=True``.
    #: A carry field for opted-in callers; deliberately excluded from ``_get_results()``/``to_dict()``.
    ancestor_descendant_raw_predictions: dict | None = None
    #: LCA-based hierarchical precision/recall/F1 (Kosmopoulos et al. 2015) per test descendant,
    #: populated only when the pipeline is called with ``lca=True``.
    lca_metric_results: LCAMetricResults | None = None

    def _get_results(self) -> Mapping[str, Any]:
        """Extend the serialized results with the hierarchical-task metrics."""
        results = dict(super()._get_results())
        if self.ancestor_descendant_metric_results is not None:
            results["ancestor_descendant_metrics"] = self.ancestor_descendant_metric_results.to_dict()
        if self.lca_metric_results is not None:
            results["lca_metrics"] = self.lca_metric_results.to_dict()
        return results


@dataclass
class HpoHierarchicalResult:
    """The result of the hierarchical HPO pipelines.

    Bundles the standard HPO study with the best trial re-fitted on the same split, so the full HPO
    surface (``hpo_result.study``, ``hpo_result.save_to_directory(...)``) stays available alongside the
    best configuration's metrics on ``result``.
    """

    #: The :class:`~pykeen.hpo.HpoPipelineResult` from the search (optuna study, serialization, ...).
    hpo_result: HpoPipelineResult
    #: The best trial re-trained on the split.
    result: HierarchicalPipelineResult | None = None


def _train_and_score_hierarchical(
    train_factory: CoreTriplesFactory,
    val_factory: CoreTriplesFactory,
    test_factory: CoreTriplesFactory,
    *,
    model: type[ERModel] | str | None,
    embedding_dim: int,
    epochs: int,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Train on ``train_factory`` and score ``test_factory``.

    Shared body of the task-specific pipelines (e.g. :func:`~pykeen.pipeline.hierarchy.hierarchy_completion_pipeline`),
    which differ only in how they build the split.
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
    return HierarchicalPipelineResult(
        **{field.name: getattr(result, field.name) for field in fields(result)},
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


def _canonical_rows(
    mapped_triples: MappedTriples,
    dataset: Dataset,
    hierarchy_relation: int | None,
) -> list[list[int]]:
    """Return the triples as row lists in canonical parent->child orientation.

    Hierarchy-relation edges are flipped when the dataset declares
    :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchy_inverted`, so heads are always
    ancestors regardless of the dataset's raw edge direction.
    """
    rows = mapped_triples.tolist()
    if getattr(dataset, "hierarchy_inverted", False):
        rows = [[t, r, h] if hierarchy_relation is None or r == hierarchy_relation else [h, r, t] for h, r, t in rows]
    return rows


def _closure_pool(
    dataset: Dataset,
    hierarchy_relation: int | None,
) -> tuple[dict[int, frozenset[int]], list[list[int]], set[tuple[int, int]], list[tuple[int, int]]]:
    """Compute ``(ancestor paths, all training rows, direct edge set, non-direct closure pool)``.

    Shared by the closure-based splits. Warns when the dataset is multi-relational but no hierarchy
    relation was resolved (all edges are then treated as hierarchy edges). When the dataset declares
    :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchy_inverted`, hierarchy edges are
    flipped to the canonical parent->child orientation before anything is built, so heads are always
    ancestors regardless of the dataset's raw edge direction.

    The ``direct`` edge set is the *transitive reduction* of the hierarchy (Ganea et al. 2018 §5's
    "basic edges"), not the raw asserted edges: a redundant asserted shortcut (``A->C`` when
    ``A->B->C`` already implies it) is a non-basic closure edge and joins the eval ``pool`` instead
    of being pinned into training. Falls back to the asserted edges (with a warning) when the
    hierarchy is not a DAG, since the transitive reduction is only defined on acyclic graphs.
    """
    if hierarchy_relation is None and dataset.num_relations > 1:
        warnings.warn(
            f"hierarchy_relation is None but the dataset has {dataset.num_relations} relations; all edges "
            "(regardless of relation) are treated as hierarchy edges. Pass hierarchy_relation to restrict "
            "the hierarchy to one relation.",
            stacklevel=4,
        )
    all_rows = _canonical_rows(dataset.training.mapped_triples, dataset, hierarchy_relation)
    paths = build_ancestor_paths(torch.as_tensor(all_rows), dataset.num_entities, hierarchy_relation)
    # Ganea et al. (2018 §5): basic edges = transitive reduction of the closure. Asserted redundant
    # (shortcut) edges are non-basic and must be eligible for the eval pool, not pinned into training.
    asserted = {(h, t) for h, r, t in all_rows if hierarchy_relation is None or r == hierarchy_relation}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(dataset.num_entities))
    graph.add_edges_from(asserted)
    if nx.is_directed_acyclic_graph(graph):
        # ponytail: transitive_reduction is ~O(V·E); run once per split, dwarfed by the closure
        # enumeration below. Fine to WN18RR/MeSH scale; revisit only if a split build becomes a bottleneck.
        direct = set(nx.transitive_reduction(graph).edges())
    else:
        warnings.warn(
            "hierarchy is not a DAG (contains a cycle); transitive reduction is undefined, falling back "
            "to asserted edges as basic edges",
            stacklevel=4,
        )
        direct = asserted
    # ponytail: enumerates the full transitive closure; fine up to MeSH scale (~200k pairs). Switch
    # to per-node descendant sampling without enumeration if WordNet-scale closures ever hurt.
    pool = sorted({(a, n) for n, ancestors in paths.items() for a in ancestors if a != n} - direct)
    return paths, all_rows, direct, pool


def _draw_negative(
    head: int,
    tail: int,
    relation: int,
    corrupt_head: bool,
    paths: Mapping[int, frozenset[int]],
    num_entities: int,
    rng: np.random.Generator,
    seen: set[tuple[int, int, int]],
) -> int | None:
    """Draw an entity replacing one slot of ``(head, tail)`` such that the pair leaves the closure.

    A tail candidate ``t'`` is valid iff ``head`` is not among its inclusive ancestors (covering
    both non-closure pairs and ``t' == head``); a head candidate ``h'`` is valid iff it is not
    among ``tail``'s inclusive ancestors. Candidates whose resulting row is already in ``seen`` are
    also rejected, so negatives are drawn without replacement per positive. Rejection-samples up to
    :data:`_MAX_ATTEMPTS_FACTOR` times, then falls back to an exact scan; returns ``None`` when no
    valid, not-yet-emitted entity exists.
    """

    def valid(entity: int) -> bool:
        if corrupt_head:
            return entity not in paths[tail] and (entity, relation, tail) not in seen
        return head not in paths[entity] and (head, relation, entity) not in seen

    for _ in range(_MAX_ATTEMPTS_FACTOR):
        candidate = int(rng.integers(num_entities))
        if valid(candidate):
            return candidate
    pool = [e for e in range(num_entities) if valid(e)]
    return int(rng.choice(pool)) if pool else None


def _sample_negatives(
    positives: Sequence[Sequence[int]],
    paths: Mapping[int, frozenset[int]],
    num_entities: int,
    rng: np.random.Generator,
    *,
    num_negatives: int = 1,
    siblings: Mapping[int, Sequence[int]] | None = None,
    two_sided: bool = False,
) -> list[list[int]]:
    """Sample ``num_negatives`` non-closure negatives per positive ``[h, r, t]``.

    By default only the tail is corrupted (``[h, r, t']``): the head entity is kept fixed
    (Bai et al. 2021), which blocks models from scoring high-level nodes uniformly high — note
    this reads "ancestor kept fixed" only for parent->child hierarchy relations (see issues.md,
    edge orientation). With ``two_sided=True`` the budget alternates between tail- and
    head-corruption (``[h', r, t]``), following Ganea et al. (2018 §5): five ``(u', v)`` and five
    ``(u, v')`` per positive at the default 10. Positives with no valid corruption left are
    skipped with a warning.

    When ``siblings`` is given (hard negatives, He et al. 2024 §4.1), a hard negative pairs an
    entity with its own sibling — the pairs hardest to tell apart. Hierarchy relations may point
    either way, so either end may be replaced by a sibling of the other end, keeping whichever
    corruption falls outside the closure; positives lacking enough valid siblings are topped up
    with random negatives to keep the positive:negative ratio.
    """
    negatives: list[list[int]] = []
    for head, relation, tail in positives:
        rows: list[list[int]] = []
        seen: set[tuple[int, int, int]] = set()  # emitted rows, to draw without replacement per positive
        if siblings is not None:
            hard = [[head, relation, s] for s in siblings.get(head, ()) if head not in paths[s]]
            hard += [[s, relation, tail] for s in siblings.get(tail, ()) if s not in paths[tail]]
            rng.shuffle(hard)
            for row in hard:
                if len(rows) >= num_negatives:
                    break
                if tuple(row) not in seen:
                    seen.add(tuple(row))
                    rows.append(row)
        while len(rows) < num_negatives:
            corrupt_head = two_sided and len(rows) % 2 == 1
            candidate = _draw_negative(head, tail, relation, corrupt_head, paths, num_entities, rng, seen)
            if candidate is None and two_sided:
                # one slot may be uncorruptable (e.g. the root as ancestor); use the other side
                corrupt_head = not corrupt_head
                candidate = _draw_negative(head, tail, relation, corrupt_head, paths, num_entities, rng, seen)
            if candidate is None:
                warnings.warn(
                    f"no valid negative corruption for pair ({head}, {tail}); skipping its remaining negatives",
                    stacklevel=2,
                )
                break
            row = [candidate, relation, tail] if corrupt_head else [head, relation, candidate]
            seen.add(tuple(row))
            rows.append(row)
        negatives.extend(rows)
    return negatives
