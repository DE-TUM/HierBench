"""Hierarchy completion — predict direct hierarchy edges removed from training.

The public surface mirrors PyKEEN's function-style pipeline API:

* :func:`hierarchy_completion_split` is pure data preparation — it returns standard
  ``(train, val, test)`` :class:`~pykeen.triples.CoreTriplesFactory` instances, so it composes with
  :func:`pykeen.pipeline.pipeline`, :func:`pykeen.hpo.hpo_pipeline`, or a hand-rolled training loop.
* :func:`hierarchy_completion_pipeline` is the one-call convenience that trains and additionally
  reports task-specific metrics.

The shared internals live in :mod:`pykeen.pipeline.hierarchical_helper`; the sibling task is
:mod:`pykeen.pipeline.transitive_ancestor_descendant` (Ganea et al. 2018; He et al. 2024).
"""

from __future__ import annotations

import warnings

import numpy as np

from .hierarchical_helper import (
    HierarchicalPipelineResult,
    HpoHierarchicalResult,
    _default_hierarchy_sampler,
    _factory_from_rows,
    _hpo_and_refit,
    _train_and_score_hierarchical,
)
from ..datasets.base import Dataset
from ..datasets.metadata import resolve_hierarchy_relation
from ..models.nbase import ERModel
from ..triples import CoreTriplesFactory

__all__ = [
    "hierarchy_completion_split",
    "hierarchy_completion_pipeline",
    "hpo_hierarchy_completion_pipeline",
]


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
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
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
    hierarchy_relation: int | str | None = None,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Remove direct hierarchy edges, train on the rest, and predict the removed edges.

    Convenience wrapper that mirrors :func:`pykeen.pipeline.pipeline`: it builds the split with
    :func:`hierarchy_completion_split` and trains and scores the model on it.

    :param dataset: A hierarchical dataset.
    :param model: Model class, string alias, or instance forwarded to
        :func:`pykeen.pipeline.pipeline`. When ``None``, uses
        :class:`~pykeen.models.unimodal.PoincareE` with ``embedding_dim``.
    :param embedding_dim: Dimensionality of entity embeddings. Only used when ``model`` is ``None``.
    :param epochs: Number of training epochs.
    :param test_ratio: Fraction of removable hierarchy edges to hold out. Default 0.1.
    :param seed: Random seed for reproducible removal and val/test splits.
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param pipeline_kwargs: Additional kwargs forwarded to :func:`pykeen.pipeline.pipeline`. Under
        sLCWA the negative sampler defaults to :class:`~pykeen.sampling.HierarchyNegativeSampler`
        (same-depth hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` to override.

    :returns: A :class:`~pykeen.pipeline.hierarchical_helper.HierarchicalPipelineResult`.
    """
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
    train_factory, val_factory, test_factory = hierarchy_completion_split(
        dataset, test_ratio=test_ratio, seed=seed, hierarchy_relation=hierarchy_relation
    )
    pipeline_kwargs = _default_hierarchy_sampler(pipeline_kwargs, hierarchy_relation)
    return _train_and_score_hierarchical(
        train_factory,
        val_factory,
        test_factory,
        model=model,
        embedding_dim=embedding_dim,
        epochs=epochs,
        **pipeline_kwargs,
    )


def hpo_hierarchy_completion_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    test_ratio: float = 0.1,
    seed: int = 42,
    hierarchy_relation: int | str | None = None,
    **hpo_kwargs,
) -> HpoHierarchicalResult:
    """Run HPO on the hierarchy-completion task, then re-fit and score the best trial.

    Builds the split with :func:`hierarchy_completion_split`, runs :func:`pykeen.hpo.hpo_pipeline`
    over it, and re-trains the winning configuration on the same split.

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param model: Model class, string alias, or ``None`` (defaults to
        :class:`~pykeen.models.unimodal.PoincareE`), forwarded to :func:`pykeen.hpo.hpo_pipeline`.
    :param test_ratio: Fraction of removable hierarchy edges to hold out. Default 0.1.
    :param seed: Random seed for reproducible removal and val/test splits.
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param hpo_kwargs: Additional kwargs forwarded to :func:`pykeen.hpo.hpo_pipeline`. Under sLCWA the
        negative sampler defaults to :class:`~pykeen.sampling.HierarchyNegativeSampler` (same-depth
        hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` to override.

    :returns: A :class:`~pykeen.pipeline.hierarchical_helper.HpoHierarchicalResult`.
    """
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
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
        **hpo_kwargs,
    )
