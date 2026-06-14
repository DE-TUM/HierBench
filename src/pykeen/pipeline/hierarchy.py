"""Ancestor-descendant prediction pipeline for hierarchical KG benchmarking."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import torch

from pykeen.datasets.base import Dataset
from pykeen.hpo import HpoPipelineResult, hpo_pipeline
from pykeen.models.nbase import ERModel
from pykeen.models.unimodal import PoincareE
from pykeen.triples import CoreTriplesFactory

from .api import PipelineResult, pipeline

__all__ = [
    "ancestor_descendant_pipeline",
    "hpo_ancestor_descendant_pipeline",
]


def ancestor_descendant_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    embedding_dim: int = 64,
    epochs: int = 100,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
    **pipeline_kwargs,
) -> PipelineResult:
    """Train on direct edges; evaluate on sampled ancestor-descendant pairs.

    Training flow based on Poincaré Embeddings for Learning Hierarchical Representations (Nickel & Kiela, NeurIPS 2017)
    and Hyperbolic Entailment Cones for Learning Hierarchical Embeddings (Ganea et al., ICML 2018).

    https://papers.nips.cc/paper_files/paper/2017/file/59dfa2df42d9e3d41f5b02bfc32229dd-Paper.pdf
    https://arxiv.org/abs/1804.01882


    Training is the regular graph (direct edges only). Held-out ancestor-descendant
    pairs are obtained by hop-balanced random-walk sampling rather than enumerating the
    transitive closure: a fixed budget ``num_pairs`` is split equally across the requested
    ``hops``, and each *k*-hop pair is produced by a length-*k* downward random walk. The
    sampled pairs are split 50/50 into validation and test, keeping the per-hop balance.

    :param dataset:
        A hierarchical dataset.
    :param model:
        Model class, string alias, or instance forwarded to :func:`pykeen.pipeline.pipeline`.
        When ``None`` (default), uses :class:`~pykeen.models.unimodal.PoincareE`
        with ``embedding_dim``.
    :param embedding_dim:
        Dimensionality of entity embeddings. Only used when ``model`` is ``None``.
    :param epochs:
        Number of training epochs.
    :param hops:
        Hop distances to sample ancestor-descendant pairs at. Default ``(2, 3, 4, 5)``.
    :param num_pairs:
        Total budget of held-out pairs, split equally across ``hops``. Default 400.
    :param seed:
        Random seed for reproducible sampling and val/test splits.
    :param pipeline_kwargs:
        Additional kwargs forwarded to :func:`pykeen.pipeline.pipeline`.

    :returns:
        A :class:`~pykeen.pipeline.PipelineResult` with the trained model and
        held-out evaluation metrics.
    """
    train_factory, val_factory, test_factory = _build_hierarchy_splits(
        dataset, hops=hops, num_pairs=num_pairs, seed=seed
    )
    resolved_model = model if model is not None else PoincareE
    model_kwargs = pipeline_kwargs.pop("model_kwargs", {})
    if model is None:
        model_kwargs.setdefault("embedding_dim", embedding_dim)
    return pipeline(
        training=train_factory,
        testing=test_factory,
        validation=val_factory,
        model=resolved_model,
        model_kwargs=model_kwargs,
        epochs=epochs,
        **pipeline_kwargs,
    )


def hpo_ancestor_descendant_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    embedding_dim: int = 64,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
    **hpo_kwargs,
) -> HpoPipelineResult:
    """Run HPO on the ancestor-descendant task.

    Applies the same split logic as :func:`ancestor_descendant_pipeline` — direct
    graph edges in training, hop-balanced random-walk sampled pairs split 50/50 into
    val/test — but delegates to :func:`pykeen.hpo.hpo_pipeline` so all Optuna HPO
    parameters (``n_trials``, ``model_kwargs_ranges``, ``timeout``, etc.) are accepted.

    :param dataset:
        A hierarchical dataset.
    :param model:
        Model class or string alias. Defaults to :class:`~pykeen.models.unimodal.PoincareE`.
    :param embedding_dim:
        Default embedding dimensionality used when ``model`` is ``None``.
    :param hops:
        Hop distances to sample ancestor-descendant pairs at. Default ``(2, 3, 4, 5)``.
    :param num_pairs:
        Total budget of held-out pairs, split equally across ``hops``. Default 400.
    :param seed:
        Random seed for reproducible sampling and splits.
    :param hpo_kwargs:
        Forwarded to :func:`pykeen.hpo.hpo_pipeline` (e.g. ``n_trials``,
        ``model_kwargs_ranges``, ``sampler``, ``storage``).

    :returns:
        A :class:`~pykeen.hpo.HpoPipelineResult`.
    """
    train_factory, val_factory, test_factory = _build_hierarchy_splits(
        dataset, hops=hops, num_pairs=num_pairs, seed=seed
    )
    resolved_model = model if model is not None else PoincareE
    model_kwargs = hpo_kwargs.pop("model_kwargs", {})
    if model is None:
        model_kwargs.setdefault("embedding_dim", embedding_dim)
    return hpo_pipeline(
        training=train_factory,
        testing=test_factory,
        validation=val_factory,
        model=resolved_model,
        model_kwargs=model_kwargs,
        **hpo_kwargs,
    )


#: Multiplier bounding sampling attempts per hop, so shallow/small graphs that cannot
#: supply the requested number of distinct k-hop pairs still terminate.
_MAX_ATTEMPTS_FACTOR = 100


def _build_hierarchy_splits(
    dataset: Dataset,
    hops: Sequence[int] = (2, 3, 4, 5),
    num_pairs: int = 400,
    seed: int = 42,
) -> tuple[CoreTriplesFactory, CoreTriplesFactory, CoreTriplesFactory]:
    """Build (train, val, test) triple factories for ancestor-descendant evaluation.

    Training set = the regular graph (direct edges only). Held-out ancestor-descendant
    pairs are sampled by hop-balanced random walks instead of enumerating the transitive
    closure: the ``num_pairs`` budget is split equally across ``hops``, and each *k*-hop
    pair is produced by a length-*k* downward random walk from a random start node. The
    sampled pairs are split 50/50 into validation and test, preserving the per-hop balance.

    :param dataset:
        The source dataset. Must expose ``training.mapped_triples``.
    :param hops:
        Hop distances to sample ancestor-descendant pairs at.
    :param num_pairs:
        Total budget of sampled pairs, split equally across ``hops`` (remainder distributed
        to the first hops).
    :param seed:
        Random seed for reproducible sampling and splits.

    :returns:
        A ``(train, val, test)`` tuple of :class:`~pykeen.triples.CoreTriplesFactory`
        instances sharing the same entity/relation vocabulary as the training set.
    """
    edge_list = [(h, t) for h, _, t in dataset.training.mapped_triples.tolist()]
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
