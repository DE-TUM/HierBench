"""Tests for the LCA-based hierarchical classification evaluator (Kosmopoulos et al. 2015)."""

import numpy as np
import pytest
import torch

from pykeen.evaluation import LCAClassificationEvaluator, LCAMetricResults
from pykeen.evaluation.lca_classification_evaluator import _lca_scores, _LCAGraph
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, SIDE_BOTH


def _ancestor_closure(edges: list[tuple[int, int]], num_nodes: int) -> dict[int, frozenset[int]]:
    """Compute the inclusive ancestor sets of a small DAG by fixpoint iteration."""
    ancestors = {n: {n} for n in range(num_nodes)}
    changed = True
    while changed:
        changed = False
        for head, tail in edges:
            new = ancestors[tail] | ancestors[head]
            if new != ancestors[tail]:
                ancestors[tail] = new
                changed = True
    return {n: frozenset(s) for n, s in ancestors.items()}


def _score_vector(num_nodes: int, predicted: list[int]) -> np.ndarray:
    """Build a score vector whose top entries are exactly the given nodes."""
    scores = np.full(num_nodes, -10.0)
    scores[predicted] = 1.0
    return scores


def _label_vector(num_nodes: int, truth: list[int]) -> np.ndarray:
    """Build a 0/1 label vector for the given true nodes."""
    labels = np.zeros(num_nodes)
    labels[truth] = 1.0
    return labels


# Fig. 8b of Kosmopoulos et al. (2015); ids:
# 0=0, 1=1, 2=2, 3=3, 4=1.1, 5=1.2, 6=2.1, 7=3.1, 8=3.2, 9=3.3, 10=3.2.1, 11=3.2.2
PAPER_DAG_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 6),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (8, 10),
    (8, 11),
]
PAPER_DAG_NUM_NODES = 12


@pytest.fixture
def paper_graph() -> _LCAGraph:
    """Build the paper's Fig. 8b DAG as an ``_LCAGraph``."""
    return _LCAGraph(edges=PAPER_DAG_EDGES, ancestors=_ancestor_closure(PAPER_DAG_EDGES, PAPER_DAG_NUM_NODES))


def test_paper_worked_example(paper_graph):
    """Reproduce the paper's Fig. 8b example: the minimal LCA graphs of Fig. 10 give P=R=F1=0.5."""
    # Y = {2.1, 3.2.1, 3.3}, Yhat = {3.1, 3.2.1, 3.2.2}
    y_true = _label_vector(PAPER_DAG_NUM_NODES, [6, 10, 9])
    y_score = _score_vector(PAPER_DAG_NUM_NODES, [7, 10, 11])
    assert _lca_scores(y_true=y_true, y_score=y_score, graph=paper_graph) == (0.5, 0.5, 0.5)


def test_perfect_prediction(paper_graph):
    """Predicting exactly the true nodes yields 1.0 across all three measures."""
    y_true = _label_vector(PAPER_DAG_NUM_NODES, [6, 10, 9])
    y_score = _score_vector(PAPER_DAG_NUM_NODES, [6, 10, 9])
    assert _lca_scores(y_true=y_true, y_score=y_score, graph=paper_graph) == (1.0, 1.0, 1.0)


def test_no_positives_returns_none(paper_graph):
    """A query without positives cannot be scored."""
    result = _lca_scores(
        y_true=np.zeros(PAPER_DAG_NUM_NODES), y_score=np.ones(PAPER_DAG_NUM_NODES), graph=paper_graph
    )
    assert result is None


def test_deredundancy(paper_graph):
    """A true set containing a node and its ancestor scores as if the ancestor were absent (p. 840)."""
    y_score = _score_vector(PAPER_DAG_NUM_NODES, [10])
    without_ancestor = _lca_scores(
        y_true=_label_vector(PAPER_DAG_NUM_NODES, [10]), y_score=y_score, graph=paper_graph
    )
    # 3.2 and 3 are ancestors of 3.2.1; note |Y| changes the top-|Y| cut, so fix Yhat via scores
    with_ancestors = _lca_scores(
        y_true=_label_vector(PAPER_DAG_NUM_NODES, [10, 8, 3]),
        y_score=_score_vector(PAPER_DAG_NUM_NODES, [10, 8, 3]),
        graph=paper_graph,
    )
    assert without_ancestor == (1.0, 1.0, 1.0)
    assert with_ancestors == (1.0, 1.0, 1.0)


def test_near_root_prediction_penalized():
    """A near-root prediction scores lower than a near-miss — the property plain hierarchical F1 lacked."""
    # chain 0 -> 1 -> 2 -> 3 with an extra branch 0 -> 4
    edges = [(0, 1), (1, 2), (2, 3), (0, 4)]
    graph = _LCAGraph(edges=edges, ancestors=_ancestor_closure(edges, 5))
    y_true = _label_vector(5, [3])
    near_miss = _lca_scores(y_true=y_true, y_score=_score_vector(5, [2]), graph=graph)
    near_root = _lca_scores(y_true=y_true, y_score=_score_vector(5, [0]), graph=graph)
    assert near_miss[2] > near_root[2]


def test_evaluator_requires_hierarchy():
    """The evaluator refuses to run without the hierarchy structures."""
    with pytest.raises(ValueError, match="edges"):
        LCAClassificationEvaluator()


def test_evaluator_finalize_and_dedup():
    """Process scores on both sides, deduplicate repeated queries, and produce per-side results."""
    edges = [(0, 1), (1, 2), (0, 3)]
    evaluator = LCAClassificationEvaluator(edges=edges, ancestors=_ancestor_closure(edges, 4))
    hrt_batch = torch.as_tensor([[1, 0, 2], [1, 0, 2]])  # duplicate query
    scores = torch.rand(2, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 2] = True
    for target in (LABEL_HEAD, LABEL_TAIL):
        evaluator.process_scores_(hrt_batch=hrt_batch, target=target, scores=scores, dense_positive_mask=mask)
        assert len(evaluator.scores[target]) == 1
    result = evaluator.finalize()
    assert isinstance(result, LCAMetricResults)
    sides = {side for side, _metric in result.data}
    assert sides == {LABEL_HEAD, LABEL_TAIL, SIDE_BOTH}
    assert all(0.0 <= value <= 1.0 for value in result.data.values())
    # finalize clears the accumulated scores
    assert not evaluator.scores


def test_evaluator_requires_positive_mask():
    """The evaluator raises when the dense positive mask is missing."""
    edges = [(0, 1)]
    evaluator = LCAClassificationEvaluator(edges=edges, ancestors=_ancestor_closure(edges, 2))
    with pytest.raises(KeyError, match="positive mask"):
        evaluator.process_scores_(
            hrt_batch=torch.as_tensor([[0, 0, 1]]), target=LABEL_TAIL, scores=torch.rand(1, 2)
        )
