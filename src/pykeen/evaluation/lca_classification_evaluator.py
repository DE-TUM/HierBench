r"""LCA-based hierarchical classification metrics (precision / recall / F1).

This implements the lowest-common-ancestor measures :math:`P_{LCA}`, :math:`R_{LCA}` and
:math:`F_{LCA}` of Kosmopoulos et al. (2015), *Evaluation measures for hierarchical
classification: a unified view and novel approaches*
(https://doi.org/10.1007/s10618-014-0382-x), §2.4.2, for knowledge-graph link prediction. For a
query, the model's per-entity scores are turned into a predicted node set via the same
top-``|Y|`` rule used by :func:`pykeen.metrics.classification.construct_indicator` (where
``|Y|`` is the number of ground-truth positives). Instead of augmenting both sets with *all*
ancestors (which rewards shallow, near-root predictions), each node is only augmented up to its
lowest common ancestor with the nearest node of the other set (paper Definitions 1–6), the
augmentation subgraphs are minimized with the paper's Algorithm 1, and precision/recall/F1 are
computed on the node sets of the minimal graphs :math:`G_t` / :math:`G_p`:

.. math::

    P_{LCA} = \\frac{|\\hat{Y}_{aug} \\cap Y_{aug}|}{|\\hat{Y}_{aug}|}, \\quad
    R_{LCA} = \\frac{|\\hat{Y}_{aug} \\cap Y_{aug}|}{|Y_{aug}|}, \\quad
    F_{LCA} = \\frac{2 \\cdot P_{LCA} \\cdot R_{LCA}}{P_{LCA} + R_{LCA}}

As in the paper, the per-query scores are averaged over all queries, reported for each side
(``head`` / ``tail``) and combined (``both``).

.. note::

    The metric is only meaningful on a hierarchy (a DAG with well-defined ancestors). It is wired
    through :func:`pykeen.pipeline.transitive_ancestor_descendant.transitive_ancestor_descendant_lca_metrics`
    rather than the generic pipeline, so the (potentially large) hierarchy structures never travel
    through the result tracker.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from typing import ClassVar, NamedTuple, cast

import numpy as np
from docdata import parse_docdata

from .evaluator import Evaluator, MetricResults
from ..constants import TARGET_TO_INDEX
from ..metrics.classification import construct_indicator, safe_divide
from ..metrics.utils import Metric, ValueRange
from ..typing import SIDE_BOTH, ExtendedTarget, FloatTensor, MappedTriples, Target, normalize_target

__all__ = [
    "LCAClassificationEvaluator",
    "LCAMetricResults",
]

#: a triple of (LCA precision, LCA recall, LCA F1)
_LCAScore = tuple[float, float, float]

_UNIT_RANGE = ValueRange(lower=0.0, lower_inclusive=True, upper=1.0, upper_inclusive=True)


@parse_docdata
class LCAPrecision(Metric):
    """The LCA precision.

    ---
    description: Precision of the LCA-augmented predicted node set.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "LCA Precision"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class LCARecall(Metric):
    """The LCA recall.

    ---
    description: Recall of the LCA-augmented true node set.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "LCA Recall"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class LCAF1(Metric):
    """The LCA F1-score.

    ---
    description: Harmonic mean of LCA precision and recall.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "LCA F1"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


#: mapping from metric key to metric class
LCA_METRICS: Mapping[str, type[Metric]] = {
    "lca_precision": LCAPrecision,
    "lca_recall": LCARecall,
    "lca_f1": LCAF1,
}


class LCAMetricKey(NamedTuple):
    """A key for LCA classification metrics."""

    side: ExtendedTarget
    metric: str


class LCAMetricResults(MetricResults[LCAMetricKey]):
    """Results from computing LCA classification metrics."""

    metrics = LCA_METRICS

    # docstr-coverage: inherited
    @classmethod
    def key_from_string(cls, s: str | None) -> LCAMetricKey:  # noqa: D102
        if s is None:
            return LCAMetricKey(side=SIDE_BOTH, metric="lca_f1")
        # side?.metric
        parts = s.split(".")
        side = normalize_target(None if len(parts) < 2 else parts[0])
        metric = parts[-1]
        return LCAMetricKey(side=side, metric=metric)


class _LCAGraph:
    """The hierarchy-side computations behind the LCA measures (Kosmopoulos et al. 2015, §2.4.2).

    Costs between nodes are exact shortest up-down path lengths through a common ancestor
    (Definitions 2–3), computed from per-node reverse-BFS distances over the direct edges.
    """

    def __init__(
        self,
        edges: Collection[tuple[int, int]],
        ancestors: Mapping[int, frozenset[int]],
        max_paths: int = 64,
    ) -> None:
        self.ancestors = ancestors
        self.max_paths = max_paths
        self.parents: dict[int, list[int]] = defaultdict(list)
        for head, tail in edges:
            self.parents[tail].append(head)
        #: distance from each (proper or improper) ancestor down to the node, lazily built
        self._up_dist: dict[int, dict[int, int]] = {}

    def up_distances(self, node: int) -> dict[int, int]:
        """Return ``{ancestor: shortest directed path length ancestor -> node}`` (inclusive)."""
        cached = self._up_dist.get(node)
        if cached is not None:
            return cached
        dist = {node: 0}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for parent in self.parents.get(current, ()):
                if parent not in dist:
                    dist[parent] = dist[current] + 1
                    queue.append(parent)
        self._up_dist[node] = dist
        return dist

    def deepest(self, nodes: Collection[int]) -> list[int]:
        """Drop every node that has a *different* node of the set among its descendants (p. 840)."""
        node_set = set(nodes)
        return [n for n in node_set if not any(n in self.ancestors.get(m, frozenset()) for m in node_set - {n})]

    def lca(self, u: int, v: int) -> tuple[float, frozenset[int]]:
        """Return ``(cost, LCA(u, v))``: the min up-down path cost and its realizing common ancestors."""
        du, dv = self.up_distances(u), self.up_distances(v)
        common = du.keys() & dv.keys()
        if not common:
            return float("inf"), frozenset()
        best = min(du[c] + dv[c] for c in common)
        return float(best), frozenset(c for c in common if du[c] + dv[c] == best)

    def lca_node_set(self, node: int, others: Collection[int]) -> frozenset[int]:
        """LCA(n, S) of Definition 4: the union of LCAs with the cost-nearest members of ``others``."""
        pairs = [self.lca(node, other) for other in others]
        best = min((cost for cost, _ in pairs), default=float("inf"))
        if best == float("inf"):
            return frozenset()
        return frozenset().union(*(lcas for cost, lcas in pairs if cost == best))

    def best_paths(self, node: int, ancestor: int, selected: set[int]) -> list[int]:
        """Pick one shortest ``ancestor -> node`` path, preferring maximal overlap with ``selected``.

        Implements the path choice of Algorithm 1's ``GetBestPaths`` (the path "that shares most
        common nodes with all other selected paths", yielding the smallest possible subgraphs).
        """
        dist = self.up_distances(node)
        # backtrack from node towards ancestor along shortest-path predecessors only
        paths: list[list[int]] = []
        stack: list[list[int]] = [[node]]
        # ponytail: cap enumeration at max_paths; dense DAG regions (e.g. MeSH) can hold
        # exponentially many equal-length paths and any shortest one is a valid path_min choice
        while stack and len(paths) < self.max_paths:
            path = stack.pop()
            current = path[-1]
            if current == ancestor:
                paths.append(path)
                continue
            stack.extend(
                [*path, parent]
                for parent in self.parents.get(current, ())
                if dist.get(parent) == dist[current] + 1 and ancestor in self.ancestors.get(parent, frozenset())
            )
        if not paths:  # disconnected; keep the node itself
            return [node]
        return max(paths, key=lambda p: (len(set(p) & selected), p))


def _get_best_lcas(connections: Mapping[tuple[int, int], frozenset[int]]) -> set[int]:
    """Algorithm 1, ``GetBestLCAs``: a minimal LCA set covering every node of Y and Ŷ.

    ``connections`` maps each (node, side) of Y ∪ Ŷ to its LCA set w.r.t. the other side. LCAs
    are greedily accumulated in descending order of how many nodes they connect, then redundant
    ones are removed in a top-down and a bottom-up pass.
    """
    counts: dict[int, int] = defaultdict(int)
    for lcas in connections.values():
        for lca in lcas:
            counts[lca] += 1
    sorted_lcas = sorted(counts, key=lambda lca: (-counts[lca], lca))

    def satisfied(candidate: set[int]) -> bool:
        return all(not lcas or lcas & candidate for lcas in connections.values())

    best: set[int] = set()
    for lca in sorted_lcas:
        if satisfied(best):
            break
        best.add(lca)
    for ordering in (sorted_lcas, reversed(sorted_lcas)):
        for lca in ordering:
            if lca in best and satisfied(best - {lca}):
                best.discard(lca)
    return best


def _lca_scores(y_true: np.ndarray, y_score: np.ndarray, graph: _LCAGraph) -> _LCAScore | None:
    """Compute (P_LCA, R_LCA, F_LCA) for a single query, or ``None`` when there are no positives."""
    true_nodes = np.nonzero(y_true)[0]
    if true_nodes.size == 0:
        return None
    # predicted set = top-|Y| scored entities (same inference rule as the F1 evaluator)
    y_pred = construct_indicator(y_score=y_score, y_true=y_true)
    pred_nodes = np.nonzero(y_pred)[0]

    truth = graph.deepest(true_nodes.tolist())
    predicted = graph.deepest(pred_nodes.tolist())

    # Definitions 4–5: LCA(n, other set) per node; sides kept apart since a node may be in both
    connections: dict[tuple[int, int], frozenset[int]] = {
        **{(n, 0): graph.lca_node_set(n, predicted) for n in truth},
        **{(n, 1): graph.lca_node_set(n, truth) for n in predicted},
    }
    best_lcas = _get_best_lcas(connections)

    # Algorithm 1, GetBestPaths + graph split (Definition 6): each selected LCA -> node path
    # segment belongs to the graph of the side its node came from — G_t for true nodes, G_p for
    # predicted ones — so a predicted node lying on a true node's root path never certifies itself
    y_aug, y_hat_aug = set(truth), set(predicted)
    sides = (y_aug, y_hat_aug)
    selected: set[int] = set()
    for (node, side), lcas in connections.items():
        for lca in lcas & best_lcas:
            path = set(graph.best_paths(node, lca, selected))
            selected |= path
            sides[side].update(path)

    intersection = len(y_hat_aug & y_aug)
    precision = safe_divide(intersection, len(y_hat_aug), zero_division=0)
    recall = safe_divide(intersection, len(y_aug), zero_division=0)
    f1 = safe_divide(2 * precision * recall, precision + recall, zero_division=0)
    return precision, recall, f1


class LCAClassificationEvaluator(Evaluator[LCAMetricKey]):
    """An evaluator computing LCA-based precision/recall/F1 (Kosmopoulos et al. 2015, §2.4.2).

    :param edges: the direct hierarchy edges, in the dataset's own edge direction.
    :param ancestors: a mapping from entity id to its inclusive ancestor set in the same
        direction. Build it with :func:`pykeen.pipeline.hierarchical_helper.build_ancestor_paths`.
    """

    metric_result_cls = LCAMetricResults

    #: per-side, per-query (deduplicated) LCA scores
    scores: MutableMapping[Target, MutableMapping[tuple[int, int], _LCAScore]]

    def __init__(
        self,
        edges: Collection[tuple[int, int]] | None = None,
        ancestors: Mapping[int, frozenset[int]] | None = None,
        max_paths: int = 64,
        **kwargs,
    ):
        """Initialize the evaluator.

        :param edges: the direct hierarchy edges (used for LCA path extraction).
        :param ancestors: a mapping from entity id to its inclusive ancestor set.
        :param max_paths: bound on the number of equal-length paths enumerated per node/LCA pair.
        :param kwargs: keyword-based parameters passed to :meth:`Evaluator.__init__`.

        :raises ValueError: if ``edges`` or ``ancestors`` is not provided.
        """
        # filtered=False so scores stay unmasked and the positive mask reflects only the
        # evaluation triples (the per-query held-out targets); requires_positive_mask gives us Y.
        super().__init__(filtered=False, requires_positive_mask=True, **kwargs)
        if edges is None or ancestors is None:
            raise ValueError("LCA evaluator requires the hierarchy `edges` and an `ancestors` map")
        self.graph = _LCAGraph(edges=edges, ancestors=ancestors, max_paths=max_paths)
        self.scores = defaultdict(dict)

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: FloatTensor,
        true_scores: FloatTensor | None = None,
        dense_positive_mask: FloatTensor | None = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError("LCA evaluator needs the positive mask!")

        scores_np = scores.detach().cpu().numpy()
        mask_np = dense_positive_mask.detach().cpu().numpy()
        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Deduplicate per (h, r) / (r, t) query so a query with several held-out targets is counted
        # once: every such row carries the same positive mask, hence the same LCA score.
        for i in range(keys.shape[0]):
            key = cast(tuple[int, int], tuple(map(int, keys[i])))
            if key in self.scores[target]:
                continue
            result = _lca_scores(y_true=mask_np[i], y_score=scores_np[i], graph=self.graph)
            if result is None:
                continue
            self.scores[target][key] = result

    # docstr-coverage: inherited
    def clear(self) -> None:  # noqa: D102
        self.scores.clear()

    @staticmethod
    def _aggregate(side: ExtendedTarget, values: Sequence[_LCAScore]) -> dict[LCAMetricKey | str, float]:
        """Average the per-query scores for one side into the three metric keys."""
        if not values:
            means = (0.0, 0.0, 0.0)
        else:
            means = cast(_LCAScore, tuple(float(np.mean(column)) for column in zip(*values, strict=True)))
        return {
            LCAMetricKey(side=side, metric=metric): value
            for metric, value in zip(LCA_METRICS, means, strict=True)
        }

    # docstr-coverage: inherited
    def finalize(self) -> LCAMetricResults:  # noqa: D102
        data: dict[LCAMetricKey | str, float] = {}
        all_values: list[_LCAScore] = []
        for target in sorted(self.scores):
            values = list(self.scores[target].values())
            all_values.extend(values)
            data.update(self._aggregate(normalize_target(target), values))
        data.update(self._aggregate(SIDE_BOTH, all_values))
        self.clear()
        return LCAMetricResults(data=data)
