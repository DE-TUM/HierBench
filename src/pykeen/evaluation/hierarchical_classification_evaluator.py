r"""Hierarchical classification metrics (precision / recall / F1) over ancestor paths.

This adapts the set-based hierarchical metrics of Kosmopoulos et al. (2015), *Evaluation measures
for hierarchical classification: a unified view and novel approaches*
(https://doi.org/10.1007/s10618-014-0382-x), to knowledge-graph link prediction. For a query, the
model's per-entity scores are turned into a predicted node set via the same top-``|Y|`` rule used
by :func:`pykeen.metrics.classification.construct_indicator` (where ``|Y|`` is the number of
ground-truth positives). Both the predicted set and the ground-truth set are augmented with the
ancestors (root-paths) of their nodes, and hierarchical precision/recall/F1 are computed on the
augmented sets:

.. math::

    hP = \\frac{|\\hat{Y}_{aug} \\cap Y_{aug}|}{|\\hat{Y}_{aug}|}, \\quad
    hR = \\frac{|\\hat{Y}_{aug} \\cap Y_{aug}|}{|Y_{aug}|}, \\quad
    hF_1 = \\frac{2 \\cdot hP \\cdot hR}{hP + hR}

As in the paper, the per-query scores are averaged over all queries, reported for each side
(``head`` / ``tail``) and combined (``both``).

.. note::

    The metric is only meaningful on a hierarchy (a DAG with well-defined ancestors). It is wired
    through :func:`pykeen.pipeline.hierarchy.hierarchy_completion_pipeline` rather than the generic
    pipeline, so the (potentially large) ``ancestors`` map never travels through the result tracker.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from typing import ClassVar, NamedTuple, cast

import numpy as np
from docdata import parse_docdata

from .evaluator import Evaluator, MetricResults
from ..constants import TARGET_TO_INDEX
from ..metrics.classification import construct_indicator, safe_divide
from ..metrics.utils import Metric, ValueRange
from ..typing import SIDE_BOTH, ExtendedTarget, FloatTensor, MappedTriples, Target, normalize_target

__all__ = [
    "HierarchicalClassificationEvaluator",
    "HierarchicalMetricResults",
]

#: a triple of (hierarchical precision, hierarchical recall, hierarchical F1)
_HScore = tuple[float, float, float]

_UNIT_RANGE = ValueRange(lower=0.0, lower_inclusive=True, upper=1.0, upper_inclusive=True)


@parse_docdata
class HierarchicalPrecision(Metric):
    """The hierarchical precision.

    ---
    description: Ancestor-augmented precision of the predicted node set.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "Hierarchical Precision"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class HierarchicalRecall(Metric):
    """The hierarchical recall.

    ---
    description: Ancestor-augmented recall of the predicted node set.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "Hierarchical Recall"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class HierarchicalF1(Metric):
    """The hierarchical F1-score.

    ---
    description: Harmonic mean of hierarchical precision and recall.
    link: https://doi.org/10.1007/s10618-014-0382-x
    """

    name: ClassVar[str] = "Hierarchical F1"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


#: mapping from metric key to metric class
HIERARCHICAL_METRICS: Mapping[str, type[Metric]] = {
    "hierarchical_precision": HierarchicalPrecision,
    "hierarchical_recall": HierarchicalRecall,
    "hierarchical_f1": HierarchicalF1,
}


class HierarchicalMetricKey(NamedTuple):
    """A key for hierarchical classification metrics."""

    side: ExtendedTarget
    metric: str


class HierarchicalMetricResults(MetricResults[HierarchicalMetricKey]):
    """Results from computing hierarchical classification metrics."""

    metrics = HIERARCHICAL_METRICS

    # docstr-coverage: inherited
    @classmethod
    def key_from_string(cls, s: str | None) -> HierarchicalMetricKey:  # noqa: D102
        if s is None:
            return HierarchicalMetricKey(side=SIDE_BOTH, metric="hierarchical_f1")
        # side?.metric
        parts = s.split(".")
        side = normalize_target(None if len(parts) < 2 else parts[0])
        metric = parts[-1]
        return HierarchicalMetricKey(side=side, metric=metric)


def _augment(nodes: np.ndarray, ancestors: Mapping[int, frozenset[int]]) -> set[int]:
    """Union the inclusive ancestor paths of the given nodes."""
    augmented: set[int] = set()
    for node in nodes.tolist():
        augmented |= ancestors.get(node, frozenset({node}))
    return augmented


def _hierarchical_scores(
    y_true: np.ndarray, y_score: np.ndarray, ancestors: Mapping[int, frozenset[int]]
) -> _HScore | None:
    """Compute (hP, hR, hF1) for a single query, or ``None`` when there are no positives."""
    true_nodes = np.nonzero(y_true)[0]
    if true_nodes.size == 0:
        return None
    # predicted set = top-|Y| scored entities (same inference rule as the F1 evaluator)
    y_pred = construct_indicator(y_score=y_score, y_true=y_true)
    pred_nodes = np.nonzero(y_pred)[0]

    y_aug = _augment(true_nodes, ancestors)
    y_hat_aug = _augment(pred_nodes, ancestors)
    intersection = len(y_hat_aug & y_aug)
    h_precision = safe_divide(intersection, len(y_hat_aug), zero_division=0)
    h_recall = safe_divide(intersection, len(y_aug), zero_division=0)
    h_f1 = safe_divide(2 * h_precision * h_recall, h_precision + h_recall, zero_division=0)
    return h_precision, h_recall, h_f1


class HierarchicalClassificationEvaluator(Evaluator[HierarchicalMetricKey]):
    """An evaluator computing hierarchical precision/recall/F1 over ancestor paths.

    :param ancestors: a mapping from entity id to its inclusive ancestor set. Build it
        with :func:`pykeen.pipeline.hierarchy.build_ancestor_paths`.
    """

    metric_result_cls = HierarchicalMetricResults

    #: per-side, per-query (deduplicated) hierarchical scores
    scores: MutableMapping[Target, MutableMapping[tuple[int, int], _HScore]]

    def __init__(self, ancestors: Mapping[int, frozenset[int]] | None = None, **kwargs):
        """Initialize the evaluator.

        :param ancestors: a mapping from entity id to its inclusive ancestor set.
        :param kwargs: keyword-based parameters passed to :meth:`Evaluator.__init__`.

        :raises ValueError: if no ``ancestors`` map is provided.
        """
        # filtered=False so scores stay unmasked and the positive mask reflects only the
        # evaluation triples (the per-query held-out targets); requires_positive_mask gives us Y.
        super().__init__(filtered=False, requires_positive_mask=True, **kwargs)
        if ancestors is None:
            raise ValueError("hierarchical evaluator requires an `ancestors` map")
        self.ancestors = ancestors
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
            raise KeyError("Hierarchical evaluator needs the positive mask!")

        scores_np = scores.detach().cpu().numpy()
        mask_np = dense_positive_mask.detach().cpu().numpy()
        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Deduplicate per (h, r) / (r, t) query so a query with several held-out targets is counted
        # once: every such row carries the same positive mask, hence the same hierarchical score.
        for i in range(keys.shape[0]):
            result = _hierarchical_scores(y_true=mask_np[i], y_score=scores_np[i], ancestors=self.ancestors)
            if result is None:
                continue
            key = cast(tuple[int, int], tuple(map(int, keys[i])))
            self.scores[target][key] = result

    # docstr-coverage: inherited
    def clear(self) -> None:  # noqa: D102
        self.scores.clear()

    @staticmethod
    def _aggregate(side: ExtendedTarget, values: Sequence[_HScore]) -> dict[HierarchicalMetricKey | str, float]:
        """Average the per-query scores for one side into the three metric keys."""
        if not values:
            means = (0.0, 0.0, 0.0)
        else:
            means = cast(_HScore, tuple(float(np.mean(column)) for column in zip(*values, strict=True)))
        return {
            HierarchicalMetricKey(side=side, metric=metric): value
            for metric, value in zip(HIERARCHICAL_METRICS, means, strict=True)
        }

    # docstr-coverage: inherited
    def finalize(self) -> HierarchicalMetricResults:  # noqa: D102
        data: dict[HierarchicalMetricKey | str, float] = {}
        all_values: list[_HScore] = []
        for target in sorted(self.scores):
            values = list(self.scores[target].values())
            all_values.extend(values)
            data.update(self._aggregate(normalize_target(target), values))
        data.update(self._aggregate(SIDE_BOTH, all_values))
        self.clear()
        return HierarchicalMetricResults(data=data)
