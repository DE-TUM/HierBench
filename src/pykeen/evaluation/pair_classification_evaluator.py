"""Pairwise classification metrics for ancestor-descendant / subsumption prediction.

Wraps the scores of :func:`pykeen.pipeline.subsumption.subsumption_prediction_pipeline` — held-out ``(h, t)`` pairs
scored against sampled negatives — in the same :class:`~pykeen.evaluation.evaluator.MetricResults`
contract used by
:class:`~pykeen.evaluation.hierarchical_classification_evaluator.HierarchicalMetricResults`, so all
hierarchical-task metrics expose the same ``.to_dict()``/``.get_metric()`` API.

mAP and AUROC reuse pykeen's existing :class:`~pykeen.metrics.classification.AveragePrecisionScore`
and :class:`~pykeen.metrics.classification.AreaUnderTheReceiverOperatingCharacteristicCurve`.
Precision/recall/F1/threshold (subsumption only) get their own metadata-only
:class:`~pykeen.metrics.utils.Metric` placeholders because they are computed with a
validation-tuned score threshold rather than pykeen's built-in top-``|Y|`` binarization
(:func:`pykeen.metrics.classification.construct_indicator`) — the same reason
:class:`~pykeen.evaluation.hierarchical_classification_evaluator.HierarchicalPrecision` and its
siblings don't reuse the confusion-matrix metrics either.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, NamedTuple

from docdata import parse_docdata

from .evaluator import MetricResults
from ..metrics.classification import AreaUnderTheReceiverOperatingCharacteristicCurve, AveragePrecisionScore
from ..metrics.utils import Metric, ValueRange

__all__ = [
    "PairClassificationMetricKey",
    "PairClassificationMetricResults",
]

_UNIT_RANGE = ValueRange(lower=0.0, lower_inclusive=True, upper=1.0, upper_inclusive=True)


@parse_docdata
class ThresholdedPrecision(Metric):
    """Precision at the validation-tuned score threshold.

    ---
    description: Precision of the model's is-a predictions at the F1-optimal threshold tuned on validation.
    link: https://arxiv.org/abs/1804.01882
    """

    name: ClassVar[str] = "Thresholded Precision"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class ThresholdedRecall(Metric):
    """Recall at the validation-tuned score threshold.

    ---
    description: Recall of the model's is-a predictions at the F1-optimal threshold tuned on validation.
    link: https://arxiv.org/abs/1804.01882
    """

    name: ClassVar[str] = "Thresholded Recall"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class ThresholdedF1Score(Metric):
    """F1 at the validation-tuned score threshold.

    ---
    description: Harmonic mean of thresholded precision and recall.
    link: https://arxiv.org/abs/1804.01882
    """

    name: ClassVar[str] = "Thresholded F1"
    value_range: ClassVar[ValueRange] = _UNIT_RANGE
    increasing: ClassVar[bool] = True


@parse_docdata
class Threshold(Metric):
    """The F1-optimal score threshold tuned on validation.

    ---
    description: The pair-score cutoff, tuned on validation, applied to test predictions.
    link: https://arxiv.org/abs/1804.01882
    """

    name: ClassVar[str] = "Threshold"
    value_range: ClassVar[ValueRange] = ValueRange()
    increasing: ClassVar[bool] = False


#: mapping from metric key to metric class
PAIR_CLASSIFICATION_METRICS: Mapping[str, type[Metric]] = {
    "average_precision": AveragePrecisionScore,
    "roc_auc": AreaUnderTheReceiverOperatingCharacteristicCurve,
    "precision": ThresholdedPrecision,
    "recall": ThresholdedRecall,
    "f1": ThresholdedF1Score,
    "threshold": Threshold,
}


class PairClassificationMetricKey(NamedTuple):
    """A key for pairwise ancestor-descendant / subsumption classification metrics."""

    metric: str


class PairClassificationMetricResults(MetricResults[PairClassificationMetricKey]):
    """Results from scoring held-out ancestor-descendant / subsumption pairs against negatives."""

    metrics = PAIR_CLASSIFICATION_METRICS

    # docstr-coverage: inherited
    @classmethod
    def key_from_string(cls, s: str | None) -> PairClassificationMetricKey:  # noqa: D102
        return PairClassificationMetricKey(metric=s if s is not None else "average_precision")
