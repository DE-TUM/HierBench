# Persist raw transitive ancestor-descendant predictions (ground truth + scores)

## Context

`benchmark_transitive_ancestor_descendant.py` currently only persists aggregated scalar metrics
(precision/recall/F1/mAP/AUROC/etc.) per run under `results/transitive-ancestor-descendant/`. The
per-pair ground truth labels, raw model scores, and thresholded predictions
that produce those metrics are computed inside
`pykeen/pipeline/transitive_ancestor_descendant.py::_transitive_ancestor_descendant_pair_metrics` but discarded once
the aggregate `PairClassificationMetricResults` is built. The user wants those
raw per-pair values persisted alongside the existing summary so individual
predictions can be inspected later (error analysis, plotting, etc.).

## Approach

Keep raw predictions fully opt-in (`return_raw: bool = False`, default off) so
every existing caller (`test_transitive_ancestor_descendant.py`, any other code using these
functions) is unaffected. Do not add fields to
`PairClassificationMetricResults`/`pair_classification_evaluator.py` — that
class's `.to_dict()` is used for the pipeline's own JSON-serialized results,
and stuffing numpy arrays into it would break that. Raw predictions travel
as a plain `dict` alongside the metrics, never through `MetricResults`.

### 1. `src/pykeen/pipeline/transitive_ancestor_descendant.py`

- `_transitive_ancestor_descendant_pair_metrics`: change the inner `_score(rows)` closure to also
  return the batch it scored (`batch.tolist()`), i.e.
  `tuple[np.ndarray, np.ndarray, list[list[int]]]` (scores, labels, rows).
  Add a `return_raw: bool = False` parameter. When `True`, after computing
  `predictions = test_scores >= threshold`, build
  `raw = {"rows": test_rows, "labels": test_labels.tolist(), "scores": test_scores.tolist(), "predictions": predictions.tolist(), "threshold": threshold}`
  and return `(PairClassificationMetricResults(...), raw)` instead of just the
  metric results. When `False` (default), behavior and return type are
  unchanged.
- `transitive_ancestor_descendant_prediction_metrics`: add `return_raw: bool = False`, forward to
  `_transitive_ancestor_descendant_pair_metrics`, mirror its conditional return
  (`PairClassificationMetricResults | None` vs.
  `tuple[PairClassificationMetricResults | None, dict | None]`).
- `transitive_ancestor_descendant_prediction_pipeline`: add `return_raw: bool = False`, forward
  to the `_transitive_ancestor_descendant_pair_metrics` call. When raw data comes back, stash it
  on a new field on the result (see below) instead of changing
  `transitive_ancestor_descendant_prediction_pipeline`'s return type (it already returns a
  `HierarchicalPipelineResult`, adding a field is the natural extension
  point).

### 2. `src/pykeen/pipeline/hierarchical_helper.py`

- `HierarchicalPipelineResult`: add
  `ancestor_descendant_raw_predictions: dict | None = None`. Do **not** add it
  to `_get_results()` — it must not leak into the pipeline's own
  `to_dict()`/serialized-results contract, it's purely a carry field for
  callers that opted in via `return_raw=True`.

### 3. `benchmark_transitive_ancestor_descendant.py`

- `_evaluate_config`: call `transitive_ancestor_descendant_prediction_pipeline(..., return_raw=True)`
  and read `result.ancestor_descendant_raw_predictions` for the random-negative
  raw data. Call `transitive_ancestor_descendant_prediction_metrics(..., hard_negatives=True, return_raw=True)`
  and unpack `(hard_scores, hard_raw)`. Return `(scores, {"rnd": rnd_raw, "hrd": hard_raw})`
  from `_evaluate_config` (both `None` entries preserved as `None` when a pass
  produced no metrics).
- `main()` / `_write_run_result`: thread the raw dict through and write it to
  a **separate** file, `results/transitive-ancestor-descendant/<dataset>/<safe_label>.predictions.json`,
  so the existing `<safe_label>.json` metrics format and `summary.csv` stay
  byte-for-byte unchanged for any existing consumers. Skip writing this file
  when the run failed (`error is not None`) or raw is empty for both keys.

## Verification

- Run `uv run python -c "..."` (or a short ad-hoc script) invoking
  `transitive_ancestor_descendant_prediction_pipeline(..., return_raw=True)` and
  `transitive_ancestor_descendant_prediction_metrics(..., return_raw=True)` on a small dataset
  (e.g. `datasets.NASA`, few epochs) and assert: `len(raw["rows"]) == len(raw["labels"]) == len(raw["scores"]) == len(raw["predictions"])`,
  and `raw["predictions"][i] == (raw["scores"][i] >= raw["threshold"])`.
- Run the existing `test_transitive_ancestor_descendant.py` (or just import-check) to confirm
  `return_raw=False` (the default) still returns the old types unchanged —
  no signature regressions for existing callers.
- Run `benchmark_transitive_ancestor_descendant.py` for one dataset/config (temporarily trim
  `DATASET_CLASSES`/`configs`, or just eyeball the loop) and confirm both
  `<label>.json` and the new `<label>.predictions.json` appear under
  `results/transitive-ancestor-descendant/<dataset>/`, and that the JSON is well-formed and
  round-trips through `json.loads`.
- `uvx ruff check` on the touched files.
