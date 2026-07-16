# LCA-based hierarchical P/R/F1 (Kosmopoulos et al. 2015) — assessment & plan

Status: **not implemented** — stored for later. This is the "LCA-based variant may
return later" from commit `1cba7804` (removal of the ancestor-augmented F_H evaluator).

Reference: Kosmopoulos et al. (2015), *Evaluation measures for hierarchical
classification: a unified view and novel approaches*, DMKD 29:820–865
(P_LCA / R_LCA / F_LCA, §2.4.2, Algorithm 1).

## Verdict

Not computable under the current pair-classification protocol; becomes meaningful
after reframing the evaluation per test descendant. Only worth adding if "how far
off are the errors" is a claim we want to make in the paper.

## Why it doesn't fit the current protocol directly

The transitive ancestor-descendant task is evaluated as pair classification:
held-out `(h, t)` pairs plus *sampled* negatives, thresholded into binary labels
(`src/pykeen/evaluation/pair_classification_evaluator.py`,
`_transitive_ancestor_descendant_pair_metrics`). LCA-P/R/F1 is defined over sets
of predicted vs. true *nodes* per instance — it needs `Ŷ` and `Y` in the hierarchy
to take LCAs of. A binary label on a single pair has no predicted-node/true-node
structure, and with sampled negatives there is no well-defined predicted set.

## The reframing that makes it work

Treat each test descendant `d` as one multi-label instance:

- `Y`  = its true transitive ancestor set (already computed in the split)
- `Ŷ` = `{a : score(a, d) ≥ threshold}` over **all** candidate ancestors, using
  the existing validation-tuned threshold

The paper's measures explicitly handle multi-label + DAG (covers MeSH). This is
also the principled fix for the exact failure that got F_H removed: full ancestor
augmentation inflates recall and rewards shallow (near-root) predictions; the LCA
variant only augments up to the lowest common ancestor with the nearest true node,
so near-root predictions get almost no credit (paper §3.2, Table 4 — matches the
`1cba7804` removal diagnosis).

## Caveats

1. **Protocol change, not a drop-in metric.** Requires a full-ranking pass —
   scoring each test descendant against all entities — instead of the sampled-
   negative pass. `|test descendants| × |E|` scores: fine for NASA/DOID,
   noticeable for WordNet/MeSH.
2. **`Y` is ancestor-closed**, an unusual regime for this metric (designed for
   small leaf-ish label sets). Behaves sensibly, but absolute values will sit
   close to plain set-F1 over the closure; the signal is in *how wrong* the
   false positives are.
3. **Cost/complexity.** Exact min-LCA over a DAG per (predicted, true) pair needs
   BFS + the paper's Algorithm 1 approximation; the authors capped path length on
   large hierarchies for compute reasons. Their C++ reference implementation
   (HEMKit, http://nlp.cs.aueb.gr/software_and_datasets/HEMKit.zip — check GitHub
   mirrors) could be shelled out to instead of reimplementing the graph code.
4. **Redundancy check.** We already report thresholded P/R/F1, mAP, AUROC, and a
   hard-negative (sibling) setting that probes error severity from another angle.
   If the hard-vs-random negative gap tells the same story, LCA-F1 adds a table
   column without adding a finding.

## Implementation order (if we do it)

1. Land `transitive_ancestor_descendant_raw_predictions_plan.md` first.
2. Build LCA-P/R/F1 as a **post-hoc consumer** of per-descendant full rankings
   (raw scores → thresholded `Ŷ` → LCA metrics), *not* wired into
   `PairClassificationMetricResults` — same reasoning as the raw-predictions
   plan: keep `MetricResults`/`.to_dict()` untouched.
3. Cap LCA search depth (paper's max-path threshold) for WordNet/MeSH.
