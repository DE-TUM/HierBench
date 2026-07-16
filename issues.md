# Known issues — transitive ancestor-descendant pipeline & benchmark

Findings from a strict review of the multi-hop transitive ancestor-descendant pipeline against Ganea et al. (2018,
Hyperbolic Entailment Cones, §5) and He et al. (2024). **Fixed so far:** two-sided evaluation
negatives, `NUM_NEGATIVES=10`, equal per-model training budgets (`TRAIN_NEGATIVES`), edge
orientation (§2), transitive-reduction basic edges (§3), α tuned on validation (§4, grid later
extended to the α=1000 regime — see §11), Poincaré burn-in (§5), deduped evaluation negatives (§6),
HyperbolicCones training/scoring (§11). The remaining items (§1, §7–§10) are open. Results
produced before these fixes are not comparable and must be rerun.

## 1. Seed variance (deferred by decision)

Every benchmark cell is a single run with seed 42 (split, negative draws, init, training). No
error bars, so model differences within a few F1 points are not interpretable.

**Recommended fix:** N full retrains per configuration with seeds 42+i, report mean±std.
Eval-only reseeding (retrain once, redraw evaluation negatives) is insufficient — it misses
training variance, usually the dominant source. If eval-only reseeding is ever used with
`closure_ratio > 0`, the split seed must be decoupled from the negative-sampling seed, otherwise
a reseeded test set can overlap the original run's extra training pairs (leakage).

## 2. Edge orientation (FIXED)

The ontology datasets' hierarchy relations (`narrower`, `has_subclass`) point parent→child, but
WN18RR's `_hypernym` points child→parent — WN18RR has no parent→child is-a relation at all
(`_hyponym` was removed when WN18 was reduced to WN18RR).

**Implemented fix:** `HierarchicalGraph.hierarchy_inverted: bool = False`
(`src/pykeen/datasets/metadata.py`), set to `True` on WN18RR; `_closure_pool` flips
hierarchy-relation edges `(h, t) → (t, h)` when inverted, before building
`paths`/`direct`/`pool`/train rows, so negatives, `_sibling_map`, and the cones' parent-slot all
see one canonical parent→child orientation. Note: on WN18RR the training triples therefore carry
`_hypernym` with flipped (parent, child) order — raw prediction rows read as `parent _hypernym
child`.

**Do not** encode both edge directions in the dataset files instead (e.g. `has_subclass` +
`has_superclass`): the split filters on one relation so the inverse edges are dead weight there,
and for any plain link-prediction use a relation plus its exact inverse is the classic WN18
test-leakage pattern.

## 3. "Basic edges" are not the transitive reduction (FIXED)

Ganea et al. define basic edges as the transitive reduction of the full closure; the code treated
the dataset's *asserted* edges as basic (`direct` in `_closure_pool`), so any redundant (transitive)
asserted shortcut edge silently moved from the eval pool into permanent training data.

**Implemented fix:** `_closure_pool` now sets `direct = nx.transitive_reduction(graph).edges()`
(basic edges) instead of the asserted edge set; redundant shortcuts fall into the eval `pool`.
No-op on clean trees (reduction == asserted). Falls back to asserted edges with a warning when the
hierarchy is not a DAG (transitive reduction is only defined on acyclic graphs).

## 4. Untuned α in the is-a scoring heuristic (FIXED)

`ISA_ALPHA = 1.0` was fixed in `benchmark_transitive_ancestor_descendant.py`; Ganea et al. tune Eq. 37's α on
validation. This handicapped exactly the baselines (PoincareE, LorentzE) the cones are compared
against.

**Implemented fix:** `ISA_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)` grid in both
`benchmark_transitive_ancestor_descendant.py` and `test_transitive_ancestor_descendant.py`; the is-a score builders take `alpha` as a
parameter instead of reading a module global. α is eval-only, so each config trains once and
`_tune_alpha` (benchmark) / `_score_pass` (test script) re-score the trained model per candidate,
picking the max validation F1 — exposed via a new `val_f1` key in `_transitive_ancestor_descendant_pair_metrics`'
`return_raw` dict (`src/pykeen/pipeline/transitive_ancestor_descendant.py`). Tuned independently per negative setting
(rnd/hrd, each on its own validation pass); the winning α is logged and stamped into the
predictions JSON. Split and negatives are re-derived from the same seed per candidate, so all α
see identical pairs and test negatives match the pre-fix draws. Ties break to the smallest α in
the grid.

## 5. No Poincaré burn-in (FIXED)

Ganea et al. call Nickel & Kiela's burn-in "essential for a good initial disentanglement"; the
pretraining pass and the Poincaré/Lorentz baseline runs skipped it. Another baseline handicap.

**Implemented fix:** `BURN_IN_EPOCHS = 10` (`benchmark_transitive_ancestor_descendant.py`), wired via
`lr_scheduler="constant", lr_scheduler_kwargs={"factor": 0.1, "total_iters": BURN_IN_EPOCHS}` into
the Poincaré pretraining pass and both the `PoincareE`/`LorentzE` baseline configs in
`build_configs`. `ConstantLR` holds lr/10 for the first `BURN_IN_EPOCHS` epochs (stepped per-epoch
via `LearningRateSchedulerTrainingCallback`), then jumps to full lr.

**Not covered:** `test_transitive_ancestor_descendant.py`'s single-dataset sweep still has no burn-in.

## 6. Duplicate evaluation negatives possible (FIXED)

`_sample_negatives` drew with replacement; the same corrupted pair could appear twice for one
positive (and the hard-negative top-up could re-draw an already-emitted pair). Low impact at 1:10,
but it biased evaluation by double-counting.

**Implemented fix:** `_sample_negatives` (`src/pykeen/pipeline/hierarchical_helper.py`) now keeps a
per-positive `seen` set of emitted `(h, r, t)` rows. The hard-negative block skips rows already in
`seen`, and `seen` (plus `relation`) is threaded into `_draw_negative`, whose validity check now
also rejects candidates whose resulting row is in `seen` — covering both the rejection loop and the
exact-scan fallback, so it returns `None` (reusing the existing skip-with-warning path) only when
the fresh, valid pool is truly exhausted. Negatives are therefore drawn without replacement per
positive across both hard and random paths. `_sample_negatives`'s signature is unchanged. Guarded by
`test_negatives_deduped_per_positive` in `tests/test_pipeline.py`.

## 7. Sibling definition broader than the papers'

`_sibling_map` counts entities sharing a direct *successor* as siblings too (co-parents), not
just entities sharing a parent. On multi-parent DAGs (DOID, MeSH) this changes hard-negative
difficulty relative to He et al.'s same-parent definition. Documented in the docstring, but worth
a sentence in any write-up. On predefined-split datasets (WordNetNoun*), ``direct`` additionally
contains the in-training closure shortcut edges (no transitive reduction is run at that scale),
which broadens the sibling definition slightly further.

## 8. API footgun: default model is mis-evaluated by default

`transitive_ancestor_descendant_prediction_pipeline` defaults to `PoincareE` — a symmetric-distance model — with
`eval_score_fn=None`, exactly the case its own docstring warns cannot distinguish ancestor from
descendant. Fix: default `eval_score_fn` to a directional score when the model is symmetric, or
warn loudly.

## 9. Not a Table 1 reproduction

Applies to the re-split datasets (WN18RR & the ontologies): dim 64 (paper: 5/10), tree root kept
(paper removes it), and WN18RR's pruned hypernym graph instead of the full WordNet noun closure
(closure pool 157,782 vs the paper's 578,477 — broken hypernym chains from the WN18→WN18RR
pruning shrink the closure). Their numbers must not be set beside Ganea et al.'s Table 1.

**Resolved for the comparison itself:** the benchmark now includes `WordNetNoun50Percent`
(`src/pykeen/datasets/wordnet_noun.py`) — Ganea's own `maxn` splits (82,114 nodes, fixed
test/valid of 28,838 closure pairs) — run at the paper's dim 5 via `DIM_OVERRIDES`. Datasets
declaring `predefined_closure_split = True` bypass the re-splitting entirely
(`_transitive_ancestor_descendant_split` uses their fixed files; no transitive reduction, no leakage of in-training
closure edges into evaluation).

## 10. No per-model HPO

Optimizers/losses/learning rates are fixed per model from the papers; comparative claims are
sensitive to these budgets. Fix: HPO per model×dataset (the `_hpo_and_refit` machinery already
exists in `hierarchical_helper.py`).

## 11. HyperbolicCones training/scoring mismatch (FIXED)

Cones scored *below* PoincareE everywhere, contradicting Ganea et al. Table 1 (parity at 0%
closure). Three causes, all in the harness, none in the cone math:

- **Loss:** `MarginRankingLoss(margin=0.01)` is *pairwise*; with radian-scale energies it
  saturates immediately (no gradient). Ganea Eq. 32 is a *pointwise* hinge. Fixed:
  `PointwiseHingeLoss(margin=0.01)` (benchmark config + `HyperbolicCones.loss_default`).
- **Optimizer:** SGD lr=1e-4 assumes the paper's batch size 10; with pykeen batching the model
  barely moved. Fixed: `RiemannianAdam lr=1e-3`.
- **Score ties:** `−relu(Ξ−ψ)` scored exactly 0.0 for every in-cone pair, degenerating MRR/mAP
  and the PR threshold search. Fixed: `HyperbolicConesInteraction.forward` returns the signed
  margin `ψ−Ξ`; the relu is recovered by the hinge loss.

Also fixed alongside: the §4 α grid capped at 5 while Nickel & Kiela use α=1000 — since ball
norms collapse near 1, small α is noise and validation always picked the grid minimum,
degenerating the is-a score to the symmetric distance (flat baselines across closure ratios).
Grid is now `(0.1, 1, 10, 100, 1000)` in both scripts. TransE was removed from the benchmark
(wins F1·rnd via locality, collapses to the all-positive floor under sibling negatives).

After the fix the closure sweep reproduces the paper's trend: WN18RR F1 0.29→0.91 for cones
(paper: 0.29→0.93) with parity vs Poincaré at 0%.
