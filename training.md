# Training recipes — subsumption prediction baselines

Shared protocol for all configs (Ganea et al. 2018, §5; He et al. 2024, §4.1), as implemented by
`subsumption_prediction_pipeline`: train on all direct hierarchy edges (plus `CLOSURE_RATIO` of the
non-direct transitive closure), hold out two `EVAL_RATIO` portions of the remaining closure pairs as
validation/test, pair each held-out positive with `NUM_NEGATIVES` negatives, tune an F1-optimal score
threshold on validation, and report Precision/Recall/F1 on test (random and hard/sibling negatives),
plus mAP/AUROC (random negatives) and rank-based MRR.

Shared settings across all baselines: `embedding_dim=64`, `epochs=200`, `seed=42`, `closure_ratio=0.0`,
`eval_ratio=0.05`, `num_negatives=50`, CPU device.

## PoincareE (Nickel & Kiela 2017)

- **Geometry**: Poincaré ball, curvature 1.0.
- **Init**: `U(-0.001, 0.001)` in tangent space, lifted via `expmap0` (model default).
- **Loss**: softmax-over-negatives ranking loss (`crossentropy`, paper Eq. 6).
- **Optimizer**: `RiemannianSGD`, lr=0.3 (paper §3.1, natural-gradient RSGD).
- **Negatives**: 50 per positive (training-time negative sampler).
- **Eval scoring**: raw Poincaré distance is symmetric and can't tell ancestor from descendant, so
  evaluation uses the directional is-a heuristic (Nickel & Kiela 2017, Eq. 8):
  `score(h,t) = -(1 + α·(‖h‖-‖t‖))·d(h,t)`, α=1.0. The sign of the norm term is calibrated per
  dataset from the trained geometry — whichever end of the hierarchy relation sits closer to the
  origin on average (the more general concepts) is treated as the "ancestor" side.

## LorentzE (Nickel & Kiela 2018)

- **Geometry**: Lorentz hyperboloid, curvature 1.0 (numerically more stable than the ball at higher dims).
- **Init**: `U(-0.001, 0.001)` (model default), lifted to the hyperboloid.
- **Loss**: softmax-over-negatives ranking loss (`crossentropy`, paper Eq. 12).
- **Optimizer**: `RiemannianSGD`, lr=0.3 (paper §3.2.2).
- **Negatives**: 50 per positive.
- **Eval scoring**: same Eq. 8 heuristic as PoincareE — hyperboloid points are first mapped to the
  Poincaré ball (`x[1:] / (x[0]+1)`, Nickel & Kiela 2018 Eq. 11) so norms encode depth the same way.

## HyperbolicCones (Ganea et al. 2018)

- **Geometry**: Poincaré ball, curvature 1.0, cone width `k=0.1` (= ε, the inner-radius exclusion).
- **Pretraining (required — random init is unstable for the cone loss)**:
  1. Train a `PoincareE` model for **100 epochs** (RiemannianSGD lr=0.3, `crossentropy` loss, 50
     negatives/positive) on the *same dataset instance* so entity ids line up.
  2. Rescale its ball embeddings by **0.7** (the paper notes plain Poincaré embeddings collapse
     toward the ball border).
  3. Map the rescaled points back to tangent space via `manifold.logmap0(...)`.
  4. Wrap as a `PretrainedInitializer` and pass as `entity_initializer` into `HyperbolicCones`.
- **Loss**: max-margin loss on entailment-cone energy (`marginranking`, margin γ=0.01, paper Eq. 32).
- **Optimizer**: `RiemannianSGD`, lr=1e-4 (paper §4.2/§5 — retraction approximation of the exponential map).
- **Negatives**: 50 per positive (training-time negative sampler).
- **Eval scoring**: no heuristic needed — the model's own directional energy
  `E(u,v) = max(0, Ξ(u,v) - ψ(u))` is already asymmetric, so the pipeline's default scorer
  (`model.predict_hrt`) is used directly.

## TransE (Euclidean baseline)

- **Geometry**: plain Euclidean space.
- **Loss/init**: model defaults.
- **Optimizer**: `Adam`, lr=0.01.
- **Eval scoring**: default translational distance score, no override needed.

## RotatE (Sun et al. 2019)

- **Geometry**: complex-valued Euclidean space (rotation in each 2D subspace).
- **Loss/init**: model defaults.
- **Optimizer**: `Adam`, lr=0.01.
- **Eval scoring**: default rotational distance score, no override needed.
