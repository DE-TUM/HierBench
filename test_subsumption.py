"""Benchmark multi-hop subsumption prediction (Ganea et al. 2018; He et al. 2024) across configs.

Follows the shared evaluation protocol of Hyperbolic Entailment Cones (Ganea et al. 2018, §5) and
Language Models as Hierarchy Encoders (He et al. 2024, §4.1 Multi-hop Inference):

* train on all direct hierarchy edges plus ``CLOSURE_RATIO`` of the non-direct transitive closure
  (Ganea's 0%/10%/25%/50% axis),
* hold out two ``EVAL_RATIO`` portions of the remaining closure pairs as validation/test,
* pair each held-out positive with ``NUM_NEGATIVES`` negatives — evaluated under both the random
  and the hard (sibling) negative settings,
* tune the F1-optimal score threshold on validation and report Precision/Recall/F1 on test,
  alongside mAP/AUROC (random negatives) and the rank-based MRR.

Each configuration trains **once** and is re-scored under the hard negative setting via
:func:`subsumption_prediction_metrics` (same seed -> same held-out pairs).
"""

from __future__ import annotations

from pykeen.datasets import NASA, DOID, Cora, ACMCCS, WN18RR
from pykeen.datasets.metadata import resolve_hierarchy_relation
from pykeen.models import HyperbolicCones, LorentzE, PoincareE
from pykeen.nn.hyperbolic import LorentzEmbedding
from pykeen.pipeline.subsumption import subsumption_prediction_metrics, subsumption_prediction_pipeline

# --- Shared experiment settings ----------------------------------------------------------
#: torch device for all pipeline runs.
DEVICE = "cpu"
DATASET = WN18RR()
HIERARCHY_RELATION = getattr(DATASET, "hierarchical_relation", None)
EPOCHS = 200
EMBEDDING_DIM = 64
SEED = 42

# --- Task settings (both papers) ---------------------------------------------------------
CLOSURE_RATIO = 0.0  # fraction of non-direct closure pairs added to training (Ganea et al. 2018 §5)
EVAL_RATIO = 0.05  # two 5% portions of the indirect subsumptions as val/test (He et al. 2024 §4.2)
NUM_NEGATIVES = 50  # 1:50 positive:negative ratio (both papers use 1:10)

#: Nickel & Kiela (2017) Eq. 8: severity of the norm (depth) penalty. Must satisfy
#: α·|‖h‖-‖t‖| < 1 so the multiplier stays positive — otherwise the score flips sign and
#: *rewards* distance, inverting the ranking (AUROC < 0.5). N&K's α=10³ was for graded HyperLex
#: ranking, not binary classification; Ganea et al. (2018 §5) tune α on validation instead.
ISA_ALPHA = 1.0


def _ball_norm(emb, indices):
    """Poincaré-ball norm of the given entities; hyperboloid points are mapped to the ball first."""
    x = emb(indices)
    if isinstance(emb, LorentzEmbedding):
        # hyperboloid -> Poincaré ball (Nickel & Kiela 2018, Eq. 11) so norms encode depth
        x = x[..., 1:] / (x[..., :1] + 1)
    return x.norm(dim=-1)


def hyperbolic_isa_score(model, batch):
    """Directional is-a score for symmetric-distance hyperbolic models (Nickel & Kiela 2017, Eq. 8).

    ``score(h is-ancestor-of t) = -(1 + α(‖h‖-‖t‖))·d(h,t)``: raw hyperbolic distance is symmetric
    and cannot tell ancestor from sibling or reversed pair; the norm term reads depth out of the
    trained embedding (general concepts settle near the origin). Ganea et al. (2018, §5) score the
    Poincaré baseline the same way. Eval-only — training keeps the plain distance objective.

    The sign of the norm term is calibrated from the trained geometry: hierarchy relations may point
    parent->child (NASA ``has_subclass``) or child->parent (WN18RR ``_hypernym``), so whichever edge
    end sits closer to the origin on average (the general end) gets the "ancestor" role.
    """
    emb = model.entity_representations[0]
    h, t = emb(batch[:, 0]), emb(batch[:, 2])
    dist = emb.manifold.dist(h, t)
    edges = DATASET.training.mapped_triples
    rel = resolve_hierarchy_relation(DATASET, HIERARCHY_RELATION)
    if rel is not None:
        edges = edges[edges[:, 1] == rel]
    head_depth = _ball_norm(emb, edges[:, 0].to(batch.device)).mean()
    tail_depth = _ball_norm(emb, edges[:, 2].to(batch.device)).mean()
    sign = 1.0 if head_depth <= tail_depth else -1.0
    return -(1 + sign * ISA_ALPHA * (_ball_norm(emb, batch[:, 0]) - _ball_norm(emb, batch[:, 2]))) * dist


# --- Configurations to compare ----------------------------------------------------------
# Each entry: (label, model, model_kwargs, extra pipeline kwargs)
CONFIGS: list[tuple[str, object, dict, dict]] = [
    (
        # Reported hyperparameters (Nickel & Kiela 2017): softmax ranking loss (Eq. 6) and
        # U(-0.001, 0.001) init are the PoincareE model defaults; the rest are set here.
        "PoincareE (Nickel & Kiela 2017)",
        PoincareE,
        {"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},  # curvature=1.0 == K=-1
        {
            "optimizer": "RiemannianSGD",  # paper §3.1 RSGD / natural gradient
            "optimizer_kwargs": {"lr": 0.3},
            "loss": "crossentropy",  # paper Eq. 6 softmax ranking loss
            "negative_sampler_kwargs": {"num_negs_per_pos": 50},  # paper §4.1
            # paper Eq. 8 directional is-a scoring for the (symmetric) distance at eval time
            "eval_score_fn": hyperbolic_isa_score,
        },
    ),
    (
        # Reported hyperparameters (Nickel & Kiela 2018, Lorentz model): same softmax ranking loss
        # (Eq. 12) and U(-0.001, 0.001) init are the LorentzE model defaults; the rest are set here.
        "LorentzE (Nickel & Kiela 2018)",
        LorentzE,
        {"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},  # curvature=1.0 == K=-1
        {
            "optimizer": "RiemannianSGD",  # paper §3.2.2 RSGD / natural gradient
            "optimizer_kwargs": {"lr": 0.3},
            "loss": "crossentropy",  # paper Eq. 12 softmax ranking loss
            "negative_sampler_kwargs": {"num_negs_per_pos": 50},  # paper §4.1
            "eval_score_fn": hyperbolic_isa_score,
        },
    ),
    # (
    #     # Reported hyperparameters (Ganea et al. 2018 §5): max-margin loss (Eq. 32) with γ=0.01,
    #     # aperture/inner-radius K=ε=0.1, 10 negatives per positive, SGD w/ retraction @ lr=1e-4.
    #     "HyperbolicCones (Ganea et al. 2018)",
    #     HyperbolicCones,
    #     {"embedding_dim": EMBEDDING_DIM, "k": 0.1, "curvature": 1.0},  # K=ε=0.1 (paper §5)
    #     {
    #         "optimizer": "RiemannianSGD",  # paper §4.2/§5: SGD w/ retraction approximation
    #         "optimizer_kwargs": {"lr": 1e-4},  # paper §5 learning rate
    #         "loss": "marginranking",  # paper Eq. 32 max-margin loss
    #         "loss_kwargs": {"margin": 0.01},  # paper §5 margin γ=0.01
    #         "negative_sampler_kwargs": {"num_negs_per_pos": 50},  # paper §5: 50 negatives/positive
    #     },
    # ),
    (
        "TransE (Euclidean baseline)",
        "TransE",
        {"embedding_dim": EMBEDDING_DIM},
        {"optimizer": "Adam", "optimizer_kwargs": {"lr": 0.01}},
    ),
    (
        "RotatE (Sun et al. 2019)",
        "RotatE",
        {"embedding_dim": EMBEDDING_DIM},
        {"optimizer": "Adam", "optimizer_kwargs": {"lr": 0.01}},
    ),
]

#: printed column -> key in the metrics dict returned by the subsumption evaluation
PAIR_METRICS = {"Prec": "precision", "Rec": "recall", "F1": "f1"}
COLUMNS = [
    *(f"{name}·rnd" for name in PAIR_METRICS),
    *(f"{name}·hrd" for name in PAIR_METRICS),
    "mAP",
    "AUROC",
    "MRR",
]


def _evaluate_config(model: object, model_kwargs: dict, extra: dict) -> dict[str, float]:
    """Train one configuration, then score the held-out subsumptions under both negative settings."""
    extra = dict(extra)
    eval_score_fn = extra.pop("eval_score_fn", None)
    result = subsumption_prediction_pipeline(
        DATASET,
        model=model,
        model_kwargs=model_kwargs,
        epochs=EPOCHS,
        closure_ratio=CLOSURE_RATIO,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        seed=SEED,
        random_seed=SEED,
        hierarchy_relation=HIERARCHY_RELATION,
        negative_sampler="basic",
        eval_score_fn=eval_score_fn,
        device=DEVICE,
        **extra,
    )
    random_scores = result.ancestor_descendant_metric_results
    hard_scores = subsumption_prediction_metrics(
        result.model,
        DATASET,
        closure_ratio=CLOSURE_RATIO,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        hard_negatives=True,
        seed=SEED,
        hierarchy_relation=HIERARCHY_RELATION,
        eval_score_fn=eval_score_fn,
    )

    def _get(results, key):
        """Look up a metric, or NaN when the results (or the whole pass) are missing."""
        return results.get_metric(key) if results is not None else float("nan")

    scores = {f"{name}·rnd": _get(random_scores, key) for name, key in PAIR_METRICS.items()}
    scores |= {f"{name}·hrd": _get(hard_scores, key) for name, key in PAIR_METRICS.items()}
    scores["mAP"] = _get(random_scores, "average_precision")
    scores["AUROC"] = _get(random_scores, "roc_auc")
    scores["MRR"] = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
    return scores


def main() -> None:
    """Run every configuration and print a side-by-side subsumption-prediction comparison."""
    print("=" * 104)
    print(f"Dataset: {type(DATASET).__name__}  task: multi-hop subsumption prediction")
    print(
        f"  hierarchy_relation={HIERARCHY_RELATION!r}  closure_ratio={CLOSURE_RATIO}  "
        f"eval_ratio={EVAL_RATIO}  negatives/positive={NUM_NEGATIVES}  epochs={EPOCHS}"
    )
    print("=" * 104)

    rows: list[tuple[str, dict[str, float]]] = []
    for label, model, model_kwargs, extra in CONFIGS:
        print(f"\n>>> {label}")
        try:
            scores = _evaluate_config(model, model_kwargs, extra)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"    FAILED: {exc}")
            scores = dict.fromkeys(COLUMNS, float("nan"))
        rows.append((label, scores))

    print("\n" + "=" * 104)
    print(f"{'Multi-hop subsumption prediction (threshold tuned on validation)':^104}")
    print("=" * 104)
    print(f"{'':<32}{'random negatives':>24}{'hard negatives':>24}")
    header = f"{'Configuration':<32}" + "".join(f"{name:>8}" for name in COLUMNS)
    print(header)
    print("-" * len(header))
    for label, scores in rows:
        cells = "".join(f"{scores[name]:>8.4f}" for name in COLUMNS)
        print(f"{label:<32}{cells}")
    print("=" * 104)
    print("Precision/Recall/F1 at the validation-tuned threshold; mAP/AUROC/MRR: higher is better.")


if __name__ == "__main__":
    main()
