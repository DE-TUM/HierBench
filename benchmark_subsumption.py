"""Benchmark every subsumption baseline (incl. HyperbolicCones) across every hierarchical dataset.

Extends ``test_subsumption.py``'s single-dataset sweep (Ganea et al. 2018; He et al. 2024 protocol,
see that module's docstring) to every dataset in this repo implementing
:class:`pykeen.datasets.metadata.HierarchicalGraph`, adds a working ``HyperbolicCones`` baseline
(warm-started from a pretrained Poincaré model, per Ganea et al. 2018 §5), reports the hierarchical
P/R/F1 (Kosmopoulos et al. 2015) already computed by the pipeline, and persists per-run results plus
a cross-run summary under ``results/subsumption/``.
"""

from __future__ import annotations

import csv
import json
import time
from functools import partial
from pathlib import Path

from pykeen import datasets
from pykeen.datasets.metadata import HierarchicalGraph, resolve_hierarchy_relation
from pykeen.models import HyperbolicCones, LorentzE, PoincareE
from pykeen.nn.hyperbolic import LorentzEmbedding
from pykeen.nn.init import PretrainedInitializer
from pykeen.pipeline.subsumption import subsumption_prediction_metrics, subsumption_prediction_pipeline

DEVICE = "cpu"
EPOCHS = 200
#: Ganea et al. (2018 §5): the Poincaré pretraining pass for HyperbolicCones always uses 100 epochs,
#: independent of EPOCHS above.
CONES_PRETRAIN_EPOCHS = 100
#: Ganea et al. (2018 §5): pretrained Poincaré embeddings collapse toward the ball border; rescale
#: before using them as the cones' tangent-space init.
CONES_PRETRAIN_RESCALE = 0.7
#: Ganea et al. (2018) Table 1's low-dimensional regime (the paper reports dim 5 and 10).
EMBEDDING_DIM = 5
SEED = 42
#: Nickel & Kiela (2017) / Ganea et al. (2018 §5): burn-in — hold lr/10 for the first
#: BURN_IN_EPOCHS epochs for a good initial disentanglement, then jump to full lr.
BURN_IN_EPOCHS = 10

#: Fraction of non-basic transitive-closure edges added to training, following Ganea et al.
#: (2018) Table 1's strongest setting. Re-split datasets sample this from their own closure
#: pool; WordNetNoun50Percent encodes it in its files (predefined split, ratio used as label).
# Full sweep: [0.0, 0.1, 0.25, 0.5]
CLOSURE_RATIOS = [0.0]
EVAL_RATIO = 0.05
#: Ganea et al. (2018 §5) / He et al. (2024): 10 evaluation negatives per positive (5 head- +
#: 5 tail-corrupted).
NUM_NEGATIVES = 10
#: Ganea et al. (2018 §5): 10 training negatives per positive — applied to *every* config so the
#: baselines get equal training budgets.
TRAIN_NEGATIVES = 10

#: Nickel & Kiela (2017) Eq. 8: severity of the norm (depth) penalty. Tuned per config on the
#: validation set (max val F1), following Ganea et al. (2018 §5). N&K use alpha=1000: since ball
#: norms all collapse near 1, the term only matters in the large-alpha regime where it acts as a
#: hard direction gate (sign flip for wrong-direction pairs). The grid must reach that regime —
#: capping it at 5 made validation always pick the smallest alpha and reduced the score to the
#: symmetric distance.
ISA_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

#: Ganea et al. (2018) Table 1 protocol (dim 5, 50% closure) extended to more taxonomies.
#: WN18RR is excluded (pruned hypernym chains, not the paper's WordNet — WordNetNoun50Percent is
#: Ganea's own split); EstatFSS is excluded (closure pool of 23 pairs, degenerate eval).
DATASET_CLASSES = [
    datasets.NASA,
    datasets.DOID,
    datasets.ACMCCS,
    datasets.EuroSciVoc,
    datasets.MeSH,
    # Ganea et al. (2018)'s actual Table 1 dataset (predefined closure split; the pipeline ignores
    # CLOSURE_RATIOS for it — the 0% variant matches the swept 0.0 ratio by construction).
    datasets.WordNetNoun0Percent,
]

RESULTS_DIR = Path(__file__).parent / "results" / "subsumption"

PAIR_METRICS = {"Prec": "precision", "Rec": "recall", "F1": "f1"}
HIER_METRICS = {"HierP": "hierarchical_precision", "HierR": "hierarchical_recall", "HierF1": "hierarchical_f1"}
COLUMNS = [
    *(f"{name}·rnd" for name in PAIR_METRICS),
    *(f"{name}·hrd" for name in PAIR_METRICS),
    "mAP",
    "AUROC",
    "MRR",
    *HIER_METRICS,
]


def _ball_norm(emb: object, indices: object) -> object:
    """Poincaré-ball norm of the given entities; hyperboloid points are mapped to the ball first."""
    x = emb(indices)
    if isinstance(emb, LorentzEmbedding):
        x = x[..., 1:] / (x[..., :1] + 1)
    return x.norm(dim=-1)


def _make_hyperbolic_isa_score(dataset: object, hierarchy_relation: int | None, alpha: float = 1.0):
    """Build a directional is-a score (Nickel & Kiela 2017, Eq. 8) closed over one dataset/relation.

    Per-dataset closure over ``test_subsumption.py``'s ``make_hyperbolic_isa_score``, which instead
    reads module-level ``DATASET``/``HIERARCHY_RELATION`` globals — not reusable across a dataset
    sweep. ``alpha`` weights the norm (depth) penalty and is tuned on validation by ``_tune_alpha``.
    """

    def score(model: object, batch: object) -> object:
        emb = model.entity_representations[0]
        h, t = emb(batch[:, 0]), emb(batch[:, 2])
        dist = emb.manifold.dist(h, t)
        edges = dataset.training.mapped_triples
        if hierarchy_relation is not None:
            edges = edges[edges[:, 1] == hierarchy_relation]
        head_depth = _ball_norm(emb, edges[:, 0].to(batch.device)).mean()
        tail_depth = _ball_norm(emb, edges[:, 2].to(batch.device)).mean()
        sign = 1.0 if head_depth <= tail_depth else -1.0
        return -(1 + sign * alpha * (_ball_norm(emb, batch[:, 0]) - _ball_norm(emb, batch[:, 2]))) * dist

    return score


def _tune_alpha(
    model: object,
    dataset: object,
    hierarchy_relation: int | None,
    score_factory,
    *,
    closure_ratio: float,
    hard_negatives: bool,
) -> tuple[object, dict | None]:
    """Tune the is-a score's alpha on the validation set (Ganea et al. 2018 §5), max val F1.

    Alpha is eval-only, so the sweep re-scores one trained model. The split and negatives are
    re-derived from ``SEED`` on every call, so all candidates (and the final test metrics) see
    identical pairs.
    """
    # ponytail: rescores dist+norms per alpha; decompose Δnorm/dist once if the sweep ever dominates
    best = None
    for alpha in ISA_ALPHAS:
        metrics, raw = subsumption_prediction_metrics(
            model,
            dataset,
            closure_ratio=closure_ratio,
            eval_ratio=EVAL_RATIO,
            num_negatives=NUM_NEGATIVES,
            hard_negatives=hard_negatives,
            seed=SEED,
            hierarchy_relation=hierarchy_relation,
            eval_score_fn=score_factory(alpha),
            return_raw=True,
        )
        if raw is None:
            return metrics, raw  # nothing was scored; no tuning possible
        raw["alpha"] = alpha
        if best is None or raw["val_f1"] > best[1]["val_f1"]:
            best = (metrics, raw)
    return best


def _pretrain_poincare_initializer(
    dataset: object, hierarchy_relation: int | None, closure_ratio: float, embedding_dim: int
) -> PretrainedInitializer:
    """Pretrain a Poincaré model and return a tangent-space initializer for HyperbolicCones.

    Ganea et al. (2018 §5): cones are warm-started from a Poincaré model pretrained for 100 epochs;
    the pretrained ball embeddings are rescaled by 0.7 (they collapse toward the border) then mapped
    back to tangent space with ``manifold.logmap0`` before being used as a tangent-space initializer.
    """
    result = subsumption_prediction_pipeline(
        dataset,
        model=PoincareE,
        model_kwargs={"embedding_dim": embedding_dim, "curvature": 1.0},
        epochs=CONES_PRETRAIN_EPOCHS,
        closure_ratio=closure_ratio,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        seed=SEED,
        random_seed=SEED,
        hierarchy_relation=hierarchy_relation,
        optimizer="RiemannianSGD",
        optimizer_kwargs={"lr": 0.3},
        lr_scheduler="constant",
        lr_scheduler_kwargs={"factor": 0.1, "total_iters": BURN_IN_EPOCHS},
        loss="crossentropy",
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": TRAIN_NEGATIVES},
        hierarchical=False,
        device=DEVICE,
    )
    emb = result.model.entity_representations[0]
    tangent = emb.manifold.logmap0(emb._embeddings.data * CONES_PRETRAIN_RESCALE)
    return PretrainedInitializer(tensor=tangent.detach().clone())


def build_configs(
    dataset: object, hierarchy_relation: int | None, closure_ratio: float, embedding_dim: int = EMBEDDING_DIM
) -> list[tuple[str, object, dict, dict]]:
    """Build the per-dataset config list, including a freshly pretrained HyperbolicCones entry."""
    cones_initializer = _pretrain_poincare_initializer(dataset, hierarchy_relation, closure_ratio, embedding_dim)
    isa_score_factory = partial(_make_hyperbolic_isa_score, dataset, hierarchy_relation)
    return [
        (
            "PoincareE (Nickel & Kiela 2017)",
            PoincareE,
            {"embedding_dim": embedding_dim, "curvature": 1.0},
            {
                "optimizer": "RiemannianSGD",
                "optimizer_kwargs": {"lr": 0.3},
                "lr_scheduler": "constant",
                "lr_scheduler_kwargs": {"factor": 0.1, "total_iters": BURN_IN_EPOCHS},
                "loss": "crossentropy",
                "negative_sampler_kwargs": {"num_negs_per_pos": TRAIN_NEGATIVES},
                "eval_score_factory": isa_score_factory,
            },
        ),
        (
            "LorentzE (Nickel & Kiela 2018)",
            LorentzE,
            {"embedding_dim": embedding_dim, "curvature": 1.0},
            {
                "optimizer": "RiemannianSGD",
                "optimizer_kwargs": {"lr": 0.3},
                "lr_scheduler": "constant",
                "lr_scheduler_kwargs": {"factor": 0.1, "total_iters": BURN_IN_EPOCHS},
                "loss": "crossentropy",
                "negative_sampler_kwargs": {"num_negs_per_pos": TRAIN_NEGATIVES},
                "eval_score_factory": isa_score_factory,
            },
        ),
        (
            "HyperbolicCones (Ganea et al. 2018)",
            HyperbolicCones,
            {
                "embedding_dim": embedding_dim,
                "k": 0.1,
                "curvature": 1.0,
                "entity_initializer": cones_initializer,
            },
            {
                # Adaptive lr: the paper's SGD lr=1e-4 assumes batch size 10 (~25-100x more steps
                # than pykeen's default batching) and barely moves the model here.
                "optimizer": "RiemannianAdam",
                "optimizer_kwargs": {"lr": 1e-3},
                # Ganea et al. (2018) Eq. 32: pointwise hinge (see HyperbolicCones.loss_default).
                "loss": "pointwisehinge",
                "loss_kwargs": {"margin": 0.01},
                "negative_sampler_kwargs": {"num_negs_per_pos": TRAIN_NEGATIVES},
            },
        ),
        (
            "RotatE (Sun et al. 2019)",
            "RotatE",
            {"embedding_dim": embedding_dim},
            {
                "optimizer": "Adam",
                "optimizer_kwargs": {"lr": 0.01},
                "negative_sampler_kwargs": {"num_negs_per_pos": TRAIN_NEGATIVES},
            },
        ),
    ]


def _get(results: object, key: str) -> float:
    """Look up a metric, or NaN when the results (or the whole pass) are missing."""
    return results.get_metric(key) if results is not None else float("nan")


def _evaluate_config(
    dataset: object,
    hierarchy_relation: int | None,
    closure_ratio: float,
    model: object,
    model_kwargs: dict,
    extra: dict,
) -> tuple[dict[str, float], dict]:
    """Train one configuration, then score the held-out subsumptions under both negative settings.

    Returns the flat metrics dict plus a ``{"rnd", "hrd"}`` dict of the raw per-pair predictions
    (each ``None`` when that pass produced no metrics) for persistence. Configs carrying an
    ``eval_score_factory`` (alpha -> score_fn) get alpha tuned on validation per pass.
    """
    extra = dict(extra)
    score_factory = extra.pop("eval_score_factory", None)
    result = subsumption_prediction_pipeline(
        dataset,
        model=model,
        model_kwargs=model_kwargs,
        epochs=EPOCHS,
        closure_ratio=closure_ratio,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        seed=SEED,
        random_seed=SEED,
        hierarchy_relation=hierarchy_relation,
        negative_sampler="basic",
        device=DEVICE,
        return_raw=True,
        **extra,
    )
    if score_factory is None:
        random_scores = result.ancestor_descendant_metric_results
        random_raw = result.ancestor_descendant_raw_predictions
        hard_scores, hard_raw = subsumption_prediction_metrics(
            result.model,
            dataset,
            closure_ratio=closure_ratio,
            eval_ratio=EVAL_RATIO,
            num_negatives=NUM_NEGATIVES,
            hard_negatives=True,
            seed=SEED,
            hierarchy_relation=hierarchy_relation,
            return_raw=True,
        )
    else:
        tune = partial(
            _tune_alpha, result.model, dataset, hierarchy_relation, score_factory, closure_ratio=closure_ratio
        )
        random_scores, random_raw = tune(hard_negatives=False)
        hard_scores, hard_raw = tune(hard_negatives=True)
        print(
            f"  tuned alpha: rnd={random_raw['alpha'] if random_raw else None}"
            f"  hrd={hard_raw['alpha'] if hard_raw else None}"
        )

    scores = {f"{name}·rnd": _get(random_scores, key) for name, key in PAIR_METRICS.items()}
    scores |= {f"{name}·hrd": _get(hard_scores, key) for name, key in PAIR_METRICS.items()}
    scores["mAP"] = _get(random_scores, "average_precision")
    scores["AUROC"] = _get(random_scores, "roc_auc")
    scores["MRR"] = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
    scores |= {
        name: _get(result.hierarchical_metric_results, f"both.{key}") for name, key in HIER_METRICS.items()
    }
    raw = {"rnd": random_raw, "hrd": hard_raw}
    return scores, raw


def _write_run_result(
    dataset_name: str,
    closure_ratio: float,
    label: str,
    scores: dict[str, float],
    duration: float,
    error: str | None,
    raw: dict | None = None,
) -> None:
    """Persist one run's full metrics plus metadata as JSON under results/subsumption/<dataset>/<closure%>/.

    When ``raw`` carries per-pair predictions and the run succeeded, they are written to a separate
    ``<config>.predictions.json`` in the same directory, keeping the metrics file/summary unchanged.
    """
    out_dir = RESULTS_DIR / dataset_name / f"closure_{int(closure_ratio * 100)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.split(" (")[0].replace(" ", "_")
    payload = {
        "dataset": dataset_name,
        "config": label,
        "epochs": EPOCHS,
        "embedding_dim": EMBEDDING_DIM,
        "seed": SEED,
        "closure_ratio": closure_ratio,
        "eval_ratio": EVAL_RATIO,
        "num_negatives": NUM_NEGATIVES,
        "duration_seconds": duration,
        "status": "failed" if error else "ok",
        "error": error,
        "metrics": scores,
    }
    (out_dir / f"{safe_label}.json").write_text(json.dumps(payload, indent=2))
    if error is None and raw and any(raw.values()):
        (out_dir / f"{safe_label}.predictions.json").write_text(json.dumps(raw, indent=2))


def _write_summary(rows: list[tuple[str, float, str, dict[str, float]]]) -> None:
    """Write one aggregated CSV: rows = dataset x closure_ratio x config, columns = all metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "closure_ratio", "config", *COLUMNS])
        for dataset_name, closure_ratio, label, scores in rows:
            writer.writerow([dataset_name, closure_ratio, label, *(scores[name] for name in COLUMNS)])


def main() -> None:
    """Run every configuration against every hierarchical dataset and closure ratio, print/persist results."""
    all_rows: list[tuple[str, float, str, dict[str, float]]] = []
    for dataset_class in DATASET_CLASSES:
        dataset = dataset_class()
        dataset_name = type(dataset).__name__
        hierarchy_relation = resolve_hierarchy_relation(
            dataset, getattr(dataset, "hierarchical_relation", None) if isinstance(dataset, HierarchicalGraph) else None
        )

        for closure_ratio in CLOSURE_RATIOS:
            print("=" * 104)
            print(f"Dataset: {dataset_name}  task: multi-hop subsumption prediction")
            print(
                f"  hierarchy_relation={hierarchy_relation!r}  closure_ratio={closure_ratio}  "
                f"eval_ratio={EVAL_RATIO}  negatives/positive={NUM_NEGATIVES}  epochs={EPOCHS}"
            )
            print("=" * 104)

            configs = build_configs(dataset, hierarchy_relation, closure_ratio)
            rows: list[tuple[str, dict[str, float]]] = []
            for label, model, model_kwargs, extra in configs:
                print(f"\n>>> {label}")
                start = time.time()
                error = None
                raw: dict | None = None
                try:
                    scores, raw = _evaluate_config(
                        dataset, hierarchy_relation, closure_ratio, model, model_kwargs, extra
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                    print(f"    FAILED: {exc}")
                    error = str(exc)
                    scores = dict.fromkeys(COLUMNS, float("nan"))
                duration = time.time() - start
                rows.append((label, scores))
                all_rows.append((dataset_name, closure_ratio, label, scores))
                _write_run_result(dataset_name, closure_ratio, label, scores, duration, error, raw)

            print("\n" + "-" * 104)
            header = f"{'Configuration':<32}" + "".join(f"{name:>8}" for name in COLUMNS)
            print(header)
            print("-" * len(header))
            for label, scores in rows:
                cells = "".join(f"{scores[name]:>8.4f}" for name in COLUMNS)
                print(f"{label:<32}{cells}")
            print("-" * 104)

    _write_summary(all_rows)
    print(f"\nDetailed per-run results written to {RESULTS_DIR}/<dataset>/<config>.json")
    print(f"Summary table written to {RESULTS_DIR}/summary.csv")


if __name__ == "__main__":
    main()
