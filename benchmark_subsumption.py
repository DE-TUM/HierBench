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
EMBEDDING_DIM = 64
SEED = 42

#: Ganea et al. (2018) Table 1: F1 sweep over the fraction of non-basic transitive-closure edges
#: added to training, at embedding dims {5, 10}.
CLOSURE_RATIOS = [0.0, 0.1, 0.25, 0.5]
EVAL_RATIO = 0.05
NUM_NEGATIVES = 50

#: Nickel & Kiela (2017) Eq. 8: severity of the norm (depth) penalty, see test_subsumption.py.
ISA_ALPHA = 1.0

DATASET_CLASSES = [
    datasets.WN18RR,
    datasets.NASA,
    datasets.DOID,
    datasets.ACMCCS,
    datasets.EstatFSS,
    datasets.EuroSciVoc,
    datasets.MeSH,
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


def _make_hyperbolic_isa_score(dataset: object, hierarchy_relation: int | None):
    """Build a directional is-a score (Nickel & Kiela 2017, Eq. 8) closed over one dataset/relation.

    Per-dataset closure over ``test_subsumption.py``'s ``hyperbolic_isa_score``, which instead reads
    module-level ``DATASET``/``HIERARCHY_RELATION`` globals — not reusable across a dataset sweep.
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
        return -(1 + sign * ISA_ALPHA * (_ball_norm(emb, batch[:, 0]) - _ball_norm(emb, batch[:, 2]))) * dist

    return score


def _pretrain_poincare_initializer(
    dataset: object, hierarchy_relation: int | None, closure_ratio: float
) -> PretrainedInitializer:
    """Pretrain a Poincaré model and return a tangent-space initializer for HyperbolicCones.

    Ganea et al. (2018 §5): cones are warm-started from a Poincaré model pretrained for 100 epochs;
    the pretrained ball embeddings are rescaled by 0.7 (they collapse toward the border) then mapped
    back to tangent space with ``manifold.logmap0`` before being used as a tangent-space initializer.
    """
    result = subsumption_prediction_pipeline(
        dataset,
        model=PoincareE,
        model_kwargs={"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},
        epochs=CONES_PRETRAIN_EPOCHS,
        closure_ratio=closure_ratio,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        seed=SEED,
        random_seed=SEED,
        hierarchy_relation=hierarchy_relation,
        optimizer="RiemannianSGD",
        optimizer_kwargs={"lr": 0.3},
        loss="crossentropy",
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": 50},
        hierarchical=False,
        device=DEVICE,
    )
    emb = result.model.entity_representations[0]
    tangent = emb.manifold.logmap0(emb._embeddings.data * CONES_PRETRAIN_RESCALE)
    return PretrainedInitializer(tensor=tangent.detach().clone())


def build_configs(
    dataset: object, hierarchy_relation: int | None, closure_ratio: float
) -> list[tuple[str, object, dict, dict]]:
    """Build the per-dataset config list, including a freshly pretrained HyperbolicCones entry."""
    cones_initializer = _pretrain_poincare_initializer(dataset, hierarchy_relation, closure_ratio)
    isa_score = _make_hyperbolic_isa_score(dataset, hierarchy_relation)
    return [
        (
            "PoincareE (Nickel & Kiela 2017)",
            PoincareE,
            {"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},
            {
                "optimizer": "RiemannianSGD",
                "optimizer_kwargs": {"lr": 0.3},
                "loss": "crossentropy",
                "negative_sampler_kwargs": {"num_negs_per_pos": 50},
                "eval_score_fn": isa_score,
            },
        ),
        (
            "LorentzE (Nickel & Kiela 2018)",
            LorentzE,
            {"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},
            {
                "optimizer": "RiemannianSGD",
                "optimizer_kwargs": {"lr": 0.3},
                "loss": "crossentropy",
                "negative_sampler_kwargs": {"num_negs_per_pos": 50},
                "eval_score_fn": isa_score,
            },
        ),
        (
            "HyperbolicCones (Ganea et al. 2018)",
            HyperbolicCones,
            {
                "embedding_dim": EMBEDDING_DIM,
                "k": 0.1,
                "curvature": 1.0,
                "entity_initializer": cones_initializer,
            },
            {
                "optimizer": "RiemannianSGD",
                "optimizer_kwargs": {"lr": 1e-4},
                "loss": "marginranking",
                "loss_kwargs": {"margin": 0.01},
                "negative_sampler_kwargs": {"num_negs_per_pos": 50},
            },
        ),
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
) -> dict[str, float]:
    """Train one configuration, then score the held-out subsumptions under both negative settings."""
    extra = dict(extra)
    eval_score_fn = extra.pop("eval_score_fn", None)
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
        eval_score_fn=eval_score_fn,
        device=DEVICE,
        **extra,
    )
    random_scores = result.ancestor_descendant_metric_results
    hard_scores = subsumption_prediction_metrics(
        result.model,
        dataset,
        closure_ratio=closure_ratio,
        eval_ratio=EVAL_RATIO,
        num_negatives=NUM_NEGATIVES,
        hard_negatives=True,
        seed=SEED,
        hierarchy_relation=hierarchy_relation,
        eval_score_fn=eval_score_fn,
    )

    scores = {f"{name}·rnd": _get(random_scores, key) for name, key in PAIR_METRICS.items()}
    scores |= {f"{name}·hrd": _get(hard_scores, key) for name, key in PAIR_METRICS.items()}
    scores["mAP"] = _get(random_scores, "average_precision")
    scores["AUROC"] = _get(random_scores, "roc_auc")
    scores["MRR"] = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
    scores |= {
        name: _get(result.hierarchical_metric_results, f"both.{key}") for name, key in HIER_METRICS.items()
    }
    return scores


def _write_run_result(
    dataset_name: str,
    closure_ratio: float,
    label: str,
    scores: dict[str, float],
    duration: float,
    error: str | None,
) -> None:
    """Persist one run's full metrics plus metadata as JSON under results/subsumption/<dataset>/<closure%>/."""
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
                try:
                    scores = _evaluate_config(dataset, hierarchy_relation, closure_ratio, model, model_kwargs, extra)
                except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                    print(f"    FAILED: {exc}")
                    error = str(exc)
                    scores = dict.fromkeys(COLUMNS, float("nan"))
                duration = time.time() - start
                rows.append((label, scores))
                all_rows.append((dataset_name, closure_ratio, label, scores))
                _write_run_result(dataset_name, closure_ratio, label, scores, duration, error)

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
