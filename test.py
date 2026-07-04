"""Benchmark the hierarchy-completion (removed-edge) task across model / representation configs.

Randomly removes direct hierarchy edges (connectivity-preserving: no node is ever cut off from the
hierarchy), trains each configuration on the remaining edges, and evaluates link prediction on the
removed edges. Hyperbolic models (PoincareE, HyperbolicCones) are paired with a Riemannian optimiser;
Euclidean baselines (TransE, DistMult) use Adam.

Under sLCWA the negative sampler defaults to ``HierarchyNegativeSampler`` (same-depth "near-miss"
hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` per config to override.

The reworked API mirrors PyKEEN's function-style pipeline:

* one-call convenience: ``hierarchy_completion_pipeline(dataset, ...)``
* data prep only: ``hierarchy_completion_split(dataset, ...)`` -> ``(train, val, test)`` factories.

Three metric families are reported side by side:

* **Rank-based** (MRR / Hits@k / AMR) — the standard link-prediction metrics.
* **Classification** (F1 / precision / recall) — via :class:`~pykeen.evaluation.ClassificationEvaluator`.
* **Hierarchical** (hF1 / hP / hR) — ancestor-path overlap metrics that account for mistake severity.
"""

from __future__ import annotations

from pykeen.datasets import Cora, ACMCCS, DOID, EuroSciVoc
from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis
from pykeen.evaluation import ClassificationEvaluator, ClassificationMetricResults
from pykeen.models import HyperbolicCones, PoincareE
from pykeen.pipeline.hierarchy import (
    HierarchicalPipelineResult,
    HpoHierarchicalResult,
    hierarchy_completion_pipeline,
    hierarchy_completion_split,
    hpo_hierarchy_completion_pipeline,
)

# --- Shared experiment settings ---------------------------------------------------------
DATASET = EuroSciVoc()
EPOCHS = 100
EMBEDDING_DIM = 128
TEST_RATIO = 0.1
SEED = 42

# --- Configurations to compare ----------------------------------------------------------
# Each entry: (label, model, model_kwargs, extra pipeline kwargs)
CONFIGS: list[tuple[str, object, dict, dict]] = [
    (
        "PoincareE (hyperbolic dist.)",
        PoincareE,
        {"embedding_dim": EMBEDDING_DIM, "curvature": 1.0},
        {"optimizer": "RiemannianAdam", "optimizer_kwargs": {"lr": 0.05}},
    ),
    (
        "HyperbolicCones (entailment)",
        HyperbolicCones,
        {"embedding_dim": EMBEDDING_DIM, "k": 0.1, "curvature": 1.0},
        {"optimizer": "RiemannianAdam", "optimizer_kwargs": {"lr": 0.05}},
    ),
    (
        "TransE (Euclidean baseline)",
        "TransE",
        {"embedding_dim": EMBEDDING_DIM},
        {"optimizer": "Adam", "optimizer_kwargs": {"lr": 0.01}},
    ),
    (
        "DistMult (bilinear baseline)",
        "DistMult",
        {"embedding_dim": EMBEDDING_DIM},
        {"optimizer": "Adam", "optimizer_kwargs": {"lr": 0.01}},
    ),
    (
        "TransE (xavier init, dim=64)",
        "TransE",
        {"embedding_dim": 64, "entity_initializer": "xavier_normal_"},
        {"optimizer": "Adam", "optimizer_kwargs": {"lr": 0.01}},
    ),
]

# Rank-based metrics, read directly from the pipeline result.
RANK_METRICS = {
    "MRR": "both.realistic.inverse_harmonic_mean_rank",
    "H@1": "both.realistic.hits_at_1",
    "H@10": "both.realistic.hits_at_10",
    "AMR": "both.realistic.arithmetic_mean_rank",
}
# Regular classification metrics, from a ClassificationEvaluator pass.
CLASSIFICATION_METRICS = {
    "F1": "both.f1_score",
    "Prec": "both.positive_predictive_value",
    "Rec": "both.true_positive_rate",
}
# Hierarchical (ancestor-path) metrics, from the attached HierarchicalMetricResults.
HIERARCHICAL_METRICS = {
    "hF1": "both.hierarchical_f1",
    "hP": "both.hierarchical_precision",
    "hR": "both.hierarchical_recall",
}
ALL_METRIC_NAMES = [*RANK_METRICS, *CLASSIFICATION_METRICS, *HIERARCHICAL_METRICS]


def _hierarchy_summary() -> None:
    """Print a short structural summary of the dataset's training hierarchy and the eval setup."""
    training = DATASET.training
    print("=" * 96)
    print(f"Dataset: {type(DATASET).__name__}")
    print(f"  entities={training.num_entities}  relations={training.num_relations}  edges={training.num_triples}")
    print(f"  eval: removed-edge completion  test_ratio={TEST_RATIO}  epochs={EPOCHS}")
    print("=" * 96)


def _collect_metrics(
    result: HierarchicalPipelineResult, classification: ClassificationMetricResults | None
) -> dict[str, float]:
    """Gather rank-based, hierarchical, and classification metrics into a flat name -> value map."""
    scores: dict[str, float] = {}
    for name, key in RANK_METRICS.items():
        scores[name] = result.get_metric(key)
    hierarchical = result.hierarchical_metric_results
    for name, key in HIERARCHICAL_METRICS.items():
        scores[name] = hierarchical.get_metric(key) if hierarchical is not None else float("nan")
    for name, key in CLASSIFICATION_METRICS.items():
        scores[name] = classification.get_metric(key) if classification is not None else float("nan")
    return scores


def _evaluate_config(model: object, model_kwargs: dict, extra: dict) -> dict[str, float]:
    """Train one configuration on the removed-edge split and return all three metric families.

    The classification metrics are computed on the same (seed-deterministic) held-out split as the
    pipeline, so they line up exactly with the rank-based and hierarchical numbers.
    """
    train, val, test = hierarchy_completion_split(DATASET, test_ratio=TEST_RATIO, seed=SEED)
    result = hierarchy_completion_pipeline(
        DATASET,
        model=model,
        model_kwargs=model_kwargs,
        epochs=EPOCHS,
        negative_sampler="basic",
        test_ratio=TEST_RATIO,
        seed=SEED,
        random_seed=SEED,
        **extra,
    )
    classification = ClassificationEvaluator().evaluate(
        result.model,
        test.mapped_triples,
        additional_filter_triples=[train.mapped_triples, val.mapped_triples],
        use_tqdm=False,
    )
    return _collect_metrics(result, classification)


HPO_N_TRIALS = 20


def _run_hpo() -> HpoHierarchicalResult:
    """Run HPO for HyperbolicCones, then re-fit the best trial and report hierarchical metrics.

    HPO optimises the rank-based objective on the held-out removed edges; the winning configuration is
    retrained by ``hpo_hierarchy_completion_pipeline`` so its hierarchical precision/recall/F1 can be
    reported alongside the best hyperparameters.
    """
    print("\n>>> HPO: HyperbolicCones")
    outcome = hpo_hierarchy_completion_pipeline(
        DATASET,
        model=HyperbolicCones,
        optimizer="RiemannianAdam",
        optimizer_kwargs={"lr": 0.05},
        epochs=EPOCHS,
        n_trials=HPO_N_TRIALS,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )
    best = outcome.hpo_result.study.best_trial
    print(f"    Best trial #{best.number}  value={best.value:.4f}")
    for param, value in best.params.items():
        print(f"      {param}: {value}")
    hierarchical = outcome.result.hierarchical_metric_results if outcome.result is not None else None
    if hierarchical is not None:
        cells = "  ".join(f"{name}={hierarchical.get_metric(key):.4f}" for name, key in HIERARCHICAL_METRICS.items())
        print(f"    Re-fitted best trial (hierarchical): {cells}")
    return outcome


def main() -> None:
    """Run every configuration and print a side-by-side metric comparison."""
    _hierarchy_summary()

    rows: list[tuple[str, dict[str, float]]] = []
    for label, model, model_kwargs, extra in CONFIGS:
        print(f"\n>>> {label}")
        try:
            scores = _evaluate_config(model, model_kwargs, extra)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"    FAILED: {exc}")
            scores = dict.fromkeys(ALL_METRIC_NAMES, float("nan"))
        rows.append((label, scores))

    # --- Comparison table ---
    print("\n" + "=" * 96)
    print(f"{'Configuration comparison':^96}")
    print("=" * 96)
    print(f"{'':<32}{'rank-based':>32}{'classification':>16}{'hierarchical':>16}")
    header = f"{'Configuration':<32}" + "".join(f"{name:>8}" for name in ALL_METRIC_NAMES)
    print(header)
    print("-" * len(header))
    for label, scores in rows:
        cells = "".join(f"{scores[name]:>8.4f}" for name in ALL_METRIC_NAMES)
        print(f"{label:<32}{cells}")
    print("=" * 96)
    print("MRR / Hits@k / F1 / hF1: higher is better.  AMR (mean rank): lower is better.")

    # --- HPO sweep ---
    # _run_hpo()


if __name__ == "__main__":
    data = DATASET
    ea = ExtendedGraphAnalysis(data)
    print(ea.is_dag)
    print(f"Number of root nodes: {len(ea.root_nodes)}")
    print(f"Number of leaf nodes: {len(ea.leaf_nodes)}")
    print(f"Balance: {ea.balance}")
    print(f"Max Hierarchy Depth: {ea.max_hierarchy_depth}")
    print(f"Min Hierarchy Depth: {ea.min_hierarchy_depth}")
    print(f"Avg Hierarchy Depth: {ea.avg_hierarchy_depth}")

    # main()
