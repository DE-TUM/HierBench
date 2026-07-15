"""Multi-hop subsumption prediction (Ganea et al. 2018; He et al. 2024).

The model trains on all direct hierarchy edges (plus an optional fraction of the non-direct
transitive closure) and is evaluated on held-out indirect subsumptions, following the shared
Multi-hop Inference protocol of Ganea et al. (2018, Hyperbolic Entailment Cones,
https://arxiv.org/abs/1804.01882, §5) and He et al. (2024, Language Models as Hierarchy Encoders,
https://arxiv.org/abs/2401.11374, §4.1).

The public surface mirrors the other hierarchical tasks (shared internals in
:mod:`pykeen.pipeline.hierarchical_helper`):

* :func:`subsumption_prediction_split` is pure data preparation — it returns standard
  ``(train, val, test)`` :class:`~pykeen.triples.CoreTriplesFactory` instances.
* :func:`subsumption_prediction_pipeline` trains and additionally reports the thresholded
  precision/recall/F1 plus mAP/AUROC.
* :func:`subsumption_prediction_metrics` re-scores an already-trained model on the
  (seed-deterministic) split, so one training run can be evaluated under both the random and the
  hard (sibling) negative settings without re-training.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from .hierarchical_helper import (
    HierarchicalPipelineResult,
    _canonical_rows,
    _closure_pool,
    _default_hierarchy_sampler,
    _factory_from_rows,
    _sample_negatives,
    _train_and_score_hierarchical,
    build_ancestor_paths,
)
from ..datasets.base import Dataset
from ..datasets.metadata import resolve_hierarchy_relation
from ..evaluation.pair_classification_evaluator import PairClassificationMetricResults
from ..metrics.classification import AreaUnderTheReceiverOperatingCharacteristicCurve, AveragePrecisionScore
from ..models.base import Model
from ..models.nbase import ERModel
from ..triples import CoreTriplesFactory
from ..typing import MappedTriples

__all__ = [
    "subsumption_prediction_split",
    "subsumption_prediction_pipeline",
    "subsumption_prediction_metrics",
]

#: stateless, reused metric instances (mirrors AveragePrecisionScore/AreaUnderTheReceiverOperatingCharacteristicCurve
#: already registered in pykeen.metrics.classification)
_AVERAGE_PRECISION = AveragePrecisionScore()
_ROC_AUC = AreaUnderTheReceiverOperatingCharacteristicCurve()


def _sibling_map(direct: set[tuple[int, int]]) -> dict[int, list[int]]:
    """Map each entity to its siblings — entities sharing a direct neighbour on the same edge side.

    Orientation-agnostic: hierarchy relations may point parent->child (e.g. NASA ``has_subclass``)
    or child->parent (e.g. WN18RR ``_hypernym``), so two entities count as siblings when they share
    a direct predecessor *or* a direct successor. On a tree this reduces to the papers' definition
    (same parent) regardless of edge direction.
    """
    by_pred: dict[int, set[int]] = {}
    by_succ: dict[int, set[int]] = {}
    for h, t in direct:
        by_pred.setdefault(h, set()).add(t)
        by_succ.setdefault(t, set()).add(h)
    siblings: dict[int, set[int]] = {}
    for group in (*by_pred.values(), *by_succ.values()):
        for member in group:
            siblings.setdefault(member, set()).update(group - {member})
    return {node: sorted(sibs) for node, sibs in siblings.items()}


#: cache of the last few computed splits, keyed by dataset identity and split parameters. The split
#: is deterministic in its arguments but expensive (ancestor map over the full hierarchy), and
#: re-scoring one trained model under several settings (e.g. an alpha sweep, or random vs. hard
#: negatives) re-derives the identical split each time. The dataset is pinned in the value both to
#: guard against ``id()`` reuse and to validate the hit. Callers must not mutate the returned
#: objects (all current callers are read-only).
_SPLIT_CACHE: dict[tuple, tuple[Dataset, tuple]] = {}
_SPLIT_CACHE_SIZE = 2


def _subsumption_split(
    dataset: Dataset,
    *,
    closure_ratio: float,
    eval_ratio: float,
    seed: int,
    hierarchy_relation: int | None,
) -> tuple[
    list[list[int]], list[list[int]], list[list[int]], dict[int, frozenset[int]], set[tuple[int, int]]
]:
    """Split the non-direct transitive closure into train-extra/val/test (cached; multi-hop inference).

    Shared body of :func:`subsumption_prediction_split`, :func:`subsumption_prediction_pipeline`,
    and :func:`subsumption_prediction_metrics`. The same ``seed`` always reproduces the same split,
    so a trained model can be re-evaluated (e.g. with hard negatives) without re-training.
    Additionally returns the inclusive ancestor map and the direct edge set for negative sampling.
    Results are memoized in :data:`_SPLIT_CACHE`.
    """
    key = (id(dataset), closure_ratio, eval_ratio, seed, hierarchy_relation)
    hit = _SPLIT_CACHE.get(key)
    if hit is not None and hit[0] is dataset:
        return hit[1]
    result = _compute_subsumption_split(
        dataset,
        closure_ratio=closure_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        hierarchy_relation=hierarchy_relation,
    )
    while len(_SPLIT_CACHE) >= _SPLIT_CACHE_SIZE:
        _SPLIT_CACHE.pop(next(iter(_SPLIT_CACHE)))
    _SPLIT_CACHE[key] = (dataset, result)
    return result


def _compute_subsumption_split(
    dataset: Dataset,
    *,
    closure_ratio: float,
    eval_ratio: float,
    seed: int,
    hierarchy_relation: int | None,
) -> tuple[
    list[list[int]], list[list[int]], list[list[int]], dict[int, frozenset[int]], set[tuple[int, int]]
]:
    """Compute the split behind :func:`_subsumption_split` (uncached body).

    Datasets declaring :attr:`~pykeen.datasets.metadata.HierarchicalGraph.predefined_closure_split`
    (e.g. the WordNetNoun* ``maxn`` splits) already encode the closure split in their files: their
    training triples are used verbatim and their fixed validation/testing triples become the eval
    positives, while ``closure_ratio``/``eval_ratio``/``seed`` are ignored for splitting (the seed
    still drives negative sampling downstream). Re-deriving a split here would leak the closure
    edges already present in training back into evaluation.
    """
    if getattr(dataset, "predefined_closure_split", False):

        def hierarchy_rows(factory: CoreTriplesFactory) -> list[list[int]]:
            rows = _canonical_rows(factory.mapped_triples, dataset, hierarchy_relation)
            return [row for row in rows if hierarchy_relation is None or row[1] == hierarchy_relation]

        train_rows = hierarchy_rows(dataset.training)
        val_rows = hierarchy_rows(dataset.validation)
        test_rows = hierarchy_rows(dataset.testing)
        # No transitive reduction here: it is O(V*E) (intractable at WordNet scale) and only the
        # asserted edges are needed — ``direct`` merely feeds negative sampling and the sibling map.
        paths = build_ancestor_paths(torch.as_tensor(train_rows), dataset.num_entities, hierarchy_relation)
        direct = {(h, t) for h, _r, t in train_rows}
        return train_rows, val_rows, test_rows, paths, direct
    if not 0 <= closure_ratio <= 1 or not 0 <= eval_ratio <= 0.5 or closure_ratio + 2 * eval_ratio > 1:
        raise ValueError(
            f"closure_ratio ({closure_ratio}) plus twice eval_ratio ({eval_ratio}) must not exceed 1, "
            "and both must be non-negative"
        )
    paths, all_rows, direct, pool = _closure_pool(dataset, hierarchy_relation)
    if not pool:
        warnings.warn(
            "The transitive closure contains no non-direct ancestor-descendant pairs; validation/test will "
            "be empty. The hierarchy is likely depth-1 (all closure pairs are direct edges). Check "
            "`hierarchy_relation` or use a deeper hierarchy.",
            stacklevel=3,
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    n_eval = int(len(pool) * eval_ratio)
    n_extra = int(len(pool) * closure_ratio)
    val_pairs = pool[:n_eval]
    test_pairs = pool[n_eval : 2 * n_eval]
    # Ganea et al. (2018 §5): a fraction of the non-basic closure edges joins the training set,
    # drawn from the pairs left over after the held-out validation/test portions.
    extra_pairs = pool[2 * n_eval : 2 * n_eval + n_extra]

    hierarchy_id = hierarchy_relation if hierarchy_relation is not None else 0
    # Training is always restricted to the hierarchy relation's direct edges: triples of other
    # relations would pull the embeddings in non-hierarchical directions while only the hierarchy
    # geometry is evaluated.
    if hierarchy_relation is not None:
        train_rows = [[h, hierarchy_relation, t] for h, t in sorted(direct)]
    else:
        train_rows = list(all_rows)
    train_rows += [[h, hierarchy_id, t] for h, t in extra_pairs]
    val_rows = [[h, hierarchy_id, t] for h, t in val_pairs]
    test_rows = [[h, hierarchy_id, t] for h, t in test_pairs]
    return train_rows, val_rows, test_rows, paths, direct


def subsumption_prediction_split(
    dataset: Dataset,
    *,
    closure_ratio: float = 0.0,
    eval_ratio: float = 0.05,
    seed: int = 42,
    hierarchy_relation: int | str | None = None,
) -> tuple[CoreTriplesFactory, CoreTriplesFactory, CoreTriplesFactory]:
    """Build ``(train, val, test)`` factories for multi-hop subsumption prediction.

    Follows the Multi-hop Inference protocol shared by Ganea et al. (2018,
    https://arxiv.org/abs/1804.01882, §5) and He et al. (2024,
    https://arxiv.org/abs/2401.11374, §4.1): the model trains on all direct hierarchy edges plus an
    optional fraction of the non-direct transitive closure, and is evaluated on two disjoint
    held-out portions of the remaining closure pairs (validation/test).

    :param dataset: A hierarchical dataset. Its training edges define the hierarchy.
    :param closure_ratio: Fraction of the non-direct closure pairs added to training — Ganea et
        al.'s 0%/10%/25%/50% transitive-closure axis. Default 0.0 (= He et al.'s Multi-hop
        Inference: train on asserted edges only).
    :param eval_ratio: Fraction of the non-direct closure pairs held out for validation and (again)
        for test. Default 0.05, following He et al. (2024, §4.2).
    :param seed: Random seed for reproducible splitting.
    :param hierarchy_relation: Relation id or label whose edges define the hierarchy. Defaults to the
        dataset's :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when it is a
        :class:`~pykeen.datasets.metadata.HierarchicalGraph`; otherwise all training edges are used.

    :returns: A ``(train, val, test)`` tuple of :class:`~pykeen.triples.CoreTriplesFactory` instances.
    """
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
    train_rows, val_rows, test_rows, _paths, _direct = _subsumption_split(
        dataset,
        closure_ratio=closure_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        hierarchy_relation=hierarchy_relation,
    )
    return (
        _factory_from_rows(train_rows, dataset),
        _factory_from_rows(val_rows, dataset),
        _factory_from_rows(test_rows, dataset),
    )


def _subsumption_pair_metrics(
    model: Model,
    val_rows: list[list[int]],
    test_rows: list[list[int]],
    paths: Mapping[int, frozenset[int]],
    direct: set[tuple[int, int]],
    num_entities: int,
    *,
    num_negatives: int,
    hard_negatives: bool,
    seed: int,
    score_fn: Callable[[Model, MappedTriples], torch.Tensor] | None = None,
    return_raw: bool = False,
) -> PairClassificationMetricResults | None | tuple[PairClassificationMetricResults | None, dict | None]:
    """Score subsumption pairs against 1:``num_negatives`` negatives and threshold on validation.

    Implements the shared evaluation of Ganea et al. (2018 §5) and He et al. (2024 §3/§4.1):
    random negatives corrupt both slots evenly (Ganea et al.'s five ``(u', v)`` plus five
    ``(u, v')`` at the default 10), the score threshold maximising F1 is picked on the validation
    pairs and applied to the test pairs, yielding precision/recall/F1; mAP/AUROC come from the
    same test scores for free.

    :param return_raw: When ``True``, additionally return a dict of the raw per-pair test
        ``rows``/``labels``/``scores``/``predictions``, the ``threshold``, and the ``val_f1``
        achieved at that threshold on validation (``None`` when nothing was scored), for error
        analysis, plotting, and validation-based hyperparameter selection.
    """
    if not val_rows or not test_rows:
        warnings.warn("empty validation or test subsumption pairs; skipping threshold metrics", stacklevel=2)
        return (None, None) if return_raw else None
    siblings = _sibling_map(direct) if hard_negatives else None
    rng = np.random.default_rng(seed)

    def _score(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray, list[list[int]]] | None:
        negatives = _sample_negatives(
            rows, paths, num_entities, rng, num_negatives=num_negatives, siblings=siblings, two_sided=True
        )
        if not negatives:
            return None
        pairs = rows + negatives
        batch = torch.tensor(pairs, dtype=torch.long).to(model.device)
        with torch.inference_mode():
            raw = model.predict_hrt(batch) if score_fn is None else score_fn(model, batch)
        scores = raw.squeeze(-1).detach().cpu().numpy()
        labels = np.concatenate([np.ones(len(rows)), np.zeros(len(negatives))])
        return scores, labels, pairs

    val = _score(val_rows)
    test = _score(test_rows)
    if val is None or test is None:
        warnings.warn("no valid negative pairs could be sampled; skipping threshold metrics", stacklevel=2)
        return (None, None) if return_raw else None
    val_scores, val_labels, _ = val
    test_scores, test_labels, test_pairs = test

    precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0)
    # the last precision/recall point (1, 0) has no threshold, hence f1[:-1]
    threshold = float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else float(val_scores.min())
    predictions = test_scores >= threshold
    results = PairClassificationMetricResults(
        data={
            "precision": float(precision_score(test_labels, predictions, zero_division=0)),
            "recall": float(recall_score(test_labels, predictions, zero_division=0)),
            "f1": float(f1_score(test_labels, predictions, zero_division=0)),
            "threshold": threshold,
            "average_precision": _AVERAGE_PRECISION(test_labels, test_scores),
            "roc_auc": _ROC_AUC(test_labels, test_scores),
        }
    )
    if not return_raw:
        return results
    raw = {
        "rows": test_pairs,
        "labels": test_labels.tolist(),
        "scores": test_scores.tolist(),
        "predictions": predictions.tolist(),
        "threshold": threshold,
        "val_f1": float(f1[:-1].max()) if len(thresholds) else 0.0,
    }
    return results, raw


def subsumption_prediction_metrics(
    model: Model,
    dataset: Dataset,
    *,
    closure_ratio: float = 0.0,
    eval_ratio: float = 0.05,
    num_negatives: int = 10,
    hard_negatives: bool = False,
    seed: int = 42,
    hierarchy_relation: int | str | None = None,
    eval_score_fn: Callable[[Model, MappedTriples], torch.Tensor] | None = None,
    return_raw: bool = False,
) -> PairClassificationMetricResults | None | tuple[PairClassificationMetricResults | None, dict | None]:
    """Evaluate a trained model on the (seed-deterministic) subsumption split.

    Rebuilds the same split as :func:`subsumption_prediction_pipeline` (given identical
    ``closure_ratio``/``eval_ratio``/``seed``/``hierarchy_relation``), samples ``num_negatives``
    negatives per positive, tunes the F1-optimal score threshold on validation, and returns test
    metrics. This lets one training run be scored under both the random and hard (sibling) negative
    settings, mirroring the side-by-side tables of He et al. (2024, Table 2) and Ganea et al.
    (2018, Table 1), without re-training.

    :param model: A trained model (e.g. from :func:`subsumption_prediction_pipeline`).
    :param dataset: The hierarchical dataset the model was trained on.
    :param closure_ratio: Must match the value used for training; see
        :func:`subsumption_prediction_split`.
    :param eval_ratio: Must match the value used for training; see
        :func:`subsumption_prediction_split`.
    :param num_negatives: Negatives sampled per positive pair. Default 10 (both papers).
    :param hard_negatives: Draw negatives from the kept entity's siblings first (an entity paired
        with its own sibling), topping up with random entities — He et al.'s hard negative setting.
    :param seed: Random seed; must match the training split's seed for the held-out pairs to line up.
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param eval_score_fn: Optional ``(model, batch) -> scores`` override of ``model.predict_hrt``;
        needed for symmetric-distance models (e.g. Poincaré embeddings), whose raw distance cannot
        tell ancestor from descendant.
    :param return_raw: When ``True``, return ``(metrics, raw)`` where ``raw`` carries the per-pair
        test ``rows``/``labels``/``scores``/``predictions``/``threshold`` (``None`` when nothing was
        evaluated).

    :returns: ``{"precision", "recall", "f1", "threshold", "average_precision", "roc_auc"}`` on the
        test pairs, or ``None`` when there is nothing to evaluate.
    """
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
    _train_rows, val_rows, test_rows, paths, direct = _subsumption_split(
        dataset,
        closure_ratio=closure_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        hierarchy_relation=hierarchy_relation,
    )
    return _subsumption_pair_metrics(
        model,
        val_rows,
        test_rows,
        paths,
        direct,
        dataset.num_entities,
        num_negatives=num_negatives,
        hard_negatives=hard_negatives,
        seed=seed,
        score_fn=eval_score_fn,
        return_raw=return_raw,
    )


def subsumption_prediction_pipeline(
    dataset: Dataset,
    *,
    model: type[ERModel] | str | None = None,
    embedding_dim: int = 64,
    epochs: int = 100,
    closure_ratio: float = 0.0,
    eval_ratio: float = 0.05,
    num_negatives: int = 10,
    hard_negatives: bool = False,
    seed: int = 42,
    hierarchical: bool = True,
    hierarchy_relation: int | str | None = None,
    eval_score_fn: Callable[[Model, MappedTriples], torch.Tensor] | None = None,
    return_raw: bool = False,
    **pipeline_kwargs,
) -> HierarchicalPipelineResult:
    """Train on the direct edges (plus optional closure fraction) and predict held-out subsumptions.

    Multi-hop subsumption prediction following Ganea et al. (2018, Hyperbolic Entailment Cones, §5)
    and He et al. (2024, Language Models as Hierarchy Encoders, §4.1 Multi-hop Inference): the split
    comes from :func:`subsumption_prediction_split`, each held-out positive is paired with
    ``num_negatives`` negatives (random, or hard sibling negatives), and the F1-optimal score
    threshold is tuned on validation and applied to test. The resulting
    precision/recall/F1/threshold plus mAP/AUROC land on ``ancestor_descendant_metric_results``.
    Unless ``hierarchical`` is ``False``, hierarchical precision/recall/F1 are reported as well.

    :param dataset: A hierarchical dataset.
    :param model: Model class, string alias, or instance forwarded to
        :func:`pykeen.pipeline.pipeline`. When ``None``, uses
        :class:`~pykeen.models.unimodal.PoincareE` with ``embedding_dim``.
    :param embedding_dim: Dimensionality of entity embeddings. Only used when ``model`` is ``None``.
    :param epochs: Number of training epochs.
    :param closure_ratio: Fraction of the non-direct closure pairs added to training (Ganea et al.'s
        0%/10%/25%/50% axis). Default 0.0.
    :param eval_ratio: Fraction of the non-direct closure pairs held out for validation and (again)
        for test. Default 0.05 (He et al. 2024, §4.2).
    :param num_negatives: Negatives sampled per held-out positive. Default 10 (both papers).
    :param hard_negatives: Use He et al.'s hard negative setting (siblings first, random top-up).
    :param seed: Random seed for reproducible splitting and negative generation.
    :param hierarchical: Whether to compute the hierarchical metrics (an extra evaluation pass).
    :param hierarchy_relation: Relation id or label defining the hierarchy; defaults to the dataset's
        :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when available.
    :param eval_score_fn: Optional ``(model, batch) -> scores`` override of ``model.predict_hrt``
        for the pair scorer; needed for symmetric-distance models (e.g. Poincaré embeddings), whose
        raw distance cannot tell ancestor from descendant.
    :param return_raw: When ``True``, the per-pair test ``rows``/``labels``/``scores``/
        ``predictions``/``threshold`` are stashed on ``ancestor_descendant_raw_predictions`` of the
        returned result for error analysis and plotting.
    :param pipeline_kwargs: Additional kwargs forwarded to :func:`pykeen.pipeline.pipeline`. Under
        sLCWA the negative sampler defaults to :class:`~pykeen.sampling.HierarchyNegativeSampler`
        (same-depth hard negatives); pass ``negative_sampler="pseudotyped"`` / ``"basic"`` to override.

    :returns: A :class:`~pykeen.pipeline.hierarchical_helper.HierarchicalPipelineResult` with
        ``ancestor_descendant_metric_results`` set to the thresholded subsumption metrics.
    """
    hierarchy_relation = resolve_hierarchy_relation(dataset, hierarchy_relation)
    train_rows, val_rows, test_rows, paths, direct = _subsumption_split(
        dataset,
        closure_ratio=closure_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        hierarchy_relation=hierarchy_relation,
    )
    train_factory = _factory_from_rows(train_rows, dataset)
    val_factory = _factory_from_rows(val_rows, dataset)
    test_factory = _factory_from_rows(test_rows, dataset)
    pipeline_kwargs = _default_hierarchy_sampler(pipeline_kwargs, hierarchy_relation)
    result = _train_and_score_hierarchical(
        dataset,
        train_factory,
        val_factory,
        test_factory,
        model=model,
        embedding_dim=embedding_dim,
        epochs=epochs,
        hierarchical=hierarchical,
        hierarchy_relation=hierarchy_relation,
        ancestors=paths,
        **pipeline_kwargs,
    )
    pair_metrics = _subsumption_pair_metrics(
        result.model,
        val_rows,
        test_rows,
        paths,
        direct,
        dataset.num_entities,
        num_negatives=num_negatives,
        hard_negatives=hard_negatives,
        seed=seed,
        score_fn=eval_score_fn,
        return_raw=return_raw,
    )
    if return_raw:
        result.ancestor_descendant_metric_results, result.ancestor_descendant_raw_predictions = pair_metrics
    else:
        result.ancestor_descendant_metric_results = pair_metrics
    return result
