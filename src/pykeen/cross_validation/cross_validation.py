"""Cross-validation pipeline for PyKEEN."""

from __future__ import annotations

import ftplib
import json
import logging
import pathlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs

from ..datasets import Dataset, dataset_resolver
from ..datasets.kfold.base import KFoldDataset, to_kfold
from ..evaluation import Evaluator
from ..losses import Loss
from ..lr_schedulers import LRScheduler
from ..models import Model
from ..nn.modules import Interaction
from ..optimizers import Optimizer
from ..pipeline import pipeline
from ..pipeline.api import PipelineResult
from ..regularizers import Regularizer
from ..sampling import NegativeSampler
from ..stoppers import Stopper
from ..trackers import ResultTracker
from ..training import TrainingLoop
from ..typing import DeviceHint, HintType
from ..utils import Result, fix_dataclass_init_docs, normalize_path
from ..version import get_git_hash, get_version

__all__ = [
    "CrossValidationPipelineResult",
    "cross_validation_pipeline",
]

logger = logging.getLogger(__name__)


@fix_dataclass_init_docs
@dataclass
class CrossValidationPipelineResult(Result):
    """Results from :func:`cross_validation_pipeline`.

    Contains per-fold :class:`~pykeen.pipeline.PipelineResult` objects as well as
    aggregated statistics (mean and standard deviation of each metric) computed
    across all folds.
    """

    #: One result per fold, in fold order
    fold_results: list[PipelineResult]

    #: The number of folds *k*
    num_folds: int

    #: Total training time in seconds, summed over all folds
    total_train_seconds: float

    #: Total evaluation time in seconds, summed over all folds
    total_evaluate_seconds: float

    #: Mean of each flat metric across folds (metric key → mean value)
    metric_means: dict[str, float]

    #: Standard deviation of each flat metric across folds (metric key → std value)
    metric_stds: dict[str, float]

    #: The version of PyKEEN used to create these results
    version: str = field(default_factory=get_version)

    #: The git hash of PyKEEN used to create these results
    git_hash: str = field(default_factory=get_git_hash)

    def get_metric(self, name: str) -> tuple[float, float]:
        """Return the ``(mean, std)`` pair for a named metric across folds.

        :param name: The metric name as returned by
            :meth:`~pykeen.metrics.MetricResults.to_flat_dict`.
        :returns: A ``(mean, std)`` tuple.
        :raises KeyError: If *name* is not found in the aggregated results.
        """
        return self.metric_means[name], self.metric_stds[name]

    def to_df(self) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` with one row per fold.

        Columns correspond to the flat metric names returned by
        :meth:`~pykeen.metrics.MetricResults.to_flat_dict`.

        :returns: A DataFrame of shape ``(num_folds, num_metrics)``.
        """
        rows = [fold.metric_results.to_flat_dict() for fold in self.fold_results]
        return pd.DataFrame(rows)

    def _get_results(self) -> dict[str, Any]:
        return {
            "num_folds": self.num_folds,
            "times": {
                "total_training": self.total_train_seconds,
                "total_evaluation": self.total_evaluate_seconds,
            },
            "metric_means": self.metric_means,
            "metric_stds": self.metric_stds,
            "version": self.version,
            "git_hash": self.git_hash,
        }

    def save_to_directory(
        self,
        directory: str | pathlib.Path,
        *,
        save_fold_results: bool = True,
        **_kwargs,
    ) -> None:
        """Save aggregate results and optionally per-fold sub-directories.

        Writes:

        - ``cv_results.json`` — aggregate statistics (means, stds, times).
        - ``fold_metrics.tsv`` — per-fold metric table (one row per fold).
        - ``fold-NNN/`` sub-directories — full :class:`~pykeen.pipeline.PipelineResult`
          saved via :meth:`~pykeen.pipeline.PipelineResult.save_to_directory`
          (only when *save_fold_results* is ``True``).

        :param directory: Destination directory (created if it does not exist).
        :param save_fold_results: Whether to save individual fold results in
            numbered sub-directories. Defaults to ``True``.
        """
        directory = normalize_path(directory, mkdir=True)
        with directory.joinpath("cv_results.json").open("w") as fh:
            json.dump(self._get_results(), fh, indent=2, sort_keys=True)
        self.to_df().to_csv(directory.joinpath("fold_metrics.tsv"), sep="\t", index=False)
        if save_fold_results:
            for i, fold_result in enumerate(self.fold_results):
                fold_dir = directory / f"fold-{i:03d}"
                fold_result.save_to_directory(str(fold_dir))

    def save_to_ftp(self, directory: str, ftp: ftplib.FTP) -> None:
        """Not implemented."""
        raise NotImplementedError("FTP saving is not implemented for CrossValidationPipelineResult.")

    def save_to_s3(self, directory: str, bucket: str, s3=None) -> None:
        """Not implemented."""
        raise NotImplementedError("S3 saving is not implemented for CrossValidationPipelineResult.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_kfold_subclasses(cls=KFoldDataset):
    """Yield all concrete (non-abstract) subclasses of *cls* recursively."""
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            yield sub
        yield from _iter_kfold_subclasses(sub)


def _resolve_kfold_dataset(
    dataset: None | str | Dataset | type[Dataset] | KFoldDataset | type[KFoldDataset],
    dataset_kwargs: Mapping[str, Any] | None,
    k: int,
    validation_ratio: float,
    kfold_random_state: int | None,
) -> KFoldDataset:
    """Resolve the *dataset* argument to a :class:`KFoldDataset`.

    Resolution order:

    1. :class:`KFoldDataset` instance → use directly.
    2. :class:`KFoldDataset` subclass → instantiate with *dataset_kwargs*.
    3. ``str`` naming a :class:`KFoldDataset` subclass → look up by class name
       (case-insensitive), then instantiate.
    4. :class:`~pykeen.datasets.Dataset` instance → wrap with
       :func:`~pykeen.datasets.kfold.base.to_kfold`.
    5. :class:`~pykeen.datasets.Dataset` subclass → instantiate, then wrap.
    6. ``str`` naming a :class:`~pykeen.datasets.Dataset` → resolve via
       :data:`~pykeen.datasets.dataset_resolver`, instantiate, then wrap.
    7. ``None`` → raises :exc:`ValueError`.

    :param dataset: The dataset specification.
    :param dataset_kwargs: Keyword arguments forwarded to the dataset constructor.
    :param k: Number of folds (used when wrapping a plain :class:`~pykeen.datasets.Dataset`).
    :param validation_ratio: Validation fraction per fold.
    :param kfold_random_state: Random seed for the k-fold split.
    :returns: A :class:`KFoldDataset`.
    :raises ValueError: If *dataset* is ``None``.
    :raises TypeError: If *dataset* has an unrecognised type.
    """
    if dataset is None:
        raise ValueError(
            "A dataset must be provided. Pass a KFoldDataset (instance or class), "
            "a regular Dataset (instance, class, or string name) together with k, "
            "or use to_kfold() before calling cross_validation_pipeline()."
        )

    # 1. Already a KFoldDataset instance
    if isinstance(dataset, KFoldDataset):
        return dataset

    # 2. KFoldDataset subclass (the class object itself)
    if isinstance(dataset, type) and issubclass(dataset, KFoldDataset):
        return dataset(**(dataset_kwargs or {}))

    # 3. String → try KFoldDataset subclasses first, then fall back to Dataset
    if isinstance(dataset, str):
        lower_name = dataset.lower()
        for kfold_cls in _iter_kfold_subclasses():
            if kfold_cls.__name__.lower() == lower_name:
                return kfold_cls(**(dataset_kwargs or {}))
        # Fall back: resolve as a regular Dataset, then wrap
        dataset_instance: Dataset = dataset_resolver.make(dataset, dataset_kwargs)
        return to_kfold(
            dataset_instance,
            k=k,
            validation_ratio=validation_ratio,
            random_state=kfold_random_state,
        )

    # 4. Dataset instance
    if isinstance(dataset, Dataset):
        return to_kfold(
            dataset,
            k=k,
            validation_ratio=validation_ratio,
            random_state=kfold_random_state,
        )

    # 5. Dataset subclass
    if isinstance(dataset, type) and issubclass(dataset, Dataset):
        dataset_instance = dataset(**(dataset_kwargs or {}))
        return to_kfold(
            dataset_instance,
            k=k,
            validation_ratio=validation_ratio,
            random_state=kfold_random_state,
        )

    raise TypeError(
        f"Cannot resolve dataset of type {type(dataset).__name__!r}: {dataset!r}. "
        "Expected a KFoldDataset/Dataset instance or subclass, or a string name."
    )


def _aggregate_fold_metrics(
    fold_results: list[PipelineResult],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-metric mean and population std across folds.

    :param fold_results: The list of :class:`~pykeen.pipeline.PipelineResult` objects,
        one per fold.
    :returns: A ``(means, stds)`` pair of flat metric dicts.
    """
    all_flat = [r.metric_results.to_flat_dict() for r in fold_results]
    keys = list(all_flat[0].keys())
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = np.array([d.get(key, float("nan")) for d in all_flat], dtype=float)
        means[key] = float(np.nanmean(values))
        stds[key] = float(np.nanstd(values, ddof=0))
    return means, stds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_validation_pipeline(
    *,
    # 0. Dataset + K-fold settings
    dataset: None | str | Dataset | type[Dataset] | KFoldDataset | type[KFoldDataset] = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    k: int = 5,
    validation_ratio: float = 0.1,
    kfold_random_state: int | None = None,
    evaluation_entity_whitelist: Collection[str] | None = None,
    evaluation_relation_whitelist: Collection[str] | None = None,
    # 1. Model
    model: None | str | Model | type[Model] = None,
    model_kwargs: Mapping[str, Any] | None = None,
    interaction: None | str | Interaction | type[Interaction] = None,
    interaction_kwargs: Mapping[str, Any] | None = None,
    dimensions: None | int | Mapping[str, int] = None,
    # 2. Loss
    loss: HintType[Loss] = None,
    loss_kwargs: Mapping[str, Any] | None = None,
    # 3. Regularizer
    regularizer: HintType[Regularizer] = None,
    regularizer_kwargs: Mapping[str, Any] | None = None,
    # 4. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    clear_optimizer: bool = True,
    # 4.1 Learning Rate Scheduler
    lr_scheduler: HintType[LRScheduler] = None,
    lr_scheduler_kwargs: Mapping[str, Any] | None = None,
    # 5. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Mapping[str, Any] | None = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Mapping[str, Any] | None = None,
    # 6. Training
    epochs: int | None = None,
    training_kwargs: Mapping[str, Any] | None = None,
    stopper: HintType[Stopper] = None,
    stopper_kwargs: Mapping[str, Any] | None = None,
    # 7. Evaluation
    evaluator: HintType[Evaluator] = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    # 8. Tracking
    result_tracker: OneOrManyHintOrType[ResultTracker] = None,
    result_tracker_kwargs: OneOrManyOptionalKwargs = None,
    # Misc
    metadata: dict[str, Any] | None = None,
    device: DeviceHint = None,
    random_seed: int | None = None,
    use_testing_data: bool = True,
    evaluation_fallback: bool = False,
    filter_validation_when_testing: bool = True,
    use_tqdm: bool | None = None,
) -> CrossValidationPipelineResult:
    """Train and evaluate a KGE model using k-fold cross-validation.

    Wraps :func:`~pykeen.pipeline.pipeline` and calls it once per fold, then
    aggregates the metric results across folds.  The function signature mirrors
    :func:`~pykeen.pipeline.pipeline` exactly, with three additional parameters
    controlling the cross-validation split (*k*, *validation_ratio*,
    *kfold_random_state*).

    :param dataset:
        A :class:`~pykeen.datasets.kfold.base.KFoldDataset` instance or subclass,
        a regular :class:`~pykeen.datasets.Dataset` instance or subclass (which will
        be split into *k* folds via
        :func:`~pykeen.datasets.kfold.base.to_kfold`), or a string name for either.
    :param dataset_kwargs:
        Keyword arguments forwarded to the dataset constructor when *dataset* is a
        class or string name.
    :param k:
        Number of folds.  Ignored when *dataset* is already a
        :class:`~pykeen.datasets.kfold.base.KFoldDataset`. Defaults to 5.
    :param validation_ratio:
        Fraction of non-test triples reserved for validation within each fold.
        Forwarded to :func:`~pykeen.datasets.kfold.base.to_kfold`.
        Ignored when *dataset* is a pre-built :class:`~pykeen.datasets.kfold.base.KFoldDataset`.
        Defaults to 0.1.
    :param kfold_random_state:
        Random seed for the k-fold triple split.  Distinct from *random_seed*,
        which governs model initialisation and training.  Ignored when *dataset* is
        a pre-built :class:`~pykeen.datasets.kfold.base.KFoldDataset`.
    :param evaluation_entity_whitelist:
        Passed to :func:`~pykeen.pipeline.pipeline` for each fold.
    :param evaluation_relation_whitelist:
        Passed to :func:`~pykeen.pipeline.pipeline` for each fold.
    :param model:
        The model to train — string name, class, or instance.
    :param model_kwargs:
        Keyword arguments for the model constructor.
    :param interaction:
        Interaction function hint (for :class:`~pykeen.models.ERModel`-based models).
    :param interaction_kwargs:
        Keyword arguments for the interaction function.
    :param dimensions:
        Embedding dimensionality shortcut.
    :param loss:
        Loss function hint.
    :param loss_kwargs:
        Keyword arguments for the loss function.
    :param regularizer:
        Regularizer hint.
    :param regularizer_kwargs:
        Keyword arguments for the regularizer.
    :param optimizer:
        Optimizer hint.
    :param optimizer_kwargs:
        Keyword arguments for the optimizer.
    :param clear_optimizer:
        Whether to clear the optimizer state after training. Defaults to ``True``.
    :param lr_scheduler:
        Learning rate scheduler hint.
    :param lr_scheduler_kwargs:
        Keyword arguments for the learning rate scheduler.
    :param training_loop:
        Training loop hint.
    :param training_loop_kwargs:
        Keyword arguments for the training loop.
    :param negative_sampler:
        Negative sampler hint (for SLCWA training).
    :param negative_sampler_kwargs:
        Keyword arguments for the negative sampler.
    :param epochs:
        Number of training epochs per fold.
    :param training_kwargs:
        Additional keyword arguments forwarded to the training loop's ``train()`` call.
    :param stopper:
        Early stopping hint.
    :param stopper_kwargs:
        Keyword arguments for the stopper.
    :param evaluator:
        Evaluator hint.
    :param evaluator_kwargs:
        Keyword arguments for the evaluator.
    :param evaluation_kwargs:
        Additional keyword arguments forwarded to the evaluator's ``evaluate()`` call.
    :param result_tracker:
        Result tracker hint.  A fresh tracker is created per fold by
        :func:`~pykeen.pipeline.pipeline`.
    :param result_tracker_kwargs:
        Keyword arguments for the result tracker.
    :param metadata:
        Additional metadata dict merged into each fold's metadata.  The keys
        ``cv_fold`` and ``cv_num_folds`` are added automatically.
    :param device:
        The device to use for training (e.g. ``"cuda"``).
    :param random_seed:
        Base random seed.  Fold *i* uses ``random_seed + i``, ensuring each fold
        is independently reproducible.  When ``None``, each fold draws a random seed.
    :param use_testing_data:
        Whether to evaluate on the test set. Defaults to ``True``.
    :param evaluation_fallback:
        Whether to fall back to a simpler evaluator on OOM errors. Defaults to ``False``.
    :param filter_validation_when_testing:
        Whether to filter validation triples during test evaluation. Defaults to ``True``.
    :param use_tqdm:
        Whether to show tqdm progress bars.
    :returns: A :class:`CrossValidationPipelineResult` aggregating all fold results.
    """
    kfold_dataset = _resolve_kfold_dataset(
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        k=k,
        validation_ratio=validation_ratio,
        kfold_random_state=kfold_random_state,
    )

    num_folds = kfold_dataset.num_folds
    fold_results: list[PipelineResult] = []

    for fold_index, fold in enumerate(kfold_dataset):
        fold_seed = (random_seed + fold_index) if random_seed is not None else None
        logger.info(
            "Running fold %d/%d (seed=%s)",
            fold_index + 1,
            num_folds,
            fold_seed,
        )

        fold_metadata: dict[str, Any] = {**(metadata or {})}
        fold_metadata["cv_fold"] = fold_index
        fold_metadata["cv_num_folds"] = num_folds
        result = pipeline(
            # Inject fold factories directly — bypasses dataset resolution in pipeline()
            training=fold.training,
            testing=fold.testing,
            validation=fold.validation,
            evaluation_entity_whitelist=evaluation_entity_whitelist,
            evaluation_relation_whitelist=evaluation_relation_whitelist,
            # Model
            model=model,
            model_kwargs=model_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            dimensions=dimensions,
            # Loss
            loss=loss,
            loss_kwargs=loss_kwargs,
            # Regularizer
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
            # Optimizer
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            clear_optimizer=clear_optimizer,
            # LR Scheduler
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            # Training Loop
            training_loop=training_loop,
            training_loop_kwargs=training_loop_kwargs,
            negative_sampler=negative_sampler,
            negative_sampler_kwargs=negative_sampler_kwargs,
            # Training
            epochs=epochs,
            training_kwargs=training_kwargs,
            stopper=stopper,
            stopper_kwargs=stopper_kwargs,
            # Evaluation
            evaluator=evaluator,
            evaluator_kwargs=evaluator_kwargs,
            evaluation_kwargs=evaluation_kwargs,
            # Tracking — pipeline() manages tracker lifecycle per fold
            result_tracker=result_tracker,
            result_tracker_kwargs=result_tracker_kwargs,
            # Misc
            metadata=fold_metadata,
            device=device,
            random_seed=fold_seed,
            use_testing_data=use_testing_data,
            evaluation_fallback=evaluation_fallback,
            filter_validation_when_testing=filter_validation_when_testing,
            use_tqdm=use_tqdm,
        )
        fold_results.append(result)

    metric_means, metric_stds = _aggregate_fold_metrics(fold_results)

    return CrossValidationPipelineResult(
        fold_results=fold_results,
        num_folds=num_folds,
        total_train_seconds=sum(r.train_seconds for r in fold_results),
        total_evaluate_seconds=sum(r.evaluate_seconds for r in fold_results),
        metric_means=metric_means,
        metric_stds=metric_stds,
    )
