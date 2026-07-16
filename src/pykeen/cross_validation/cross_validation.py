"""Cross-validation pipeline for PyKEEN."""

from __future__ import annotations

import ftplib
import inspect
import json
import logging
import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..datasets import Dataset, dataset_resolver
from ..datasets.kfold.base import KFoldDataset, to_kfold
from ..pipeline import pipeline
from ..pipeline.api import PipelineResult
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
        rows = [_flat_metrics(fold) for fold in self.fold_results]
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


def _flat_metrics(result: PipelineResult) -> dict[str, Any]:
    """Flatten a fold result's rank-based metrics plus any task-specific metrics.

    Task pipelines (e.g. :func:`~pykeen.pipeline.subsumption.subsumption_prediction_pipeline`)
    return a :class:`~pykeen.pipeline.hierarchical_helper.HierarchicalPipelineResult` carrying an
    extra ``ancestor_descendant_metric_results``; its metrics are merged under an
    ``ancestor_descendant.`` prefix so they aggregate alongside the standard ones.

    :param result: A fold's pipeline result.
    :returns: A flat ``metric name -> value`` dict.
    """
    flat = dict(result.metric_results.to_flat_dict())
    extra = getattr(result, "ancestor_descendant_metric_results", None)
    if extra is not None:
        flat.update({f"ancestor_descendant.{k}": v for k, v in extra.to_flat_dict().items()})
    return flat


def _aggregate_fold_metrics(
    fold_results: list[PipelineResult],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-metric mean and population std across folds.

    :param fold_results: The list of :class:`~pykeen.pipeline.PipelineResult` objects,
        one per fold.
    :returns: A ``(means, stds)`` pair of flat metric dicts.
    """
    all_flat = [_flat_metrics(r) for r in fold_results]
    keys = sorted({k for d in all_flat for k in d})
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
    # K-fold settings
    dataset: None | str | Dataset | type[Dataset] | KFoldDataset | type[KFoldDataset] = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    k: int = 5,
    validation_ratio: float = 0.1,
    kfold_random_state: int | None = None,
    # The per-fold pipeline and its (cross-cutting) arguments
    pipeline: Callable[..., PipelineResult] = pipeline,
    random_seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    **pipeline_kwargs: Any,
) -> CrossValidationPipelineResult:
    """Train and evaluate a model using k-fold cross-validation.

    Splits *dataset* into *k* folds and calls *pipeline* once per fold — passing each fold as
    ``dataset=`` — then aggregates the metric results across folds.  Any pipeline that accepts a
    :class:`~pykeen.datasets.Dataset` works: the stock :func:`~pykeen.pipeline.pipeline` (default),
    :func:`~pykeen.pipeline.subsumption.subsumption_prediction_pipeline`,
    :func:`~pykeen.pipeline.hierarchy.hierarchy_completion_pipeline`, etc.  All pipeline-specific
    settings (``model``, ``epochs``, ``training_kwargs``, ``closure_ratio``, …) are forwarded via
    *pipeline_kwargs*.

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
    :param pipeline:
        The per-fold pipeline callable.  Must accept a ``dataset=`` keyword argument and return a
        :class:`~pykeen.pipeline.PipelineResult`.  Defaults to :func:`~pykeen.pipeline.pipeline`.
    :param random_seed:
        Base random seed.  Fold *i* uses ``random_seed + i``, ensuring each fold is independently
        reproducible.  Injected into *pipeline* under whichever seed argument it exposes
        (``random_seed`` or ``seed``), unless already given in *pipeline_kwargs*.  When ``None``,
        each fold draws its own seed.
    :param metadata:
        Additional metadata dict merged into each fold's metadata (when the pipeline accepts a
        ``metadata`` argument).  The keys ``cv_fold`` and ``cv_num_folds`` are added automatically.
    :param pipeline_kwargs:
        Additional keyword arguments forwarded verbatim to *pipeline* for every fold.
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

    # Introspect the target pipeline once to route the cross-cutting seed/metadata arguments.
    sig = inspect.signature(pipeline).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.values())
    seed_key = "random_seed" if "random_seed" in sig else ("seed" if "seed" in sig else None)
    accepts_metadata = "metadata" in sig or has_var_kw

    for fold_index, fold in enumerate(kfold_dataset):
        call_kwargs = dict(pipeline_kwargs)

        if random_seed is not None and seed_key and seed_key not in call_kwargs:
            call_kwargs[seed_key] = random_seed + fold_index

        if accepts_metadata:
            call_kwargs["metadata"] = {
                **(metadata or {}),
                **(call_kwargs.get("metadata") or {}),
                "cv_fold": fold_index,
                "cv_num_folds": num_folds,
            }

        logger.info(
            "Running fold %d/%d (%s=%s)",
            fold_index + 1,
            num_folds,
            seed_key or "seed",
            call_kwargs.get(seed_key) if seed_key else None,
        )
        fold_results.append(pipeline(dataset=fold, **call_kwargs))

    metric_means, metric_stds = _aggregate_fold_metrics(fold_results)

    return CrossValidationPipelineResult(
        fold_results=fold_results,
        num_folds=num_folds,
        total_train_seconds=sum(r.train_seconds for r in fold_results),
        total_evaluate_seconds=sum(r.evaluate_seconds for r in fold_results),
        metric_means=metric_means,
        metric_stds=metric_stds,
    )
