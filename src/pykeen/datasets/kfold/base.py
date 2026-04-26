"""Base classes and utilities for k-fold dataset splitting."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch

from pykeen.datasets import Dataset
from pykeen.datasets.base import EagerDataset

__all__ = ["EagerKFoldDataset", "KFoldDataset", "to_kfold"]

logger = logging.getLogger(__name__)


class KFoldDataset(ABC):
    """Abstract base for k-fold dataset collections.

    A :class:`KFoldDataset` stores *k* pre-defined splits, each of which is a
    fully-formed :class:`~pykeen.datasets.Dataset` with ``training``,
    ``testing``, and ``validation`` triples factories.  It is intended for use
    exclusively with :func:`~pykeen.corss_validation.cross_validation.cross_validation_pipeline`.

    Subclasses must implement :meth:`_load_folds`.  Loading is lazy: folds are
    not materialised until the :attr:`datasets` property is first accessed.

    Example subclass skeleton::

        class MyKFoldDataset(KFoldDataset):
            def _load_folds(self) -> list[Dataset]:
                # download / read / construct k Dataset objects
                ...
    """

    _datasets: list[Dataset] | None = None

    @abstractmethod
    def _load_folds(self) -> list[Dataset]:
        """Load and return the list of k pre-defined dataset splits.

        :returns: A list of :class:`~pykeen.datasets.Dataset` objects, one per fold.
        """

    @property
    def datasets(self) -> list[Dataset]:
        """The list of k dataset folds, lazily loaded on first access."""
        if self._datasets is None:
            logger.debug("Loading folds for %s", self.__class__.__name__)
            self._datasets = self._load_folds()
        return self._datasets

    @property
    def num_folds(self) -> int:
        """The number of folds *k*."""
        return len(self.datasets)

    def __len__(self) -> int:
        return self.num_folds

    def __iter__(self) -> Iterator[Dataset]:
        return iter(self.datasets)

    def __getitem__(self, index: int) -> Dataset:
        return self.datasets[index]


class EagerKFoldDataset(KFoldDataset):
    """A :class:`KFoldDataset` backed by a pre-computed list of dataset folds.

    Mirrors :class:`~pykeen.datasets.base.EagerDataset` for the k-fold setting.
    Typically created via :func:`to_kfold` rather than directly.
    """

    def __init__(self, datasets: list[Dataset]) -> None:
        """Initialize with a pre-built list of dataset folds.

        :param datasets: The list of k :class:`~pykeen.datasets.Dataset` objects, one per fold.
        """
        self._datasets = datasets

    def _load_folds(self) -> list[Dataset]:  # noqa: D102
        datasets = self._datasets
        assert datasets is not None
        return datasets


def to_kfold(
    dataset: Dataset,
    k: int = 5,
    *,
    validation_ratio: float = 0.1,
    random_state: int | None = None,
) -> EagerKFoldDataset:
    """Create a k-fold dataset by re-splitting an existing dataset's triples.

    All triples from training, testing, and validation are pooled, shuffled,
    then divided into *k* equal chunks.  For each fold *i*, chunk *i* becomes
    the test set and the remaining chunks are used for training (with an optional
    held-out validation portion).

    Entity and relation mappings — as well as numeric literals for
    :class:`~pykeen.triples.TriplesNumericLiteralsFactory` — are preserved via
    :meth:`~pykeen.triples.TriplesFactory.clone_and_exchange_triples`.

    :param dataset: The source dataset whose triples are pooled and re-split.
    :param k: The number of folds. Defaults to 5.
    :param validation_ratio: Fraction of the non-test triples to reserve for
        validation.  Pass ``0.0`` to omit validation entirely. Defaults to 0.1.
    :param random_state: Optional integer seed for reproducibility.
    :returns: An :class:`EagerKFoldDataset` containing *k* folds.
    :raises ValueError: If *k* < 2.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")

    # Pool all triples
    parts = [dataset.training.mapped_triples, dataset.testing.mapped_triples]
    if dataset.validation is not None:
        parts.append(dataset.validation.mapped_triples)
    all_triples = torch.cat(parts, dim=0)

    # Shuffle
    generator = torch.Generator()
    if random_state is not None:
        generator.manual_seed(random_state)
    perm = torch.randperm(len(all_triples), generator=generator)
    all_triples = all_triples[perm]

    # Divide into k chunks (may differ by ≤1 triple — acceptable for CV)
    chunks = torch.chunk(all_triples, k, dim=0)

    factory = dataset.training  # reference factory; preserves mappings + numeric literals
    folds: list[Dataset] = []
    for i in range(k):
        test_triples = chunks[i]
        remaining = torch.cat([chunks[j] for j in range(k) if j != i], dim=0)

        test_factory = factory.clone_and_exchange_triples(test_triples)

        if validation_ratio > 0.0:
            train_factory, val_factory = factory.clone_and_exchange_triples(remaining).split(1.0 - validation_ratio)
        else:
            train_factory = factory.clone_and_exchange_triples(remaining)
            val_factory = None

        folds.append(EagerDataset(training=train_factory, testing=test_factory, validation=val_factory))

    return EagerKFoldDataset(folds)
