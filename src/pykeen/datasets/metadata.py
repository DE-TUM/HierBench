"""Metadata-aware dataset classes with optional per-entity and per-relation context."""

from __future__ import annotations

import logging
import pathlib
from typing import Any, ClassVar, Mapping, Sequence

import pandas as pd
from typing import TypeAlias
from ..constants import PYKEEN_DATASETS
from ..triples import TriplesFactory
from ..typing import TorchRandomHint
from ..utils import normalize_path
from .base import EagerDataset

__all__ = [
    "MetadataDataset",
    "RemoteMetadataDataset",
]

logger = logging.getLogger(__name__)
Metadata: TypeAlias = Mapping[str, Any]


def _load_metadata_file(path: pathlib.Path) -> pd.DataFrame:
    """Load a metadata file into a DataFrame, auto-detecting TSV / JSONL / CSV by extension."""
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep="\t")



class MetadataDataset(EagerDataset):
    """A dataset that optionally pairs KG triples with per-entity and per-relation metadata.

    Metadata is exposed as :class:`pandas.DataFrame` via :attr:`entity_metadata` and
    :attr:`relation_metadata`. All conversion (to tensors, embeddings, etc.) is left
    to the caller.

    Graph-level attributes (name, description, stats, etc.) belong in the inherited
    :attr:`~pykeen.datasets.base.Dataset.metadata` mapping.

    Example::

        ds = Cora()
        df = ds.entity_metadata          # pandas DataFrame, one row per entity
        features = torch.tensor(df[numeric_cols].values, dtype=torch.float32)
    """

    def __init__(
        self,
        training: TriplesFactory,
        testing: TriplesFactory,
        validation: TriplesFactory | None = None,
        *,
        entity_metadata: pd.DataFrame | None = None,
        relation_metadata: pd.DataFrame | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Initialize the metadata dataset.

        :param training: Triples factory for training triples.
        :param testing: Triples factory for testing triples.
        :param validation: Optional triples factory for validation triples.
        :param entity_metadata: DataFrame with one row per entity (ordered by entity ID).
            Columns are freely chosen by the dataset author. ``None`` means no metadata.
        :param relation_metadata: Optional DataFrame with one row per relation.
        :param metadata: Dataset-level metadata (name, description, etc.) stored in the
            inherited :attr:`~pykeen.datasets.base.Dataset.metadata` mapping.
        """
        super().__init__(training, testing, validation, metadata=metadata)
        self.entity_metadata: pd.DataFrame | None = entity_metadata
        self.relation_metadata: pd.DataFrame | None = relation_metadata


class RemoteMetadataDataset(MetadataDataset):
    """A :class:`MetadataDataset` that downloads its triples and metadata from URLs.

    This is the recommended base class for concrete datasets. Dataset authors
    only need to set class-level URL attributes — no ``__init__`` boilerplate
    required::

        class Cora(RemoteMetadataDataset):
            triples_url          = "https://example.com/cora/edges.tsv"
            entity_metadata_url  = "https://example.com/cora/node_features.tsv"
            relation_metadata_url = "https://example.com/cora/rel_features.tsv"

    Override :meth:`_load_entity_metadata` or :meth:`_load_relation_metadata` to
    support custom file formats (NumPy, HDF5, pickle, …)::

        class MyDataset(RemoteMetadataDataset):
            triples_url         = "https://example.com/edges.tsv"
            entity_metadata_url = "https://example.com/features.npy"

            def _load_entity_metadata(self, path):
                import numpy as np
                arr = np.load(path)
                return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
    """

    #: URL of the tab-separated triples file (head, relation, tail).
    triples_url: ClassVar[str]
    #: URL of the entity metadata file. ``None`` means no entity metadata.
    entity_metadata_url: ClassVar[str | None] = None
    #: URL of the relation metadata file. ``None`` means no relation metadata.
    relation_metadata_url: ClassVar[str | None] = None
    #: Train / test / validation split ratios.
    ratios: ClassVar[Sequence[float]] = (0.8, 0.1, 0.1)

    def __init__(
        self,
        *,
        cache_root: str | pathlib.Path | None = None,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        download_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the remote metadata dataset.

        :param cache_root: Local directory for cached files. Defaults to a
            sub-directory of :data:`pykeen.constants.PYKEEN_DATASETS` named
            after the concrete subclass.
        :param create_inverse_triples: Whether to add inverse triples to training.
        :param random_state: Random state for the train/test/val split.
        :param download_kwargs: Extra keyword arguments forwarded to
            :func:`pystow.utils.download`.
        :param kwargs: Forwarded to :class:`MetadataDataset`.
        """
        from pystow.utils import download

        dataset_root = normalize_path(
            cache_root,
            type(self).__name__.lower(),
            mkdir=True,
            default=PYKEEN_DATASETS,
        )
        download_kwargs = download_kwargs or {}

        triples_path = dataset_root / "dataset.tsv"
        if not triples_path.is_file():
            logger.info("downloading triples from %s", self.triples_url)
            download(url=self.triples_url, path=triples_path, **download_kwargs)  # noqa: S310

        entity_metadata = self._download_and_load(
            self.entity_metadata_url, "entity_metadata", dataset_root, download, download_kwargs
        )
        relation_metadata = self._download_and_load(
            self.relation_metadata_url, "relation_metadata", dataset_root, download, download_kwargs
        )

        full_factory = TriplesFactory.from_path(triples_path, create_inverse_triples=create_inverse_triples)
        training, testing, validation = full_factory.split(ratios=self.ratios, random_state=random_state)

        super().__init__(
            training=training,
            testing=testing,
            validation=validation,
            entity_metadata=entity_metadata,
            relation_metadata=relation_metadata,
            **kwargs,
        )

    def _download_and_load(
        self,
        url: str | None,
        stem: str,
        dataset_root: pathlib.Path,
        download_fn: Any,
        download_kwargs: dict[str, Any],
    ) -> pd.DataFrame | None:
        if url is None:
            return None
        suffix = pathlib.Path(url).suffix or ".tsv"
        path = dataset_root / f"{stem}{suffix}"
        if not path.is_file():
            logger.info("downloading %s from %s", stem, url)
            download_fn(url=url, path=path, **download_kwargs)  # noqa: S310
        if stem == "entity_metadata":
            return self._load_entity_metadata(path)
        return self._load_relation_metadata(path)

    # ------------------------------------------------------------------
    # Subclass hooks — override to support custom file formats
    # ------------------------------------------------------------------

    def _load_entity_metadata(self, path: pathlib.Path) -> pd.DataFrame | None:
        """Load entity metadata from *path* into a :class:`~pandas.DataFrame`.

        Override to support custom formats (NumPy, HDF5, pickle, …).
        The default implementation handles TSV, CSV, and JSONL via
        :func:`_load_metadata_file`.
        """
        return _load_metadata_file(path)

    def _load_relation_metadata(self, path: pathlib.Path) -> pd.DataFrame | None:
        """Load relation metadata from *path* into a :class:`~pandas.DataFrame`.

        Override to support custom formats. Default handles TSV, CSV, and JSONL.
        """
        return _load_metadata_file(path)
