"""Metadata-aware dataset classes with optional per-entity and per-relation context."""

from __future__ import annotations

import logging
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import pandas as pd

from .base import EagerDataset
from ..constants import PYKEEN_DATASETS
from ..triples import TriplesFactory
from ..typing import TorchRandomHint
from ..utils import normalize_path

if TYPE_CHECKING:
    from .base import Dataset

__all__ = [
    "LOCAL_DATA_URL",
    "HierarchicalGraph",
    "MetadataDataset",
    "RemoteMetadataDataset",
    "SingleFileRemoteMetadataDataset",
    "load_closure_pool",
    "resolve_hierarchy_relation",
]

logger = logging.getLogger(__name__)
Metadata: TypeAlias = Mapping[str, Any]

#: ``file://`` base URL of the repo-local benchmark data in ``datasets/`` at the project root (a
#: checkout sibling of ``src/``). The taxonomy datasets point their ``_BASE_URL`` here so the data
#: ships with the checkout and downloads copy from disk (urllib's ``file://`` handler) instead of the
#: remote host.
LOCAL_DATA_URL = pathlib.Path(__file__).resolve().parents[3].joinpath("datasets").as_uri() + "/"


class HierarchicalGraph:
    """Mixin marking a dataset whose triples form a taxonomy/hierarchy.

    Concrete datasets set :attr:`hierarchical_relation` to the relation label
    forming the parent/child tree edges (e.g. ``narrower``, ``has_subclass``).
    Detect taxonomy datasets with ``isinstance(ds, HierarchicalGraph)``.
    """

    #: Name of the relation forming the parent/child hierarchy edges.
    hierarchical_relation: ClassVar[str]

    #: Whether the hierarchy edges point child->parent (e.g. WN18RR's ``_hypernym``) instead of the
    #: canonical parent->child (``narrower``, ``has_subclass``). Closure-based splits flip inverted
    #: edges so that heads are always ancestors, keeping metrics comparable across datasets.
    hierarchy_inverted: ClassVar[bool] = False

    #: Whether the dataset ships its own train/validation/test closure split (e.g. the
    #: WordNetNoun* ``maxn`` splits of Ganea et al. 2018). Closure-based splits then use the
    #: dataset's fixed validation/testing triples as eval positives and the training triples
    #: verbatim, instead of re-deriving a split from the training graph's transitive closure
    #: (which would leak closure edges already present in training back into evaluation).
    predefined_closure_split: ClassVar[bool] = False

    #: Optional URL of the precomputed *non-direct* transitive closure: a label TSV of every
    #: (ancestor, hierarchy relation, descendant) closure pair that is not a basic/direct edge, in
    #: canonical parent->child orientation (regenerate with
    #: ``scripts/export_transitive_ancestor_descendant_splits.py``). When set, closure-based splits
    #: load it via :func:`load_closure_pool` instead of computing the transitive reduction and
    #: closure enumeration; ``None`` (or a failed download) falls back to on-demand computation.
    closure_url: ClassVar[str | None] = None


def resolve_hierarchy_relation(dataset: Dataset, hierarchy_relation: int | str | None) -> int | None:
    """Resolve the hierarchy relation to a relation id.

    Precedence: an explicit ``hierarchy_relation`` (id or label) wins; otherwise, if ``dataset`` is a
    :class:`HierarchicalGraph`, its :attr:`hierarchical_relation` label is used. Returns ``None``
    when neither is available; callers treat ``None`` as "all edges are hierarchy edges", so a
    dataset without a designated hierarchy relation transparently falls back to using every relation.
    """
    if hierarchy_relation is None and isinstance(dataset, HierarchicalGraph):
        hierarchy_relation = dataset.hierarchical_relation
    if isinstance(hierarchy_relation, str):
        try:
            return dataset.training.relations_to_ids([hierarchy_relation])[0]
        except (AttributeError, KeyError) as exc:
            raise KeyError(f"hierarchy relation {hierarchy_relation!r} not found in dataset relations") from exc
    return hierarchy_relation


def load_closure_pool(dataset: Dataset) -> list[tuple[int, int]] | None:
    """Load the precomputed non-direct transitive closure pool of a dataset, if it ships one.

    Downloads :attr:`HierarchicalGraph.closure_url` into the dataset's cache directory (skipped
    when already cached), maps the label pairs to the dataset's entity ids, and returns them
    sorted — the exact ``pool`` that on-demand computation would produce, so seeded splits are
    reproduced identically. Returns ``None`` when the dataset declares no ``closure_url`` or the
    download fails (with a warning), letting callers fall back to computing the closure.
    """
    url = getattr(dataset, "closure_url", None)
    if url is None:
        return None
    path = PYKEEN_DATASETS.joinpath(type(dataset).__name__.lower(), "closure.tsv")
    if not path.is_file():
        from pystow.utils import download

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            download(url=url, path=path)  # noqa: S310
        except Exception as exc:  # noqa: BLE001 - any download failure must fall back, not crash
            path.unlink(missing_ok=True)  # drop a partial download so the next call retries
            warnings.warn(
                f"could not download closure file {url} ({exc}); computing the closure on demand",
                stacklevel=2,
            )
            return None
    entity_to_id = dataset.training.entity_to_id
    return sorted(
        (entity_to_id[head], entity_to_id[tail])
        for head, _relation, tail in (line.split("\t") for line in path.read_text(encoding="utf-8").splitlines())
    )


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


class SingleFileRemoteMetadataDataset(MetadataDataset):
    """A :class:`MetadataDataset` that downloads a single triples file and splits it locally.

    Dataset authors only need to set class-level URL attributes — no ``__init__``
    boilerplate required::

        class Cora(SingleFileRemoteMetadataDataset):
            triples_url          = "https://example.com/cora/edges.tsv"
            entity_metadata_url  = "https://example.com/cora/node_features.tsv"
            relation_metadata_url = "https://example.com/cora/rel_features.tsv"

    The split is performed locally with a fixed ``random_state`` (default: 0), so
    results are reproducible within the same PyKEEN version but not guaranteed across
    versions.  For published benchmarks that require identical splits for every user,
    prefer :class:`RemoteMetadataDataset`.

    Override :meth:`_load_entity_metadata` or :meth:`_load_relation_metadata` to
    support custom file formats (NumPy, HDF5, pickle, …)::

        class MyDataset(SingleFileRemoteMetadataDataset):
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
    #: Train / test / validation split ratios. ``None`` skips splitting entirely — training,
    #: testing, and validation all point at the full triples factory. This is needed for tree-shaped
    #: datasets (e.g. taxonomies) where most entities have degree 1 and a random split cannot cover
    #: all entities in the training portion; see :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split`.
    ratios: ClassVar[Sequence[float] | None] = (0.8, 0.1, 0.1)

    def __init__(
        self,
        *,
        cache_root: str | pathlib.Path | None = None,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = 0,
        download_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the remote metadata dataset.

        :param cache_root: Local directory for cached files. Defaults to a
            sub-directory of :data:`pykeen.constants.PYKEEN_DATASETS` named
            after the concrete subclass.
        :param create_inverse_triples: Whether to add inverse triples to training.
        :param random_state: Random state for the train/test/val split. Defaults to 0.
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
        if self.ratios is None:
            training = testing = validation = full_factory
        else:
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


class RemoteMetadataDataset(MetadataDataset):
    """A :class:`MetadataDataset` with predefined train / test / validation splits served as separate files.

    This is the recommended base class for benchmark datasets where reproducibility
    requires fixed, server-side splits that are identical for every user.  It mirrors
    the pattern of :class:`~pykeen.datasets.base.UnpackedRemoteDataset` but adds
    optional per-entity and per-relation metadata.

    Test and validation factories share the entity and relation index built from the
    training file, following the standard PyKEEN convention.

    Dataset authors only need to set class-level URL attributes::

        class MyBenchmark(RemoteMetadataDataset):
            training_url         = "https://example.com/train.tsv"
            testing_url          = "https://example.com/test.tsv"
            validation_url       = "https://example.com/valid.tsv"
            entity_metadata_url  = "https://example.com/node_features.tsv"

    Override :meth:`_load_entity_metadata` or :meth:`_load_relation_metadata` to
    support custom file formats (NumPy, HDF5, pickle, …).
    """

    #: URL of the training triples file (TSV: head, relation, tail).
    training_url: ClassVar[str]
    #: URL of the testing triples file.
    testing_url: ClassVar[str]
    #: URL of the validation triples file.
    validation_url: ClassVar[str]
    #: URL of the entity metadata file. ``None`` means no entity metadata.
    entity_metadata_url: ClassVar[str | None] = None
    #: URL of the relation metadata file. ``None`` means no relation metadata.
    relation_metadata_url: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        cache_root: str | pathlib.Path | None = None,
        create_inverse_triples: bool = False,
        download_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset.

        :param cache_root: Local directory for cached files. Defaults to a
            sub-directory of :data:`pykeen.constants.PYKEEN_DATASETS` named
            after the concrete subclass.
        :param create_inverse_triples: Whether to add inverse triples to training.
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

        # Download the three split files
        training_path = dataset_root / "train.tsv"
        testing_path = dataset_root / "test.tsv"
        validation_path = dataset_root / "valid.tsv"
        for url, path in [
            (self.training_url, training_path),
            (self.testing_url, testing_path),
            (self.validation_url, validation_path),
        ]:
            if not path.is_file():
                logger.info("downloading %s from %s", path.name, url)
                download(url=url, path=path, **download_kwargs)  # noqa: S310

        # Load triples — test/valid share the training entity+relation index
        training = TriplesFactory.from_path(
            training_path,
            create_inverse_triples=create_inverse_triples,
        )
        testing = TriplesFactory.from_path(
            testing_path,
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
            create_inverse_triples=False,
        )
        validation = TriplesFactory.from_path(
            validation_path,
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
            create_inverse_triples=False,
        )

        # Download and load metadata
        entity_metadata = self._download_and_load(
            self.entity_metadata_url, "entity_metadata", dataset_root, download, download_kwargs
        )
        relation_metadata = self._download_and_load(
            self.relation_metadata_url, "relation_metadata", dataset_root, download, download_kwargs
        )

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

    def _load_entity_metadata(self, path: pathlib.Path) -> pd.DataFrame | None:
        """Load entity metadata from *path* into a :class:`~pandas.DataFrame`.

        Override to support custom formats (NumPy, HDF5, pickle, …).
        The default implementation handles TSV, CSV, and JSONL.
        """
        return _load_metadata_file(path)

    def _load_relation_metadata(self, path: pathlib.Path) -> pd.DataFrame | None:
        """Load relation metadata from *path* into a :class:`~pandas.DataFrame`.

        Override to support custom formats. Default handles TSV, CSV, and JSONL.
        """
        return _load_metadata_file(path)
