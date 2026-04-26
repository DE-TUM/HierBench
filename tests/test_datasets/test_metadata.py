"""Tests for MetadataDataset, RemoteMetadataDataset, and _load_metadata_file."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from pykeen.datasets.base import EagerDataset
from pykeen.datasets.metadata import MetadataDataset, RemoteMetadataDataset, _load_metadata_file
from pykeen.triples import TriplesFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRIPLES = [
    ["A", "rel1", "B"],
    ["B", "rel2", "C"],
    ["C", "rel1", "A"],
    ["A", "rel2", "C"],
    ["B", "rel1", "A"],
]


def _make_factory() -> TriplesFactory:
    return TriplesFactory.from_labeled_triples(np.array(_TRIPLES))


def _entity_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"e{i}" for i in range(n)], "feat": [float(i) for i in range(n)]})


def _relation_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"weight": [float(i) for i in range(n)]})


@pytest.fixture
def factory() -> TriplesFactory:
    """Return a minimal TriplesFactory for testing."""
    return _make_factory()


@pytest.fixture
def dataset(factory: TriplesFactory) -> MetadataDataset:
    """Return a MetadataDataset with entity and relation metadata."""
    return MetadataDataset(
        training=factory,
        testing=factory,
        validation=factory,
        entity_metadata=_entity_df(factory.num_entities),
        relation_metadata=_relation_df(factory.num_relations),
    )


# ---------------------------------------------------------------------------
# MetadataDataset construction
# ---------------------------------------------------------------------------


class TestMetadataDatasetConstruction:
    """Tests for MetadataDataset construction and attribute access."""

    def test_is_eager_dataset(self, dataset: MetadataDataset) -> None:
        assert isinstance(dataset, EagerDataset)

    def test_entity_metadata_stored(self, dataset: MetadataDataset) -> None:
        assert isinstance(dataset.entity_metadata, pd.DataFrame)
        assert len(dataset.entity_metadata) == dataset.num_entities

    def test_relation_metadata_stored(self, dataset: MetadataDataset) -> None:
        assert isinstance(dataset.relation_metadata, pd.DataFrame)
        assert len(dataset.relation_metadata) == dataset.num_relations

    def test_entity_metadata_defaults_none(self, factory: TriplesFactory) -> None:
        ds = MetadataDataset(training=factory, testing=factory)
        assert ds.entity_metadata is None

    def test_relation_metadata_defaults_none(self, factory: TriplesFactory) -> None:
        ds = MetadataDataset(training=factory, testing=factory)
        assert ds.relation_metadata is None

    def test_validation_optional(self, factory: TriplesFactory) -> None:
        ds = MetadataDataset(training=factory, testing=factory)
        assert ds.validation is None

    def test_dataset_level_metadata_stored(self, factory: TriplesFactory) -> None:
        ds = MetadataDataset(training=factory, testing=factory, metadata={"name": "test"})
        assert ds.metadata["name"] == "test"

    def test_num_entities_from_base(self, dataset: MetadataDataset, factory: TriplesFactory) -> None:
        assert dataset.num_entities == factory.num_entities

    def test_num_relations_from_base(self, dataset: MetadataDataset, factory: TriplesFactory) -> None:
        assert dataset.num_relations == factory.num_relations


# ---------------------------------------------------------------------------
# _load_metadata_file
# ---------------------------------------------------------------------------


class TestLoadMetadataFile:
    """Tests for _load_metadata_file format detection."""

    def test_tsv(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        df = _load_metadata_file(p)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_csv(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("x,y\n10,20\n")
        df = _load_metadata_file(p)
        assert list(df.columns) == ["x", "y"]
        assert df["x"].iloc[0] == 10

    def test_jsonl(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.jsonl"
        lines = [json.dumps({"k": i}) for i in range(3)]
        p.write_text("\n".join(lines) + "\n")
        df = _load_metadata_file(p)
        assert "k" in df.columns
        assert len(df) == 3

    def test_json_extension_treated_as_jsonl(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.json"
        lines = [json.dumps({"v": i}) for i in range(2)]
        p.write_text("\n".join(lines) + "\n")
        df = _load_metadata_file(p)
        assert len(df) == 2

    def test_unknown_extension_falls_back_to_tsv(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "data.dat"
        p.write_text("col1\tcol2\n5\t6\n")
        df = _load_metadata_file(p)
        assert list(df.columns) == ["col1", "col2"]


# ---------------------------------------------------------------------------
# RemoteMetadataDataset
# ---------------------------------------------------------------------------


class TestRemoteMetadataDataset:
    """Tests use a minimal concrete subclass with URL downloads mocked via monkeypatch."""

    def _make_class(
        self, triples_url="http://x/triples.tsv", entity_url=None, relation_url=None, ratios=(0.6, 0.2, 0.2)
    ):
        class _DS(RemoteMetadataDataset):
            pass

        _DS.triples_url = triples_url
        _DS.entity_metadata_url = entity_url
        _DS.relation_metadata_url = relation_url
        _DS.ratios = ratios
        return _DS

    def _write_triples(self, path: pathlib.Path) -> None:
        path.write_text("\n".join("\t".join(t) for t in _TRIPLES) + "\n")

    def test_downloads_triples_when_missing(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        downloaded = []

        def fake_download(url, path, **kw):
            self._write_triples(path)
            downloaded.append(url)

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", fake_download)

        ds_class = self._make_class()
        ds = ds_class(cache_root=tmp_path)
        assert len(downloaded) == 1
        assert ds.training is not None

    def test_skips_download_when_file_exists(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        # RemoteMetadataDataset puts files in <cache_root>/<classname>/
        ds_class = self._make_class()
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        existing = subdir / "dataset.tsv"
        self._write_triples(existing)

        downloaded = []

        def fake_download(url, path, **kw):
            downloaded.append(url)

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", fake_download)
        ds = ds_class(cache_root=tmp_path)
        assert downloaded == [], "should not re-download existing file"
        assert ds.training is not None

    def test_no_entity_metadata_when_url_none(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        ds_class = self._make_class(entity_url=None)
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = ds_class(cache_root=tmp_path)
        assert ds.entity_metadata is None

    def test_entity_metadata_loaded_when_url_given(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        ds_class = self._make_class(entity_url="http://x/entities.tsv")
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")

        factory = TriplesFactory.from_path(subdir / "dataset.tsv")
        n = factory.num_entities
        (subdir / "entity_metadata.tsv").write_text("feat\n" + "\n".join(str(float(i)) for i in range(n)) + "\n")

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = ds_class(cache_root=tmp_path)
        assert ds.entity_metadata is not None
        assert "feat" in ds.entity_metadata.columns

    def test_relation_metadata_loaded_when_url_given(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        ds_class = self._make_class(relation_url="http://x/relations.tsv")
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")

        factory = TriplesFactory.from_path(subdir / "dataset.tsv")
        r = factory.num_relations
        (subdir / "relation_metadata.tsv").write_text("w\n" + "\n".join(str(float(i)) for i in range(r)) + "\n")

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = ds_class(cache_root=tmp_path)
        assert ds.relation_metadata is not None
        assert "w" in ds.relation_metadata.columns

    def test_split_produces_three_splits(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        ds_class = self._make_class()
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = ds_class(cache_root=tmp_path)
        assert ds.training is not None
        assert ds.testing is not None
        assert ds.validation is not None

    def test_custom_load_entity_metadata_hook(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """_load_entity_metadata override is called instead of the default."""

        class CustomDS(RemoteMetadataDataset):
            triples_url = "http://x/triples.tsv"
            entity_metadata_url = "http://x/entities.npy"
            ratios = (0.6, 0.2, 0.2)

            def _load_entity_metadata(self, path):
                return pd.DataFrame({"custom": [1, 2, 3]})

        subdir = tmp_path / CustomDS.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")
        (subdir / "entity_metadata.npy").write_bytes(b"")  # placeholder so download is skipped

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = CustomDS(cache_root=tmp_path)
        assert ds.entity_metadata is not None
        assert "custom" in ds.entity_metadata.columns

    def test_ratios_respected(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        ds_class = self._make_class(ratios=(0.8, 0.1, 0.1))
        subdir = tmp_path / ds_class.__name__.lower()
        subdir.mkdir()
        self._write_triples(subdir / "dataset.tsv")

        import pystow.utils as pu

        monkeypatch.setattr(pu, "download", lambda **kw: None)
        ds = ds_class(cache_root=tmp_path)
        total = ds.training.num_triples + ds.testing.num_triples + ds.validation.num_triples
        assert total == len(_TRIPLES)
