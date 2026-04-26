"""Test the cross-validation pipeline."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from pykeen.cross_validation.cross_validation import (
    CrossValidationPipelineResult,
    _aggregate_fold_metrics,
    _resolve_kfold_dataset,
    cross_validation_pipeline,
)
from pykeen.datasets import Nations
from pykeen.datasets.kfold.base import EagerKFoldDataset, KFoldDataset, to_kfold

# Shared minimal kwargs for every pipeline call — keeps tests fast
_FAST_KWARGS = {
    "model": "TransE",
    "training_kwargs": {"num_epochs": 1, "use_tqdm": False},
    "evaluation_kwargs": {"use_tqdm": False},
}


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


class _SimpleKFoldDataset(KFoldDataset):
    """Minimal concrete KFoldDataset subclass for testing."""

    def _load_folds(self):
        return to_kfold(Nations(), k=2, random_state=0).datasets


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


class TestDatasetResolution(unittest.TestCase):
    """Unit tests for _resolve_kfold_dataset."""

    @classmethod
    def setUpClass(cls):
        cls.nations = Nations()
        cls.kfold = to_kfold(cls.nations, k=2, random_state=0)

    def test_kfold_instance_passthrough(self):
        """A KFoldDataset instance must be returned unchanged."""
        result = _resolve_kfold_dataset(self.kfold, None, 2, 0.1, None)
        assert result is self.kfold

    def test_kfold_subclass_instantiated(self):
        """A KFoldDataset subclass must be instantiated (no extra kwargs needed)."""
        result = _resolve_kfold_dataset(_SimpleKFoldDataset, None, 2, 0.1, None)
        assert isinstance(result, KFoldDataset)
        assert result.num_folds == 2

    def test_dataset_instance_wrapped(self):
        """A Dataset instance must be wrapped into EagerKFoldDataset."""
        result = _resolve_kfold_dataset(self.nations, None, 2, 0.1, 42)
        assert isinstance(result, EagerKFoldDataset)
        assert result.num_folds == 2

    def test_dataset_class_instantiated_and_wrapped(self):
        """A Dataset subclass must be instantiated and wrapped."""
        result = _resolve_kfold_dataset(Nations, None, 3, 0.1, 42)
        assert isinstance(result, EagerKFoldDataset)
        assert result.num_folds == 3

    def test_string_name_resolved_and_wrapped(self):
        """A string dataset name must be resolved via dataset_resolver and wrapped."""
        result = _resolve_kfold_dataset("nations", None, 2, 0.1, 42)
        assert isinstance(result, EagerKFoldDataset)
        assert result.num_folds == 2

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="dataset"):
            _resolve_kfold_dataset(None, None, 5, 0.1, None)

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            _resolve_kfold_dataset(42, None, 5, 0.1, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestCrossValidationResult(unittest.TestCase):
    """Unit tests for CrossValidationPipelineResult methods."""

    @classmethod
    def setUpClass(cls):
        cls.result: CrossValidationPipelineResult = cross_validation_pipeline(
            dataset=Nations,
            k=2,
            kfold_random_state=0,
            random_seed=0,
            **_FAST_KWARGS,
        )

    def test_result_type(self):
        assert isinstance(self.result, CrossValidationPipelineResult)

    def test_num_folds(self):
        assert self.result.num_folds == 2
        assert len(self.result.fold_results) == 2

    def test_metric_means_stds_same_keys(self):
        assert set(self.result.metric_means) == set(self.result.metric_stds)

    def test_get_metric_returns_float_pair(self):
        key = next(iter(self.result.metric_means))
        mean, std = self.result.get_metric(key)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0.0

    def test_get_metric_unknown_key_raises(self):
        with pytest.raises(KeyError):
            self.result.get_metric("this_metric_does_not_exist")

    def test_to_df_shape(self):
        df = self.result.to_df()
        assert len(df) == 2
        assert len(df.columns) > 0

    def test_to_df_columns_match_metric_means(self):
        df = self.result.to_df()
        assert set(df.columns) == set(self.result.metric_means)

    def test_times_positive(self):
        assert self.result.total_train_seconds > 0.0
        assert self.result.total_evaluate_seconds > 0.0

    def test_version_and_git_hash_present(self):
        assert isinstance(self.result.version, str)
        assert isinstance(self.result.git_hash, str)

    def test_get_results_keys(self):
        d = self.result._get_results()
        for key in ("num_folds", "times", "metric_means", "metric_stds", "version", "git_hash"):
            assert key in d
        assert d["num_folds"] == 2


# ---------------------------------------------------------------------------
# Save to directory
# ---------------------------------------------------------------------------


class TestSaveToDirectory(unittest.TestCase):
    """Test filesystem persistence of CrossValidationPipelineResult."""

    @classmethod
    def setUpClass(cls):
        cls.result: CrossValidationPipelineResult = cross_validation_pipeline(
            dataset=Nations,
            k=2,
            kfold_random_state=0,
            random_seed=0,
            **_FAST_KWARGS,
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cv_results_json_created(self):
        self.result.save_to_directory(self.directory)
        json_path = self.directory / "cv_results.json"
        assert json_path.exists()
        with json_path.open() as fh:
            data = json.load(fh)
        assert data["num_folds"] == 2
        assert "metric_means" in data
        assert "metric_stds" in data

    def test_fold_metrics_tsv_created(self):
        self.result.save_to_directory(self.directory)
        assert (self.directory / "fold_metrics.tsv").exists()

    def test_fold_subdirs_created_by_default(self):
        self.result.save_to_directory(self.directory)
        for i in range(2):
            assert (self.directory / f"fold-{i:03d}").is_dir()

    def test_save_without_fold_subdirs(self):
        self.result.save_to_directory(self.directory, save_fold_results=False)
        for i in range(2):
            assert not (self.directory / f"fold-{i:03d}").exists()

    def test_save_to_ftp_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.result.save_to_ftp("some/dir", None)

    def test_save_to_s3_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.result.save_to_s3("some/dir", "bucket")


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------


class TestCrossValidationPipeline(unittest.TestCase):
    """Integration tests for cross_validation_pipeline."""

    def test_with_dataset_class(self):
        result = cross_validation_pipeline(dataset=Nations, k=2, random_seed=0, **_FAST_KWARGS)
        assert isinstance(result, CrossValidationPipelineResult)
        assert result.num_folds == 2

    def test_with_dataset_instance(self):
        result = cross_validation_pipeline(dataset=Nations(), k=2, random_seed=0, **_FAST_KWARGS)
        assert isinstance(result, CrossValidationPipelineResult)
        assert result.num_folds == 2

    def test_with_string_name(self):
        result = cross_validation_pipeline(dataset="nations", k=2, random_seed=0, **_FAST_KWARGS)
        assert isinstance(result, CrossValidationPipelineResult)

    def test_with_kfold_dataset_instance(self):
        kfold = to_kfold(Nations(), k=2, random_state=0)
        result = cross_validation_pipeline(dataset=kfold, random_seed=0, **_FAST_KWARGS)
        assert isinstance(result, CrossValidationPipelineResult)
        assert result.num_folds == 2

    def test_no_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset"):
            cross_validation_pipeline(model="TransE")

    def test_reproducibility_with_seeds(self):
        """Same kfold and model seeds must produce identical metric means."""
        kfold = to_kfold(Nations(), k=2, random_state=42)
        kwargs = {"dataset": kfold, "random_seed": 7, **_FAST_KWARGS}
        r1 = cross_validation_pipeline(**kwargs)
        r2 = cross_validation_pipeline(**kwargs)
        assert r1.metric_means == r2.metric_means

    def test_no_random_seed(self):
        """Running without seeds must not raise."""
        result = cross_validation_pipeline(dataset=Nations, k=2, kfold_random_state=0, random_seed=None, **_FAST_KWARGS)
        assert isinstance(result, CrossValidationPipelineResult)

    def test_custom_metadata_propagated(self):
        """User metadata keys must appear in every fold's pipeline metadata."""
        result = cross_validation_pipeline(
            dataset=Nations,
            k=2,
            kfold_random_state=0,
            metadata={"experiment": "cv_test"},
            **_FAST_KWARGS,
        )
        for fold_result in result.fold_results:
            assert fold_result.metadata.get("experiment") == "cv_test"

    def test_cv_fold_metadata_injected(self):
        """cv_fold and cv_num_folds must be injected into each fold's metadata."""
        result = cross_validation_pipeline(dataset=Nations, k=2, kfold_random_state=0, **_FAST_KWARGS)
        for i, fold_result in enumerate(result.fold_results):
            assert fold_result.metadata.get("cv_fold") == i
            assert fold_result.metadata.get("cv_num_folds") == 2


# ---------------------------------------------------------------------------
# Unit test for aggregate helper
# ---------------------------------------------------------------------------


def test_aggregate_fold_metrics_correctness():
    """_aggregate_fold_metrics must compute correct mean and population std."""

    def _fake_result(metrics: dict):
        r = MagicMock()
        r.metric_results.to_flat_dict.return_value = metrics
        return r

    fold_results = [
        _fake_result({"hits_at_1": 0.2, "mrr": 0.4}),
        _fake_result({"hits_at_1": 0.4, "mrr": 0.6}),
    ]
    means, stds = _aggregate_fold_metrics(fold_results)

    assert means["hits_at_1"] == pytest.approx(0.3)
    assert means["mrr"] == pytest.approx(0.5)
    assert stds["hits_at_1"] == pytest.approx(0.1)
    assert stds["mrr"] == pytest.approx(0.1)


def test_aggregate_fold_metrics_single_fold():
    """With a single fold std must be zero."""

    def _fake_result(metrics: dict):
        r = MagicMock()
        r.metric_results.to_flat_dict.return_value = metrics
        return r

    means, stds = _aggregate_fold_metrics([_fake_result({"mrr": 0.5})])
    assert means["mrr"] == pytest.approx(0.5)
    assert stds["mrr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Slower tests (marked so they can be skipped in fast test runs)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCrossValidationSlow(unittest.TestCase):
    """Heavier cross-validation tests."""

    def test_k3_full_run(self):
        result = cross_validation_pipeline(
            dataset=Nations,
            k=3,
            kfold_random_state=0,
            random_seed=0,
            model="TransE",
            training_kwargs={"num_epochs": 3, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
        )
        assert result.num_folds == 3
        assert len(result.fold_results) == 3
        df = result.to_df()
        assert len(df) == 3

    def test_save_k3_to_directory(self):
        result = cross_validation_pipeline(
            dataset=Nations,
            k=3,
            kfold_random_state=0,
            random_seed=0,
            model="TransE",
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
        )
        with tempfile.TemporaryDirectory() as directory:
            result.save_to_directory(directory)
            d = pathlib.Path(directory)
            assert (d / "cv_results.json").exists()
            assert (d / "fold_metrics.tsv").exists()
            for i in range(3):
                assert (d / f"fold-{i:03d}").is_dir()
