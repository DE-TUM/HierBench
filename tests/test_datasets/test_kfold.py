"""Tests for KFoldDataset, EagerKFoldDataset, and to_kfold."""

from __future__ import annotations

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.base import EagerDataset
from pykeen.datasets.kfold import EagerKFoldDataset, KFoldDataset, to_kfold
from pykeen.datasets.nations import NationsLiteral

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nations() -> Nations:
    """Return the Nations dataset (cached for the module)."""
    return Nations()


@pytest.fixture(scope="module")
def nations_kfold(nations: Nations) -> EagerKFoldDataset:
    """Return a 5-fold split of Nations with a fixed random seed."""
    return to_kfold(nations, k=5, random_state=42)


# ---------------------------------------------------------------------------
# EagerKFoldDataset structural tests
# ---------------------------------------------------------------------------


class TestEagerKFoldDatasetStructure:
    """Verify the structural contract of EagerKFoldDataset."""

    def test_is_kfold_dataset(self, nations_kfold: EagerKFoldDataset) -> None:
        """EagerKFoldDataset must be a subclass of KFoldDataset."""
        assert isinstance(nations_kfold, KFoldDataset)

    def test_len(self, nations_kfold: EagerKFoldDataset) -> None:
        """__len__ must equal k."""
        assert len(nations_kfold) == 5

    def test_num_folds(self, nations_kfold: EagerKFoldDataset) -> None:
        """num_folds property must equal k."""
        assert nations_kfold.num_folds == 5

    def test_iter(self, nations_kfold: EagerKFoldDataset) -> None:
        """Iterating must yield exactly k Dataset objects."""
        folds = list(nations_kfold)
        assert len(folds) == 5

    def test_getitem(self, nations_kfold: EagerKFoldDataset) -> None:
        """Index access must return the correct fold."""
        for i in range(5):
            fold = nations_kfold[i]
            assert fold is nations_kfold.datasets[i]

    def test_datasets_property(self, nations_kfold: EagerKFoldDataset) -> None:
        """Datasets property must return a list of Dataset objects."""
        assert isinstance(nations_kfold.datasets, list)
        assert len(nations_kfold.datasets) == 5


# ---------------------------------------------------------------------------
# Per-fold content tests
# ---------------------------------------------------------------------------


class TestFoldContent:
    """Verify that each fold contains valid training / testing / validation splits."""

    def test_training_not_none(self, nations_kfold: EagerKFoldDataset) -> None:
        """Every fold must have a training factory."""
        for i, fold in enumerate(nations_kfold):
            assert fold.training is not None, f"fold {i}: training is None"

    def test_testing_not_none(self, nations_kfold: EagerKFoldDataset) -> None:
        """Every fold must have a testing factory."""
        for i, fold in enumerate(nations_kfold):
            assert fold.testing is not None, f"fold {i}: testing is None"

    def test_validation_not_none(self, nations_kfold: EagerKFoldDataset) -> None:
        """Validation must exist when validation_ratio > 0 (default 0.1)."""
        for i, fold in enumerate(nations_kfold):
            assert fold.validation is not None, f"fold {i}: validation is None"

    def test_folds_are_eager_datasets(self, nations_kfold: EagerKFoldDataset) -> None:
        """Each fold must be an EagerDataset."""
        for i, fold in enumerate(nations_kfold):
            assert isinstance(fold, EagerDataset), f"fold {i} is not EagerDataset"

    def test_no_triple_overlap_train_test(self, nations_kfold: EagerKFoldDataset) -> None:
        """Training and test sets of each fold must be disjoint."""
        for i, fold in enumerate(nations_kfold):
            train_set = {tuple(t) for t in fold.training.mapped_triples.tolist()}
            test_set = {tuple(t) for t in fold.testing.mapped_triples.tolist()}
            overlap = train_set & test_set
            assert not overlap, f"fold {i}: {len(overlap)} triples appear in both train and test"

    def test_no_triple_overlap_train_val(self, nations_kfold: EagerKFoldDataset) -> None:
        """Training and validation sets of each fold must be disjoint."""
        for i, fold in enumerate(nations_kfold):
            if fold.validation is None:
                continue
            train_set = {tuple(t) for t in fold.training.mapped_triples.tolist()}
            val_set = {tuple(t) for t in fold.validation.mapped_triples.tolist()}
            overlap = train_set & val_set
            assert not overlap, f"fold {i}: {len(overlap)} triples appear in both train and val"


# ---------------------------------------------------------------------------
# Triple conservation
# ---------------------------------------------------------------------------


class TestTripleConservation:
    """Verify that all triples are preserved across folds."""

    def test_total_triples_conserved(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """The union of all test fold triples must equal the full dataset."""
        total_triples = (
            nations.training.num_triples
            + nations.testing.num_triples
            + (nations.validation.num_triples if nations.validation else 0)
        )
        # Each triple appears in exactly one test fold
        test_triples_total = sum(fold.testing.num_triples for fold in nations_kfold)
        assert test_triples_total == total_triples

    def test_fold_sizes_follow_chunk_distribution(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """Test fold sizes match torch.chunk distribution: first k-1 chunks are ceil(n/k), last gets remainder."""
        import math

        total = (
            nations.training.num_triples
            + nations.testing.num_triples
            + (nations.validation.num_triples if nations.validation else 0)
        )
        k = len(nations_kfold)
        chunk_size = math.ceil(total / k)
        last_size = total - (k - 1) * chunk_size
        for i, fold in enumerate(nations_kfold):
            expected = last_size if i == k - 1 else chunk_size
            assert fold.testing.num_triples == expected, (
                f"fold {i}: expected {expected} test triples, got {fold.testing.num_triples}"
            )


# ---------------------------------------------------------------------------
# Entity / relation mapping preservation
# ---------------------------------------------------------------------------


class TestMappingPreservation:
    """Verify that entity and relation mappings are consistent across folds."""

    def test_entity_to_id_preserved(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """All folds must use the same entity_to_id mapping as the source dataset."""
        expected = nations.training.entity_to_id
        for i, fold in enumerate(nations_kfold):
            assert fold.training.entity_to_id == expected, f"fold {i}: entity_to_id mismatch"

    def test_relation_to_id_preserved(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """All folds must use the same relation_to_id mapping as the source dataset."""
        expected = nations.training.relation_to_id
        for i, fold in enumerate(nations_kfold):
            assert fold.training.relation_to_id == expected, f"fold {i}: relation_to_id mismatch"

    def test_num_entities_consistent(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """num_entities must match the source dataset in every fold."""
        expected = nations.num_entities
        for i, fold in enumerate(nations_kfold):
            assert fold.training.num_entities == expected, f"fold {i}: num_entities mismatch"

    def test_num_relations_consistent(self, nations: Nations, nations_kfold: EagerKFoldDataset) -> None:
        """num_relations must match the source dataset in every fold."""
        expected = nations.num_relations
        for i, fold in enumerate(nations_kfold):
            assert fold.training.num_relations == expected, f"fold {i}: num_relations mismatch"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify deterministic behaviour with a fixed random seed."""

    def test_same_seed_same_result(self, nations: Nations) -> None:
        """Two calls with the same seed must produce identical fold splits."""
        kf1 = to_kfold(nations, k=3, random_state=0)
        kf2 = to_kfold(nations, k=3, random_state=0)
        for i in range(3):
            assert torch.equal(kf1[i].testing.mapped_triples, kf2[i].testing.mapped_triples), (
                f"fold {i}: test triples differ despite same seed"
            )

    def test_different_seeds_differ(self, nations: Nations) -> None:
        """Two calls with different seeds should (almost certainly) produce different folds."""
        kf1 = to_kfold(nations, k=3, random_state=1)
        kf2 = to_kfold(nations, k=3, random_state=2)
        any_different = any(
            not torch.equal(kf1[i].testing.mapped_triples, kf2[i].testing.mapped_triples) for i in range(3)
        )
        assert any_different, "Different seeds produced identical folds — extremely unlikely"


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge-case behaviour of to_kfold."""

    def test_k_less_than_2_raises(self, nations: Nations) -> None:
        """to_kfold must raise ValueError for k < 2."""
        with pytest.raises(ValueError, match="k must be at least 2"):
            to_kfold(nations, k=1)

    def test_no_validation(self, nations: Nations) -> None:
        """validation_ratio=0.0 must produce folds where validation is None."""
        kf = to_kfold(nations, k=3, validation_ratio=0.0)
        for i, fold in enumerate(kf):
            assert fold.validation is None, f"fold {i}: expected validation=None"

    def test_k_equals_2(self, nations: Nations) -> None:
        """k=2 is the minimum legal value; must succeed."""
        kf = to_kfold(nations, k=2, random_state=0)
        assert len(kf) == 2

    def test_returns_eager_kfold_dataset(self, nations: Nations) -> None:
        """to_kfold must return an EagerKFoldDataset instance."""
        kf = to_kfold(nations, k=3)
        assert isinstance(kf, EagerKFoldDataset)


# ---------------------------------------------------------------------------
# Numeric literals preservation
# ---------------------------------------------------------------------------


class TestNumericLiteralsPreservation:
    """Verify that numeric literals are preserved when the source uses TriplesNumericLiteralsFactory."""

    @pytest.fixture(scope="class")
    def nations_literal_kfold(self) -> EagerKFoldDataset:
        """Return a 3-fold split of NationsLiteral."""
        return to_kfold(NationsLiteral(), k=3, random_state=0)

    def test_literals_to_id_preserved(self, nations_literal_kfold: EagerKFoldDataset) -> None:
        """literals_to_id must be identical across all folds."""
        from pykeen.triples import TriplesNumericLiteralsFactory

        first = nations_literal_kfold[0].training
        assert isinstance(first, TriplesNumericLiteralsFactory)
        expected_literals = first.literals_to_id
        for i, fold in enumerate(nations_literal_kfold):
            factory = fold.training
            assert isinstance(factory, TriplesNumericLiteralsFactory), f"fold {i}: wrong factory type"
            assert factory.literals_to_id == expected_literals, f"fold {i}: literals_to_id mismatch"

    def test_numeric_literals_shape_preserved(self, nations_literal_kfold: EagerKFoldDataset) -> None:
        """numeric_literals matrix shape must be identical across all folds."""
        from pykeen.triples import TriplesNumericLiteralsFactory

        first_shape = nations_literal_kfold[0].training.numeric_literals.shape
        for i, fold in enumerate(nations_literal_kfold):
            factory = fold.training
            assert isinstance(factory, TriplesNumericLiteralsFactory)
            assert factory.numeric_literals.shape == first_shape, f"fold {i}: numeric_literals shape mismatch"
