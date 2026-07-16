"""Tests for hyperbolic representation modules."""

from __future__ import annotations

import warnings

import pytest
import torch

from pykeen.nn.hyperbolic import LorentzEmbedding, PoincareEmbedding
from pykeen.nn.init import PretrainedInitializer
from tests import cases


class PoincareEmbeddingTests(cases.RepresentationTestCase):
    """Standard representation contract tests for PoincareEmbedding."""

    cls = PoincareEmbedding
    kwargs = {"embedding_dim": 5}

    def _pre_instantiation_hook(self, kwargs):
        kwargs = super()._pre_instantiation_hook(kwargs)
        # Suppress the Riemannian optimizer warning during test setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
        return kwargs

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super().setUp()


class LorentzEmbeddingTests(cases.RepresentationTestCase):
    """Standard representation contract tests for LorentzEmbedding."""

    cls = LorentzEmbedding
    kwargs = {"embedding_dim": 5}

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super().setUp()


class TestPoincareProperties:
    """Targeted invariant tests for PoincareEmbedding."""

    @pytest.fixture
    def emb(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return PoincareEmbedding(max_id=10, embedding_dim=4, curvature=1.0)

    def test_shape_matches_embedding_dim(self, emb):
        assert emb.shape == (4,)

    def test_on_manifold_after_init(self, emb):
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_on_manifold_after_projection(self, emb):
        with torch.no_grad():
            emb._embeddings.data += 2.0  # push off manifold
        emb.post_parameter_update()
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_trainable_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = PoincareEmbedding(max_id=5, embedding_dim=3, trainable=False)
        assert not emb._embeddings.requires_grad

    def test_trainable_curvature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = PoincareEmbedding(max_id=5, embedding_dim=3, trainable_curvature=True)
        assert isinstance(emb._curvature, torch.nn.Parameter)
        assert emb._curvature in list(emb.parameters())

    def test_pretrained_initializer(self):
        tensor = torch.randn(10, 4) * 1e-3
        init = PretrainedInitializer(tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = PoincareEmbedding(max_id=10, embedding_dim=4, initializer=init)
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_forward_all_indices(self, emb):
        out = emb(indices=None)
        assert out.shape == (10, 4)

    def test_forward_2d_indices(self, emb):
        idx = torch.randint(10, size=(3, 4))
        out = emb(indices=idx)
        assert out.shape == (3, 4, 4)

    def test_is_manifold_parameter(self, emb):
        import geoopt

        assert isinstance(emb._embeddings, geoopt.ManifoldParameter)

    def test_construction_is_silent(self):
        """Construction no longer warns about optimizers — the check moved to the training loop."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            PoincareEmbedding(max_id=5, embedding_dim=3)

    def test_registered_in_resolver(self):
        from pykeen.nn import representation_resolver

        assert representation_resolver.lookup("PoincareEmbedding") is PoincareEmbedding


class TestLorentzProperties:
    """Targeted invariant tests for LorentzEmbedding."""

    @pytest.fixture
    def emb(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return LorentzEmbedding(max_id=10, embedding_dim=4, curvature=1.0)

    def test_shape_is_d_plus_one(self, emb):
        # embedding_dim=4 → shape=(5,)
        assert emb.shape == (5,)

    def test_internal_storage_is_d_plus_one(self, emb):
        assert emb._embeddings.data.shape == (10, 5)

    def test_on_manifold_after_init(self, emb):
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_on_manifold_after_projection(self, emb):
        with torch.no_grad():
            emb._embeddings.data[:, 0] -= 5.0  # break time coord constraint
        emb.post_parameter_update()
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_forward_returns_full_lorentz_vector(self, emb):
        out = emb(indices=None)
        assert out.shape == (10, 5)  # (max_id, d+1), not (max_id, d)

    def test_forward_2d_indices(self, emb):
        idx = torch.randint(10, size=(2, 3))
        out = emb(indices=idx)
        assert out.shape == (2, 3, 5)

    def test_trainable_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = LorentzEmbedding(max_id=5, embedding_dim=3, trainable=False)
        assert not emb._embeddings.requires_grad

    def test_trainable_curvature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = LorentzEmbedding(max_id=5, embedding_dim=3, trainable_curvature=True)
        assert isinstance(emb._curvature, torch.nn.Parameter)
        assert emb._curvature in list(emb.parameters())

    def test_pretrained_initializer_spatial_only(self):
        # Initializer takes (N, d) spatial tensor, not (N, d+1)
        tensor = torch.randn(10, 4) * 1e-3
        init = PretrainedInitializer(tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            emb = LorentzEmbedding(max_id=10, embedding_dim=4, initializer=init)
        assert emb.manifold.check_point_on_manifold(emb._embeddings.data)

    def test_construction_is_silent(self):
        """Construction no longer warns about optimizers — the check moved to the training loop."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LorentzEmbedding(max_id=5, embedding_dim=3)

    def test_registered_in_resolver(self):
        from pykeen.nn import representation_resolver

        assert representation_resolver.lookup("LorentzEmbedding") is LorentzEmbedding
