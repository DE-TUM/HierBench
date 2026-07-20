"""Tests for the manifold/non-Riemannian optimizer warning in the training loop."""

from __future__ import annotations

import warnings

import geoopt
import pytest
import torch
from torch import nn

from pykeen.training.training_loop import _warn_if_manifold_without_riemannian


class _ManifoldModule(nn.Module):
    """Minimal module holding a single Poincaré-ball manifold parameter."""

    def __init__(self):
        super().__init__()
        manifold = geoopt.PoincareBall()
        self.p = geoopt.ManifoldParameter(manifold.origin(3, 2), manifold=manifold)


@pytest.fixture
def manifold_model():
    """Return a module carrying geoopt manifold parameters."""
    return _ManifoldModule()


def _assert_warns(model, optimizer):
    with pytest.warns(UserWarning, match="non-Riemannian"):
        _warn_if_manifold_without_riemannian(model, optimizer)


def _assert_silent(model, optimizer):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _warn_if_manifold_without_riemannian(model, optimizer)


def test_warns_with_euclidean_optimizer(manifold_model):
    """A manifold model trained with plain Adam triggers the warning."""
    _assert_warns(manifold_model, torch.optim.Adam(manifold_model.parameters()))


def test_silent_with_riemannian_adam(manifold_model):
    """RiemannianAdam is on-manifold, so no warning fires."""
    _assert_silent(manifold_model, geoopt.optim.RiemannianAdam(manifold_model.parameters()))


def test_silent_with_riemannian_sgd(manifold_model):
    """RiemannianSGD is on-manifold, so no warning fires."""
    _assert_silent(manifold_model, geoopt.optim.RiemannianSGD(manifold_model.parameters(), lr=0.1))


def test_silent_for_euclidean_model():
    """A model without manifold parameters never warns, even with a Euclidean optimizer."""
    model = nn.Linear(2, 2)
    _assert_silent(model, torch.optim.Adam(model.parameters()))
