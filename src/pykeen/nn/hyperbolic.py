"""Hyperbolic representation modules using geoopt manifolds."""

from __future__ import annotations

import logging
from typing import Any

import geoopt
import torch
from torch import nn

from .init import initializer_resolver
from .representation import Representation, process_max_id, process_shape
from ..typing import FloatTensor, Hint, Initializer, LongTensor

__all__ = [
    "PoincareEmbedding",
    "LorentzEmbedding",
]

logger = logging.getLogger(__name__)


class PoincareEmbedding(Representation):
    """Trainable embeddings on the Poincaré ball manifold (curvature -c).

    Points live in B^d_c = {x ∈ ℝ^d : ||x||² < 1/c}.

    Uses :class:`geoopt.ManifoldParameter` so the embeddings can be optimized
    with :class:`geoopt.optim.RiemannianAdam` (recommended) or with standard
    Adam + manifold projection via :meth:`post_parameter_update`.

    ---
    name: Poincaré Embedding
    """

    def __init__(
        self,
        max_id: int | None = None,
        num_embeddings: int | None = None,
        embedding_dim: int | None = None,
        shape: None | int = None,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: dict[str, Any] | None = None,
        trainable: bool = True,
        **kwargs,
    ):
        """Initialize Poincaré ball embeddings.

        :param max_id:
            The number of embeddings.
        :param num_embeddings:
            Alias for max_id (deprecated).
        :param embedding_dim:
            The dimensionality d of the Poincaré ball. Ball radius = 1/sqrt(c).
        :param shape:
            Alternative to embedding_dim; must be 1-D for Poincaré.
        :param curvature:
            Absolute curvature c > 0. Actual sectional curvature = -c.
        :param trainable_curvature:
            If True, wraps curvature in nn.Parameter so it is learned during training.
        :param initializer:
            Optional callable on a (max_id, d) zero tensor; result is lifted to
            the manifold via expmap0. Compatible with PretrainedInitializer.
        :param initializer_kwargs:
            Additional kwargs for the initializer.
        :param trainable:
            Whether the embedding weights require gradient.
        :param kwargs:
            Passed to :class:`~pykeen.nn.representation.Representation`.
        """
        max_id = process_max_id(max_id, num_embeddings)
        _embedding_dim, _shape = process_shape(embedding_dim, shape)

        # Set up nn.Module state first — required before assigning nn.Parameter
        super().__init__(max_id=max_id, shape=_shape, **kwargs)

        if trainable_curvature:
            self._curvature = nn.Parameter(torch.tensor(float(curvature)))
            self.manifold = geoopt.PoincareBall(c=self._curvature)
        else:
            self._curvature = None
            self.manifold = geoopt.PoincareBall(c=float(curvature))

        self.initializer = initializer_resolver.make_safe(initializer, initializer_kwargs)

        data = self.manifold.origin(max_id, _embedding_dim)
        self._embeddings = geoopt.ManifoldParameter(data, manifold=self.manifold, requires_grad=trainable)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # noqa: D102
        """Reset embeddings by sampling in tangent space and mapping with expmap0."""
        if not hasattr(self, "_embeddings") or not hasattr(self, "manifold"):
            return
        with torch.no_grad():
            if self.initializer is not None:
                v = self.initializer(torch.zeros_like(self._embeddings.data))
            else:
                v = torch.randn_like(self._embeddings.data) * 1e-3
            self._embeddings.data.copy_(self.manifold.expmap0(v))

    def post_parameter_update(self) -> None:  # noqa: D102
        """Project embedding parameters back onto the Poincare manifold."""
        with torch.no_grad():
            self._embeddings.data.copy_(self.manifold.projx(self._embeddings.data))

    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        if indices is None:
            return self._embeddings
        return self._embeddings[indices.to(self._embeddings.device)]

    def iter_extra_repr(self):  # noqa: D102
        """Yield extra representation fields for module string formatting."""
        yield from super().iter_extra_repr()
        c = self._curvature.item() if self._curvature is not None else self.manifold.c
        yield f"curvature={c:.4f}"
        if self._curvature is not None:
            yield "trainable_curvature=True"


class LorentzEmbedding(Representation):
    """Trainable embeddings on the Lorentz (hyperboloid) manifold (curvature -k).

    Points live in L^d_k = {x ∈ ℝ^(d+1) : ⟨x,x⟩_L = -1/k, x₀ > 0},
    where ⟨u,v⟩_L = -u₀v₀ + Σᵢ uᵢvᵢ is the Lorentzian inner product.

    Numerically more stable than the Poincaré ball for larger dimensions.
    The full (d+1)-dimensional vector is returned by :meth:`forward` because
    downstream interaction functions need x₀ to compute Lorentzian distances
    and inner products.

    Uses :class:`geoopt.ManifoldParameter` so the embeddings can be optimized
    with :class:`geoopt.optim.RiemannianAdam` (recommended) or with standard
    Adam + manifold projection via :meth:`post_parameter_update`.

    ---
    name: Lorentz Embedding
    """

    def __init__(
        self,
        max_id: int | None = None,
        num_embeddings: int | None = None,
        embedding_dim: int | None = None,
        shape: None | int = None,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: dict[str, Any] | None = None,
        trainable: bool = True,
        **kwargs,
    ):
        """Initialize Lorentz hyperboloid embeddings.

        :param max_id:
            The number of embeddings.
        :param num_embeddings:
            Alias for max_id (deprecated).
        :param embedding_dim:
            The spatial dimension d. Embeddings are stored and returned as
            (d+1)-dimensional Lorentz vectors; shape is set to (d+1,).
        :param shape:
            Alternative to embedding_dim; must be 1-D.
        :param curvature:
            Absolute curvature k > 0. Actual sectional curvature = -k.
        :param trainable_curvature:
            If True, wraps curvature in nn.Parameter so it is learned during training.
        :param initializer:
            Optional callable on a (max_id, d) zero tensor of spatial coords;
            result is lifted to the hyperboloid via expmap0 at the origin.
            Compatible with PretrainedInitializer(tensor of shape (N, d)).
        :param initializer_kwargs:
            Additional kwargs for the initializer.
        :param trainable:
            Whether the embedding weights require gradient.
        :param kwargs:
            Passed to :class:`~pykeen.nn.representation.Representation`.
        """
        max_id = process_max_id(max_id, num_embeddings)
        _embedding_dim, _shape = process_shape(embedding_dim, shape)
        # Internal storage and public shape are both (d+1,)
        internal_dim = _embedding_dim + 1
        lorentz_shape = (internal_dim,)

        super().__init__(max_id=max_id, shape=lorentz_shape, **kwargs)

        if trainable_curvature:
            self._curvature = nn.Parameter(torch.tensor(float(curvature)))
            self.manifold = geoopt.Lorentz(k=self._curvature)
        else:
            self._curvature = None
            self.manifold = geoopt.Lorentz(k=float(curvature))

        self._internal_dim = internal_dim
        self.initializer = initializer_resolver.make_safe(initializer, initializer_kwargs)

        data = self.manifold.origin(max_id, internal_dim)
        self._embeddings = geoopt.ManifoldParameter(data, manifold=self.manifold, requires_grad=trainable)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # noqa: D102
        """Reset embeddings by sampling spatial tangent vectors and applying expmap0."""
        if not hasattr(self, "_embeddings") or not hasattr(self, "manifold") or not hasattr(self, "_internal_dim"):
            return
        with torch.no_grad():
            # Build tangent vector at origin: time component must be 0
            v = torch.zeros(self.max_id, self._internal_dim)
            if self.initializer is not None:
                # Initializer operates on (max_id, d) spatial coordinates
                spatial = self.initializer(torch.zeros(self.max_id, self._internal_dim - 1))
                v[:, 1:] = spatial
            else:
                v[:, 1:] = torch.randn(self.max_id, self._internal_dim - 1) * 1e-3
            self._embeddings.data.copy_(self.manifold.expmap0(v))

    def post_parameter_update(self) -> None:  # noqa: D102
        """Project embedding parameters back onto the Lorentz manifold."""
        with torch.no_grad():
            self._embeddings.data.copy_(self.manifold.projx(self._embeddings.data))

    def _plain_forward(self, indices: LongTensor | None = None) -> FloatTensor:  # noqa: D102
        # Return full (d+1) Lorentz vector — x₀ is needed for ⟨u,v⟩_L
        if indices is None:
            return self._embeddings
        return self._embeddings[indices.to(self._embeddings.device)]

    def iter_extra_repr(self):  # noqa: D102
        """Yield extra representation fields for module string formatting."""
        yield from super().iter_extra_repr()
        k = self._curvature.item() if self._curvature is not None else self.manifold.k
        yield f"curvature={k:.4f}"
        if self._curvature is not None:
            yield "trainable_curvature=True"
