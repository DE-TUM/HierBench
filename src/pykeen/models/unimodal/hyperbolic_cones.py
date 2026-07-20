"""Hyperbolic Entailment Cones model for hierarchical link prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pykeen.losses import PointwiseHingeLoss
from pykeen.models.nbase import ERModel
from pykeen.nn.hyperbolic import HyperbolicConesEmbedding
from pykeen.nn.modules import HyperbolicConesInteraction
from pykeen.typing import FloatTensor, Hint, Initializer

__all__ = ["HyperbolicCones"]


class HyperbolicCones(ERModel[FloatTensor, tuple[()], FloatTensor]):
    """Hyperbolic Entailment Cones for learning hierarchical embeddings.

    Embeds entities on the Poincaré ball and scores parent→child entailment by measuring
    how far the child lies outside the parent's angular entailment cone.

    Each parent entity ``h`` defines a cone of half-angle
    ``ψ(h) = arcsin(K(1−‖h‖²)/‖h‖)``. A child ``t`` is scored by the signed margin
    ``ψ(h) − Ξ(h,t)`` where ``Ξ`` is its exterior angle; a non-negative score means
    ``t`` lies inside the cone. The paper's relu'd cone energy is recovered by the
    default :class:`pykeen.losses.PointwiseHingeLoss`.

    Designed for hierarchical datasets; best paired with
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_pipeline` and a
    Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.

    Introduced by [ganea2018]_.

    .. warning::

        This model uses manifold parameters. For correct Riemannian gradient updates use
        a Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.
    """

    #: Ganea et al. (2018) Eq. 32 is a *pointwise* hinge: positives are unconditionally driven to
    #: zero energy, negatives to energy >= margin. A pairwise margin-ranking loss saturates almost
    #: immediately here (energies are angles in radians vs a 0.01 margin) and yields no gradient.
    loss_default = PointwiseHingeLoss
    loss_default_kwargs = {"margin": 0.01, "reduction": "mean"}

    hpo_default = {
        "embedding_dim": {"type": int, "low": 5, "high": 200, "q": 5},
        "k": {"type": float, "low": 0.01, "high": 0.5, "log": True},
        "curvature": {"type": float, "low": 0.1, "high": 2.0},
    }

    def __init__(
        self,
        *,
        embedding_dim: int = 5,
        k: float = 0.1,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        entity_initializer: Hint[Initializer] = None,
        entity_initializer_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialise the model.

        :param embedding_dim:
            Dimensionality of the Poincaré ball embeddings.
        :param k:
            Cone width parameter (K in the paper). Controls the opening half-angle ψ and
            sets the inner-radius constraint ``inner_radius = 2k/(1+√(1+4k²))``.
            Larger k → wider cones. Typical value: 0.1.
        :param curvature:
            Curvature ``c`` of the Poincaré ball (must be positive).
        :param trainable_curvature:
            Whether to learn the curvature jointly with the embeddings.
        :param entity_initializer:
            Tangent-space initializer for the embeddings (lifted to the ball via ``expmap0``).
            Defaults to a small random tangent vector. Ganea et al. (2018 §5) instead warm-start
            from a pretrained Poincaré model (see :class:`pykeen.models.PoincareE`), rescaled by
            0.7 and mapped to tangent space via ``manifold.logmap0`` before wrapping in
            :class:`pykeen.nn.init.PretrainedInitializer`.
        :param entity_initializer_kwargs:
            Additional keyword arguments for the entity initializer.
        :param kwargs:
            Additional keyword arguments forwarded to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=HyperbolicConesInteraction,
            interaction_kwargs={"k": k},
            entity_representations=HyperbolicConesEmbedding,
            entity_representations_kwargs={
                "shape": embedding_dim,
                "curvature": curvature,
                "trainable_curvature": trainable_curvature,
                "k": k,
                "initializer": entity_initializer,
                "initializer_kwargs": entity_initializer_kwargs,
            },
            relation_representations=[],
            **kwargs,
        )
