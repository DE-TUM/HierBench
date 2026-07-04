"""Hyperbolic Entailment Cones model for hierarchical link prediction."""

from __future__ import annotations

from pykeen.losses import MarginRankingLoss
from pykeen.models.nbase import ERModel
from pykeen.nn.hyperbolic import HyperbolicConesEmbedding
from pykeen.nn.modules import HyperbolicConesInteraction
from pykeen.typing import FloatTensor

__all__ = ["HyperbolicCones"]


class HyperbolicCones(ERModel[FloatTensor, tuple[()], FloatTensor]):
    """Hyperbolic Entailment Cones for learning hierarchical embeddings.

    Embeds entities on the Poincaré ball and scores parent→child entailment by measuring
    how far the child lies outside the parent's angular entailment cone.

    Each parent entity ``h`` defines a cone of half-angle
    ``ψ(h) = arcsin(K(1−‖h‖²)/‖h‖)``. A child ``t`` is scored by how much
    its exterior angle ``Ξ(h,t)`` exceeds ``ψ(h)``; a zero penalty means ``t``
    lies inside the cone.

    Designed for hierarchical datasets; best paired with
    :func:`pykeen.pipeline.hierarchy.transitive_ancestor_descendant_pipeline` and a
    Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.

    .. warning::

        This model uses manifold parameters. For correct Riemannian gradient updates use
        a Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.

    .. [ganea2018] Ganea, O.-E., Bécigneul, G., & Hofmann, T. (2018).
       `Hyperbolic Entailment Cones for Learning Hierarchical Embeddings
       <https://arxiv.org/abs/1804.01882>`_. ICML 2018.
    """

    loss_default = MarginRankingLoss
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
            },
            relation_representations=[],
            **kwargs,
        )
