"""Poincaré Embedding model for hierarchical link prediction."""

from __future__ import annotations

from pykeen.models.nbase import ERModel
from pykeen.nn.hyperbolic import PoincareEmbedding
from pykeen.nn.modules import PoincareEInteraction
from pykeen.typing import FloatTensor

__all__ = ["PoincareE"]


class PoincareE(ERModel[FloatTensor, tuple[()], FloatTensor]):
    """Poincaré Embeddings for hierarchical knowledge graph completion.

    Embeds entities on the Poincaré ball and scores triples by negative Poincaré distance.
    Designed for hierarchical datasets; best paired with
    :func:`pykeen.pipeline.hierarchy.ancestor_descendant_pipeline`.

    .. warning::

        This model uses manifold parameters. For correct Riemannian gradient updates use a
        Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.

    .. [nickel2017] Nickel, M., & Kiela, D. (2017). `Poincaré embeddings for learning hierarchical
       representations <https://arxiv.org/abs/1705.08039>`_. NeurIPS 2017.
    """

    hpo_default = {
        "embedding_dim": {"type": int, "low": 8, "high": 256, "q": 8},
        "curvature": {"type": float, "low": 0.1, "high": 2.0},
    }

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        **kwargs,
    ) -> None:
        """Initialise the model.

        :param embedding_dim:
            Dimensionality of the Poincaré ball embeddings.
        :param curvature:
            Curvature ``c`` of the Poincaré ball (must be positive).
        :param trainable_curvature:
            Whether to learn the curvature jointly with the embeddings.
        :param kwargs:
            Additional keyword arguments forwarded to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=PoincareEInteraction,
            interaction_kwargs={"curvature": curvature, "trainable_curvature": trainable_curvature},
            entity_representations=PoincareEmbedding,
            entity_representations_kwargs={"shape": embedding_dim, "curvature": curvature},
            relation_representations=[],
            **kwargs,
        )
