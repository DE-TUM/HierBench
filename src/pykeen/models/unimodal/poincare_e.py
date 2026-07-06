"""Poincaré Embedding model for hierarchical link prediction."""

from __future__ import annotations

import functools

import torch

from pykeen.losses import CrossEntropyLoss
from pykeen.models.nbase import ERModel
from pykeen.nn.hyperbolic import PoincareEmbedding
from pykeen.nn.modules import PoincareEInteraction
from pykeen.typing import FloatTensor, Hint, Initializer

__all__ = ["PoincareE"]

#: Nickel & Kiela (2017) §3.2: initialise embeddings from U(-0.001, 0.001) in tangent space.
_NICKEL_INITIALIZER = functools.partial(torch.nn.init.uniform_, a=-0.001, b=0.001)


class PoincareE(ERModel[FloatTensor, tuple[()], FloatTensor]):
    """Poincaré Embeddings for hierarchical knowledge graph completion.

    Embeds entities on the Poincaré ball and scores triples by negative Poincaré distance.
    Designed for hierarchical datasets; best paired with
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_pipeline`.

    .. warning::

        This model uses manifold parameters. For correct Riemannian gradient updates use a
        Riemannian optimiser such as :class:`geoopt.optim.RiemannianAdam`.

    Introduced by [nickel2017]_.

    .. [nickel2017] Nickel, M., & Kiela, D. (2017). `Poincaré embeddings for learning hierarchical
       representations <https://arxiv.org/abs/1705.08039>`_. NeurIPS 2017.
    """

    #: Softmax-over-negatives ranking loss (Nickel & Kiela 2017, Eq. 6).
    loss_default = CrossEntropyLoss
    loss_default_kwargs = {"reduction": "mean"}

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
        entity_initializer: Hint[Initializer] = _NICKEL_INITIALIZER,
        **kwargs,
    ) -> None:
        """Initialise the model.

        :param embedding_dim:
            Dimensionality of the Poincaré ball embeddings.
        :param curvature:
            Curvature ``c`` of the Poincaré ball (must be positive).
        :param trainable_curvature:
            Whether to learn the curvature jointly with the embeddings.
        :param entity_initializer:
            Tangent-space initializer for the embeddings (lifted to the ball via ``expmap0``).
            Defaults to the paper's ``U(-0.001, 0.001)``.
        :param kwargs:
            Additional keyword arguments forwarded to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=PoincareEInteraction,
            interaction_kwargs={"curvature": curvature, "trainable_curvature": trainable_curvature},
            entity_representations=PoincareEmbedding,
            entity_representations_kwargs={
                "shape": embedding_dim,
                "curvature": curvature,
                "initializer": entity_initializer,
            },
            relation_representations=[],
            **kwargs,
        )
