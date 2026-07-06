"""Lorentz-model hyperbolic embeddings for hierarchical link prediction."""

from __future__ import annotations

import functools

import torch

from pykeen.losses import CrossEntropyLoss
from pykeen.models.nbase import ERModel
from pykeen.nn.hyperbolic import LorentzEmbedding
from pykeen.nn.modules import LorentzInteraction
from pykeen.typing import FloatTensor, Hint, Initializer

__all__ = ["LorentzE"]

#: Nickel & Kiela (2018) §3.2.2: initialise embeddings from U(-0.001, 0.001) in tangent space.
_NICKEL_INITIALIZER = functools.partial(torch.nn.init.uniform_, a=-0.001, b=0.001)


class LorentzE(ERModel[FloatTensor, tuple[()], FloatTensor]):
    """Lorentz-model embeddings for hierarchical knowledge graph completion.

    Embeds entities on the Lorentz (hyperboloid) manifold and scores triples by negative
    Lorentzian distance. The Lorentz parameterisation avoids the numerical instabilities of
    the Poincaré ball and yields higher-quality embeddings, especially in low dimensions.
    Best paired with :func:`pykeen.pipeline.hierarchy.hierarchy_completion_pipeline`.

    .. warning::

        This model uses manifold parameters. For correct Riemannian gradient updates use a
        Riemannian optimiser such as :class:`geoopt.optim.RiemannianSGD`.

    Introduced by [nickel2018]_.

    .. [nickel2018] Nickel, M., & Kiela, D. (2018). `Learning continuous hierarchies in the
       Lorentz model of hyperbolic geometry <https://arxiv.org/abs/1806.03417>`_. ICML 2018.
    """

    #: Softmax-over-negatives ranking loss (Nickel & Kiela 2018, Eq. 12).
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
            Spatial dimensionality of the Lorentz embeddings (stored internally as ``d+1``).
        :param curvature:
            Absolute curvature ``k`` of the Lorentz manifold (must be positive).
        :param trainable_curvature:
            Whether to learn the curvature jointly with the embeddings.
        :param entity_initializer:
            Tangent-space initializer for the embeddings (lifted to the hyperboloid via ``expmap0``).
            Defaults to the paper's ``U(-0.001, 0.001)``.
        :param kwargs:
            Additional keyword arguments forwarded to :class:`pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=LorentzInteraction,
            interaction_kwargs={"curvature": curvature, "trainable_curvature": trainable_curvature},
            entity_representations=LorentzEmbedding,
            entity_representations_kwargs={
                "shape": embedding_dim,
                "curvature": curvature,
                "initializer": entity_initializer,
            },
            relation_representations=[],
            **kwargs,
        )
