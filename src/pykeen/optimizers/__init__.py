"""Optimizers available in PyKEEN, based on :class:`pykeen.optim.optimizer.Optimizer`."""

from collections.abc import Mapping
from typing import Any

from class_resolver.contrib.torch import optimizer_resolver
from geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

__all__ = [
    "Optimizer",
    "optimizers_hpo_defaults",
    "optimizer_resolver",
]

# geoopt's Riemannian optimisers are not ``torch.optim`` subclasses discovered by the resolver,
# so register them explicitly to make them selectable by name (e.g. ``optimizer="RiemannianAdam"``).
# RiemannianSGD is the Nickel & Kiela (2017) RSGD / natural-gradient path for PoincareE.
optimizer_resolver.register(RiemannianAdam, raise_on_conflict=False)
optimizer_resolver.register(RiemannianSGD, raise_on_conflict=False)

#: The default strategy for optimizing the optimizers' hyper-parameters (yo dawg)
optimizers_hpo_defaults: Mapping[type[Optimizer], Mapping[str, Any]] = {
    Adagrad: {
        "lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
    },
    Adam: {
        "lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
    },
    Adamax: {
        "lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
    },
    AdamW: {
        "lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
    },
    SGD: {
        "lr": {"type": float, "low": 0.001, "high": 0.1, "scale": "log"},
    },
}
