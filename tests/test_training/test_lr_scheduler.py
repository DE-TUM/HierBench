"""Tests for LR schedulers."""

import pytest
from class_resolver import HintOrType, OptionalKwargs
from torch.optim import lr_scheduler

from pykeen.pipeline import pipeline
from pykeen.training.callbacks import TrainingCallback


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(None, None), ("CosineAnnealingWarmRestarts", {"T_0": 10})],
)
def test_lr_scheduler(cls: HintOrType[lr_scheduler.LRScheduler], kwargs: OptionalKwargs) -> None:
    """Smoke-test for training with learning rate schedule."""
    pipeline(
        dataset="nations",
        model="mure",
        model_kwargs={"embedding_dim": 2},
        training_kwargs={"num_epochs": 1},
        lr_scheduler=cls,
        lr_scheduler_kwargs=kwargs,
    )


def test_lr_scheduler_rebuild_does_not_compound_factor() -> None:
    """Rebuilding the optimizer/scheduler at train start must not re-apply a construction-time lr scaling."""
    lr, factor = 0.3, 0.1
    seen: list[float] = []

    class _LRRecorder(TrainingCallback):
        """Record the optimizer's actual lr after each epoch."""

        def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
            """Store the current lr of the first parameter group."""
            seen.append(self.training_loop.optimizer.param_groups[0]["lr"])

    pipeline(
        dataset="nations",
        model="mure",
        model_kwargs={"embedding_dim": 2},
        training_kwargs={"num_epochs": 1, "callbacks": [_LRRecorder()]},
        optimizer_kwargs={"lr": lr},
        lr_scheduler="constant",
        lr_scheduler_kwargs={"factor": factor, "total_iters": 10},
    )
    # ConstantLR scales at construction; a compounding rebuild would yield lr * factor**2 or worse
    assert seen == pytest.approx([lr * factor])
