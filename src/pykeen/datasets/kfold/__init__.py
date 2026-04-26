"""K-fold dataset utilities for cross-validation."""

from .base import EagerKFoldDataset, KFoldDataset, to_kfold

__all__ = ["EagerKFoldDataset", "KFoldDataset", "to_kfold"]
