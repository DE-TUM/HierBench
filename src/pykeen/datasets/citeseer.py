"""The CiteSeer citation network dataset."""

from __future__ import annotations

from .numeric import SingleRemoteNumericDataset

__all__ = [
    "CiteSeer",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiSVG9qjMcqKgKXuLF7v8v/citeseer/"
_DATASET_URL = _BASE_URL + "dataset.tsv"
_LITERALS_URL = _BASE_URL + "literals.tsv"


class CiteSeer(SingleRemoteNumericDataset):
    """The CiteSeer citation network dataset.

    Nodes are scientific publications; edges represent citations between them.
    Each node has a literal feature vector derived from the paper's bag-of-words
    representation and one of 6 class labels (research areas).

    Source: https://linqs.soe.ucsc.edu/data
    (Sen et al., 2008)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the CiteSeer dataset.

        :param kwargs: Additional keyword arguments forwarded to
            :class:`~pykeen.datasets.numeric.SingleRemoteNumericDataset`.
        """
        kwargs.setdefault("read_csv_kwargs", {})
        kwargs["read_csv_kwargs"].setdefault("dtype", str)
        super().__init__(
            url=_DATASET_URL,
            literals_url=_LITERALS_URL,
            **kwargs,
        )
