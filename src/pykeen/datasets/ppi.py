"""The PPI (Protein-Protein Interaction) dataset."""

from __future__ import annotations

from .numeric import SingleRemoteNumericDataset

__all__ = [
    "PPI",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi5Uw1bNRWJ4SaSC7NBCo8/PPI/"
_DATASET_URL = _BASE_URL + "dataset.tsv"
_LITERALS_URL = _BASE_URL + "literals.tsv"


class PPI(SingleRemoteNumericDataset):
    """The PPI (Protein-Protein Interaction) dataset.

    Nodes are proteins; edges represent protein-protein interactions between them.
    Each node has a literal feature vector.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the PPI dataset.

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
