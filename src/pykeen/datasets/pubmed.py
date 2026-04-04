"""The PubMed citation graph dataset."""

from __future__ import annotations

from .numeric import SingleRemoteNumericDataset

__all__ = [
    "PubMed",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiKQMfkFSLYAXMA4pgsNak/pubmed/"
_DATASET_URL = _BASE_URL + "dataset.tsv"
_LITERALS_URL = _BASE_URL + "literals.tsv"


class PubMed(SingleRemoteNumericDataset):
    """The PubMed citation graph dataset.

    Nodes are PubMed articles; edges represent citation links between them.
    Each node has a literal feature vector and one of 3 class labels
    (diabetes types: Experimentally induced, Type 1, Type 2).

    Source: https://linqs.org/datasets/#pubmed-diabetes
    (Sen et al., 2008)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the PubMed dataset.

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
