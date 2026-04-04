"""The Wikipedia Squirrel page-page graph dataset."""

from __future__ import annotations

from .numeric import SingleRemoteNumericDataset

__all__ = [
    "Squirrel",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiTXqqQ77hDtEfgC6Su546/squirrel/"
_DATASET_URL = _BASE_URL + "dataset.tsv"
_LITERALS_URL = _BASE_URL + "literals.tsv"


class Squirrel(SingleRemoteNumericDataset):
    """The Wikipedia Squirrel page-page graph dataset.

    Nodes are Wikipedia pages about squirrels; edges connect pages that share
    mutual links. Each node has a literal feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html
    (MUSAE, Rozemberczki et al., 2021)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Squirrel dataset.

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
