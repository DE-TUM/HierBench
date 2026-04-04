"""The Wikipedia Crocodile page-page graph dataset."""

from __future__ import annotations

from .numeric import SingleRemoteNumericDataset

__all__ = [
    "Crocodile",
]
_BASE_URL = "https://syncandshare.lrz.de/dl/fiSthwLrG6VmKqbr9eVsnA/crocodile/"
_DATASET_URL = _BASE_URL + "dataset.tsv"
_LITERALS_URL = _BASE_URL + "literals.tsv"


class Crocodile(SingleRemoteNumericDataset):
    """The Wikipedia Crocodile page-page graph dataset.

    Nodes are Wikipedia pages about crocodiles; edges connect pages that share
    mutual links. Each node has a literal feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html
    (MUSAE, Rozemberczki et al., 2021)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Crocodile dataset.

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
