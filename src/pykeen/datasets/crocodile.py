"""The Wikipedia Crocodile page-page graph dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "Crocodile",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiSthwLrG6VmKqbr9eVsnA/crocodile/"


class Crocodile(RemoteMetadataDataset):
    """The Wikipedia Crocodile page-page graph dataset.

    Nodes are Wikipedia pages about crocodiles; edges connect pages that share
    mutual links. Each node has a feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html
    (MUSAE, Rozemberczki et al., 2021)
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
