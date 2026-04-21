"""The Wikipedia Squirrel page-page graph dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "Squirrel",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiTXqqQ77hDtEfgC6Su546/squirrel/"


class Squirrel(RemoteMetadataDataset):
    """The Wikipedia Squirrel page-page graph dataset.

    Nodes are Wikipedia pages about squirrels; edges connect pages that share
    mutual links. Each node has a feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html
    (MUSAE, Rozemberczki et al., 2021)
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
