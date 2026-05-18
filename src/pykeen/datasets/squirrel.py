"""The Wikipedia Squirrel page-page graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "Squirrel",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiTXqqQ77hDtEfgC6Su546/squirrel/"


@parse_docdata
class Squirrel(RemoteMetadataDataset):
    """The Wikipedia Squirrel page-page graph dataset.

    Nodes are Wikipedia pages about squirrels; edges connect pages that share
    mutual links. Each node has a feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html

    ---
    name: Squirrel
    citation:
        author: Rozemberczki
        year: 2021
        link: https://arxiv.org/abs/2106.11181
        github: benedekrozemberczki/MUSAE
    statistics:
        entities: 5201
        relations: 1
        training: 158794
        testing: 19850
        validation: 19849
        triples: 198493
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
