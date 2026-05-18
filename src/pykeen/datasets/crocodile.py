"""The Wikipedia Crocodile page-page graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "Crocodile",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiSthwLrG6VmKqbr9eVsnA/crocodile/"


@parse_docdata
class Crocodile(RemoteMetadataDataset):
    """The Wikipedia Crocodile page-page graph dataset.

    Nodes are Wikipedia pages about crocodiles; edges connect pages that share
    mutual links. Each node has a feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html

    ---
    name: Crocodile
    citation:
        author: Rozemberczki
        year: 2021
        link: https://arxiv.org/abs/2106.11181
        github: benedekrozemberczki/MUSAE
    statistics:
        entities: 11631
        relations: 1
        training: 136734
        testing: 17092
        validation: 17092
        triples: 170918
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
