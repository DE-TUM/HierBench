"""The Wikipedia Chameleon page-page graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "Chameleon",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi6LqgLVY3P3AEtC24Yzpq/chameleon/"


@parse_docdata
class Chameleon(RemoteMetadataDataset):
    """The Wikipedia Chameleon page-page graph dataset.

    Nodes are Wikipedia pages about chameleons; edges connect pages that share
    mutual links. Each node has a feature vector derived from the page's nouns
    and one of 5 class labels (page traffic categories).

    Source: https://snap.stanford.edu/data/wikipedia-article-networks.html
    (MUSAE, Rozemberczki et al., 2021)

    ---
    name: Chameleon
    citation:
        author: Rozemberczki
        year: 2021
        link: https://arxiv.org/abs/2106.11181
        github: benedekrozemberczki/MUSAE
    statistics:
        entities: 2277
        relations: 1
        training: 24884
        testing: 3109
        validation: 3109
        triples: 31102
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
