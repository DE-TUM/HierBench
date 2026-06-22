"""The Wikipedia Chameleon page-page graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "Chameleon",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi6LqgLVY3P3AEtC24Yzpq/chameleon/"


@parse_docdata
class Chameleon(SingleFileRemoteMetadataDataset):
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
        link: https://arxiv.org/abs/1909.13021
        github: benedekrozemberczki/MUSAE
    statistics:
        entities: 2277
        relations: 1
        training: 28880
        testing: 3610
        validation: 3611
        triples: 36101
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
