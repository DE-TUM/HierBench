"""The PubMed citation graph dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "PubMed",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiKQMfkFSLYAXMA4pgsNak/pubmed/"


class PubMed(RemoteMetadataDataset):
    """The PubMed citation graph dataset.

    Nodes are PubMed articles; edges represent citation links between them.
    Each node has a literal feature vector and one of 3 class labels
    (diabetes types: Experimentally induced, Type 1, Type 2).

    Source: https://linqs.org/datasets/#pubmed-diabetes
    (Sen et al., 2008)
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
