"""The Cora citation network dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "Cora",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiRjW4AR7yjLBjut8onBg5/cora/"


class Cora(RemoteMetadataDataset):
    """The Cora citation network dataset.

    Nodes are scientific publications; edges represent citations between them.
    Each node has a feature vector derived from the paper's bag-of-words
    representation and one of 7 class labels (research topics).

    Source: https://relational.fit.cvut.cz/dataset/CORA
    (McCallum et al., 2000)
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
