"""The CiteSeer citation network dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "CiteSeer",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiSVG9qjMcqKgKXuLF7v8v/citeseer/"


class CiteSeer(RemoteMetadataDataset):
    """The CiteSeer citation network dataset.

    Nodes are scientific publications; edges represent citations between them.
    Each node has a feature vector derived from the paper's bag-of-words
    representation and one of 6 class labels (research areas).

    Source: https://linqs.soe.ucsc.edu/data
    (Sen et al., 2008)
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
