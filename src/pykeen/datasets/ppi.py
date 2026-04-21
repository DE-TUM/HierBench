"""The PPI (Protein-Protein Interaction) dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "PPI",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi5Uw1bNRWJ4SaSC7NBCo8/PPI/"


class PPI(RemoteMetadataDataset):
    """The PPI (Protein-Protein Interaction) dataset.

    Nodes are proteins; edges represent protein-protein interactions between them.
    Each node has a feature vector.
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
