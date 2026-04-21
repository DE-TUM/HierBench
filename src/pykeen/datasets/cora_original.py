"""The Cora (McCallum) original dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "CoraOriginal",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiQHnjbWoTtkw1D9mud59W/cora_mccallum/"


class CoraOriginal(RemoteMetadataDataset):
    """The original Cora citation network dataset (McCallum et al., 2000).

    Nodes are scientific publications; edges represent citations between them.
    This is the unprocessed original version with node feature metadata.

    Source: McCallum et al., 2000
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
