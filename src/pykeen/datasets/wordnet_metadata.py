"""A WordNet dataset with entity metadata."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "WordNet",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiMJE3wfCbXEy6PzjHcHoX/wordnet/"


class WordNet(RemoteMetadataDataset):
    """WordNet knowledge graph with per-entity metadata.

    Triples file: three tab-separated columns (head, relation, tail).
    Metadata file: one row per entity in entity-ID order.
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
