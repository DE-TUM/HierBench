"""The Disease Ontology (DOID) dataset."""

from __future__ import annotations

from .metadata import RemoteMetadataDataset

__all__ = [
    "DOID",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiUW4yvpeJj7TQEZ9qbBra/doid/"


class DOID(RemoteMetadataDataset):
    """The Disease Ontology knowledge graph with per-entity metadata.

    Triples file: three tab-separated columns (head, relation, tail).
    Metadata file: one row per entity in entity-ID order.
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
