"""The Disease Ontology (DOID) dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "DOID",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/doid/"


@parse_docdata
class DOID(SingleFileRemoteMetadataDataset, HierarchicalGraph):
    """The Disease Ontology knowledge graph with per-entity metadata.

    Nodes are disease terms from the Human Disease Ontology; edges represent
    hierarchical relationships (e.g. is-a) between disease concepts.
    Each entity carries structured metadata.

    Source: https://disease-ontology.org/downloads/

    ---
    name: DOID
    citation:
        author: Schriml
        year:
        link:
        github: DiseaseOntology/HumanDiseaseOntology
    statistics:
        entities: 12221
        relations: 2
        training: 13928
        testing: 1741
        validation: 1741
        triples: 17410
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    hierarchical_relation = "has_subclass"
