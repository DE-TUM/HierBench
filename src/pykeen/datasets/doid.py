"""The Disease Ontology (DOID) dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "DOID",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiUW4yvpeJj7TQEZ9qbBra/doid/"


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
        entities: 12079
        relations: 2
        training: 13710
        testing: 1714
        validation: 1714
        triples: 17138
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    hierarchical_relation = "has_subclass"
