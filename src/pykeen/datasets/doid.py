"""The Disease Ontology (DOID) dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "DOID",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiUW4yvpeJj7TQEZ9qbBra/doid/"


@parse_docdata
class DOID(RemoteMetadataDataset):
    """The Disease Ontology knowledge graph with per-entity metadata.

    Nodes are disease terms from the Human Disease Ontology; edges represent
    hierarchical relationships (e.g. is-a) between disease concepts.
    Each entity carries structured metadata.

    Source: https://disease-ontology.org/downloads/

    ---
    name: DOID
    citation:
        author: Schriml
        year: 2022
        link: https://doi.org/10.1093/nar/gkac1048
        github: DiseaseOntology/HumanDiseaseOntology
    statistics:
        entities: 0
        relations: 0
        training: 0
        testing: 0
        validation: 0
        triples: 0
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
