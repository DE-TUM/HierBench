"""The ACM Computing Classification System (CCS) taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "ACMCCS",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/acmccm/"


@parse_docdata
class ACMCCS(SingleFileRemoteMetadataDataset, HierarchicalGraph):
    """The ACM Computing Classification System (CCS) taxonomy.

    Nodes are ACM CCS 2012 concepts; edges are ``narrower`` / ``hasTopConcept``
    relations forming the classification hierarchy. Each node carries a
    human-readable label.

    Source: https://www.acm.org/publications/class-2012

    Nearly every concept is a leaf with a single ``narrower``/``hasTopConcept`` edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    ---
    name: ACM-CCS
    citation:
        author: ACM
        year: 2012
        link: https://www.acm.org/publications/class-2012
    statistics:
        entities: 2114
        relations: 2
        training: 2113
        testing: 2113
        validation: 2113
        triples: 2113
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    ratios = None
    hierarchical_relation = "narrower"
