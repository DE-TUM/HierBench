"""The Eurostat Farm Structure Survey (FSS) taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "EstatFSS",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/estat-fss/"


@parse_docdata
class EstatFSS(SingleFileRemoteMetadataDataset, HierarchicalGraph):
    """The Eurostat Farm Structure Survey (FSS) taxonomy.

    Nodes are FSS concepts; edges are ``narrower``, ``hasTopConcept``,
    ``correspondsTo``, and ``closeMatch`` relations forming the classification
    hierarchy. Each node carries a human-readable label.

    Source: https://data.europa.eu/data/datasets?query=fss_2010-16

    Nearly every concept is a leaf with a single incoming edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    ---
    name: Estat-FSS
    citation:
        author: Eurostat
        year: 2016
        link: https://data.europa.eu/data/datasets?query=fss_2010-16
    statistics:
        entities: 355
        relations: 4
        training: 452
        testing: 452
        validation: 452
        triples: 452
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    ratios = None
    hierarchical_relation = "narrower"
