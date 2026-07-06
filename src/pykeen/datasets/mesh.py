"""The Medical Subject Headings (MeSH) taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "MeSH",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi3K4YDeddSRQqrTFLC6nt/MeSH/"


@parse_docdata
class MeSH(SingleFileRemoteMetadataDataset, HierarchicalGraph):
    """The Medical Subject Headings (MeSH) taxonomy.

    Nodes are MeSH descriptors; edges are ``narrower``, ``pharmacologicalAction``,
    and ``seeAlso`` relations. Each node carries a human-readable label.

    Source: https://www.nlm.nih.gov/mesh/meshhome.html

    Nearly every concept is a leaf with a single ``narrower`` edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    ---
    name: MeSH
    citation:
        author: National Library of Medicine
        year: 2024
        link: https://www.nlm.nih.gov/mesh/meshhome.html
    statistics:
        entities: 30837
        relations: 3
        training: 56383
        testing: 56383
        validation: 56383
        triples: 56383
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    ratios = None
    hierarchical_relation = "narrower"
