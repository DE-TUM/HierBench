"""The Medical Subject Headings (MeSH) taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "MeSH",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/mesh/"


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

    MeSH's hierarchy is not a DAG, so the transitive reduction is undefined; ships a precomputed
    transitive closure (``closure_url``) consumed by :func:`pykeen.datasets.metadata.load_closure_pool`
    computed with the asserted-edges fallback, so hierarchy-closure splits skip re-deriving it; see
    :class:`~pykeen.datasets.ancestor_descendant.MeSHTransitive0Percent` for a frozen
    ancestor-descendant split that already bakes in the transitive edges.

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
    closure_url = _BASE_URL + "closure.tsv"
    ratios = None
    hierarchical_relation = "narrower"
