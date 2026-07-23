"""The NASA taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, SingleFileRemoteMetadataDataset

__all__ = [
    "NASA",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/nasa/"


@parse_docdata
class NASA(SingleFileRemoteMetadataDataset, HierarchicalGraph):
    """The 2024 NASA Technology Taxonomy.

    Nodes are NASA technology areas; edges are ``has_subclass`` relations forming the
    classification hierarchy. Each node carries a human-readable title, definition, and
    example technologies.

    Nearly every concept is a leaf with a single incoming edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    Ships a precomputed transitive closure (``closure_url``) consumed by
    :func:`pykeen.datasets.metadata.load_closure_pool`, so hierarchy-closure splits skip the
    transitive-reduction computation; see :class:`~pykeen.datasets.ancestor_descendant.NASATransitive0Percent`
    for a frozen ancestor-descendant split that already bakes in the transitive edges.

    ---
    name: NASA
    citation:
        author: NASA
        year: 2024
        link: https://www.nasa.gov/technology/technology-taxonomy/
    statistics:
        entities: 495
        relations: 1
        training: 478
        testing: 478
        validation: 478
        triples: 478
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    closure_url = _BASE_URL + "closure.tsv"
    ratios = None
    hierarchical_relation = "has_subclass"
