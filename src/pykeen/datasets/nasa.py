"""The NASA taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "NASA",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiHdHacRoQetsxARSBLcak/NASA/"


@parse_docdata
class NASA(SingleFileRemoteMetadataDataset):
    """The 2024 NASA Technology Taxonomy.

    Nodes are NASA technology areas; edges are ``has_subclass`` relations forming the
    classification hierarchy. Each node carries a human-readable title, definition, and
    example technologies.

    Nearly every concept is a leaf with a single incoming edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    ---
    name: NASA
    citation:
        author: NASA
        year: 2024
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
    ratios = None
