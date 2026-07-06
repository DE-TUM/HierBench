"""The Cora citation network dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "Cora",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiRjW4AR7yjLBjut8onBg5/cora/"


@parse_docdata
class Cora(SingleFileRemoteMetadataDataset):
    """The Cora citation network dataset.

    Nodes are scientific publications classified into one of seven subject
    areas; edges represent citation links between them. Each node has a
    1433-dimensional bag-of-words feature vector.

    Source: https://graphsandnetworks.com/the-cora-dataset/

    ---
    name: Cora
    citation:
        author: McCallum
        year: 2000
        link: https://link.springer.com/article/10.1023/A:1007379606734
    statistics:
        entities: 2708
        relations: 1
        training: 4343
        testing: 543
        validation: 543
        triples: 5429
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
