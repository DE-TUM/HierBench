"""The PPI (Protein-Protein Interaction) dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "PPI",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fi5Uw1bNRWJ4SaSC7NBCo8/PPI/"


@parse_docdata
class PPI(SingleFileRemoteMetadataDataset):
    """The PPI (Protein-Protein Interaction) dataset.

    Nodes are proteins; edges represent protein-protein interactions.
    Each node has a feature vector encoding biological properties.

    Source: https://snap.stanford.edu/graphsage/

    ---
    name: PPI
    citation:
        author: Hamilton
        year: 2017
        link: https://arxiv.org/abs/1706.02216
        github: williamleif/GraphSAGE
    statistics:
        entities: 56944
        relations: 1
        training: 654972
        testing: 81872
        validation: 81872
        triples: 818716
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
