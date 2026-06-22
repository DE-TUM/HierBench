"""The CiteSeer citation network dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "CiteSeer",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiSVG9qjMcqKgKXuLF7v8v/citeseer/"


@parse_docdata
class CiteSeer(SingleFileRemoteMetadataDataset):
    """The CiteSeer citation network dataset.

    Nodes are scientific publications; edges represent citations between them.
    Each node has a feature vector derived from the paper's bag-of-words
    representation and one of 6 class labels (research areas).

    Source: https://linqs.soe.ucsc.edu/data
    (Sen et al., 2008)

    ---
    name: CiteSeer
    citation:
        author: Sen
        year: 2008
        link: https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz
    statistics:
        entities: 3327
        relations: 1
        training: 3785
        testing: 473
        validation: 474
        triples: 4732
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
