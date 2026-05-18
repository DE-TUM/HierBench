"""The PubMed citation graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "PubMed",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiKQMfkFSLYAXMA4pgsNak/pubmed/"


@parse_docdata
class PubMed(RemoteMetadataDataset):
    """The PubMed citation graph dataset.

    Nodes are PubMed articles about diabetes research; edges represent citation
    links between them. Each node has a TF/IDF weighted bag-of-words feature
    vector and one of 3 class labels (Experimentally induced, Type 1, Type 2).

    Source: https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz

    ---
    name: PubMed
    citation:
        author: Sen
        year: 2008
        link: https://jmlr.csail.mit.edu/papers/v11/sen10a.html
    statistics:
        entities: 19717
        relations: 1
        training: 35461
        testing: 4433
        validation: 4433
        triples: 44327
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
