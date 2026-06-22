"""The PubMed citation graph dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "PubMed",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiKQMfkFSLYAXMA4pgsNak/pubmed/"


@parse_docdata
class PubMed(SingleFileRemoteMetadataDataset):
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
        link: https://linqs.org/datasets/#pubmed-diabetes
    statistics:
        entities: 19717
        relations: 1
        training: 35470
        testing: 4434
        validation: 4434
        triples: 44338
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
