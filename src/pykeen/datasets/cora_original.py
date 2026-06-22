"""The Cora (McCallum) original dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "CoraOriginal",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiQHnjbWoTtkw1D9mud59W/cora_mccallum/"


@parse_docdata
class CoraOriginal(SingleFileRemoteMetadataDataset):
    """The original Cora research-paper classification dataset (McCallum et al., 2000).

    Research papers classified into a topic hierarchy with 73 leaf categories;
    citations provide relational edges between papers. Each paper node carries
    extracted feature metadata.

    Source: https://people.cs.umass.edu/~mccallum/data.html

    ---
    name: Cora (Original)
    citation:
        author: McCallum
        year: 2000
        link: https://people.cs.umass.edu/~mccallum/data.html
    statistics:
        entities: 225026
        relations: 1
        training: 571412
        testing: 71427
        validation: 71427
        triples: 714266
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
