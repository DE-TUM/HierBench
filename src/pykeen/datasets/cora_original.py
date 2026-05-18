"""The Cora (McCallum) original dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "CoraOriginal",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiQHnjbWoTtkw1D9mud59W/cora_mccallum/"


@parse_docdata
class CoraOriginal(RemoteMetadataDataset):
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
        entities: 0
        relations: 0
        training: 0
        testing: 0
        validation: 0
        triples: 0
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
