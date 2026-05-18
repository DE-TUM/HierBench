"""A WordNet dataset with entity metadata."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import RemoteMetadataDataset

__all__ = [
    "WordNet",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiMJE3wfCbXEy6PzjHcHoX/wordnet/"


@parse_docdata
class WordNet(RemoteMetadataDataset):
    """WordNet knowledge graph with per-entity metadata.

    Nodes are WordNet synsets; edges represent lexical/semantic relations
    (e.g. hypernymy, meronymy). Each entity carries structured metadata.

    Source: https://wordnet.princeton.edu/

    ---
    name: WordNet
    citation:
        author: Miller
        year: 1995
        link: https://doi.org/10.1145/219717.219748
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
