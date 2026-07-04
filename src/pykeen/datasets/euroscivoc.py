"""The European Science Vocabulary (EuroSciVoc) taxonomy dataset."""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import SingleFileRemoteMetadataDataset

__all__ = [
    "EuroSciVoc",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiCbKyE5KfNYjeU9YJPyhi/EuroSciVoc/"


@parse_docdata
class EuroSciVoc(SingleFileRemoteMetadataDataset):
    """The European Science Vocabulary (EuroSciVoc) taxonomy.

    Nodes are EuroSciVoc concepts covering all fields of science; edges are
    top-down ``narrower`` / ``hasTopConcept`` relations forming the
    classification hierarchy. Each node carries a human-readable label.

    Source: https://op.europa.eu/en/web/eu-vocabularies/euroscivoc

    Nearly every concept is a leaf with a single incoming edge, so a random
    train/test/validation split cannot cover all entities. Training, testing, and validation
    therefore all point at the full hierarchy; use
    :func:`pykeen.pipeline.hierarchy.hierarchy_completion_split` to build an evaluation split.

    ---
    name: EuroSciVoc
    citation:
        author: Publications Office of the European Union
        year: 2019
        link: https://op.europa.eu/en/web/eu-vocabularies/euroscivoc
    statistics:
        entities: 1065
        relations: 2
        training: 1064
        testing: 1064
        validation: 1064
        triples: 1064
    """

    triples_url = _BASE_URL + "dataset.tsv"
    entity_metadata_url = _BASE_URL + "metadata.tsv"
    ratios = None
