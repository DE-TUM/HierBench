"""Predefined multi-hop transitive ancestor-descendant splits of the taxonomy datasets.

These are frozen copies of the split that
:func:`pykeen.pipeline.transitive_ancestor_descendant.transitive_ancestor_descendant_prediction_split`
derives from :class:`~pykeen.datasets.nasa.NASA` & co. at a fixed ``eval_ratio=0.05``, ``seed=42``
and a given ``closure_ratio`` — the setting benchmarked in
``benchmarking/benchmark_transitive_ancestor_descendant.py``. Training holds the basic (direct)
hierarchy edges plus the ``closure_ratio`` fraction of the non-direct transitive closure folded in,
validation and testing hold two disjoint 5% portions of the remaining closure (Ganea et al. 2018 §5;
He et al. 2024 §4.1). The ``<N>Percent`` suffix names that closure fraction (the WordNetNoun*
convention): ``0Percent`` folds none of the closure into training.

Unlike the plain taxonomy classes (:class:`~pykeen.datasets.nasa.NASA` & co.), which hold only the
asserted hierarchy edges, these classes' ``train.tsv``/``valid.tsv``/``test.tsv`` files *already
contain the transitive (non-direct) ancestor-descendant edges* worked into the split. Loading one of
these datasets requires no split derivation at all —
:attr:`~pykeen.datasets.metadata.HierarchicalGraph.predefined_closure_split` tells
:func:`~pykeen.pipeline.transitive_ancestor_descendant.transitive_ancestor_descendant_prediction_split`
to use the shipped train/validation/testing triples verbatim.

Shipping the split as files makes the benchmark reproducible across PyKEEN versions and machines: the
closure-splitting code no longer has to be re-run (and re-agree) to compare numbers. Regenerate with
``scripts/export_transitive_ancestor_descendant_splits.py``.

Adding another closure fraction (e.g. 5 / 10 / 50 percent) is one small subclass per dataset:
derive the split-file URLs with :func:`_split_urls`, point ``entity_metadata_url`` at the shared
``{slug}/metadata.tsv``, and set ``predefined_closure_split = True``. The remote layout is flat — one
folder ``{slug}-ancestor-descendant-closure{percent}/`` per (dataset, fraction), holding just the
three split files.
"""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, RemoteMetadataDataset

__all__ = [
    "ACMCCSTransitive0Percent",
    "DOIDTransitive0Percent",
    "EuroSciVocTransitive0Percent",
    "MeSHTransitive0Percent",
    "NASATransitive0Percent",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/"


def _split_urls(slug: str, closure_percent: int) -> tuple[str, str, str]:
    """Return the ``(training, testing, validation)`` URLs of a predefined closure split.

    Each split lives in its own flat folder ``{slug}-ancestor-descendant-closure{closure_percent}/``
    directly under :data:`_BASE_URL` (no nesting under the source dataset's directory), holding just
    ``train.tsv`` / ``test.tsv`` / ``valid.tsv``. The per-entity metadata is not duplicated here; the
    split classes reuse the source dataset's shared ``{slug}/metadata.tsv``.
    """
    folder = f"{_BASE_URL}{slug}-ancestor-descendant-closure{closure_percent}/"
    return folder + "train.tsv", folder + "test.tsv", folder + "valid.tsv"


@parse_docdata
class NASATransitive0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """The NASA technology taxonomy with a predefined ancestor-descendant split.

    Training holds the direct ``has_subclass`` edges; validation and testing already hold disjoint
    held-out portions of the non-direct transitive closure — the transitive edges are baked into
    the shipped files, no split derivation needed at load time. Frozen copy of the ``seed=42``,
    ``eval_ratio=0.05``, ``closure_ratio=0.0`` split of :class:`~pykeen.datasets.nasa.NASA`.

    ---
    name: NASA-Transitive-0percent
    citation:
        author: NASA
        year: 2024
        link: https://www.nasa.gov/technology/technology-taxonomy/
    statistics:
        entities: 495
        relations: 1
        training: 478
        testing: 19
        validation: 19
        triples: 516
    """

    training_url, testing_url, validation_url = _split_urls("nasa", 0)
    entity_metadata_url = _BASE_URL + "nasa/metadata.tsv"
    hierarchical_relation = "has_subclass"
    predefined_closure_split = True


@parse_docdata
class DOIDTransitive0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """The Disease Ontology with a predefined ancestor-descendant split.

    Training holds the direct ``has_subclass`` edges; validation and testing already hold disjoint
    held-out portions of the non-direct transitive closure — the transitive edges are baked into
    the shipped files, no split derivation needed at load time. Frozen copy of the ``seed=42``,
    ``eval_ratio=0.05``, ``closure_ratio=0.0`` split of :class:`~pykeen.datasets.doid.DOID`.

    ---
    name: DOID-Transitive-0percent
    citation:
        author: Schriml
        year:
        link:
        github: DiseaseOntology/HumanDiseaseOntology
    statistics:
        entities: 12190
        relations: 1
        training: 13802
        testing: 1192
        validation: 1192
        triples: 16186
    """

    training_url, testing_url, validation_url = _split_urls("doid", 0)
    entity_metadata_url = _BASE_URL + "doid/metadata.tsv"
    hierarchical_relation = "has_subclass"
    predefined_closure_split = True


@parse_docdata
class ACMCCSTransitive0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """The ACM CCS classification with a predefined ancestor-descendant split.

    Training holds the direct ``narrower`` edges; validation and testing already hold disjoint
    held-out portions of the non-direct transitive closure — the transitive edges are baked into
    the shipped files, no split derivation needed at load time. Frozen copy of the ``seed=42``,
    ``eval_ratio=0.05``, ``closure_ratio=0.0`` split of :class:`~pykeen.datasets.acm_ccs.ACMCCS`.

    ---
    name: ACM-CCS-Transitive-0percent
    citation:
        author: ACM
        year: 2012
        link: https://www.acm.org/publications/class-2012
    statistics:
        entities: 2113
        relations: 1
        training: 2100
        testing: 195
        validation: 195
        triples: 2490
    """

    training_url, testing_url, validation_url = _split_urls("acmccm", 0)
    entity_metadata_url = _BASE_URL + "acmccm/metadata.tsv"
    hierarchical_relation = "narrower"
    predefined_closure_split = True


@parse_docdata
class EuroSciVocTransitive0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """The EuroSciVoc taxonomy with a predefined ancestor-descendant split.

    Training holds the direct ``narrower`` edges; validation and testing already hold disjoint
    held-out portions of the non-direct transitive closure — the transitive edges are baked into
    the shipped files, no split derivation needed at load time. Frozen copy of the ``seed=42``,
    ``eval_ratio=0.05``, ``closure_ratio=0.0`` split of
    :class:`~pykeen.datasets.euroscivoc.EuroSciVoc`.

    ---
    name: EuroSciVoc-Transitive-0percent
    citation:
        author: Publications Office of the European Union
        year: 2019
        link: https://op.europa.eu/en/web/eu-vocabularies/euroscivoc
    statistics:
        entities: 1064
        relations: 1
        training: 1058
        testing: 106
        validation: 106
        triples: 1270
    """

    training_url, testing_url, validation_url = _split_urls("euroscivoc", 0)
    entity_metadata_url = _BASE_URL + "euroscivoc/metadata.tsv"
    hierarchical_relation = "narrower"
    predefined_closure_split = True


@parse_docdata
class MeSHTransitive0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """MeSH with a predefined ancestor-descendant split.

    Training holds the asserted ``narrower`` edges (MeSH's hierarchy is not a DAG, so the transitive
    reduction is undefined and the asserted edges are used as basic edges); validation and testing
    already hold disjoint held-out portions of the non-direct transitive closure — the transitive
    edges are baked into the shipped files, no split derivation needed at load time. Frozen copy of
    the ``seed=42``, ``eval_ratio=0.05``, ``closure_ratio=0.0`` split of
    :class:`~pykeen.datasets.mesh.MeSH`.

    ---
    name: MeSH-Transitive-0percent
    citation:
        author: National Library of Medicine
        year: 2024
        link: https://www.nlm.nih.gov/mesh/meshhome.html
    statistics:
        entities: 30837
        relations: 1
        training: 42114
        testing: 8647
        validation: 8647
        triples: 59408
    """

    training_url, testing_url, validation_url = _split_urls("mesh", 0)
    entity_metadata_url = _BASE_URL + "mesh/metadata.tsv"
    hierarchical_relation = "narrower"
    predefined_closure_split = True
