"""Built-in datasets for PyKEEN.

New datasets (inheriting from :class:`pykeen.datasets.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py`, `setup.cfg`, `pyproject.toml`, or other
package configuration. They are loaded automatically with :func:`importlib.metadata.entry_points` via
:mod:`class_resolver`.
"""

import logging

from class_resolver import ClassResolver

from .acm_ccs import ACMCCS
from .aristo import AristoV4
from .base import (  # noqa:F401
    CompressedSingleDataset,
    Dataset,
    EagerDataset,
    LazyDataset,
    PackedZipRemoteDataset,
    PathDataset,
    RemoteDataset,
    SingleTabbedDataset,
    TabbedDataset,
    TarFileRemoteDataset,
    TarFileSingleDataset,
    UnpackedRemoteDataset,
    ZipSingleDataset,
)
from .biokg import BioKG
from .cameleon import Chameleon
from .citeseer import CiteSeer
from .ckg import CKG
from .codex import CoDExLarge, CoDExMedium, CoDExSmall
from .conceptnet import ConceptNet
from .cora import Cora
from .cora_original import CoraOriginal
from .countries import Countries
from .crocodile import Crocodile
from .cskg import CSKG
from .db100k import DB100K
from .dbpedia import DBpedia50
from .doid import DOID
from .drkg import DRKG
from .ea import CN3l, EADataset, MTransEDataset, OpenEA, WK3l15k, WK3l120k
from .estat_fss import EstatFSS
from .euroscivoc import EuroSciVoc
from .freebase import FB15k, FB15k237
from .globi import Globi
from .hetionet import Hetionet
from .kinships import Kinships
from .mesh import MeSH
from .metadata import HierarchicalGraph, MetadataDataset, RemoteMetadataDataset, SingleFileRemoteMetadataDataset
from .nasa import NASA
from .nations import Nations
from .numeric import (
    NumericPathDataset,
    SingleRemoteNumericDataset,
    TabbedNumericDataset,
    UnpackedRemoteNumericDataset,
)
from .ogb import OGBBioKG, OGBLoader, OGBWikiKG2
from .openbiolink import OpenBioLink, OpenBioLinkLQ
from .pharmebinet import PharMeBINet
from .pharmkg import PharmKG, PharmKG8k
from .ppi import PPI
from .primekg import PrimeKG
from .pubmed import PubMed
from .squirrel import Squirrel
from .umls import UMLS
from .utils import get_dataset
from .wd50k import WD50KT
from .wikidata5m import Wikidata5M
from .wordnet import WN18, WN18RR
from .wordnet_noun import (
    WordNetNoun0Percent,
    WordNetNoun10Percent,
    WordNetNoun25Percent,
    WordNetNoun50Percent,
    WordNetNoun90Percent,
)
from .yago import YAGO310

__all__ = [
    # Utilities
    "dataset_resolver",
    "get_dataset",
    "has_dataset",
    # Base Classes
    "Dataset",
    # Concrete Classes
    "ACMCCS",
    "AristoV4",
    "Hetionet",
    "Kinships",
    "Nations",
    "OpenBioLink",
    "OpenBioLinkLQ",
    "CoDExSmall",
    "CoDExMedium",
    "CoDExLarge",
    "CN3l",
    "OGBBioKG",
    "OGBWikiKG2",
    "UMLS",
    "FB15k",
    "FB15k237",
    "WK3l15k",
    "WK3l120k",
    "WN18",
    "WN18RR",
    "WordNetNoun0Percent",
    "WordNetNoun10Percent",
    "WordNetNoun25Percent",
    "WordNetNoun50Percent",
    "WordNetNoun90Percent",
    "YAGO310",
    "DRKG",
    "BioKG",
    "ConceptNet",
    "CKG",
    "CSKG",
    "DBpedia50",
    "DB100K",
    "DOID",
    "EuroSciVoc",
    "OpenEA",
    "Countries",
    "WD50KT",
    "Wikidata5M",
    "PharmKG8k",
    "PharmKG",
    "PrimeKG",
    "HierarchicalGraph",
    "MetadataDataset",
    "RemoteMetadataDataset",
    "SingleFileRemoteMetadataDataset",
    "UnpackedRemoteNumericDataset",
    "Chameleon",
    "CiteSeer",
    "Cora",
    "CoraOriginal",
    "Crocodile",
    "PPI",
    "PubMed",
    "Squirrel",
    "Globi",
    "PharMeBINet",
    "NASA",
    "MeSH",
    "EstatFSS",
]

logger = logging.getLogger(__name__)

#: A resolver for datasets
dataset_resolver: ClassResolver[Dataset] = ClassResolver.from_subclasses(
    base=Dataset,
    skip={
        EagerDataset,
        LazyDataset,
        PathDataset,
        RemoteDataset,
        UnpackedRemoteDataset,
        TarFileRemoteDataset,
        PackedZipRemoteDataset,
        CompressedSingleDataset,
        TarFileSingleDataset,
        ZipSingleDataset,
        TabbedDataset,
        SingleTabbedDataset,
        NumericPathDataset,
        SingleRemoteNumericDataset,
        TabbedNumericDataset,
        UnpackedRemoteNumericDataset,
        MTransEDataset,
        OGBLoader,
        EADataset,
        MetadataDataset,
        RemoteMetadataDataset,
        SingleFileRemoteMetadataDataset,
    },
)
dataset_resolver.register_entrypoint("pykeen.datasets")


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return dataset_resolver.lookup(key) is not None
