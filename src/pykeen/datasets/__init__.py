"""Built-in datasets for PyKEEN.

New datasets (inheriting from :class:`pykeen.datasets.Dataset`) can be registered with PyKEEN using the
:mod:`pykeen.datasets` group in Python entrypoints in your own `setup.py`, `setup.cfg`, `pyproject.toml`, or other
package configuration. They are loaded automatically with :func:`importlib.metadata.entry_points` via
:mod:`class_resolver`.
"""

import logging

from class_resolver import ClassResolver

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
from .freebase import FB15k, FB15k237
from .globi import Globi
from .hetionet import Hetionet
from .kinships import Kinships
from .metadata import MetadataDataset, RemoteMetadataDataset
from .nations import Nations
from .numeric import (
    NumericPathDataset,
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
from .wordnet_metadata import WordNet
from .yago import YAGO310

__all__ = [
    # Utilities
    "dataset_resolver",
    "get_dataset",
    "has_dataset",
    # Base Classes
    "Dataset",
    # Concrete Classes
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
    "WordNet",
    "YAGO310",
    "DRKG",
    "BioKG",
    "ConceptNet",
    "CKG",
    "CSKG",
    "DBpedia50",
    "DB100K",
    "DOID",
    "OpenEA",
    "Countries",
    "WD50KT",
    "Wikidata5M",
    "PharmKG8k",
    "PharmKG",
    "PrimeKG",
    "MetadataDataset",
    "RemoteMetadataDataset",
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
        TabbedNumericDataset,
        UnpackedRemoteNumericDataset,
        MTransEDataset,
        OGBLoader,
        EADataset,
        MetadataDataset,
        RemoteMetadataDataset,
    },
)
dataset_resolver.register_entrypoint("pykeen.datasets")


def has_dataset(key: str) -> bool:
    """Return if the dataset is registered in PyKEEN."""
    return dataset_resolver.lookup(key) is not None
