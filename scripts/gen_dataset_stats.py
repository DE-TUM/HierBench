"""Generate dataset statistics for docdata YAML blocks."""

from pykeen.datasets.cameleon import Chameleon
from pykeen.datasets.citeseer import CiteSeer
from pykeen.datasets.cora import Cora
from pykeen.datasets.cora_original import CoraOriginal
from pykeen.datasets.crocodile import Crocodile
from pykeen.datasets.doid import DOID
from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis
from pykeen.datasets.ppi import PPI
from pykeen.datasets.pubmed import PubMed
from pykeen.datasets.squirrel import Squirrel
from pykeen.datasets.wordnet_metadata import WordNet

for cls in [Chameleon, CiteSeer, Cora, CoraOriginal, Crocodile, DOID, PPI, PubMed, Squirrel, WordNet]:
    print(f"Loading {cls.__name__}...")
    ds = cls()
    ega = ExtendedGraphAnalysis(ds, split="full")
    factory = ega._factory
    train_n = ds.training.num_triples
    test_n = ds.testing.num_triples
    val_n = ds.validation.num_triples
    print(f"# {cls.__name__}")
    print(f"entities: {factory.num_entities}")
    print(f"relations: {factory.num_relations}")
    print(f"training: {train_n}")
    print(f"testing: {test_n}")
    print(f"validation: {val_n}")
    print(f"triples: {train_n + test_n + val_n}")
    print()
