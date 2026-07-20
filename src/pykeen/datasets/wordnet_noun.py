"""WordNet noun-hypernymy taxonomy datasets (hyperbolic-cones ``maxn`` splits).

These are the WordNet noun hypernymy hierarchies from the ``maxn`` benchmark of
Ganea et al. (`hyperbolic_cones <https://github.com/dalab/hyperbolic_cones>`_).
Edges are stored top-down as ``<parent> hyponym <child>``. Every variant shares the
same fixed test / validation splits and differs only in how much of the transitive
closure is added to the training basic (direct-hypernym) edges.
"""

from __future__ import annotations

from docdata import parse_docdata

from .metadata import HierarchicalGraph, RemoteMetadataDataset

__all__ = [
    "WordNetNoun0Percent",
    "WordNetNoun10Percent",
    "WordNetNoun25Percent",
    "WordNetNoun50Percent",
    "WordNetNoun90Percent",
]

_BASE_URL = "https://syncandshare.lrz.de/dl/fiLx8c9PzQoaLR4LjvZjyg/"


@parse_docdata
class WordNetNoun0Percent(RemoteMetadataDataset, HierarchicalGraph):
    """WordNet noun hypernymy with basic (direct-hypernym) training edges only.

    Nodes are WordNet noun synsets; edges are top-down ``hyponym`` relations.
    Training contains only the basic direct-hypernym edges (no transitive closure);
    testing and validation are the fixed held-out transitive edges.

    Source: https://github.com/dalab/hyperbolic_cones (``data/maxn``)

    ---
    name: WordNet-Noun-0percent
    citation:
        author: Ganea
        year: 2018
        link: https://arxiv.org/abs/1804.01882
    statistics:
        entities: 82114
        relations: 1
        training: 84363
        testing: 28838
        validation: 28838
    """

    training_url = _BASE_URL + "wordnet-noun-0percent/train.tsv"
    testing_url = _BASE_URL + "wordnet-noun-0percent/test.tsv"
    validation_url = _BASE_URL + "wordnet-noun-0percent/valid.tsv"
    hierarchical_relation = "hyponym"
    predefined_closure_split = True


@parse_docdata
class WordNetNoun10Percent(RemoteMetadataDataset, HierarchicalGraph):
    """WordNet noun hypernymy: basic edges + 10% of the transitive closure in training.

    Nodes are WordNet noun synsets; edges are top-down ``hyponym`` relations.
    Testing and validation are the fixed held-out transitive edges.

    Source: https://github.com/dalab/hyperbolic_cones (``data/maxn``)

    ---
    name: WordNet-Noun-10percent
    citation:
        author: Ganea
        year: 2018
        link: https://arxiv.org/abs/1804.01882
    statistics:
        entities: 82114
        relations: 1
        training: 142039
        testing: 28838
        validation: 28838
    """

    training_url = _BASE_URL + "wordnet-noun-10percent/train.tsv"
    testing_url = _BASE_URL + "wordnet-noun-10percent/test.tsv"
    validation_url = _BASE_URL + "wordnet-noun-10percent/valid.tsv"
    hierarchical_relation = "hyponym"
    predefined_closure_split = True


@parse_docdata
class WordNetNoun25Percent(RemoteMetadataDataset, HierarchicalGraph):
    """WordNet noun hypernymy: basic edges + 25% of the transitive closure in training.

    Nodes are WordNet noun synsets; edges are top-down ``hyponym`` relations.
    Testing and validation are the fixed held-out transitive edges.

    Source: https://github.com/dalab/hyperbolic_cones (``data/maxn``)

    ---
    name: WordNet-Noun-25percent
    citation:
        author: Ganea
        year: 2018
        link: https://arxiv.org/abs/1804.01882
    statistics:
        entities: 82114
        relations: 1
        training: 228554
        testing: 28838
        validation: 28838
    """

    training_url = _BASE_URL + "wordnet-noun-25percent/train.tsv"
    testing_url = _BASE_URL + "wordnet-noun-25percent/test.tsv"
    validation_url = _BASE_URL + "wordnet-noun-25percent/valid.tsv"
    hierarchical_relation = "hyponym"
    predefined_closure_split = True


@parse_docdata
class WordNetNoun50Percent(RemoteMetadataDataset, HierarchicalGraph):
    """WordNet noun hypernymy: basic edges + 50% of the transitive closure in training.

    Nodes are WordNet noun synsets; edges are top-down ``hyponym`` relations.
    Testing and validation are the fixed held-out transitive edges.

    Source: https://github.com/dalab/hyperbolic_cones (``data/maxn``)

    ---
    name: WordNet-Noun-50percent
    citation:
        author: Ganea
        year: 2018
        link: https://arxiv.org/abs/1804.01882
    statistics:
        entities: 82114
        relations: 1
        training: 372745
        testing: 28838
        validation: 28838
    """

    training_url = _BASE_URL + "wordnet-noun-50percent/train.tsv"
    testing_url = _BASE_URL + "wordnet-noun-50percent/test.tsv"
    validation_url = _BASE_URL + "wordnet-noun-50percent/valid.tsv"
    hierarchical_relation = "hyponym"
    predefined_closure_split = True


@parse_docdata
class WordNetNoun90Percent(RemoteMetadataDataset, HierarchicalGraph):
    """WordNet noun hypernymy: basic edges + 90% of the transitive closure in training.

    Nodes are WordNet noun synsets; edges are top-down ``hyponym`` relations.
    Testing and validation are the fixed held-out transitive edges.

    Source: https://github.com/dalab/hyperbolic_cones (``data/maxn``)

    ---
    name: WordNet-Noun-90percent
    citation:
        author: Ganea
        year: 2018
        link: https://arxiv.org/abs/1804.01882
    statistics:
        entities: 82114
        relations: 1
        training: 603450
        testing: 28838
        validation: 28838
    """

    training_url = _BASE_URL + "wordnet-noun-90percent/train.tsv"
    testing_url = _BASE_URL + "wordnet-noun-90percent/test.tsv"
    validation_url = _BASE_URL + "wordnet-noun-90percent/valid.tsv"
    hierarchical_relation = "hyponym"
    predefined_closure_split = True
