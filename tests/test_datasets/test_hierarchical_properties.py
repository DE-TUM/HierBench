"""Tests for :class:`pykeen.datasets.hierarchy_analysis.GraphAnalysis`."""

from __future__ import annotations

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.base import EagerDataset
from pykeen.datasets.hierarchy_analysis import GraphAnalysis
from pykeen.triples import CoreTriplesFactory


@pytest.fixture(scope="module")
def nations() -> Nations:
    """Fixture returning the Nations dataset."""
    return Nations()


def _chain_dataset() -> EagerDataset:
    """Return a chain graph 0→1→2→3 with a single relation."""
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    return EagerDataset(training=tf, testing=tf)


def _star_dataset() -> EagerDataset:
    """Return a star graph with root 0 → {1, 2, 3} and a single relation."""
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    return EagerDataset(training=tf, testing=tf)


# ---------------------------------------------------------------------------
# GraphAnalysis construction / split handling
# ---------------------------------------------------------------------------


def test_split_train_matches_training(nations: Nations) -> None:
    """The default split analyses the training factory."""
    ha = GraphAnalysis(nations, split="train")
    assert ha.total_edges == nations.training.num_triples


def test_split_full_uses_merged(nations: Nations) -> None:
    """The 'full' split merges all triples factories."""
    ha = GraphAnalysis(nations, split="full")
    assert ha.total_edges == nations.merged().num_triples


def test_split_validation(nations: Nations) -> None:
    """The 'validation' split analyses the validation factory when present."""
    assert nations.validation is not None
    ha = GraphAnalysis(nations, split="validation")
    assert ha.total_edges == nations.validation.num_triples


def test_split_invalid(nations: Nations) -> None:
    """An unknown split raises ValueError."""
    with pytest.raises(ValueError):
        GraphAnalysis(nations, split="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Task 1: node_count, edge_count, root_nodes, leaf_nodes
# ---------------------------------------------------------------------------


def test_root_nodes_type(nations: Nations) -> None:
    """root_nodes returns a frozenset."""
    assert isinstance(GraphAnalysis(nations).root_nodes, frozenset)


def test_root_nodes_have_no_parents(nations: Nations) -> None:
    """Every root node never appears as a tail."""
    tails = set(nations.training.mapped_triples[:, 2].tolist())
    for r in GraphAnalysis(nations).root_nodes:
        assert r not in tails


def test_leaf_nodes_type(nations: Nations) -> None:
    """leaf_nodes returns a frozenset."""
    assert isinstance(GraphAnalysis(nations).leaf_nodes, frozenset)


def test_leaf_nodes_have_no_children(nations: Nations) -> None:
    """Every leaf node never appears as a head."""
    heads = set(nations.training.mapped_triples[:, 0].tolist())
    for leaf in GraphAnalysis(nations).leaf_nodes:
        assert leaf not in heads


def test_root_and_leaf_chain() -> None:
    """Chain 0→1→2→3: only node 0 is root, only node 3 is leaf."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.root_nodes == frozenset({0})
    assert ha.leaf_nodes == frozenset({3})


def test_root_and_leaf_star() -> None:
    """Star 0→{1,2,3}: root is 0, leaves are {1,2,3}."""
    ha = GraphAnalysis(_star_dataset())
    assert ha.root_nodes == frozenset({0})
    assert ha.leaf_nodes == frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# Task 2: is_dag
# ---------------------------------------------------------------------------


def test_is_dag_returns_bool(nations: Nations) -> None:
    """is_dag returns a bool."""
    assert isinstance(GraphAnalysis(nations).is_dag, bool)


def test_is_dag_acyclic() -> None:
    """An acyclic chain is a DAG."""
    assert GraphAnalysis(_chain_dataset()).is_dag is True


def test_is_dag_cyclic() -> None:
    """A cycle 0→1→2→0 is not a DAG."""
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 0]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert GraphAnalysis(ds).is_dag is False


# ---------------------------------------------------------------------------
# Task 3: max_hierarchy_depth, avg_hierarchy_depth
# ---------------------------------------------------------------------------


def test_max_hierarchy_depth_int(nations: Nations) -> None:
    """max_hierarchy_depth is a non-negative int."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.max_hierarchy_depth, int)
    assert ha.max_hierarchy_depth >= 0


def test_avg_hierarchy_depth_float(nations: Nations) -> None:
    """avg_hierarchy_depth is a non-negative float."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.avg_hierarchy_depth, float)
    assert ha.avg_hierarchy_depth >= 0.0


def test_max_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3 has max depth 3."""
    assert GraphAnalysis(_chain_dataset()).max_hierarchy_depth == 3


def test_avg_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3: depths are 0,1,2,3 so mean is 1.5."""
    assert GraphAnalysis(_chain_dataset()).avg_hierarchy_depth == pytest.approx(1.5)


def test_max_hierarchy_depth_star() -> None:
    """Star 0→{1,2,3}: max depth is 1."""
    assert GraphAnalysis(_star_dataset()).max_hierarchy_depth == 1


def test_levels_alias_chain() -> None:
    """levels is an alias for max_hierarchy_depth."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.levels == ha.max_hierarchy_depth == 3


# ---------------------------------------------------------------------------
# Task 4: average_fan_out, max_fan_out, balance
# ---------------------------------------------------------------------------


def test_average_fan_out_float(nations: Nations) -> None:
    """average_fan_out is a non-negative float."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.average_fan_out, float)
    assert ha.average_fan_out >= 0.0


def test_max_fan_out_int(nations: Nations) -> None:
    """max_fan_out is a non-negative int."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.max_fan_out, int)
    assert ha.max_fan_out >= 0


def test_fan_out_star() -> None:
    """Star 0→{1,2,3}: max_fan_out=3, average_fan_out=0.75."""
    ha = GraphAnalysis(_star_dataset())
    assert ha.max_fan_out == 3
    assert ha.average_fan_out == pytest.approx(0.75)


def test_balance_range(nations: Nations) -> None:
    """Balance is a float in [0, 1]."""
    b = GraphAnalysis(nations).balance
    assert isinstance(b, float)
    assert 0.0 <= b <= 1.0


def test_balance_in_range_star() -> None:
    """Balance of a star graph is in [0, 1]."""
    assert 0.0 <= GraphAnalysis(_star_dataset()).balance <= 1.0


def test_balance_uniform_depths() -> None:
    """Graph with no edges: all depths are 0, so balance is 1.0."""
    triples = torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert GraphAnalysis(ds).balance == 1.0


# ---------------------------------------------------------------------------
# Task 5: get_ancestors, get_descendants, nearest_common_ancestor
# ---------------------------------------------------------------------------


def test_ancestors_type(nations: Nations) -> None:
    """get_ancestors returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 2].item())
    assert isinstance(GraphAnalysis(nations).get_ancestors(node), frozenset)


def test_descendants_type(nations: Nations) -> None:
    """get_descendants returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 0].item())
    assert isinstance(GraphAnalysis(nations).get_descendants(node), frozenset)


def test_ancestors_chain() -> None:
    """In chain 0→1→2→3, ancestors of 3 are {0,1,2} and root 0 has none."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.get_ancestors(3) == frozenset({0, 1, 2})
    assert ha.get_ancestors(0) == frozenset()


def test_descendants_chain() -> None:
    """In chain 0→1→2→3, descendants of 0 are {1,2,3} and leaf 3 has none."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.get_descendants(0) == frozenset({1, 2, 3})
    assert ha.get_descendants(3) == frozenset()


def test_nca_chain() -> None:
    """NCA in a chain: NCA(1,3)=1, NCA(2,3)=2, NCA(0,3)=0."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.nearest_common_ancestor(1, 3) == 1
    assert ha.nearest_common_ancestor(2, 3) == 2
    assert ha.nearest_common_ancestor(0, 3) == 0


def test_nca_same_node() -> None:
    """NCA of a node with itself is the node."""
    assert GraphAnalysis(_chain_dataset()).nearest_common_ancestor(2, 2) == 2


def test_nca_no_common_ancestor() -> None:
    """Two disconnected components have no common ancestor."""
    triples = torch.tensor([[0, 0, 1], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert GraphAnalysis(ds).nearest_common_ancestor(1, 3) is None


def test_nca_dag_with_diamond() -> None:
    """Diamond DAG (0→1,0→2,1→3,2→3): NCA(1,2)=0, NCA(1,3)=1."""
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 3], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    ha = GraphAnalysis(ds)
    assert ha.nearest_common_ancestor(1, 2) == 0
    assert ha.nearest_common_ancestor(1, 3) == 1


# ---------------------------------------------------------------------------
# Helpers for paper metric tests (Zloch et al. 2019 §3.2)
# ---------------------------------------------------------------------------


def _make_dataset(triples: list[list[int]], num_entities: int, num_relations: int = 1) -> EagerDataset:
    """Return an EagerDataset from a raw triple list."""
    t = torch.tensor(triples, dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=t, num_entities=num_entities, num_relations=num_relations)
    return EagerDataset(training=tf, testing=tf)


def _reciprocal_dataset() -> EagerDataset:
    """Return a fully bidirectional graph: 0→1 and 1→0."""
    return _make_dataset([[0, 0, 1], [1, 0, 0]], num_entities=2)


def _mixed_reciprocal_dataset() -> EagerDataset:
    """Return a partially bidirectional graph: 0→1, 1→0, 0→2 (2 of 3 total edges have a reverse)."""
    return _make_dataset([[0, 0, 1], [1, 0, 0], [0, 0, 2]], num_entities=3)


def _parallel_edge_dataset() -> EagerDataset:
    """Return a graph with parallel edges: 0→1 via relation 0 and relation 1."""
    return _make_dataset([[0, 0, 1], [0, 1, 1]], num_entities=2, num_relations=2)


def _high_h_index_dataset() -> EagerDataset:
    """Return a graph where entities 3, 4, 5 each have in-degree 3 → h_index = 3."""
    triples = [[0, 0, 3], [1, 0, 3], [2, 0, 3], [0, 0, 4], [1, 0, 4], [2, 0, 4], [0, 0, 5], [1, 0, 5], [2, 0, 5]]
    return _make_dataset(triples, num_entities=6)


def _empty_dataset() -> EagerDataset:
    """Return a dataset with no edges (3 isolated entities)."""
    t = torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=t, num_entities=3, num_relations=1)
    return EagerDataset(training=tf, testing=tf)


def _uniform_degree_dataset() -> EagerDataset:
    """Return a graph where every entity has in-degree 1 and out-degree 1: 0→2, 1→3, 2→0, 3→1."""
    return _make_dataset([[0, 0, 2], [1, 0, 3], [2, 0, 0], [3, 0, 1]], num_entities=4)


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------


def test_density_float(nations: Nations) -> None:
    """Density is a non-negative float."""
    d = GraphAnalysis(nations).density
    assert isinstance(d, float)
    assert d >= 0.0


def test_density_star() -> None:
    """Star with 3 triples and 4 entities has density 3/16."""
    assert GraphAnalysis(_star_dataset()).density == pytest.approx(3 / 16)


def test_density_empty() -> None:
    """Empty graph has density 0.0."""
    assert GraphAnalysis(_empty_dataset()).density == 0.0


# ---------------------------------------------------------------------------
# unique_edge_density
# ---------------------------------------------------------------------------


def test_unique_edge_density_float(nations: Nations) -> None:
    """Unique_edge_density is a non-negative float no greater than density."""
    ha = GraphAnalysis(nations)
    ued = ha.unique_edge_density
    assert isinstance(ued, float)
    assert 0.0 <= ued <= ha.density


def test_unique_edge_density_star() -> None:
    """Star has no parallel edges, so unique_edge_density equals density."""
    ha = GraphAnalysis(_star_dataset())
    assert ha.unique_edge_density == pytest.approx(ha.density)


def test_unique_edge_density_parallel_edges() -> None:
    """Parallel-edge graph has strictly smaller unique_edge_density than density."""
    ha = GraphAnalysis(_parallel_edge_dataset())
    assert ha.unique_edge_density < ha.density


def test_unique_edge_density_empty() -> None:
    """Empty graph has unique_edge_density 0.0."""
    assert GraphAnalysis(_empty_dataset()).unique_edge_density == 0.0


# ---------------------------------------------------------------------------
# reciprocity
# ---------------------------------------------------------------------------


def test_reciprocity_range(nations: Nations) -> None:
    """Reciprocity is a float in [0, 1]."""
    r = GraphAnalysis(nations).reciprocity
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


def test_reciprocity_chain() -> None:
    """Chain 0→1→2→3 has no back-edges, so reciprocity is 0.0."""
    assert GraphAnalysis(_chain_dataset()).reciprocity == pytest.approx(0.0)


def test_reciprocity_fully_bidirectional() -> None:
    """Fully bidirectional graph 0↔1 has reciprocity 1.0."""
    assert GraphAnalysis(_reciprocal_dataset()).reciprocity == pytest.approx(1.0)


def test_reciprocity_mixed() -> None:
    """Graph with 2 of 3 total edges having a reverse has reciprocity 2/3."""
    assert GraphAnalysis(_mixed_reciprocal_dataset()).reciprocity == pytest.approx(2 / 3)


def test_reciprocity_parallel_edges() -> None:
    """Parallel-edge graph: 3 edges all have a reverse, so reciprocity is 1.0."""
    ds = _make_dataset([[0, 0, 1], [0, 1, 1], [1, 0, 0]], num_entities=2, num_relations=2)
    assert GraphAnalysis(ds).reciprocity == pytest.approx(1.0)


def test_reciprocity_parallel_no_reverse() -> None:
    """Parallel edges with no reverse: 2 edges, m_bi=0, reciprocity is 0.0."""
    ds = _make_dataset([[0, 0, 1], [0, 1, 1]], num_entities=2, num_relations=2)
    assert GraphAnalysis(ds).reciprocity == pytest.approx(0.0)


def test_reciprocity_empty() -> None:
    """Empty graph has reciprocity 0.0."""
    assert GraphAnalysis(_empty_dataset()).reciprocity == 0.0


# ---------------------------------------------------------------------------
# h_index
# ---------------------------------------------------------------------------


def test_h_index_int(nations: Nations) -> None:
    """H_index is a non-negative int."""
    h = GraphAnalysis(nations).h_index
    assert isinstance(h, int)
    assert h >= 0


def test_h_index_chain() -> None:
    """Chain in-degrees [0,1,1,1]: exactly 1 entity with in-degree ≥ 1 → h_index = 1."""
    assert GraphAnalysis(_chain_dataset()).h_index == 1


def test_h_index_high() -> None:
    """Graph with 3 entities of in-degree 3 each has h_index = 3."""
    assert GraphAnalysis(_high_h_index_dataset()).h_index == 3


def test_h_index_empty() -> None:
    """Empty graph has h_index 0."""
    assert GraphAnalysis(_empty_dataset()).h_index == 0


# ---------------------------------------------------------------------------
# degree_variance_in / degree_variance_out
# ---------------------------------------------------------------------------


def test_degree_variance_in_float(nations: Nations) -> None:
    """Degree_variance_in is a non-negative float."""
    v = GraphAnalysis(nations).degree_variance_in
    assert isinstance(v, float)
    assert v >= 0.0


def test_degree_variance_out_float(nations: Nations) -> None:
    """Degree_variance_out is a non-negative float."""
    v = GraphAnalysis(nations).degree_variance_out
    assert isinstance(v, float)
    assert v >= 0.0


def test_degree_variance_chain() -> None:
    """Chain in/out-degrees both [0,1,1,1]/[1,1,1,0]: population variance is 0.1875."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.degree_variance_in == pytest.approx(0.1875)
    assert ha.degree_variance_out == pytest.approx(0.1875)


def test_degree_variance_empty() -> None:
    """Empty graph has degree variances of 0.0."""
    ha = GraphAnalysis(_empty_dataset())
    assert ha.degree_variance_in == 0.0
    assert ha.degree_variance_out == 0.0


# ---------------------------------------------------------------------------
# degree_std_in / degree_std_out
# ---------------------------------------------------------------------------


def test_degree_std_in_float(nations: Nations) -> None:
    """Degree_std_in is a non-negative float equal to sqrt(degree_variance_in)."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.degree_std_in, float)
    assert ha.degree_std_in >= 0.0
    assert ha.degree_std_in == pytest.approx(ha.degree_variance_in**0.5)


def test_degree_std_out_float(nations: Nations) -> None:
    """Degree_std_out is a non-negative float equal to sqrt(degree_variance_out)."""
    ha = GraphAnalysis(nations)
    assert isinstance(ha.degree_std_out, float)
    assert ha.degree_std_out >= 0.0
    assert ha.degree_std_out == pytest.approx(ha.degree_variance_out**0.5)


def test_degree_std_chain() -> None:
    """Chain degree std is sqrt(0.1875) for both in and out."""
    ha = GraphAnalysis(_chain_dataset())
    assert ha.degree_std_in == pytest.approx(0.1875**0.5)
    assert ha.degree_std_out == pytest.approx(0.1875**0.5)


# ---------------------------------------------------------------------------
# coefficient_of_variation_in / coefficient_of_variation_out
# ---------------------------------------------------------------------------


def test_cv_in_float(nations: Nations) -> None:
    """Coefficient_of_variation_in is a non-negative float."""
    cv = GraphAnalysis(nations).coefficient_of_variation_in
    assert isinstance(cv, float)
    assert cv >= 0.0


def test_cv_out_float(nations: Nations) -> None:
    """Coefficient_of_variation_out is a non-negative float."""
    cv = GraphAnalysis(nations).coefficient_of_variation_out
    assert isinstance(cv, float)
    assert cv >= 0.0


def test_cv_uniform_degree() -> None:
    """Graph where every entity has in- and out-degree 1 has cv_in = cv_out = 0.0."""
    ha = GraphAnalysis(_uniform_degree_dataset())
    assert ha.coefficient_of_variation_in == pytest.approx(0.0)
    assert ha.coefficient_of_variation_out == pytest.approx(0.0)


def test_cv_empty() -> None:
    """Empty graph has cv_in = cv_out = 0.0."""
    ha = GraphAnalysis(_empty_dataset())
    assert ha.coefficient_of_variation_in == 0.0
    assert ha.coefficient_of_variation_out == 0.0
