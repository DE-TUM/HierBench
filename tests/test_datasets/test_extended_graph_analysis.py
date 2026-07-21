"""Tests for :class:`pykeen.datasets.extended_graph_analysis.ExtendedGraphAnalysis`."""

from __future__ import annotations

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.base import EagerDataset
from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis
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
# ExtendedGraphAnalysis construction / split handling
# ---------------------------------------------------------------------------


def test_split_train_matches_training(nations: Nations) -> None:
    """The default split analyses the training factory."""
    ha = ExtendedGraphAnalysis(nations, split="train")
    assert ha.total_edges == nations.training.num_triples


def test_split_full_uses_merged(nations: Nations) -> None:
    """The 'full' split merges all triples factories."""
    ha = ExtendedGraphAnalysis(nations, split="full")
    assert ha.total_edges == nations.merged().num_triples


def test_split_invalid(nations: Nations) -> None:
    """An unknown split raises ValueError."""
    with pytest.raises(ValueError, match="split must be one of"):
        ExtendedGraphAnalysis(nations, split="nonsense")  # type: ignore[arg-type]


def test_split_test_raises(nations: Nations) -> None:
    """The 'test' split is no longer supported."""
    with pytest.raises(ValueError, match="split must be one of"):
        ExtendedGraphAnalysis(nations, split="test")  # type: ignore[arg-type]


def test_split_validation_raises(nations: Nations) -> None:
    """The 'validation' split is no longer supported."""
    with pytest.raises(ValueError, match="split must be one of"):
        ExtendedGraphAnalysis(nations, split="validation")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Task 1: node_count, edge_count, root_nodes, leaf_nodes
# ---------------------------------------------------------------------------


def test_root_nodes_type(nations: Nations) -> None:
    """root_nodes returns a frozenset."""
    assert isinstance(ExtendedGraphAnalysis(nations).root_nodes, frozenset)


def test_root_nodes_have_no_parents(nations: Nations) -> None:
    """Every root node never appears as a tail."""
    tails = set(nations.training.mapped_triples[:, 2].tolist())
    for r in ExtendedGraphAnalysis(nations).root_nodes:
        assert r not in tails


def test_leaf_nodes_type(nations: Nations) -> None:
    """leaf_nodes returns a frozenset."""
    assert isinstance(ExtendedGraphAnalysis(nations).leaf_nodes, frozenset)


def test_leaf_nodes_have_no_children(nations: Nations) -> None:
    """Every leaf node never appears as a head."""
    heads = set(nations.training.mapped_triples[:, 0].tolist())
    for leaf in ExtendedGraphAnalysis(nations).leaf_nodes:
        assert leaf not in heads


def test_root_and_leaf_chain() -> None:
    """Chain 0→1→2→3: only node 0 is root, only node 3 is leaf."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.root_nodes == frozenset({0})
    assert ha.leaf_nodes == frozenset({3})


def test_root_and_leaf_star() -> None:
    """Star 0→{1,2,3}: root is 0, leaves are {1,2,3}."""
    ha = ExtendedGraphAnalysis(_star_dataset())
    assert ha.root_nodes == frozenset({0})
    assert ha.leaf_nodes == frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# Task 2: is_dag
# ---------------------------------------------------------------------------


def test_is_dag_returns_bool(nations: Nations) -> None:
    """is_dag returns a bool."""
    assert isinstance(ExtendedGraphAnalysis(nations).is_dag, bool)


def test_is_dag_acyclic() -> None:
    """An acyclic chain is a DAG."""
    assert ExtendedGraphAnalysis(_chain_dataset()).is_dag is True


def test_is_dag_cyclic() -> None:
    """A cycle 0→1→2→0 is not a DAG."""
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 0]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ExtendedGraphAnalysis(ds).is_dag is False


# ---------------------------------------------------------------------------
# Task 3: max_hierarchy_depth, avg_hierarchy_depth
# ---------------------------------------------------------------------------


def test_max_hierarchy_depth_int(nations: Nations) -> None:
    """max_hierarchy_depth is a non-negative int."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.max_hierarchy_depth, int)
    assert ha.max_hierarchy_depth >= 0


def test_avg_hierarchy_depth_float(nations: Nations) -> None:
    """avg_hierarchy_depth is a non-negative float."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.avg_hierarchy_depth, float)
    assert ha.avg_hierarchy_depth >= 0.0


def test_max_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3 has max depth 3."""
    assert ExtendedGraphAnalysis(_chain_dataset()).max_hierarchy_depth == 3


def test_avg_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3: depths are 0,1,2,3 so mean is 1.5."""
    assert ExtendedGraphAnalysis(_chain_dataset()).avg_hierarchy_depth == pytest.approx(1.5)


def test_max_hierarchy_depth_star() -> None:
    """Star 0→{1,2,3}: max depth is 1."""
    assert ExtendedGraphAnalysis(_star_dataset()).max_hierarchy_depth == 1


def test_levels_alias_chain() -> None:
    """Levels is an alias for max_hierarchy_depth."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.levels == ha.max_hierarchy_depth == 3


# ---------------------------------------------------------------------------
# Task 4: avg_fan_out, max_fan_out, leaf_depth_variance
# ---------------------------------------------------------------------------


def test_avg_fan_out_float(nations: Nations) -> None:
    """avg_fan_out is a non-negative float."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.avg_fan_out, float)
    assert ha.avg_fan_out >= 0.0


def test_max_fan_out_int(nations: Nations) -> None:
    """max_fan_out is a non-negative int."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.max_fan_out, int)
    assert ha.max_fan_out >= 0


def test_fan_out_star() -> None:
    """Star 0→{1,2,3}: max_fan_out=3, avg_fan_out=0.75."""
    ha = ExtendedGraphAnalysis(_star_dataset())
    assert ha.max_fan_out == 3
    assert ha.avg_fan_out == pytest.approx(0.75)


def test_leaf_depth_variance_range(nations: Nations) -> None:
    """Leaf-depth variance is a non-negative float."""
    b = ExtendedGraphAnalysis(nations).leaf_depth_variance
    assert isinstance(b, float)
    assert b >= 0.0


def test_leaf_depth_variance_in_range_star() -> None:
    """Star graph: all leaves share depth 1, so variance of leaf depths is 0.0."""
    assert ExtendedGraphAnalysis(_star_dataset()).leaf_depth_variance == 0.0


def test_leaf_depth_variance_uniform_depths() -> None:
    """Graph with no edges: every leaf has depth 0, so variance is 0.0."""
    triples = torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ExtendedGraphAnalysis(ds).leaf_depth_variance == 0.0


def test_balance_range(nations: Nations) -> None:
    """J¹ balance is a float in [0, 1]."""
    b = ExtendedGraphAnalysis(nations).balance
    assert isinstance(b, float)
    assert 0.0 <= b <= 1.0


def test_balance_star() -> None:
    """Star 0→{1,2,3}: fully symmetric, so J¹ balance is 1.0."""
    assert ExtendedGraphAnalysis(_star_dataset()).balance == pytest.approx(1.0)


def test_balance_chain() -> None:
    """Chain 0→1→2→3: fully linear, so J¹ balance is 0.0."""
    assert ExtendedGraphAnalysis(_chain_dataset()).balance == pytest.approx(0.0)


def test_balance_empty() -> None:
    """Graph with no edges has no internal nodes, so J¹ balance is 0.0."""
    triples = torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ExtendedGraphAnalysis(ds).balance == 0.0


def test_balance_symmetric_beats_caterpillar() -> None:
    """A balanced binary tree on 4 leaves scores higher than a caterpillar with the same leaves."""
    # Balanced: 0→{1,2}, 1→{3,4}, 2→{5,6} (leaves 3,4,5,6).
    balanced = _make_dataset([[0, 0, 1], [0, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 5], [2, 0, 6]], num_entities=7)
    # Caterpillar: 0→{1,2}, 2→{3,4}, 4→{5,6} (leaves 1,3,5,6).
    caterpillar = _make_dataset([[0, 0, 1], [0, 0, 2], [2, 0, 3], [2, 0, 4], [4, 0, 5], [4, 0, 6]], num_entities=7)
    assert ExtendedGraphAnalysis(balanced).balance > ExtendedGraphAnalysis(caterpillar).balance


# ---------------------------------------------------------------------------
# Task 5: get_ancestors, get_descendants, nearest_common_ancestor
# ---------------------------------------------------------------------------


def test_ancestors_type(nations: Nations) -> None:
    """get_ancestors returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 2].item())
    assert isinstance(ExtendedGraphAnalysis(nations).get_ancestors(node), frozenset)


def test_descendants_type(nations: Nations) -> None:
    """get_descendants returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 0].item())
    assert isinstance(ExtendedGraphAnalysis(nations).get_descendants(node), frozenset)


def test_ancestors_chain() -> None:
    """In chain 0→1→2→3, ancestors of 3 are {0,1,2} and root 0 has none."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.get_ancestors(3) == frozenset({0, 1, 2})
    assert ha.get_ancestors(0) == frozenset()


def test_descendants_chain() -> None:
    """In chain 0→1→2→3, descendants of 0 are {1,2,3} and leaf 3 has none."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.get_descendants(0) == frozenset({1, 2, 3})
    assert ha.get_descendants(3) == frozenset()


def test_nca_chain() -> None:
    """NCA in a chain: NCA(1,3)=1, NCA(2,3)=2, NCA(0,3)=0."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.nearest_common_ancestor(1, 3) == 1
    assert ha.nearest_common_ancestor(2, 3) == 2
    assert ha.nearest_common_ancestor(0, 3) == 0


def test_nca_same_node() -> None:
    """NCA of a node with itself is the node."""
    assert ExtendedGraphAnalysis(_chain_dataset()).nearest_common_ancestor(2, 2) == 2


def test_nca_no_common_ancestor() -> None:
    """Two disconnected components have no common ancestor."""
    triples = torch.tensor([[0, 0, 1], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ExtendedGraphAnalysis(ds).nearest_common_ancestor(1, 3) is None


def test_nca_dag_with_diamond() -> None:
    """Diamond DAG (0→1,0→2,1→3,2→3): NCA(1,2)=0, NCA(1,3)=1."""
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 3], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    ha = ExtendedGraphAnalysis(ds)
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
    d = ExtendedGraphAnalysis(nations).density
    assert isinstance(d, float)
    assert d >= 0.0


def test_density_star() -> None:
    """Star with 3 triples and 4 entities has density 3/16."""
    assert ExtendedGraphAnalysis(_star_dataset()).density == pytest.approx(3 / 16)


def test_density_empty() -> None:
    """Empty graph has density 0.0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).density == 0.0


# ---------------------------------------------------------------------------
# unique_edge_density
# ---------------------------------------------------------------------------


def test_unique_edge_density_float(nations: Nations) -> None:
    """Unique_edge_density is a non-negative float no greater than density."""
    ha = ExtendedGraphAnalysis(nations)
    ued = ha.unique_edge_density
    assert isinstance(ued, float)
    assert 0.0 <= ued <= ha.density


def test_unique_edge_density_star() -> None:
    """Star has no parallel edges, so unique_edge_density equals density."""
    ha = ExtendedGraphAnalysis(_star_dataset())
    assert ha.unique_edge_density == pytest.approx(ha.density)


def test_unique_edge_density_parallel_edges() -> None:
    """Parallel-edge graph has strictly smaller unique_edge_density than density."""
    ha = ExtendedGraphAnalysis(_parallel_edge_dataset())
    assert ha.unique_edge_density < ha.density


def test_unique_edge_density_empty() -> None:
    """Empty graph has unique_edge_density 0.0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).unique_edge_density == 0.0


# ---------------------------------------------------------------------------
# reciprocity
# ---------------------------------------------------------------------------


def test_reciprocity_range(nations: Nations) -> None:
    """Reciprocity is a float in [0, 1]."""
    r = ExtendedGraphAnalysis(nations).reciprocity
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


def test_reciprocity_chain() -> None:
    """Chain 0→1→2→3 has no back-edges, so reciprocity is 0.0."""
    assert ExtendedGraphAnalysis(_chain_dataset()).reciprocity == pytest.approx(0.0)


def test_reciprocity_fully_bidirectional() -> None:
    """Fully bidirectional graph 0↔1 has reciprocity 1.0."""
    assert ExtendedGraphAnalysis(_reciprocal_dataset()).reciprocity == pytest.approx(1.0)


def test_reciprocity_mixed() -> None:
    """Graph with 2 of 3 total edges having a reverse has reciprocity 2/3."""
    assert ExtendedGraphAnalysis(_mixed_reciprocal_dataset()).reciprocity == pytest.approx(2 / 3)


def test_reciprocity_parallel_edges() -> None:
    """Parallel-edge graph: 3 edges all have a reverse, so reciprocity is 1.0."""
    ds = _make_dataset([[0, 0, 1], [0, 1, 1], [1, 0, 0]], num_entities=2, num_relations=2)
    assert ExtendedGraphAnalysis(ds).reciprocity == pytest.approx(1.0)


def test_reciprocity_parallel_no_reverse() -> None:
    """Parallel edges with no reverse: 2 edges, m_bi=0, reciprocity is 0.0."""
    ds = _make_dataset([[0, 0, 1], [0, 1, 1]], num_entities=2, num_relations=2)
    assert ExtendedGraphAnalysis(ds).reciprocity == pytest.approx(0.0)


def test_reciprocity_empty() -> None:
    """Empty graph has reciprocity 0.0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).reciprocity == 0.0


# ---------------------------------------------------------------------------
# h_index
# ---------------------------------------------------------------------------


def test_h_index_int(nations: Nations) -> None:
    """H_index is a non-negative int."""
    h = ExtendedGraphAnalysis(nations).h_index
    assert isinstance(h, int)
    assert h >= 0


def test_h_index_chain() -> None:
    """Chain in-degrees [0,1,1,1]: exactly 1 entity with in-degree ≥ 1 → h_index = 1."""
    assert ExtendedGraphAnalysis(_chain_dataset()).h_index == 1


def test_h_index_high() -> None:
    """Graph with 3 entities of in-degree 3 each has h_index = 3."""
    assert ExtendedGraphAnalysis(_high_h_index_dataset()).h_index == 3


def test_h_index_empty() -> None:
    """Empty graph has h_index 0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).h_index == 0


# ---------------------------------------------------------------------------
# variance_in_degree / variance_out_degree
# ---------------------------------------------------------------------------


def test_variance_in_degree_float(nations: Nations) -> None:
    """Degree_variance_in is a non-negative float."""
    v = ExtendedGraphAnalysis(nations).variance_in_degree
    assert isinstance(v, float)
    assert v >= 0.0


def test_variance_out_degree_float(nations: Nations) -> None:
    """Degree_variance_out is a non-negative float."""
    v = ExtendedGraphAnalysis(nations).variance_out_degree
    assert isinstance(v, float)
    assert v >= 0.0


def test_degree_variance_chain() -> None:
    """Chain in/out-degrees both [0,1,1,1]/[1,1,1,0]: population variance is 0.1875."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.variance_in_degree == pytest.approx(0.1875)
    assert ha.variance_out_degree == pytest.approx(0.1875)


def test_degree_variance_empty() -> None:
    """Empty graph has degree variances of 0.0."""
    ha = ExtendedGraphAnalysis(_empty_dataset())
    assert ha.variance_in_degree == 0.0
    assert ha.variance_out_degree == 0.0


# ---------------------------------------------------------------------------
# std_in_degree / std_out_degree
# ---------------------------------------------------------------------------


def test_std_in_degree_float(nations: Nations) -> None:
    """Degree_std_in is a non-negative float equal to sqrt(variance_in_degree)."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.std_in_degree, float)
    assert ha.std_in_degree >= 0.0
    assert ha.std_in_degree == pytest.approx(ha.variance_in_degree**0.5)


def test_std_out_degree_float(nations: Nations) -> None:
    """Degree_std_out is a non-negative float equal to sqrt(variance_out_degree)."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.std_out_degree, float)
    assert ha.std_out_degree >= 0.0
    assert ha.std_out_degree == pytest.approx(ha.variance_out_degree**0.5)


def test_degree_std_chain() -> None:
    """Chain degree std is sqrt(0.1875) for both in and out."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.std_in_degree == pytest.approx(0.1875**0.5)
    assert ha.std_out_degree == pytest.approx(0.1875**0.5)


# ---------------------------------------------------------------------------
# coefficient_of_variation_in_degree / coefficient_of_variation_out_degree
# ---------------------------------------------------------------------------


def test_cv_in_float(nations: Nations) -> None:
    """Coefficient_of_variation_in_degree is a non-negative float."""
    cv = ExtendedGraphAnalysis(nations).coefficient_of_variation_in_degree
    assert isinstance(cv, float)
    assert cv >= 0.0


def test_cv_out_float(nations: Nations) -> None:
    """Coefficient_of_variation_out_degree is a non-negative float."""
    cv = ExtendedGraphAnalysis(nations).coefficient_of_variation_out_degree
    assert isinstance(cv, float)
    assert cv >= 0.0


def test_cv_uniform_degree() -> None:
    """Graph where every entity has in- and out-degree 1 has cv_in = cv_out = 0.0."""
    ha = ExtendedGraphAnalysis(_uniform_degree_dataset())
    assert ha.coefficient_of_variation_in_degree == pytest.approx(0.0)
    assert ha.coefficient_of_variation_out_degree == pytest.approx(0.0)


def test_cv_empty() -> None:
    """Empty graph has cv_in = cv_out = 0.0."""
    ha = ExtendedGraphAnalysis(_empty_dataset())
    assert ha.coefficient_of_variation_in_degree == 0.0
    assert ha.coefficient_of_variation_out_degree == 0.0


# ---------------------------------------------------------------------------
# undirected_h_index
# ---------------------------------------------------------------------------


def test_undirected_h_index_int(nations: Nations) -> None:
    """Undirected_h_index is a non-negative int."""
    h = ExtendedGraphAnalysis(nations).undirected_h_index
    assert isinstance(h, int)
    assert h >= 0


def test_undirected_h_index_chain() -> None:
    """Chain total degrees [1,2,2,1] sorted desc [2,2,1,1]: h=2."""
    assert ExtendedGraphAnalysis(_chain_dataset()).undirected_h_index == 2


def test_undirected_h_index_empty() -> None:
    """Empty graph has undirected_h_index 0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).undirected_h_index == 0


def test_undirected_h_index_ge_h_index(nations: Nations) -> None:
    """Undirected h-index (total degree) >= directed h-index (in-degree only)."""
    ha = ExtendedGraphAnalysis(nations)
    assert ha.undirected_h_index >= ha.h_index


# ---------------------------------------------------------------------------
# degree_centrality_max
# ---------------------------------------------------------------------------


def test_degree_centrality_max_int(nations: Nations) -> None:
    """Degree_centrality_max is a non-negative int equal to max_degree."""
    ha = ExtendedGraphAnalysis(nations)
    assert isinstance(ha.degree_centrality_max, int)
    assert ha.degree_centrality_max == ha.max_degree


def test_degree_centrality_max_star() -> None:
    """Star 0→{1,2,3}: max degree is 3 (node 0 has out-degree 3)."""
    assert ExtendedGraphAnalysis(_star_dataset()).degree_centrality_max == 3


def test_degree_centrality_max_empty() -> None:
    """Empty graph has degree_centrality_max 0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).degree_centrality_max == 0


# ---------------------------------------------------------------------------
# pagerank_max
# ---------------------------------------------------------------------------


def test_pagerank_max_float(nations: Nations) -> None:
    """Pagerank_max is a positive float no greater than 1.0."""
    pr = ExtendedGraphAnalysis(nations).pagerank_max
    assert isinstance(pr, float)
    assert 0.0 < pr <= 1.0


def test_pagerank_max_empty() -> None:
    """Empty graph (no edges, 3 entities) has uniform pagerank 1/3."""
    assert ExtendedGraphAnalysis(_empty_dataset()).pagerank_max == pytest.approx(1 / 3)


def test_pagerank_max_star() -> None:
    """Star graph has positive pagerank_max <= 1.0."""
    pr = ExtendedGraphAnalysis(_star_dataset()).pagerank_max
    assert 0.0 < pr <= 1.0


# ---------------------------------------------------------------------------
# graph_centralization
# ---------------------------------------------------------------------------


def test_graph_centralization_float(nations: Nations) -> None:
    """Graph_centralization is a non-negative float."""
    c = ExtendedGraphAnalysis(nations).graph_centralization
    assert isinstance(c, float)
    assert c >= 0.0


def test_graph_centralization_star() -> None:
    """Star 0→{1,2,3}: C_D = (3*4 - 6) / (3*2) = 1.0."""
    assert ExtendedGraphAnalysis(_star_dataset()).graph_centralization == pytest.approx(1.0)


def test_graph_centralization_chain() -> None:
    """Chain 0→1→2→3: d_max=2, sum=6, C_D = (2*4 - 6) / (3*2) = 1/3."""
    assert ExtendedGraphAnalysis(_chain_dataset()).graph_centralization == pytest.approx(1 / 3)


def test_graph_centralization_empty() -> None:
    """Empty graph with 3 entities: all degrees 0, so C_D = 0.0."""
    assert ExtendedGraphAnalysis(_empty_dataset()).graph_centralization == 0.0


def test_graph_centralization_two_nodes() -> None:
    """Graph with n <= 2 returns 0.0."""
    ds = _make_dataset([[0, 0, 1]], num_entities=2)
    assert ExtendedGraphAnalysis(ds).graph_centralization == 0.0


# ---------------------------------------------------------------------------
# power_law_exponent / power_law_exponent_in / power_law_minimum_cutoff
# ---------------------------------------------------------------------------


def test_power_law_exponent_float(nations: Nations) -> None:
    """Power_law_exponent is a non-negative float."""
    alpha = ExtendedGraphAnalysis(nations).power_law_exponent
    assert isinstance(alpha, float)
    assert alpha >= 0.0


def test_power_law_exponent_in_float(nations: Nations) -> None:
    """Power_law_exponent_in is a non-negative float."""
    alpha = ExtendedGraphAnalysis(nations).power_law_exponent_in
    assert isinstance(alpha, float)
    assert alpha >= 0.0


def test_power_law_minimum_cutoff_int(nations: Nations) -> None:
    """Power_law_minimum_cutoff is a non-negative int."""
    d_min = ExtendedGraphAnalysis(nations).power_law_minimum_cutoff
    assert isinstance(d_min, int)
    assert d_min >= 0


def test_power_law_empty() -> None:
    """Empty graph: fewer than 3 positive degrees, returns 0.0 / 0."""
    ha = ExtendedGraphAnalysis(_empty_dataset())
    assert ha.power_law_exponent == 0.0
    assert ha.power_law_exponent_in == 0.0
    assert ha.power_law_minimum_cutoff == 0


def test_power_law_chain_in_degree() -> None:
    """Chain in-degrees [0,1,1,1]: all positive values equal, so alpha_in is 0.0."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.power_law_exponent_in == pytest.approx(0.0)


def test_power_law_chain_total_degree() -> None:
    """Chain total degrees [1,2,2,1]: positive, varied, alpha > 0, d_min = 1."""
    ha = ExtendedGraphAnalysis(_chain_dataset())
    assert ha.power_law_exponent > 0.0
    assert ha.power_law_minimum_cutoff == 1


# ---------------------------------------------------------------------------
# min_branch_out / max_branch_out
# ---------------------------------------------------------------------------


def test_branch_out_star() -> None:
    """A star's single parent (out-degree 3) sets min, max and avg branch-out to 3."""
    ha = ExtendedGraphAnalysis(_star_dataset())
    assert ha.min_branch_out == 3
    assert ha.max_branch_out == 3
    assert ha.avg_branch_out == pytest.approx(3.0)


def test_branch_out_mixed() -> None:
    """With parents of out-degree 2 and 1, branch-out spans [1, 2]."""
    ds = _make_dataset([[0, 0, 1], [0, 0, 2], [1, 0, 3]], num_entities=4)
    ha = ExtendedGraphAnalysis(ds)
    assert ha.min_branch_out == 1
    assert ha.max_branch_out == 2
    assert ha.min_branch_out <= ha.avg_branch_out <= ha.max_branch_out


def test_max_branch_out_equals_max_fan_out(nations: Nations) -> None:
    """The largest-out-degree node is always a parent, so max_branch_out == max_fan_out."""
    ha = ExtendedGraphAnalysis(nations)
    assert ha.max_branch_out == ha.max_fan_out


# ---------------------------------------------------------------------------
# hierarchy_relation filtering
# ---------------------------------------------------------------------------


def _mixed_relation_dataset() -> EagerDataset:
    """Return a chain 0->1->2 on relation 0, plus an unrelated cycle-closing edge 2->0 on relation 1."""
    return _make_dataset([[0, 0, 1], [1, 0, 2], [2, 1, 0]], num_entities=3, num_relations=2)


def test_hierarchy_relation_filters_edges() -> None:
    """Restricting to relation 0 drops the relation-1 edge that turns the chain into a cycle."""
    ds = _mixed_relation_dataset()
    assert not ExtendedGraphAnalysis(ds).is_dag
    ha = ExtendedGraphAnalysis(ds, hierarchy_relation=0)
    assert ha.is_dag
    assert ha.root_nodes == frozenset({0})
    assert ha.leaf_nodes == frozenset({2})
    assert ha.min_branch_out <= ha.avg_branch_out <= ha.max_branch_out


# ---------------------------------------------------------------------------
# num_entities / num_relations
# ---------------------------------------------------------------------------


def test_num_entities_matches_factory(nations: Nations) -> None:
    """Num_entities matches the training factory's entity count."""
    ha = ExtendedGraphAnalysis(nations, split="train")
    assert ha.num_entities == nations.training.num_entities


def test_num_relations_matches_factory(nations: Nations) -> None:
    """Num_relations matches the training factory's relation count."""
    ha = ExtendedGraphAnalysis(nations, split="train")
    assert ha.num_relations == nations.training.num_relations


def test_num_entities_star() -> None:
    """Star dataset has 4 entities."""
    assert ExtendedGraphAnalysis(_star_dataset()).num_entities == 4


def test_num_relations_star() -> None:
    """Star dataset has 1 relation."""
    assert ExtendedGraphAnalysis(_star_dataset()).num_relations == 1
