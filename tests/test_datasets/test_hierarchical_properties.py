"""Tests for hierarchical graph properties on Dataset."""

from __future__ import annotations

import pytest
import torch

from pykeen.datasets import Nations
from pykeen.datasets.base import EagerDataset
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
# Task 1: node_count, edge_count, root_nodes, leaf_nodes
# ---------------------------------------------------------------------------


def test_root_nodes_type(nations: Nations) -> None:
    """root_nodes returns a frozenset."""
    assert isinstance(nations.root_nodes, frozenset)


def test_root_nodes_have_no_parents(nations: Nations) -> None:
    """Every root node never appears as a tail."""
    tails = set(nations.training.mapped_triples[:, 2].tolist())
    for r in nations.root_nodes:
        assert r not in tails


def test_leaf_nodes_type(nations: Nations) -> None:
    """leaf_nodes returns a frozenset."""
    assert isinstance(nations.leaf_nodes, frozenset)


def test_leaf_nodes_have_no_children(nations: Nations) -> None:
    """Every leaf node never appears as a head."""
    heads = set(nations.training.mapped_triples[:, 0].tolist())
    for leaf in nations.leaf_nodes:
        assert leaf not in heads


def test_root_and_leaf_chain() -> None:
    """Chain 0→1→2→3: only node 0 is root, only node 3 is leaf."""
    ds = _chain_dataset()
    assert ds.root_nodes == frozenset({0})
    assert ds.leaf_nodes == frozenset({3})


def test_root_and_leaf_star() -> None:
    """Star 0→{1,2,3}: root is 0, leaves are {1,2,3}."""
    ds = _star_dataset()
    assert ds.root_nodes == frozenset({0})
    assert ds.leaf_nodes == frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# Task 2: is_dag
# ---------------------------------------------------------------------------


def test_is_dag_returns_bool(nations: Nations) -> None:
    """is_dag returns a bool."""
    assert isinstance(nations.is_dag, bool)


def test_is_dag_acyclic() -> None:
    """An acyclic chain is a DAG."""
    ds = _chain_dataset()
    assert ds.is_dag is True


def test_is_dag_cyclic() -> None:
    """A cycle 0→1→2→0 is not a DAG."""
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 0]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ds.is_dag is False


# ---------------------------------------------------------------------------
# Task 3: hierarchy_depth, average_hierarchy_depth
# ---------------------------------------------------------------------------


def test_hierarchy_depth_int(nations: Nations) -> None:
    """hierarchy_depth is a non-negative int."""
    assert isinstance(nations.hierarchy_depth, int)
    assert nations.hierarchy_depth >= 0


def test_average_hierarchy_depth_float(nations: Nations) -> None:
    """average_hierarchy_depth is a non-negative float."""
    assert isinstance(nations.average_hierarchy_depth, float)
    assert nations.average_hierarchy_depth >= 0.0


def test_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3 has max depth 3."""
    ds = _chain_dataset()
    assert ds.hierarchy_depth == 3


def test_average_hierarchy_depth_chain() -> None:
    """Chain 0→1→2→3: depths are 0,1,2,3 so mean is 1.5."""
    ds = _chain_dataset()
    assert ds.average_hierarchy_depth == pytest.approx(1.5)


def test_hierarchy_depth_star() -> None:
    """Star 0→{1,2,3}: max depth is 1."""
    ds = _star_dataset()
    assert ds.hierarchy_depth == 1


# ---------------------------------------------------------------------------
# Task 4: average_fan_out, max_fan_out, balance
# ---------------------------------------------------------------------------


def test_average_fan_out_float(nations: Nations) -> None:
    """average_fan_out is a non-negative float."""
    assert isinstance(nations.average_fan_out, float)
    assert nations.average_fan_out >= 0.0


def test_max_fan_out_int(nations: Nations) -> None:
    """max_fan_out is a non-negative int."""
    assert isinstance(nations.max_fan_out, int)
    assert nations.max_fan_out >= 0


def test_fan_out_star() -> None:
    """Star 0→{1,2,3}: max_fan_out=3, average_fan_out=0.75."""
    ds = _star_dataset()
    assert ds.max_fan_out == 3
    assert ds.average_fan_out == pytest.approx(0.75)


def test_balance_range(nations: Nations) -> None:
    """Balance is a float in [0, 1]."""
    b = nations.balance
    assert isinstance(b, float)
    assert 0.0 <= b <= 1.0


def test_balance_in_range_star() -> None:
    """Balance of a star graph is in [0, 1]."""
    ds = _star_dataset()
    assert 0.0 <= ds.balance <= 1.0


def test_balance_uniform_depths() -> None:
    """Graph with no edges: all depths are 0, so balance is 1.0."""
    triples = torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=3, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ds.balance == 1.0


# ---------------------------------------------------------------------------
# Task 5: get_ancestors, get_descendants, nearest_common_ancestor
# ---------------------------------------------------------------------------


def test_ancestors_type(nations: Nations) -> None:
    """get_ancestors returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 2].item())
    assert isinstance(nations.get_ancestors(node), frozenset)


def test_descendants_type(nations: Nations) -> None:
    """get_descendants returns a frozenset."""
    node = int(nations.training.mapped_triples[0, 0].item())
    assert isinstance(nations.get_descendants(node), frozenset)


def test_ancestors_chain() -> None:
    """In chain 0→1→2→3, ancestors of 3 are {0,1,2} and root 0 has none."""
    ds = _chain_dataset()
    assert ds.get_ancestors(3) == frozenset({0, 1, 2})
    assert ds.get_ancestors(0) == frozenset()


def test_descendants_chain() -> None:
    """In chain 0→1→2→3, descendants of 0 are {1,2,3} and leaf 3 has none."""
    ds = _chain_dataset()
    assert ds.get_descendants(0) == frozenset({1, 2, 3})
    assert ds.get_descendants(3) == frozenset()


def test_nca_chain() -> None:
    """NCA in a chain: NCA(1,3)=1, NCA(2,3)=2, NCA(0,3)=0."""
    ds = _chain_dataset()
    assert ds.nearest_common_ancestor(1, 3) == 1
    assert ds.nearest_common_ancestor(2, 3) == 2
    assert ds.nearest_common_ancestor(0, 3) == 0


def test_nca_same_node() -> None:
    """NCA of a node with itself is the node."""
    ds = _chain_dataset()
    assert ds.nearest_common_ancestor(2, 2) == 2


def test_nca_no_common_ancestor() -> None:
    """Two disconnected components have no common ancestor."""
    triples = torch.tensor([[0, 0, 1], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ds.nearest_common_ancestor(1, 3) is None


def test_nca_dag_with_diamond() -> None:
    """Diamond DAG (0→1,0→2,1→3,2→3): NCA(1,2)=0, NCA(1,3)=1."""
    triples = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 3], [2, 0, 3]], dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=triples, num_entities=4, num_relations=1)
    ds = EagerDataset(training=tf, testing=tf)
    assert ds.nearest_common_ancestor(1, 2) == 0
    assert ds.nearest_common_ancestor(1, 3) == 1
