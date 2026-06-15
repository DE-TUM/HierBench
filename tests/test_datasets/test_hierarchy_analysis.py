"""Tests for :mod:`pykeen.datasets.hierarchy_analysis` using the Cora dataset.

The Cora citation network is a real-world directed graph (papers citing
papers), which makes it a good complement to the small synthetic graphs used
in ``test_hierarchical_properties.py``: it contains citation cycles, multiple
root/leaf nodes, and a non-trivial hierarchy structure.
"""

from __future__ import annotations

import pytest

from pykeen.datasets import Cora
from pykeen.datasets.hierarchy_analysis import _SpanningTreeView

# downloads and caches the Cora dataset on first use
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cora() -> Cora:
    """Return the Cora citation network dataset."""
    return Cora()


class TestRootAndLeafNodes:
    """Tests for root_nodes and leaf_nodes."""

    def test_root_nodes_type(self, cora: Cora) -> None:
        """root_nodes returns a frozenset."""
        assert isinstance(cora.root_nodes, frozenset)

    def test_root_nodes_have_no_incoming_edges(self, cora: Cora) -> None:
        """Every root node never appears as a tail."""
        tails = set(cora.training.mapped_triples[:, 2].tolist())
        for r in cora.root_nodes:
            assert r not in tails

    def test_root_nodes_within_entity_range(self, cora: Cora) -> None:
        """Root node IDs are valid entity IDs."""
        assert cora.root_nodes
        assert all(0 <= r < cora.num_entities for r in cora.root_nodes)

    def test_leaf_nodes_type(self, cora: Cora) -> None:
        """leaf_nodes returns a frozenset."""
        assert isinstance(cora.leaf_nodes, frozenset)

    def test_leaf_nodes_have_no_outgoing_edges(self, cora: Cora) -> None:
        """Every leaf node never appears as a head."""
        heads = set(cora.training.mapped_triples[:, 0].tolist())
        for leaf in cora.leaf_nodes:
            assert leaf not in heads

    def test_leaf_nodes_within_entity_range(self, cora: Cora) -> None:
        """Leaf node IDs are valid entity IDs."""
        assert cora.leaf_nodes
        assert all(0 <= leaf < cora.num_entities for leaf in cora.leaf_nodes)


class TestIsDag:
    """Tests for is_dag."""

    def test_is_dag_returns_bool(self, cora: Cora) -> None:
        """is_dag returns a bool."""
        assert isinstance(cora.is_dag, bool)

    def test_cora_contains_citation_cycles(self, cora: Cora) -> None:
        """The Cora citation graph contains cycles, so it is not a DAG."""
        assert cora.is_dag is False


class TestDepthMetrics:
    """Tests for hierarchy_depth, average_hierarchy_depth, and levels."""

    def test_hierarchy_depth_non_negative_int(self, cora: Cora) -> None:
        """hierarchy_depth is a non-negative int."""
        depth = cora.hierarchy_depth
        assert isinstance(depth, int)
        assert depth > 0

    def test_average_hierarchy_depth_non_negative_float(self, cora: Cora) -> None:
        """average_hierarchy_depth is a non-negative float."""
        avg_depth = cora.average_hierarchy_depth
        assert isinstance(avg_depth, float)
        assert avg_depth >= 0.0

    def test_average_depth_does_not_exceed_max_depth(self, cora: Cora) -> None:
        """The average BFS depth cannot exceed the maximum BFS depth."""
        assert cora.average_hierarchy_depth <= cora.hierarchy_depth

    def test_levels_is_alias_for_hierarchy_depth(self, cora: Cora) -> None:
        """levels is an alias for hierarchy_depth."""
        assert cora.levels == cora.hierarchy_depth


class TestFanOutMetrics:
    """Tests for average_fan_out, max_fan_out, and average_branch_out."""

    def test_average_fan_out_non_negative_float(self, cora: Cora) -> None:
        """average_fan_out is a non-negative float."""
        afo = cora.average_fan_out
        assert isinstance(afo, float)
        assert afo >= 0.0

    def test_average_fan_out_matches_edge_to_entity_ratio(self, cora: Cora) -> None:
        """average_fan_out equals num_triples / num_entities."""
        expected = cora.training.num_triples / cora.num_entities
        assert cora.average_fan_out == pytest.approx(expected)

    def test_max_fan_out_non_negative_int(self, cora: Cora) -> None:
        """max_fan_out is a non-negative int."""
        mfo = cora.max_fan_out
        assert isinstance(mfo, int)
        assert mfo > 0

    def test_max_fan_out_at_least_average_fan_out(self, cora: Cora) -> None:
        """The maximum out-degree is at least the average out-degree."""
        assert cora.max_fan_out >= cora.average_fan_out

    def test_average_branch_out_non_negative_float(self, cora: Cora) -> None:
        """average_branch_out is a non-negative float."""
        abo = cora.average_branch_out
        assert isinstance(abo, float)
        assert abo >= 0.0

    def test_average_branch_out_at_least_average_fan_out(self, cora: Cora) -> None:
        """Excluding leaves from the average can only raise the value."""
        assert cora.average_branch_out >= cora.average_fan_out

    def test_average_branch_out_at_most_max_fan_out(self, cora: Cora) -> None:
        """The average over parent nodes cannot exceed the global maximum."""
        assert cora.average_branch_out <= cora.max_fan_out


class TestBalance:
    """Tests for the balance property."""

    def test_balance_in_unit_interval(self, cora: Cora) -> None:
        """balance is a float in [0, 1]."""
        balance = cora.balance
        assert isinstance(balance, float)
        assert 0.0 <= balance <= 1.0


class TestAncestryQueries:
    """Tests for get_ancestors, get_descendants, and nearest_common_ancestor."""

    def test_ancestors_and_descendants_types(self, cora: Cora) -> None:
        """get_ancestors and get_descendants return frozensets."""
        head, _, tail = cora.training.mapped_triples[0].tolist()
        assert isinstance(cora.get_ancestors(tail), frozenset)
        assert isinstance(cora.get_descendants(head), frozenset)

    def test_ancestors_excludes_self(self, cora: Cora) -> None:
        """A node is never its own ancestor."""
        leaf = next(iter(cora.leaf_nodes))
        assert leaf not in cora.get_ancestors(leaf)

    def test_descendants_excludes_self(self, cora: Cora) -> None:
        """A node is never its own descendant."""
        root = next(iter(cora.root_nodes))
        assert root not in cora.get_descendants(root)

    def test_root_node_has_no_ancestors(self, cora: Cora) -> None:
        """A root node (in-degree zero) has no ancestors."""
        root = next(iter(cora.root_nodes))
        assert cora.get_ancestors(root) == frozenset()

    def test_leaf_node_has_no_descendants(self, cora: Cora) -> None:
        """A leaf node (out-degree zero) has no descendants."""
        leaf = next(iter(cora.leaf_nodes))
        assert cora.get_descendants(leaf) == frozenset()

    def test_ancestor_descendant_consistency(self, cora: Cora) -> None:
        """If b is a descendant of a, then a is an ancestor of b."""
        head, _, tail = (int(x) for x in cora.training.mapped_triples[0])
        assert tail in cora.get_descendants(head)
        assert head in cora.get_ancestors(tail)

    def test_nearest_common_ancestor_of_node_with_itself(self, cora: Cora) -> None:
        """The NCA of a node with itself is the node."""
        head, _, _ = cora.training.mapped_triples[0].tolist()
        assert cora.nearest_common_ancestor(head, head) == head

    def test_nearest_common_ancestor_return_type(self, cora: Cora) -> None:
        """nearest_common_ancestor returns an entity ID or None."""
        head, _, tail = cora.training.mapped_triples[0].tolist()
        result = cora.nearest_common_ancestor(head, tail)
        assert result is None or isinstance(result, int)

    def test_nearest_common_ancestor_is_common_ancestor(self, cora: Cora) -> None:
        """If an NCA exists, it must be an ancestor (or itself) of both nodes."""
        head, _, tail = (int(x) for x in cora.training.mapped_triples[0])
        nca = cora.nearest_common_ancestor(head, tail)
        if nca is not None:
            assert nca == head or nca in cora.get_ancestors(head)
            assert nca == tail or nca in cora.get_ancestors(tail)


class TestSpanningTree:
    """Tests for spanning_tree and _SpanningTreeView."""

    def test_invalid_mode_raises(self, cora: Cora) -> None:
        """An invalid mode raises a ValueError."""
        with pytest.raises(ValueError, match="mode"):
            cora.spanning_tree(mode="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_returns_view(self, cora: Cora, mode: str) -> None:
        """spanning_tree returns a _SpanningTreeView."""
        tree = cora.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert isinstance(tree, _SpanningTreeView)

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_is_dag(self, cora: Cora, mode: str) -> None:
        """A spanning tree/forest is always acyclic, even though Cora is not."""
        tree = cora.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.is_dag is True

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_preserves_num_entities(self, cora: Cora, mode: str) -> None:
        """The spanning tree retains the same number of entities."""
        tree = cora.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.num_entities == cora.num_entities

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_edge_count(self, cora: Cora, mode: str) -> None:
        """A spanning forest has fewer edges than the original graph and at most one per node."""
        tree = cora.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.training.num_triples < cora.training.num_triples
        assert tree.training.num_triples <= tree.num_entities

    def test_bfs_spanning_tree_is_shallower_than_dfs(self, cora: Cora) -> None:
        """BFS produces shallower trees than DFS, which follows long chains first."""
        bfs_tree = cora.spanning_tree(mode="bfs")
        dfs_tree = cora.spanning_tree(mode="dfs")
        assert bfs_tree.hierarchy_depth <= dfs_tree.hierarchy_depth

    def test_spanning_tree_exposes_hierarchy_properties(self, cora: Cora) -> None:
        """The spanning tree view exposes all HierarchyAnalysis properties."""
        tree = cora.spanning_tree(mode="bfs")
        assert isinstance(tree.root_nodes, frozenset)
        assert isinstance(tree.leaf_nodes, frozenset)
        assert isinstance(tree.balance, float)
        assert isinstance(tree.average_fan_out, float)
        assert isinstance(tree.max_fan_out, int)
