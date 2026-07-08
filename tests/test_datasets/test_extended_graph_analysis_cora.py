"""Tests for hierarchical properties of :class:`ExtendedGraphAnalysis` using the Cora dataset.

Only hierarchical properties (roots/leaves, DAG, depth, fan-out, leaf-depth variance,
ancestry, spanning tree) are tested here. Paper-aligned graph measures
(Zloch et al. 2019) are covered in ``test_extended_graph_analysis.py`` and
``test_extended_graph_analysis_random.py``.

The Cora citation network is a real-world directed graph (papers citing
papers), which makes it a good complement to the small synthetic graphs used
in ``test_extended_graph_analysis.py``: it contains citation cycles, multiple
root/leaf nodes, and a non-trivial hierarchy structure.
"""

from __future__ import annotations

import pytest

from pykeen.datasets import Cora
from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis, _SpanningTreeView

# downloads and caches the Cora dataset on first use
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cora() -> Cora:
    """Return the Cora citation network dataset."""
    return Cora()


@pytest.fixture(scope="module")
def ha(cora: Cora) -> ExtendedGraphAnalysis:
    """Return a ExtendedGraphAnalysis over Cora's training split."""
    return ExtendedGraphAnalysis(cora, split="train")


class TestRootAndLeafNodes:
    """Tests for root_nodes and leaf_nodes."""

    def test_root_nodes_type(self, ha: ExtendedGraphAnalysis) -> None:
        """root_nodes returns a frozenset."""
        assert isinstance(ha.root_nodes, frozenset)

    def test_root_nodes_have_no_incoming_edges(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """Every root node never appears as a tail."""
        tails = set(cora.training.mapped_triples[:, 2].tolist())
        for r in ha.root_nodes:
            assert r not in tails

    def test_root_nodes_within_entity_range(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """Root node IDs are valid entity IDs."""
        assert ha.root_nodes
        assert all(0 <= r < cora.num_entities for r in ha.root_nodes)

    def test_leaf_nodes_type(self, ha: ExtendedGraphAnalysis) -> None:
        """leaf_nodes returns a frozenset."""
        assert isinstance(ha.leaf_nodes, frozenset)

    def test_leaf_nodes_have_no_outgoing_edges(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """Every leaf node never appears as a head."""
        heads = set(cora.training.mapped_triples[:, 0].tolist())
        for leaf in ha.leaf_nodes:
            assert leaf not in heads

    def test_leaf_nodes_within_entity_range(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """Leaf node IDs are valid entity IDs."""
        assert ha.leaf_nodes
        assert all(0 <= leaf < cora.num_entities for leaf in ha.leaf_nodes)


class TestIsDag:
    """Tests for is_dag."""

    def test_is_dag_returns_bool(self, ha: ExtendedGraphAnalysis) -> None:
        """is_dag returns a bool."""
        assert isinstance(ha.is_dag, bool)

    def test_cora_contains_citation_cycles(self, ha: ExtendedGraphAnalysis) -> None:
        """The Cora citation graph contains cycles, so it is not a DAG."""
        assert ha.is_dag is False


class TestDepthMetrics:
    """Tests for max_hierarchy_depth, avg_hierarchy_depth, and levels."""

    def test_max_hierarchy_depth_non_negative_int(self, ha: ExtendedGraphAnalysis) -> None:
        """max_hierarchy_depth is a non-negative int."""
        depth = ha.max_hierarchy_depth
        assert isinstance(depth, int)
        assert depth > 0

    def test_avg_hierarchy_depth_non_negative_float(self, ha: ExtendedGraphAnalysis) -> None:
        """avg_hierarchy_depth is a non-negative float."""
        avg_depth = ha.avg_hierarchy_depth
        assert isinstance(avg_depth, float)
        assert avg_depth >= 0.0

    def test_average_depth_does_not_exceed_max_depth(self, ha: ExtendedGraphAnalysis) -> None:
        """The average depth cannot exceed the maximum depth."""
        assert ha.avg_hierarchy_depth <= ha.max_hierarchy_depth

    def test_levels_is_alias_for_max_hierarchy_depth(self, ha: ExtendedGraphAnalysis) -> None:
        """Levels is an alias for max_hierarchy_depth."""
        assert ha.levels == ha.max_hierarchy_depth


class TestFanOutMetrics:
    """Tests for avg_fan_out, max_fan_out, and avg_branch_out."""

    def test_avg_fan_out_non_negative_float(self, ha: ExtendedGraphAnalysis) -> None:
        """avg_fan_out is a non-negative float."""
        afo = ha.avg_fan_out
        assert isinstance(afo, float)
        assert afo >= 0.0

    def test_avg_fan_out_matches_edge_to_entity_ratio(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """avg_fan_out equals num_triples / num_entities."""
        expected = cora.training.num_triples / cora.num_entities
        assert ha.avg_fan_out == pytest.approx(expected)

    def test_max_fan_out_non_negative_int(self, ha: ExtendedGraphAnalysis) -> None:
        """max_fan_out is a non-negative int."""
        mfo = ha.max_fan_out
        assert isinstance(mfo, int)
        assert mfo > 0

    def test_max_fan_out_at_least_avg_fan_out(self, ha: ExtendedGraphAnalysis) -> None:
        """The maximum out-degree is at least the average out-degree."""
        assert ha.max_fan_out >= ha.avg_fan_out

    def test_avg_branch_out_non_negative_float(self, ha: ExtendedGraphAnalysis) -> None:
        """avg_branch_out is a non-negative float."""
        abo = ha.avg_branch_out
        assert isinstance(abo, float)
        assert abo >= 0.0

    def test_avg_branch_out_at_least_avg_fan_out(self, ha: ExtendedGraphAnalysis) -> None:
        """Excluding leaves from the average can only raise the value."""
        assert ha.avg_branch_out >= ha.avg_fan_out

    def test_avg_branch_out_at_most_max_fan_out(self, ha: ExtendedGraphAnalysis) -> None:
        """The average over parent nodes cannot exceed the global maximum."""
        assert ha.avg_branch_out <= ha.max_fan_out


class TestLeafDepthVariance:
    """Tests for the leaf_depth_variance property."""

    def test_leaf_depth_variance_non_negative(self, ha: ExtendedGraphAnalysis) -> None:
        """Leaf-depth variance is a non-negative float."""
        leaf_depth_variance = ha.leaf_depth_variance
        assert isinstance(leaf_depth_variance, float)
        assert leaf_depth_variance >= 0.0


class TestAncestryQueries:
    """Tests for get_ancestors, get_descendants, and nearest_common_ancestor."""

    def test_ancestors_and_descendants_types(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """get_ancestors and get_descendants return frozensets."""
        head, _, tail = cora.training.mapped_triples[0].tolist()
        assert isinstance(ha.get_ancestors(tail), frozenset)
        assert isinstance(ha.get_descendants(head), frozenset)

    def test_ancestors_excludes_self(self, ha: ExtendedGraphAnalysis) -> None:
        """A node is never its own ancestor."""
        leaf = next(iter(ha.leaf_nodes))
        assert leaf not in ha.get_ancestors(leaf)

    def test_descendants_excludes_self(self, ha: ExtendedGraphAnalysis) -> None:
        """A node is never its own descendant."""
        root = next(iter(ha.root_nodes))
        assert root not in ha.get_descendants(root)

    def test_root_node_has_no_ancestors(self, ha: ExtendedGraphAnalysis) -> None:
        """A root node (in-degree zero) has no ancestors."""
        root = next(iter(ha.root_nodes))
        assert ha.get_ancestors(root) == frozenset()

    def test_leaf_node_has_no_descendants(self, ha: ExtendedGraphAnalysis) -> None:
        """A leaf node (out-degree zero) has no descendants."""
        leaf = next(iter(ha.leaf_nodes))
        assert ha.get_descendants(leaf) == frozenset()

    def test_ancestor_descendant_consistency(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """If b is a descendant of a, then a is an ancestor of b."""
        head, _, tail = (int(x) for x in cora.training.mapped_triples[0])
        assert tail in ha.get_descendants(head)
        assert head in ha.get_ancestors(tail)

    def test_nearest_common_ancestor_of_node_with_itself(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """The NCA of a node with itself is the node."""
        head, _, _ = cora.training.mapped_triples[0].tolist()
        assert ha.nearest_common_ancestor(head, head) == head

    def test_nearest_common_ancestor_return_type(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """nearest_common_ancestor returns an entity ID or None."""
        head, _, tail = cora.training.mapped_triples[0].tolist()
        result = ha.nearest_common_ancestor(head, tail)
        assert result is None or isinstance(result, int)

    def test_nearest_common_ancestor_is_common_ancestor(self, cora: Cora, ha: ExtendedGraphAnalysis) -> None:
        """If an NCA exists, it must be an ancestor (or itself) of both nodes."""
        head, _, tail = (int(x) for x in cora.training.mapped_triples[0])
        nca = ha.nearest_common_ancestor(head, tail)
        if nca is not None:
            assert nca == head or nca in ha.get_ancestors(head)
            assert nca == tail or nca in ha.get_ancestors(tail)


class TestSpanningTree:
    """Tests for spanning_tree and _SpanningTreeView."""

    def test_invalid_mode_raises(self, ha: ExtendedGraphAnalysis) -> None:
        """An invalid mode raises a ValueError."""
        with pytest.raises(ValueError, match="mode"):
            ha.spanning_tree(mode="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_returns_view(self, ha: ExtendedGraphAnalysis, mode: str) -> None:
        """spanning_tree returns a _SpanningTreeView."""
        tree = ha.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert isinstance(tree, _SpanningTreeView)

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_is_dag(self, ha: ExtendedGraphAnalysis, mode: str) -> None:
        """A spanning tree/forest is always acyclic, even though Cora is not."""
        tree = ha.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.is_dag is True

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_preserves_num_entities(self, cora: Cora, ha: ExtendedGraphAnalysis, mode: str) -> None:
        """The spanning tree retains the same number of entities."""
        tree = ha.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.total_vertices == cora.num_entities

    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_edge_count(self, ha: ExtendedGraphAnalysis, mode: str) -> None:
        """A spanning forest has fewer edges than the original graph and at most one per node."""
        tree = ha.spanning_tree(mode=mode)  # type: ignore[arg-type]
        assert tree.total_edges < ha.total_edges
        assert tree.total_edges <= tree.total_vertices

    def test_bfs_spanning_tree_is_shallower_than_dfs(self, ha: ExtendedGraphAnalysis) -> None:
        """BFS produces shallower trees than DFS, which follows long chains first."""
        bfs_tree = ha.spanning_tree(mode="bfs")
        dfs_tree = ha.spanning_tree(mode="dfs")
        assert bfs_tree.max_hierarchy_depth <= dfs_tree.max_hierarchy_depth

    def test_spanning_tree_exposes_hierarchy_properties(self, ha: ExtendedGraphAnalysis) -> None:
        """The spanning tree view exposes all ExtendedGraphAnalysis properties."""
        tree = ha.spanning_tree(mode="bfs")
        assert isinstance(tree.root_nodes, frozenset)
        assert isinstance(tree.leaf_nodes, frozenset)
        assert isinstance(tree.leaf_depth_variance, float)
        assert isinstance(tree.avg_fan_out, float)
        assert isinstance(tree.max_fan_out, int)
