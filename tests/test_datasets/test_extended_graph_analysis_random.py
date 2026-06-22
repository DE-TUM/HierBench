"""Property-based random-graph tests for :class:`ExtendedGraphAnalysis`.

These complement the example-based tests in ``test_extended_graph_analysis.py``
(tiny hand-built graphs with hard-coded numbers) and the loose-bound Cora tests
in ``test_extended_graph_analysis_cora.py``. Here we instead generate *many* random graphs
across several shape families (general digraphs, DAGs, rooted trees) and assert
that the metrics hold against an **independent NetworkX oracle** built freshly
from the same edge list -- never reusing the :class:`ExtendedGraphAnalysis` internals.

NetworkX is used as much as possible for both graph *generation* and the
*oracles*. Even where ``ExtendedGraphAnalysis`` calls NetworkX internally, the oracle
still validates its wiring: triples -> graph construction, torch-tensor
aggregation, and the paper formulas (density ``m/n^2``, h-index, balance,
parallel/unique counts) layered on top.

``hypothesis`` is not available, so we follow the repo's lightweight pattern
(see ``test_dataset_restriction.py``): deterministic per-case seeds driven by
``@pytest.mark.parametrize("seed", ...)``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from pykeen.datasets.base import EagerDataset
from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis
from pykeen.triples import CoreTriplesFactory

# Number of random graphs drawn per shape family.
SEEDS = list(range(20))


# ---------------------------------------------------------------------------
# Graph generation (NetworkX generators -> labelled triples)
# ---------------------------------------------------------------------------


def _node_count(seed: int) -> int:
    """Vary the number of entities in ``5..15`` for shape diversity."""
    return 5 + (seed % 11)


def _dataset_from_edges(edges: list[tuple[int, int, int]], n: int, num_relations: int) -> EagerDataset:
    """Wrap a list of ``(head, relation, tail)`` triples in an EagerDataset.

    :param edges: The labelled edges.
    :param n: Number of entities (kept even if some are isolated).
    :param num_relations: Number of relation types.

    :returns: An :class:`EagerDataset` using the same factory for train/test.
    """
    mapped = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 3), dtype=torch.long)
    tf = CoreTriplesFactory.create(mapped_triples=mapped, num_entities=n, num_relations=max(num_relations, 1))
    return EagerDataset(training=tf, testing=tf)


def _random_digraph_edges(seed: int, n: int) -> tuple[list[tuple[int, int, int]], int]:
    """Random directed graph (may contain cycles), via ``nx.gnp_random_graph``.

    Every third seed adds parallel multi-relational edges (a reverse copy under a
    second relation) so the multigraph / parallel-edge semantics get exercised.
    """
    graph = nx.gnp_random_graph(n, 0.25, seed=seed, directed=True)
    edges = [(u, 0, v) for u, v in graph.edges()]
    num_relations = 1
    if seed % 3 == 0 and edges:
        num_relations = 2
        # add a parallel edge (same head/tail, different relation) for a few edges
        for u, _, v in edges[: max(1, len(edges) // 4)]:
            edges.append((u, 1, v))
    return edges, num_relations


def _random_dag_edges(seed: int, n: int) -> tuple[list[tuple[int, int, int]], int]:
    """Random DAG via ``nx.gn_graph`` (a growing-network tree-like DAG)."""
    graph = nx.gn_graph(n, seed=seed)
    edges = [(u, 0, v) for u, v in graph.edges()]
    return edges, 1


def _random_tree(seed: int, n: int) -> tuple[list[tuple[int, int, int]], int, int]:
    """Random rooted tree oriented away from its root.

    :returns: ``(edges, num_relations, root)`` where every non-root node has
        in-degree 1 and ``root`` is the unique source.
    """
    undirected = nx.random_labeled_rooted_tree(n, seed=seed)
    root = undirected.graph["root"]
    directed = nx.bfs_tree(undirected, root)
    edges = [(u, 0, v) for u, v in directed.edges()]
    return edges, 1, root


# ---------------------------------------------------------------------------
# Independent NetworkX oracle
# ---------------------------------------------------------------------------


def _oracle_graphs(edges: list[tuple[int, int, int]], n: int) -> tuple[nx.MultiDiGraph, nx.DiGraph]:
    """Build fresh ``(multigraph, digraph)`` from the edge list, mirroring ExtendedGraphAnalysis.

    The multigraph keeps parallel (multi-relational) edges; the digraph collapses
    them. Both include all ``n`` nodes so isolated entities are represented.
    """
    multi = nx.MultiDiGraph()
    multi.add_nodes_from(range(n))
    di = nx.DiGraph()
    di.add_nodes_from(range(n))
    for head, relation, tail in edges:
        multi.add_edge(head, tail, key=relation)
        di.add_edge(head, tail)
    return multi, di


def _oracle_roots(di: nx.DiGraph) -> set[int]:
    """Nodes with in-degree zero."""
    return {node for node, degree in di.in_degree() if degree == 0}


def _oracle_leaves(di: nx.DiGraph) -> set[int]:
    """Nodes with out-degree zero."""
    return {node for node, degree in di.out_degree() if degree == 0}


def _oracle_depths(di: nx.DiGraph, n: int) -> dict[int, int]:
    """Shortest-path depth from the nearest root (unreachable -> 0)."""
    roots = _oracle_roots(di)
    if not roots:
        return dict.fromkeys(range(n), 0)
    depths = dict(nx.multi_source_dijkstra_path_length(di, roots))
    for node in range(n):
        depths.setdefault(node, 0)
    return depths


def _oracle_max_depth(di: nx.DiGraph) -> int:
    """Deepest shortest-path distance reachable from any root."""
    roots = _oracle_roots(di)
    depth = 0
    for root in roots:
        lengths = nx.single_source_shortest_path_length(di, root)
        depth = max(depth, max(lengths.values(), default=0))
    return depth


def _oracle_h_index(in_degrees: list[int]) -> int:
    """Largest h such that at least h nodes have in-degree >= h."""
    h = 0
    for i, d in enumerate(sorted(in_degrees, reverse=True), start=1):
        if d >= i:
            h = i
        else:
            break
    return h


def _oracle_diameter(di: nx.DiGraph) -> int:
    """Max diameter over the connected components of the undirected projection."""
    undirected = di.to_undirected()
    return max(
        (nx.diameter(undirected.subgraph(component)) for component in nx.connected_components(undirected)),
        default=0,
    )


def _oracle_undirected_h_index(total_degrees: list[int]) -> int:
    """Largest h such that at least h nodes have total degree >= h."""
    h = 0
    for i, d in enumerate(sorted(total_degrees, reverse=True), start=1):
        if d >= i:
            h = i
        else:
            break
    return h


def _oracle_graph_centralization(di: nx.DiGraph, n: int) -> float:
    """Graph centralization C_D from unique-edge degrees."""
    if n <= 2:
        return 0.0
    total_deg = {node: di.in_degree(node) + di.out_degree(node) for node in range(n)}
    d_max = max(total_deg.values())
    denominator = (n - 1) * (n - 2)
    if denominator <= 0:
        return 0.0
    return (d_max * n - sum(total_deg.values())) / denominator


def _oracle_power_law(degrees: list[int]) -> tuple[float, int]:
    """Fit a power-law tail via log-log OLS and return (alpha, d_min)."""
    values = np.array(degrees, dtype=float)
    values = values[values > 0]
    if values.size < 3:
        return 0.0, 0
    values = np.sort(values)[::-1]
    ranks = np.arange(1, values.size + 1, dtype=float)
    slope, _ = np.polyfit(np.log(ranks), np.log(values), deg=1)
    return float(abs(slope)), int(values.min())


# ---------------------------------------------------------------------------
# TestStructuralInvariants: general random digraphs
# ---------------------------------------------------------------------------


class TestStructuralInvariants:
    """Metrics on arbitrary random digraphs, checked against a NetworkX oracle."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_counts_and_roots_leaves(self, seed: int) -> None:
        """Vertex/edge counts and root/leaf sets match the oracle."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        assert ha.total_vertices == n
        assert ha.total_edges == len(edges)
        assert set(ha.root_nodes) == _oracle_roots(di)
        assert set(ha.leaf_nodes) == _oracle_leaves(di)

        tails = {t for _, _, t in edges}
        heads = {h for h, _, _ in edges}
        assert all(r not in tails for r in ha.root_nodes)
        assert all(leaf not in heads for leaf in ha.leaf_nodes)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_unique_and_parallel_edges(self, seed: int) -> None:
        """Unique-edge count and parallel-edge count match the oracle."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        assert ha.unique_edges == di.number_of_edges()
        assert ha.parallel_edges == len(edges) - di.number_of_edges()
        assert ha.parallel_edges >= 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_degree_metrics(self, seed: int) -> None:
        """Max/average degrees and per-node degrees match the multigraph oracle."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, _di = _oracle_graphs(edges, n)

        in_deg = dict(multi.in_degree())
        out_deg = dict(multi.out_degree())
        total_deg = {node: in_deg[node] + out_deg[node] for node in range(n)}

        assert ha.max_fan_out == max(out_deg.values())
        assert ha.max_out_degree == max(out_deg.values())
        assert ha.max_in_degree == max(in_deg.values())
        assert ha.max_degree == max(total_deg.values())

        assert ha.average_out_degree == pytest.approx(sum(out_deg.values()) / n)
        assert ha.average_in_degree == pytest.approx(sum(in_deg.values()) / n)
        assert ha.average_degree == pytest.approx(sum(total_deg.values()) / n)
        assert ha.average_fan_out == pytest.approx(len(edges) / n)
        # Excluding leaves can only raise the mean; tolerance covers float32 rounding
        # in the torch mean when there are no leaves and the two are mathematically equal.
        assert ha.average_branch_out >= ha.average_fan_out - 1e-6

        for node in range(n):
            assert ha.in_degree(node) == in_deg[node]
            assert ha.out_degree(node) == out_deg[node]
            assert ha.total_degree(node) == total_deg[node]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_density_metrics(self, seed: int) -> None:
        """Density formula and the unique-edge density bound hold."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        assert ha.density == pytest.approx(len(edges) / (n * n))
        assert ha.unique_edge_density == pytest.approx(di.number_of_edges() / (n * n))
        assert 0.0 <= ha.unique_edge_density <= ha.density

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reciprocity(self, seed: int) -> None:
        """Reciprocity matches the parallel-inclusive oracle (and nx for simple graphs)."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, di = _oracle_graphs(edges, n)

        m = len(edges)
        directed_pairs = set(di.edges())
        m_bi = sum(1 for u, v in multi.edges() if (v, u) in directed_pairs)
        expected = m_bi / m if m else 0.0
        assert ha.reciprocity == pytest.approx(expected)
        assert 0.0 <= ha.reciprocity <= 1.0

        # For simple digraphs (no parallels) this also equals nx.overall_reciprocity.
        if num_relations == 1 and m > 0:
            assert ha.reciprocity == pytest.approx(nx.overall_reciprocity(di))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_h_index_and_diameter(self, seed: int) -> None:
        """h-index matches the sorted-degree oracle; diameter matches the nx oracle."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, di = _oracle_graphs(edges, n)

        in_degrees = [deg for _, deg in multi.in_degree()]
        assert ha.h_index == _oracle_h_index(in_degrees)
        assert 0 <= ha.h_index <= n
        assert ha.diameter == _oracle_diameter(di)
        assert ha.diameter >= 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_variance_and_std(self, seed: int) -> None:
        """Degree variances are non-negative and std equals sqrt(variance)."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))

        assert ha.degree_variance_in >= 0.0
        assert ha.degree_variance_out >= 0.0
        assert ha.degree_std_in == pytest.approx(ha.degree_variance_in**0.5)
        assert ha.degree_std_out == pytest.approx(ha.degree_variance_out**0.5)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_depth_metrics(self, seed: int) -> None:
        """Average/max depth and balance match the nx shortest-path oracle / bounds."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        depths = _oracle_depths(di, n)
        assert ha.avg_hierarchy_depth == pytest.approx(sum(depths.values()) / n)
        assert ha.max_hierarchy_depth == _oracle_max_depth(di)
        assert ha.avg_hierarchy_depth <= ha.max_hierarchy_depth
        assert 0.0 <= ha.balance <= 1.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_is_dag_matches_networkx(self, seed: int) -> None:
        """is_dag agrees with nx.is_directed_acyclic_graph on the oracle digraph."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)
        assert ha.is_dag == nx.is_directed_acyclic_graph(di)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_undirected_h_index_and_centrality(self, seed: int) -> None:
        """undirected_h_index, degree_centrality_max, and graph_centralization match oracles."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, di = _oracle_graphs(edges, n)

        pairs = zip(multi.in_degree(), multi.out_degree(), strict=False)
        total_degrees = [in_d + out_d for (_, in_d), (_, out_d) in pairs]
        assert ha.undirected_h_index == _oracle_undirected_h_index(total_degrees)
        assert ha.undirected_h_index >= ha.h_index

        assert ha.degree_centrality_max == ha.max_degree

        assert ha.graph_centralization == pytest.approx(_oracle_graph_centralization(di, n))
        assert ha.graph_centralization >= 0.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pagerank_max(self, seed: int) -> None:
        """pagerank_max matches NetworkX pagerank on the multigraph."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, _di = _oracle_graphs(edges, n)

        if edges:
            scores = nx.pagerank(multi)
            expected = max(scores.values())
        else:
            expected = 1.0 / n
        assert ha.pagerank_max == pytest.approx(expected)
        assert 0.0 < ha.pagerank_max <= 1.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_power_law(self, seed: int) -> None:
        """Power-law exponents and cutoff are non-negative and match the oracle."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, _di = _oracle_graphs(edges, n)

        in_degrees = [deg for _, deg in multi.in_degree()]
        out_degrees = [deg for _, deg in multi.out_degree()]
        total_degrees = [i + o for i, o in zip(in_degrees, out_degrees, strict=False)]

        alpha, d_min = _oracle_power_law(total_degrees)
        assert ha.power_law_exponent == pytest.approx(alpha)
        assert ha.power_law_minimum_cutoff == d_min

        alpha_in, _ = _oracle_power_law(in_degrees)
        assert ha.power_law_exponent_in == pytest.approx(alpha_in)

        assert ha.power_law_exponent >= 0.0
        assert ha.power_law_exponent_in >= 0.0
        assert ha.power_law_minimum_cutoff >= 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ancestor_descendant_duality(self, seed: int) -> None:
        """B is a descendant of a iff a is an ancestor of b; no self-membership."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        for node in range(n):
            assert ha.get_ancestors(node) == frozenset(nx.ancestors(di, node))
            assert ha.get_descendants(node) == frozenset(nx.descendants(di, node))
            assert node not in ha.get_ancestors(node)
            assert node not in ha.get_descendants(node)

        for a in range(n):
            for b in ha.get_descendants(a):
                assert a in ha.get_ancestors(b)


# ---------------------------------------------------------------------------
# TestDagProperties: random DAGs
# ---------------------------------------------------------------------------


class TestDagProperties:
    """Properties that must hold for acyclic random graphs."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_is_dag(self, seed: int) -> None:
        """A generated DAG is acyclic."""
        n = _node_count(seed)
        edges, num_relations = _random_dag_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        assert ha.is_dag is True

    @pytest.mark.parametrize("seed", SEEDS)
    def test_roots_have_no_ancestors_leaves_no_descendants(self, seed: int) -> None:
        """Roots have no ancestors and leaves have no descendants."""
        n = _node_count(seed)
        edges, num_relations = _random_dag_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        for root in ha.root_nodes:
            assert ha.get_ancestors(root) == frozenset()
        for leaf in ha.leaf_nodes:
            assert ha.get_descendants(leaf) == frozenset()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_depth_matches_oracle(self, seed: int) -> None:
        """max_hierarchy_depth equals the oracle longest shortest-path from roots."""
        n = _node_count(seed)
        edges, num_relations = _random_dag_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)
        assert ha.max_hierarchy_depth == _oracle_max_depth(di)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_nca_is_common_ancestor(self, seed: int) -> None:
        """If an NCA exists it is a common ancestor (or one of the inputs) of both."""
        n = _node_count(seed)
        edges, num_relations = _random_dag_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        for a in range(n):
            for b in range(a + 1, n):
                nca = ha.nearest_common_ancestor(a, b)
                if nca is not None:
                    assert nca == a or nca in ha.get_ancestors(a)
                    assert nca == b or nca in ha.get_ancestors(b)


# ---------------------------------------------------------------------------
# TestTreeProperties: random rooted trees
# ---------------------------------------------------------------------------


class TestTreeProperties:
    """Properties of directed rooted trees (the cleanest hierarchy shape)."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_tree_shape(self, seed: int) -> None:
        """One root, n-1 edges, every non-root has in-degree 1, no parallels/cycles."""
        n = _node_count(seed)
        edges, num_relations, root = _random_tree(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        multi, _di = _oracle_graphs(edges, n)

        assert ha.root_nodes == frozenset({root})
        assert ha.total_edges == n - 1
        assert ha.parallel_edges == 0
        assert ha.reciprocity == pytest.approx(0.0)
        assert ha.is_dag is True
        for node, deg in multi.in_degree():
            assert deg == (0 if node == root else 1)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_tree_depth_matches_nx(self, seed: int) -> None:
        """Max depth equals the nx DAG longest-path length; avg depth matches BFS."""
        n = _node_count(seed)
        edges, num_relations, root = _random_tree(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)

        assert ha.max_hierarchy_depth == nx.dag_longest_path_length(di)
        lengths = nx.single_source_shortest_path_length(di, root)
        assert ha.avg_hierarchy_depth == pytest.approx(sum(lengths.values()) / n)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_tree_nca_matches_lca(self, seed: int) -> None:
        """nearest_common_ancestor matches the unique tree LCA from NetworkX."""
        n = _node_count(seed)
        edges, num_relations, _root = _random_tree(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        _multi, di = _oracle_graphs(edges, n)
        for a in range(n):
            for b in range(a + 1, n):
                expected = nx.lowest_common_ancestor(di, a, b)
                assert ha.nearest_common_ancestor(a, b) == expected


# ---------------------------------------------------------------------------
# TestSpanningTreeProperties: spanning trees of possibly-cyclic digraphs
# ---------------------------------------------------------------------------


class TestSpanningTreeProperties:
    """Invariants of the BFS/DFS spanning-tree views over random digraphs."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("mode", ["bfs", "dfs"])
    def test_spanning_tree_is_acyclic_forest(self, seed: int, mode: str) -> None:
        """The spanning tree is a DAG with preserved vertices and a subset of edges."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        tree = ha.spanning_tree(mode=mode)  # type: ignore[arg-type]

        assert tree.is_dag is True
        assert tree.total_vertices == n
        assert tree.total_edges <= ha.total_edges
        assert tree.total_edges <= tree.total_vertices

        original_pairs = {(h, t) for h, _, t in edges}
        spanning_pairs = {(int(h), int(t)) for h, _, t in tree._factory.mapped_triples.tolist()}
        assert spanning_pairs <= original_pairs

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bfs_not_deeper_than_dfs(self, seed: int) -> None:
        """BFS spanning trees are no deeper than DFS spanning trees."""
        n = _node_count(seed)
        edges, num_relations = _random_digraph_edges(seed, n)
        ha = ExtendedGraphAnalysis(_dataset_from_edges(edges, n, num_relations))
        bfs = ha.spanning_tree(mode="bfs")
        dfs = ha.spanning_tree(mode="dfs")
        assert bfs.max_hierarchy_depth <= dfs.max_hierarchy_depth
