"""Standalone graph/hierarchy analysis for datasets.

This module exposes :class:`ExtendedGraphAnalysis`, a NetworkX-style helper that takes a
:class:`pykeen.datasets.base.Dataset` (or any object exposing ``training`` /
``testing`` / ``validation`` triples factories and ``merged()``) and computes
hierarchical and general graph measures over a chosen split.

Usage::

    from pykeen.datasets import Nations
    from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis

    ha = ExtendedGraphAnalysis(Nations(), split="train", hierarchy_relation="accusation")
    ha.max_hierarchy_depth
    ha.density
    ha.pagerank_max

The analysis object materialises a single NetworkX graph plus the shared degree
tensors once (via :func:`functools.cached_property`) and reuses them across all
metrics, so computing a full report does not recompute these intermediates.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections import deque
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

from .metadata import resolve_hierarchy_relation
from ..triples import CoreTriplesFactory

if TYPE_CHECKING:
    from .base import Dataset

__all__ = [
    "ExtendedGraphAnalysis",
]

Split = Literal["full", "train"]


class ExtendedGraphAnalysis:
    """Hierarchical and general graph measures computed over one split of a dataset.

    All metrics operate on a single :class:`~pykeen.triples.CoreTriplesFactory`
    resolved from the requested split. Shared intermediates (the NetworkX graph,
    in/out-degree tensors, BFS depths) are cached on the instance, so accessing
    many metrics on the same instance is cheap.
    """

    def __init__(
        self,
        dataset: Dataset,
        split: Split = "train",
        hierarchy_relation: int | str | None = None,
    ) -> None:
        """Initialize the analysis for a dataset split.

        :param dataset: The dataset to analyse. Only ``training`` and ``merged()``
            are required.
        :param split: Which subset to analyse: ``"train"`` (default) or ``"full"``
            (all triples merged). Test and validation splits share the same entity
            vocabulary as training and are not meaningful for graph analysis.
        :param hierarchy_relation: Restrict the analysis to edges of this relation (id or
            label). Defaults to the dataset's
            :attr:`~pykeen.datasets.metadata.HierarchicalGraph.hierarchical_relation` when it is
            a :class:`~pykeen.datasets.metadata.HierarchicalGraph`; otherwise all edges are used.
            Multi-relational datasets mix unrelated relations into the "hierarchy" edges when this
            is left unset, which can turn a true DAG into a graph with cycles.
        """
        self._factory: CoreTriplesFactory = self._resolve_factory(dataset, split)
        self._num_entities: int = self._factory.num_entities
        self._split: str = split
        self._hierarchy_relation: int | None = resolve_hierarchy_relation(dataset, hierarchy_relation)

    @staticmethod
    def _resolve_factory(dataset: Dataset, split: Split) -> CoreTriplesFactory:
        """Resolve the triples factory for the requested split.

        :param dataset: The dataset to analyse.
        :param split: One of ``"full"`` or ``"train"``.

        :returns: The corresponding :class:`~pykeen.triples.CoreTriplesFactory`.

        :raises ValueError: If ``split`` is not a recognized value.
        """
        if split == "train":
            return dataset.training
        if split == "full":
            return dataset.merged()
        raise ValueError(f"split must be one of 'full', 'train', got {split!r}")

    # ------------------------------------------------------------------
    # Cached materialised structures
    # ------------------------------------------------------------------

    @functools.cached_property
    def _multigraph(self) -> nx.MultiDiGraph:
        """A multi-directed graph preserving parallel (multi-relational) edges."""
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(self._num_entities))
        triples = self._factory.mapped_triples
        for head, relation, tail in triples.tolist():
            if self._hierarchy_relation is not None and relation != self._hierarchy_relation:
                continue
            graph.add_edge(int(head), int(tail), key=int(relation), relation=int(relation))
        return graph

    @functools.cached_property
    def _digraph(self) -> nx.DiGraph:
        """A simple directed graph with parallel edges collapsed (for path algorithms)."""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self._num_entities))
        triples = self._factory.mapped_triples
        if self._hierarchy_relation is not None:
            triples = triples[triples[:, 1] == self._hierarchy_relation]
        if triples.numel():
            graph.add_edges_from(triples[:, [0, 2]].tolist())
        return graph

    @functools.cached_property
    def _in_out_degrees(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(in_degree, out_degree)`` tensors of length ``num_entities``.

        Degrees are read from :attr:`_multigraph`, so parallel (multi-relational)
        edges are counted, matching the Zloch et al. (2019) convention.
        """
        return self._degree_tensors(self._multigraph)

    @functools.cached_property
    def _unique_in_out_degrees(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return in- and out-degree tensors using unique edges only (Zloch et al. 2019).

        Degrees are read from :attr:`_digraph`, which collapses parallel edges, so
        each ``(head, tail)`` pair is counted once.
        """
        return self._degree_tensors(self._digraph)

    def _degree_tensors(self, graph: nx.DiGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(in_degree, out_degree)`` torch tensors from a NetworkX graph.

        :param graph: A directed (multi-)graph whose nodes are entity IDs.

        :returns: Two length-``num_entities`` tensors indexed by entity ID.
        """
        n = self._num_entities
        in_deg = torch.zeros(n, dtype=torch.long)
        out_deg = torch.zeros(n, dtype=torch.long)
        for node, degree in graph.in_degree():
            in_deg[node] = degree
        for node, degree in graph.out_degree():
            out_deg[node] = degree
        return in_deg, out_deg

    @functools.cached_property
    def _node_depths(self) -> dict[int, int]:
        """Shortest-path depth of each node from its nearest root node.

        Multi-source shortest paths from all root nodes simultaneously. Nodes
        unreachable from any root (isolated nodes or nodes only inside cycles)
        are assigned depth 0.

        :returns: Mapping ``{entity_id: depth}``.
        """
        roots = self.root_nodes
        if not roots:
            return dict.fromkeys(range(self._num_entities), 0)
        # Single multi-source traversal: edges are unweighted, so Dijkstra with the
        # default unit weights yields the same depths as a multi-source BFS. The
        # path-length variant returns only the ``{node: distance}`` mapping.
        depths = dict(nx.multi_source_dijkstra_path_length(self._digraph, set(roots)))
        for n in range(self._num_entities):
            depths.setdefault(n, 0)  # isolated / cycle-only nodes → depth 0
        return depths

    # ------------------------------------------------------------------
    # Hierarchical graph properties
    # ------------------------------------------------------------------

    @functools.cached_property
    def root_nodes(self) -> frozenset[int]:
        r"""Top-level nodes: entities that never appear as a tail.

        .. math:: R = \{v \in V \mid d^-(v) = 0\}

        where :math:`d^-(v)` is the in-degree of :math:`v`.

        :returns: Frozenset of entity IDs with in-degree zero.
        """
        return frozenset(node for node, degree in self._digraph.in_degree() if degree == 0)

    @functools.cached_property
    def leaf_nodes(self) -> frozenset[int]:
        r"""Bottom-level nodes: entities that never appear as a head.

        .. math:: L = \{v \in V \mid d^+(v) = 0\}

        where :math:`d^+(v)` is the out-degree of :math:`v`.

        :returns: Frozenset of entity IDs with out-degree zero.
        """
        return frozenset(node for node, degree in self._digraph.out_degree() if degree == 0)

    @functools.cached_property
    def is_dag(self) -> bool:
        """Whether the graph is a Directed Acyclic Graph (DAG).

        :returns: ``True`` if no directed cycle exists, ``False`` otherwise.
        """
        return nx.is_directed_acyclic_graph(self._digraph)

    @functools.cached_property
    def max_hierarchy_depth(self) -> int:
        r"""Maximum shortest-path distance reachable from any root node.

        The *depth* :math:`\delta(v)` of a node is its shortest-path distance from
        the nearest root node, and

        .. math:: \text{Max depth} = \max_{v \in V} \delta(v)

        For each root node, the shortest-path distances to all reachable nodes
        are computed; the deepest such distance across all roots is returned.
        This uses shortest paths (rather than an ambiguous spanning-tree longest
        path) and is well-defined when cycles do not involve the root nodes.

        :returns: The deepest level below a root. Returns 0 if there are no roots
            or no edges.
        """
        roots = self.root_nodes
        if not roots:
            return 0
        graph = self._digraph
        depth = 0
        for root in roots:
            lengths = nx.single_source_shortest_path_length(graph, root)
            depth = max(depth, max(lengths.values(), default=0))
        return depth

    @property
    def avg_hierarchy_depth(self) -> float:
        r"""Mean shortest-path depth across all nodes (from the nearest root).

        .. math:: \text{Avg depth} = \frac{1}{|V|} \sum_{v \in V} \delta(v)

        where :math:`\delta(v)` is the shortest-path distance of :math:`v` from
        its nearest root node.

        :returns: Average depth from roots. Returns 0.0 for an empty graph.
        """
        depths = self._node_depths
        if not depths:
            return 0.0
        return sum(depths.values()) / len(depths)

    @property
    def min_hierarchy_depth(self) -> int:
        r"""Minimum shortest-path depth across all nodes (from the nearest root).

        .. math:: \text{Min depth} = \min_{v \in V} \delta(v)

        where :math:`\delta(v)` is the shortest-path distance of :math:`v` from
        its nearest root node. This is 0 whenever a root node exists.

        :returns: Minimum depth from roots. Returns 0 for an empty or rootless graph.
        """
        depths = self._node_depths
        if not depths:
            return 0
        return min(depths.values())

    @property
    def levels(self) -> int:
        """Alias for :attr:`max_hierarchy_depth`.

        :returns: The deepest level below a root.
        """
        return self.max_hierarchy_depth

    @property
    def avg_fan_out(self) -> float:
        r"""Average number of children per node (average out-degree).

        .. math:: \bar z_\text{out} = \frac{m}{n}

        where :math:`m` is the number of edges and :math:`n = |V|` the number of
        nodes; equivalently the mean out-degree over *all* nodes, including leaves.

        :returns: Mean out-degree. Returns 0.0 for a graph with no nodes.
        """
        if self._num_entities == 0:
            return 0.0
        return int(self._factory.num_triples) / self._num_entities

    @property
    def max_fan_out(self) -> int:
        r"""Maximum number of children any single node has.

        .. math:: z_\text{out,max} = \max_{v \in V} d^+(v)

        where :math:`d^+(v)` is the out-degree of :math:`v`.

        :returns: Maximum out-degree across all entity IDs. Returns 0 for an
            empty graph.
        """
        _, out_deg = self._in_out_degrees
        return int(out_deg.max().item()) if out_deg.numel() > 0 else 0

    @property
    def avg_branch_out(self) -> float:
        r"""Mean out-degree of non-leaf nodes (nodes with at least one child).

        .. math:: \text{Avg branch-out} = \frac{1}{|\widetilde V|}
            \sum_{v \in \widetilde V} d^+(v)

        where :math:`\widetilde V = \{v \in V \mid d^+(v) \ge 1\}` is the set of
        internal (non-leaf) nodes.

        This is **not** the paper's z_out (Zloch et al. 2019), which averages
        out-degree over *all* entities (see :attr:`avg_fan_out`). Excluding
        leaves makes this a hierarchy-specific branching factor.

        :returns: Average out-degree of parent nodes. Returns 0.0 if there are no
            parent nodes.
        """
        _, out_deg = self._in_out_degrees
        parent_counts = out_deg[out_deg > 0]
        if parent_counts.numel() == 0:
            return 0.0
        return float(parent_counts.float().mean().item())

    @property
    def min_branch_out(self) -> int:
        r"""Minimum out-degree among non-leaf nodes (nodes with at least one child).

        .. math:: \text{Min branch-out} = \min_{v \in \widetilde V} d^+(v)

        where :math:`\widetilde V` is the set of internal (non-leaf) nodes.

        :returns: Smallest branching factor of a parent node. Returns 0 if there
            are no parent nodes.
        """
        _, out_deg = self._in_out_degrees
        parent_counts = out_deg[out_deg > 0]
        return int(parent_counts.min().item()) if parent_counts.numel() else 0

    @property
    def max_branch_out(self) -> int:
        r"""Maximum out-degree among non-leaf nodes.

        .. math:: \text{Max branch-out} = \max_{v \in \widetilde V} d^+(v)

        where :math:`\widetilde V` is the set of internal (non-leaf) nodes.

        Equals :attr:`max_fan_out` for any non-empty graph (the node with the
        largest out-degree is by definition a parent); kept for symmetry with
        :attr:`min_branch_out` and :attr:`avg_branch_out`.

        :returns: Largest branching factor of a parent node. Returns 0 if there
            are no parent nodes.
        """
        _, out_deg = self._in_out_degrees
        parent_counts = out_deg[out_deg > 0]
        return int(parent_counts.max().item()) if parent_counts.numel() else 0

    @property
    def leaf_depth_variance(self) -> float:
        r"""Variance of the leaves' depths V (Sackin's original balance index).

        Following Coronado, Mir, Rosselló & Rotger (2020), *"On Sackin's original
        proposal: the variance of the leaves' depths as a phylogenetic balance
        index"*, BMC Bioinformatics 21:154 (see also the TreeBalance compendium,
        https://treebalance.wordpress.com/variance-of-leaf-depths/):

        .. math:: V(T) = \frac{1}{n} \sum_{x \in L(T)} (\delta(x) - \bar\delta)^2

        where :math:`L(T)` is the set of **leaves** (out-degree-zero nodes),
        :math:`\delta(x)` the depth of leaf ``x`` from its nearest root, and
        :math:`\bar\delta` the mean leaf depth. It is the *raw* population
        variance over leaf depths only (not a coefficient of variation, not over
        all nodes).

        This is an *imbalance* index: ``0.0`` means all leaves share the same
        depth (maximally balanced, e.g. a fully symmetric tree); larger values
        indicate a more skewed hierarchy (a comb/caterpillar maximises it).

        The index is defined for bifurcating phylogenetic trees; here it is
        applied to a general multi-root DAG, so leaf depths use the shortest
        path from the nearest root (:attr:`_node_depths`).

        :returns: Variance of leaf depths V >= 0.0. Returns 0.0 for fewer than
            two leaves.
        """
        if not self.is_dag:
            warnings.warn(
                "leaf_depth_variance is a tree-balance index; the graph has a cycle "
                "(not a DAG), so leaf depths are ill-defined and the value may not be "
                "meaningful.",
                stacklevel=2,
            )
        depths = self._node_depths
        leaf_depths = [depths[leaf] for leaf in self.leaf_nodes]
        n = len(leaf_depths)
        if n <= 1:
            return 0.0
        sackin = sum(leaf_depths)  # S(T): sum of leaf depths
        sackin_sq = sum(d * d for d in leaf_depths)  # S^(2)(T): sum of squared leaf depths
        return sackin_sq / n - (sackin / n) ** 2

    @property
    def balance(self) -> float:
        r"""J¹ robust, universal tree balance index (Lemant, Le Sueur, Manojlović & Noble 2021).

        Following Lemant, Le Sueur, Manojlović & Noble (2021), *"Robust, Universal
        Tree Balance Indices"* (bioRxiv 2021.08.25.457695), the recommended index
        J¹ is a weighted average of the normalised Shannon entropies of the internal
        nodes. For cladograms with equally-sized leaves and zero-size internal nodes
        (the case here, where entities carry no size) it takes the closed form of
        their Equation 5:

        .. math:: J^1(T) = \frac{-1}{\sum_{i \in \widetilde V} n_i}
            \sum_{i \in \widetilde V} \sum_{j \in C(i)} n_j \log_{d^+(i)} \frac{n_j}{n_i}

        where :math:`\widetilde V` is the set of internal nodes (out-degree >= 1),
        :math:`C(i)` the children of ``i``, :math:`d^+(i)` its out-degree, and
        :math:`n_i` the number of leaves of the subtree rooted at ``i``. The
        denominator is exactly Sackin's index (summed over internal nodes).

        Unlike Sackin's index (see :attr:`leaf_depth_variance`), J¹ is intrinsically
        normalised to ``[0, 1]``: ``0.0`` for a linear (or single-node) graph and
        ``1.0`` for a fully symmetric tree. Linear nodes (out-degree 1) get balance
        score 0, so they never contribute to the numerator (the base-1 logarithm is
        undefined) but still count toward the denominator.

        Like :attr:`leaf_depth_variance`, this operates directly on the DAG
        (:attr:`_digraph`), generalising the tree index to a multi-root DAG: ``n_i``
        is the number of *distinct* leaf descendants of ``i``. For strict tree
        semantics, compute it on :meth:`spanning_tree` instead.

        :returns: J¹ in ``[0, 1]``. Returns 0.0 for a graph with no internal nodes
            (empty, single-node, or fully linear).
        """
        if not self.is_dag:
            warnings.warn(
                "balance is a tree-balance index; the graph has a cycle (not a DAG), "
                "so the notion of leaf depth is ill-defined and the value may not be "
                "meaningful.",
                stacklevel=2,
            )
        graph = self._digraph
        leaves = self.leaf_nodes
        # n_i: distinct leaves reachable from each node. Each leaf contributes 1 to
        # itself and to every ancestor (nx.ancestors handles cycles pragmatically).
        # ponytail: O(L*(V+E)) via per-leaf ancestor walks; switch to a reverse-topo
        # pass only if this shows up in profiling on large DAGs.
        leaf_count: dict[int, int] = dict.fromkeys(leaves, 1)
        for leaf in leaves:
            for anc in nx.ancestors(graph, leaf):
                leaf_count[anc] = leaf_count.get(anc, 0) + 1

        numerator = 0.0
        sackin = 0  # sum of n_i over internal nodes — Sackin's index (denominator)
        for i, out_deg in graph.out_degree():
            if out_deg == 0:
                continue  # leaf, not an internal node
            n_i = leaf_count.get(i, 0)
            sackin += n_i
            if out_deg < 2 or n_i == 0:
                continue  # W^1_i = 0 for linear nodes (base-1 log undefined)
            for j in graph.successors(i):
                n_j = leaf_count.get(j, 0)
                if n_j > 0:
                    numerator += n_j * math.log(n_j / n_i, out_deg)
        if sackin == 0:
            return 0.0
        return -numerator / sackin

    def get_ancestors(self, node_id: int) -> frozenset[int]:
        """Return all ancestors (transitive predecessors) of the given node.

        :param node_id: The entity ID to query.

        :returns: Frozenset of entity IDs that have a directed path to
            ``node_id``. Returns an empty frozenset for root nodes.
        """
        return frozenset(nx.ancestors(self._digraph, node_id))

    def get_descendants(self, node_id: int) -> frozenset[int]:
        """Return all descendants (transitive successors) of the given node.

        :param node_id: The entity ID to query.

        :returns: Frozenset of entity IDs reachable from ``node_id`` via directed
            edges. Returns an empty frozenset for leaf nodes.
        """
        return frozenset(nx.descendants(self._digraph, node_id))

    def nearest_common_ancestor(self, node_a: int, node_b: int) -> int | None:
        """Compute the nearest (lowest) common ancestor of two nodes.

        Finds the deepest node that is an ancestor of both inputs. Works on
        general DAGs with multiple roots.

        :param node_a: First entity ID.
        :param node_b: Second entity ID.

        :returns: The entity ID of the nearest common ancestor, or ``None`` if no
            common ancestor exists.
        """
        if node_a == node_b:
            return node_a
        ancestors_a = self.get_ancestors(node_a) | {node_a}
        ancestors_b = self.get_ancestors(node_b) | {node_b}
        common = ancestors_a & ancestors_b
        if not common:
            return None
        depths = self._node_depths
        return max(common, key=lambda n: depths.get(n, 0))

    # TODO: mayeb solve using networkx
    def spanning_tree(self, mode: Literal["bfs", "dfs"] = "bfs") -> _SpanningTreeView:
        """Return a spanning-tree view of this graph for hierarchical analysis.

        Breaks cycles by retaining only the edges that first discover each node
        during a traversal. The result is always a DAG (spanning forest, one tree
        per weakly-connected component).

        :param mode: Traversal order. ``'bfs'`` (default) produces shallower trees
            (shortest paths from roots); ``'dfs'`` produces deeper trees.

        :returns: A :class:`_SpanningTreeView` exposing all :class:`ExtendedGraphAnalysis`
            metrics over the spanning-tree edges.

        :raises ValueError: If ``mode`` is not ``'bfs'`` or ``'dfs'``.
        """
        if mode not in ("bfs", "dfs"):
            raise ValueError(f"mode must be 'bfs' or 'dfs', got {mode!r}")

        triples = self._factory.mapped_triples
        if self._hierarchy_relation is not None:
            triples = triples[triples[:, 1] == self._hierarchy_relation]
        n = self._num_entities

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for row in triples.tolist():
            adj[int(row[0])].append((int(row[1]), int(row[2])))

        roots = self.root_nodes
        start_nodes = sorted(roots) if roots else list(range(min(1, n)))

        visited: set[int] = set()
        spanning_edges: list[list[int]] = []

        if mode == "bfs":
            queue: deque[tuple[int, int, int | None]] = deque()
            for s in start_nodes:
                if s not in visited:
                    visited.add(s)
                    queue.append((s, -1, None))
            while True:
                while queue:
                    node, rel, parent = queue.popleft()
                    if parent is not None:
                        spanning_edges.append([parent, rel, node])
                    for r, child in adj[node]:
                        if child not in visited:
                            visited.add(child)
                            queue.append((child, r, node))
                next_start = next((i for i in range(n) if i not in visited), None)
                if next_start is None:
                    break
                visited.add(next_start)
                queue.append((next_start, -1, None))

        else:  # dfs
            stack: list[tuple[int, int, int | None]] = [(s, -1, None) for s in reversed(start_nodes)]
            while True:
                while stack:
                    node, rel, parent = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    if parent is not None:
                        spanning_edges.append([parent, rel, node])
                    for r, child in reversed(adj[node]):
                        if child not in visited:
                            stack.append((child, r, node))
                next_start = next((i for i in range(n) if i not in visited), None)
                if next_start is None:
                    break
                stack.append((next_start, -1, None))

        new_triples = (
            torch.tensor(spanning_edges, dtype=torch.long) if spanning_edges else torch.zeros((0, 3), dtype=torch.long)
        )
        new_tf = CoreTriplesFactory.create(
            mapped_triples=new_triples,
            num_entities=n,
            num_relations=self._factory.num_relations,
        )
        return _SpanningTreeView(new_tf)

    # ------------------------------------------------------------------
    # Paper-aligned graph measures (Zloch et al. 2019)
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        """Number of entities (vertices) in the analysed split."""
        return self._num_entities

    @property
    def num_relations(self) -> int:
        """Number of relations in the analysed split."""
        return self._factory.num_relations

    @property
    def total_vertices(self) -> int:
        """Total number of vertices ``|V|`` (Zloch et al. 2019)."""
        return self._num_entities

    @property
    def total_edges(self) -> int:
        """Total number of edges m (Zloch et al. 2019)."""
        return self._factory.num_triples

    @property
    def parallel_edges(self) -> int:
        """Parallel edges m_p: edges sharing the same source and target (Zloch et al. 2019)."""
        return int(self.total_edges - self._digraph.number_of_edges())

    @property
    def unique_edges(self) -> int:
        """Unique edges m_u = m - m_p (Zloch et al. 2019)."""
        return int(self._digraph.number_of_edges())

    def total_degree(self, node_id: int) -> int:
        """Total degree d(v) = d_in(v) + d_out(v) for a single vertex (Zloch et al. 2019)."""
        return int(self._multigraph.degree(node_id))

    def in_degree(self, node_id: int) -> int:
        """In-degree d_in(v) for a single vertex (Zloch et al. 2019)."""
        return int(self._multigraph.in_degree(node_id))

    def out_degree(self, node_id: int) -> int:
        """Out-degree d_out(v) for a single vertex (Zloch et al. 2019)."""
        return int(self._multigraph.out_degree(node_id))

    def node_depth(self, node_id: int) -> int:
        r"""Shortest-path depth of a single vertex from its nearest root node.

        .. math:: \delta(v) = \min_{r \in R} d(r, v)

        where :math:`R` is the set of root nodes (see :attr:`root_nodes`) and
        :math:`d(r, v)` the directed shortest-path distance. Roots have depth 0;
        a node under several roots takes the shallower one. Computed by a single
        multi-source traversal from all roots (:attr:`_node_depths`), so cycles
        never inflate it. Isolated or cycle-only nodes (unreachable from any
        root) are assigned depth 0.

        The mean and max of this quantity over all nodes are
        :attr:`avg_hierarchy_depth` and :attr:`max_hierarchy_depth`.

        :param node_id: The entity ID to query.

        :returns: Shortest-path depth >= 0.

        :raises KeyError: If ``node_id`` is not a valid entity ID.
        """
        return self._node_depths[node_id]

    @property
    def max_degree(self) -> int:
        """Maximum total degree d_max across all vertices (Zloch et al. 2019)."""
        in_deg, out_deg = self._in_out_degrees
        total_deg = in_deg + out_deg
        return int(total_deg.max().item()) if total_deg.numel() > 0 else 0

    @property
    def max_in_degree(self) -> int:
        """Maximum in-degree d_max,in across all vertices (Zloch et al. 2019)."""
        in_deg, _ = self._in_out_degrees
        return int(in_deg.max().item()) if in_deg.numel() > 0 else 0

    @property
    def max_out_degree(self) -> int:
        """Maximum out-degree d_max,out across all vertices (Zloch et al. 2019)."""
        _, out_deg = self._in_out_degrees
        return int(out_deg.max().item()) if out_deg.numel() > 0 else 0

    @property
    def avg_degree(self) -> float:
        """Average total degree z across all vertices (Zloch et al. 2019)."""
        in_deg, out_deg = self._in_out_degrees
        total_deg = in_deg + out_deg
        return float(total_deg.float().mean().item()) if total_deg.numel() > 0 else 0.0

    @property
    def avg_in_degree(self) -> float:
        """Average in-degree z_in across all vertices (Zloch et al. 2019)."""
        in_deg, _ = self._in_out_degrees
        return float(in_deg.float().mean().item()) if in_deg.numel() > 0 else 0.0

    @property
    def avg_out_degree(self) -> float:
        """Average out-degree z_out across all vertices (Zloch et al. 2019)."""
        _, out_deg = self._in_out_degrees
        return float(out_deg.float().mean().item()) if out_deg.numel() > 0 else 0.0

    @property
    def h_index(self) -> int:
        """Directed h-index h_d (Zloch et al. 2019).

        The largest integer h such that at least h entities have in-degree >= h.

        :returns: Directed h-index >= 0.
        """
        in_deg, _ = self._in_out_degrees
        sorted_deg = torch.sort(in_deg, descending=True)[0]
        h = 0
        for i, d in enumerate(sorted_deg.tolist(), start=1):
            if d >= i:
                h = i
            else:
                break
        return h

    @property
    def undirected_h_index(self) -> int:
        """Undirected h-index h_u based on total degree (Zloch et al. 2019)."""
        in_deg, out_deg = self._in_out_degrees
        total_deg = in_deg + out_deg
        sorted_deg = torch.sort(total_deg, descending=True)[0]
        h = 0
        for i, d in enumerate(sorted_deg.tolist(), start=1):
            if d >= i:
                h = i
            else:
                break
        return h

    @property
    def degree_centrality_max(self) -> int:
        """Maximum degree centrality C_D,max, equal to d_max (Zloch et al. 2019)."""
        return self.max_degree

    @property
    def pagerank_max(self) -> float:
        """Maximum PageRank score PR_max (Zloch et al. 2019)."""
        if self._num_entities == 0:
            return 0.0
        if self._factory.mapped_triples.numel() == 0:
            return 1.0 / self._num_entities
        scores = nx.pagerank(self._multigraph)
        return float(max(scores.values(), default=0.0))

    @property
    def graph_centralization(self) -> float:
        """Graph centralization C_D using unique edges (Zloch et al. 2019)."""
        n = self._num_entities
        if n <= 2:
            return 0.0
        in_deg, out_deg = self._unique_in_out_degrees
        total_deg = in_deg + out_deg
        d_max = int(total_deg.max().item()) if total_deg.numel() > 0 else 0
        denominator = (n - 1) * (n - 2)
        if denominator <= 0:
            return 0.0
        return float((d_max * n - total_deg.sum().item()) / denominator)

    @property
    def density(self) -> float:
        """Edge density p = m / n^2 (Zloch et al. 2019).

        :returns: Density in ``[0, 1]``. Returns 0.0 for a graph with no nodes.
        """
        n = self._num_entities
        if n == 0:
            return 0.0
        return self._factory.num_triples / (n * n)

    @property
    def unique_edge_density(self) -> float:
        """Unique-edge density p_u = m_u / n^2 (Zloch et al. 2019).

        Like :attr:`density` but parallel edges are counted only once.

        :returns: Unique-edge density in ``[0, 1]``.
        """
        n = self._num_entities
        if n == 0:
            return 0.0
        return self._digraph.number_of_edges() / (n * n)

    @property
    def reciprocity(self) -> float:
        """Fraction of bidirectional edges y = m_bi / m (Zloch et al. 2019).

        m_bi counts individual edges (incl. parallel multi-relational ones) for
        which a reverse (v, u) also exists. Relation labels are ignored.

        :returns: Reciprocity in ``[0, 1]``. Returns 0.0 for an empty graph.
        """
        m = self.total_edges
        if m == 0:
            return 0.0
        directed_pairs = set(self._digraph.edges())
        m_bi = sum(1 for u, v in self._multigraph.edges() if (v, u) in directed_pairs)
        return m_bi / m

    @functools.cached_property
    def diameter(self) -> int:
        """Graph diameter d: the longest shortest path between any two vertices (Zloch et al. 2019).

        Computed on the undirected projection, as the maximum diameter over all
        connected components (handles disconnected graphs).

        :returns: Graph diameter >= 0.
        """
        undirected = self._digraph.to_undirected()
        return max(
            (nx.diameter(undirected.subgraph(component)) for component in nx.connected_components(undirected)),
            default=0,
        )

    def gromov_hyperbolicity(self, max_sample_nodes: int = 200, seed: int = 0) -> float:
        r"""Gromov 4-point :math:`\delta`-hyperbolicity of the giant component.

        Following Adcock, Sullivan & Mahoney (2013), *"Tree-like structure in large
        social and information networks"* (ICDM), Definition 1: for a quadruplet of
        nodes, take the three perfect-matching sums of pairwise shortest-path distances
        :math:`S_1 \ge S_2 \ge S_3`; the quadruplet's :math:`\delta` is
        :math:`(S_1 - S_2)/2`. The graph's :math:`\delta` is the maximum over all
        quadruplets. Small :math:`\delta` means metrically tree-like (a tree is
        0-hyperbolic); a 4-cycle has :math:`\delta = 1`.

        Distances use the **undirected** shortest-path metric (the paper models networks
        as undirected) and are only finite within a connected component, so the metric
        is computed on the **largest connected component** of :attr:`_digraph`'s
        undirected projection (the paper's "giant component" convention), matching the
        per-component treatment of :attr:`diameter`.

        Exact :math:`\delta` is :math:`O(n^4)` over a full distance matrix, infeasible at
        KG scale, which is why the paper samples quadruplets on large graphs. Here we
        sample ``max_sample_nodes`` **landmark** nodes, compute their pairwise distances
        with a single batched BFS (:func:`scipy.sparse.csgraph.shortest_path`), and take
        the exact :math:`\delta` over that submatrix. When the giant component has at most
        ``max_sample_nodes`` nodes every node is a landmark, so the result is the *exact*
        :math:`\delta`; otherwise it is a lower-bound estimate (the same guarantee
        direction as the paper's quadruplet sampling).

        :param max_sample_nodes: Number of landmark nodes to sample from the giant
            component. The result is exact when the component is no larger than this.
        :param seed: Seed for the landmark sampling RNG (unused when sampling is not
            needed).

        :returns: :math:`\delta \ge 0`, a multiple of 0.5 for unweighted graphs. Returns
            0.0 when the giant component has fewer than four nodes.
        """
        undirected = self._digraph.to_undirected()
        components = list(nx.connected_components(undirected))
        if not components:
            return 0.0
        nodes = list(max(components, key=len))
        if len(nodes) < 4:
            return 0.0
        sub = undirected.subgraph(nodes)
        csr = nx.to_scipy_sparse_array(sub, nodelist=nodes, format="csr")
        if len(nodes) > max_sample_nodes:
            rng = np.random.default_rng(seed)
            landmark_pos = np.sort(rng.choice(len(nodes), size=max_sample_nodes, replace=False))
        else:
            landmark_pos = np.arange(len(nodes))
        dist = shortest_path(csr, method="D", unweighted=True, directed=False, indices=landmark_pos)
        return self._delta_from_distance_matrix(dist[:, landmark_pos])

    @staticmethod
    def _delta_from_distance_matrix(distances: np.ndarray) -> float:
        r"""Exact 4-point :math:`\delta` over a finite pairwise-distance matrix.

        :param distances: A symmetric ``(s, s)`` matrix of finite shortest-path
            distances (all nodes within one connected component).

        :returns: :math:`\max_{\text{quadruplets}} (S_1 - S_2)/2`. Returns 0.0 for fewer
            than four nodes.
        """
        # ponytail: O(s^4) exact over the s-landmark submatrix (semi-vectorised, O(s^2)
        # memory); upgrade path is Cohen-Coudert-Lancin (2015) far-apart-pair pruning if
        # the landmark count must grow past a few hundred.
        s = distances.shape[0]
        if s < 4:
            return 0.0
        triu = np.triu(np.ones((s, s), dtype=bool), k=1)  # strict k < l (each quadruplet once)
        best = 0.0
        for i in range(s):
            di = distances[i]
            for j in range(i + 1, s):
                dj = distances[j]
                # Three perfect matchings of {i, j, k, l}, broadcast over the (k, l) grid.
                a = distances[i, j] + distances  # {i,j}{k,l}
                b = di[:, None] + dj[None, :]  # {i,k}{j,l}
                c = di[None, :] + dj[:, None]  # {i,l}{j,k}
                sums = np.stack((a, b, c))
                sums.sort(axis=0)  # ascending: [-1] largest S1, [-2] second S2
                delta = (sums[-1] - sums[-2]) / 2.0
                valid = triu.copy()
                valid[(i, j), :] = False
                valid[:, (i, j)] = False
                if valid.any():
                    best = max(best, float(delta[valid].max()))
        return best

    @property
    def variance_in_degree(self) -> float:
        """Population variance sigma^2_in of in-degree distribution (Zloch et al. 2019)."""
        in_deg, _ = self._in_out_degrees
        return float(in_deg.float().var(correction=0).item())

    @property
    def variance_out_degree(self) -> float:
        """Population variance sigma^2_out of out-degree distribution (Zloch et al. 2019)."""
        _, out_deg = self._in_out_degrees
        return float(out_deg.float().var(correction=0).item())

    @property
    def std_in_degree(self) -> float:
        """Population standard deviation sigma_in of in-degree distribution (Zloch et al. 2019)."""
        return self.variance_in_degree**0.5

    @property
    def std_out_degree(self) -> float:
        """Population standard deviation sigma_out of out-degree distribution (Zloch et al. 2019)."""
        return self.variance_out_degree**0.5

    @property
    def coefficient_of_variation_in_degree(self) -> float:
        """Coefficient of variation cv_in = (sigma_in / z_in) * 100 (Zloch et al. 2019)."""
        in_deg, _ = self._in_out_degrees
        z_in = float(in_deg.float().mean().item())
        if z_in == 0.0:
            return 0.0
        return (self.std_in_degree / z_in) * 100.0

    @property
    def coefficient_of_variation_out_degree(self) -> float:
        """Coefficient of variation cv_out = (sigma_out / z_out) * 100 (Zloch et al. 2019)."""
        _, out_deg = self._in_out_degrees
        z_out = float(out_deg.float().mean().item())
        if z_out == 0.0:
            return 0.0
        return (self.std_out_degree / z_out) * 100.0

    @staticmethod
    def _fit_power_law(degrees: torch.Tensor) -> tuple[float, int]:
        """Fit a power-law tail and return ``(alpha, d_min)`` (Zloch et al. 2019)."""
        values = degrees.detach().cpu().numpy().astype(float)
        values = values[values > 0]
        if values.size < 3:
            return 0.0, 0
        values = np.sort(values)[::-1]
        ranks = np.arange(1, values.size + 1, dtype=float)
        log_ranks = np.log(ranks)
        log_values = np.log(values)
        slope, _intercept = np.polyfit(log_ranks, log_values, deg=1)
        alpha = float(abs(slope))
        return alpha, int(values.min())

    @property
    def power_law_exponent(self) -> float:
        """Power-law exponent alpha estimated from total degree (Zloch et al. 2019)."""
        in_deg, out_deg = self._in_out_degrees
        alpha, _ = self._fit_power_law(in_deg + out_deg)
        return alpha

    @property
    def power_law_exponent_in(self) -> float:
        """Power-law exponent alpha_in estimated from in-degree (Zloch et al. 2019)."""
        in_deg, _ = self._in_out_degrees
        alpha, _ = self._fit_power_law(in_deg)
        return alpha

    @property
    def power_law_minimum_cutoff(self) -> int:
        """Minimum power-law cutoff d_min estimated from total degree (Zloch et al. 2019)."""
        in_deg, out_deg = self._in_out_degrees
        _, d_min = self._fit_power_law(in_deg + out_deg)
        return d_min


class _SpanningTreeView(ExtendedGraphAnalysis):
    """Lightweight :class:`ExtendedGraphAnalysis` over a spanning-tree triples factory.

    Returned by :meth:`ExtendedGraphAnalysis.spanning_tree`. Exposes all metrics over the
    spanning-tree edge set without wrapping a full dataset.
    """

    def __init__(self, training: CoreTriplesFactory) -> None:
        """Initialize the view directly from a spanning-tree triples factory.

        :param training: The spanning-tree :class:`~pykeen.triples.CoreTriplesFactory`.
        """
        self._factory = training
        self._num_entities = training.num_entities
        self._split = "spanning_tree"
        self._hierarchy_relation = None
