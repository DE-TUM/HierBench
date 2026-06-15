"""Hierarchical graph properties for datasets."""

from __future__ import annotations

import statistics
from collections import deque
from typing import Literal

import torch

from ..triples import CoreTriplesFactory


class HierarchyAnalysis:
    """Hierarchical graph properties computed over the training triples.

    Intended to be used as a base class for :class:`pykeen.datasets.base.Dataset`.
    All methods rely solely on ``self.training.mapped_triples`` and
    ``self.num_entities``, both of which are guaranteed by ``Dataset``.
    """

    # ------------------------------------------------------------------
    # Hierarchical graph properties
    # ------------------------------------------------------------------

    @property
    def root_nodes(self) -> frozenset[int]:
        """Top-level nodes: entities that never appear as a tail.

        :returns: Frozenset of entity IDs with in-degree zero.
        """
        tails = set(self.training.mapped_triples[:, 2].tolist())
        return frozenset(i for i in range(self.num_entities) if i not in tails)

    @property
    def leaf_nodes(self) -> frozenset[int]:
        """Bottom-level nodes: entities that never appear as a head.

        :returns: Frozenset of entity IDs with out-degree zero.
        """
        heads = set(self.training.mapped_triples[:, 0].tolist())
        return frozenset(i for i in range(self.num_entities) if i not in heads)

    @property
    def is_dag(self) -> bool:
        """Check whether the training graph is a Directed Acyclic Graph (DAG).

        Runs an iterative depth-first search with three-colour marking.
        No auxiliary structures are stored on ``self``.

        :returns: ``True`` if no directed cycle exists, ``False`` otherwise.
        """
        triples = self.training.mapped_triples
        adj: list[list[int]] = [[] for _ in range(self.num_entities)]
        for h, t in zip(triples[:, 0].tolist(), triples[:, 2].tolist(), strict=False):
            adj[h].append(t)

        color = [0] * self.num_entities  # 0 white, 1 grey, 2 black

        for start in range(self.num_entities):
            if color[start] != 0:
                continue
            stack: list[tuple[int, int]] = [(start, 0)]  # (node, child_index)
            color[start] = 1
            while stack:
                node, idx = stack[-1]
                if idx < len(adj[node]):
                    stack[-1] = (node, idx + 1)
                    child = adj[node][idx]
                    if color[child] == 1:
                        return False  # back-edge → cycle
                    if color[child] == 0:
                        color[child] = 1
                        stack.append((child, 0))
                else:
                    color[node] = 2
                    stack.pop()
        return True

    def _node_depths(self) -> dict[int, int]:
        """Compute the BFS depth of each node from any root node.

        Multi-source BFS starting from all root nodes simultaneously.
        Nodes unreachable from any root are assigned depth 0.

        :returns: Mapping ``{entity_id: depth}`` where depth is the minimum
            number of hops from any root. Not cached; recomputed on each call.
        """
        triples = self.training.mapped_triples
        heads = triples[:, 0]
        tails = triples[:, 2]

        roots = self.root_nodes
        depths: dict[int, int] = dict.fromkeys(roots, 0)
        frontier: set[int] = set(roots)
        current_depth = 0

        while frontier:
            current_depth += 1
            frontier_t = torch.tensor(sorted(frontier), dtype=torch.long)
            mask = torch.isin(heads, frontier_t)
            children = set(tails[mask].tolist())
            next_frontier = children - set(depths.keys())
            for c in next_frontier:
                depths[c] = current_depth
            frontier = next_frontier

        # Nodes unreachable from any root (e.g. isolated or in a pure cycle) → depth 0
        for n in range(self.num_entities):
            depths.setdefault(n, 0)
        return depths

    @property
    def hierarchy_depth(self) -> int:
        """Maximum BFS depth of any node measured from the nearest root node.

        This is **not** the graph diameter δ (longest shortest path between any
        two vertices). It is the deepest level in a BFS tree rooted at the
        set of source nodes (entities with in-degree zero).

        :returns: The largest BFS depth across all nodes. Returns 0 if the
            graph has no edges.
        """
        depths = self._node_depths()
        return max(depths.values(), default=0)

    @property
    def average_hierarchy_depth(self) -> float:
        """Mean BFS depth across all nodes.

        :returns: Average depth from roots. Returns 0.0 for an empty graph.
        """
        depths = self._node_depths()
        if not depths:
            return 0.0
        return sum(depths.values()) / len(depths)

    @property
    def average_fan_out(self) -> float:
        """Average number of children per node (average out-degree).

        Computed as ``total_edges / num_entities``.

        :returns: Mean out-degree. Returns 0.0 for a graph with no nodes.
        """
        if self.num_entities == 0:
            return 0.0
        return int(self.training.num_triples) / self.num_entities

    @property
    def max_fan_out(self) -> int:
        """Maximum number of children any single node has.

        :returns: Maximum out-degree across all entity IDs. Returns 0 for an
            empty graph.
        """
        if self.num_entities == 0:
            return 0
        triples = self.training.mapped_triples
        if triples.numel() == 0:
            return 0
        heads = triples[:, 0]
        counts = torch.bincount(heads, minlength=self.num_entities)
        return int(counts.max().item())

    @property
    def balance(self) -> float:
        """Hierarchy shape uniformity: ``1 − CoV(BFS depths)``, clamped to ``[0, 1]``.

        This is a **custom** hierarchy-shape metric.  It is *not* the paper's
        coefficient of variation cv_in / cv_out (Zloch et al. 2019), which
        measures degree-distribution heterogeneity.  Here CoV is applied to the
        BFS depth values of all nodes, not to degree values.

        A value of ``1.0`` means all nodes share the same BFS depth (maximally
        uniform tree-like structure). Lower values indicate a more skewed or
        imbalanced hierarchy.

        :returns: Balance score in ``[0, 1]``.
        """
        depths = list(self._node_depths().values())
        if len(depths) <= 1:
            return 1.0
        mean = statistics.mean(depths)
        if mean == 0:
            return 1.0
        stdev = statistics.pstdev(depths)
        cov = stdev / mean
        return float(max(0.0, 1.0 - cov))

    @property
    def levels(self) -> int:
        """Alias for :attr:`hierarchy_depth`.

        Returns the maximum BFS depth from root nodes, **not** the graph
        diameter δ (longest shortest path between any two vertices).

        :returns: The largest BFS depth across all nodes. Returns 0 if the graph has no edges.
        """
        return self.hierarchy_depth

    @property
    def average_branch_out(self) -> float:
        """Mean out-degree of non-leaf nodes (nodes with at least one child).

        This is **not** the paper's z_out (Zloch et al. 2019), which is
        the average out-degree over *all* entities (= m / n, see
        :attr:`average_fan_out`).  Excluding leaves inflates this value and
        makes it a hierarchy-specific branching factor rather than a standard
        graph metric.

        :returns: Average out-degree of parent nodes. Returns 0.0 if there are no parent nodes.
        """
        triples = self.training.mapped_triples
        if triples.numel() == 0:
            return 0.0
        heads = triples[:, 0]
        counts = torch.bincount(heads, minlength=self.num_entities)
        parent_counts = counts[counts > 0]
        if parent_counts.numel() == 0:
            return 0.0
        return float(parent_counts.float().mean().item())

    def get_ancestors(self, node_id: int) -> frozenset[int]:
        """Return all ancestors (transitive predecessors) of the given node.

        Performs a BFS backwards through parent edges using only local variables.

        :param node_id: The entity ID to query.

        :returns: Frozenset of entity IDs that have a directed path to
            ``node_id``. Returns an empty frozenset for root nodes.
        """
        triples = self.training.mapped_triples
        heads = triples[:, 0]
        tails = triples[:, 2]

        visited: set[int] = set()
        frontier: set[int] = {node_id}

        while frontier:
            frontier_t = torch.tensor(sorted(frontier), dtype=torch.long)
            mask = torch.isin(tails, frontier_t)
            parents = set(heads[mask].tolist())
            next_frontier = parents - visited - {node_id}
            visited.update(next_frontier)
            frontier = next_frontier

        return frozenset(visited)

    def get_descendants(self, node_id: int) -> frozenset[int]:
        """Return all descendants (transitive successors) of the given node.

        Performs a BFS forward through child edges using only local variables.

        :param node_id: The entity ID to query.

        :returns: Frozenset of entity IDs reachable from ``node_id`` via
            directed edges. Returns an empty frozenset for leaf nodes.
        """
        triples = self.training.mapped_triples
        heads = triples[:, 0]
        tails = triples[:, 2]

        visited: set[int] = set()
        frontier: set[int] = {node_id}

        while frontier:
            frontier_t = torch.tensor(sorted(frontier), dtype=torch.long)
            mask = torch.isin(heads, frontier_t)
            children = set(tails[mask].tolist())
            next_frontier = children - visited - {node_id}
            visited.update(next_frontier)
            frontier = next_frontier

        return frozenset(visited)

    def nearest_common_ancestor(self, node_a: int, node_b: int) -> int | None:
        """Compute the nearest (lowest) common ancestor of two nodes.

        Finds the deepest node that is an ancestor of both inputs.
        Works on general DAGs with multiple roots.

        :param node_a: First entity ID.
        :param node_b: Second entity ID.

        :returns: The entity ID of the nearest common ancestor, or ``None``
            if no common ancestor exists (e.g. disconnected components).
        """
        if node_a == node_b:
            return node_a
        ancestors_a = self.get_ancestors(node_a) | {node_a}
        ancestors_b = self.get_ancestors(node_b) | {node_b}
        common = ancestors_a & ancestors_b
        if not common:
            return None
        depths = self._node_depths()
        return max(common, key=lambda n: depths.get(n, 0))

    def spanning_tree(self, mode: Literal["bfs", "dfs"] = "bfs") -> "_SpanningTreeView":
        """Return a spanning-tree view of this graph for hierarchical analysis.

        Breaks cycles by retaining only the edges that first discover each node
        during a traversal. The result is always a DAG (spanning forest — one
        tree per weakly-connected component).

        :param mode: Traversal order used to build the spanning tree.
            ``'bfs'`` (default) produces shallower trees (shortest paths from
            roots); ``'dfs'`` produces deeper trees that follow long chains first.
        :returns: A :class:`_SpanningTreeView` exposing all
            :class:`HierarchyAnalysis` properties over the spanning-tree edges.
        :raises ValueError: If ``mode`` is not ``'bfs'`` or ``'dfs'``.
        """
        if mode not in ("bfs", "dfs"):
            raise ValueError(f"mode must be 'bfs' or 'dfs', got {mode!r}")

        triples = self.training.mapped_triples
        n = self.num_entities

        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for row in triples.tolist():
            adj[int(row[0])].append((int(row[1]), int(row[2])))

        roots = self.root_nodes
        start_nodes = sorted(roots) if roots else [0]

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
            stack: list[tuple[int, int, int | None]] = [
                (s, -1, None) for s in reversed(start_nodes)
            ]
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
            torch.tensor(spanning_edges, dtype=torch.long)
            if spanning_edges
            else torch.zeros((0, 3), dtype=torch.long)
        )
        new_tf = CoreTriplesFactory.create(
            mapped_triples=new_triples,
            num_entities=n,
            num_relations=self.training.num_relations,
        )
        return _SpanningTreeView(training=new_tf)


class _SpanningTreeView(HierarchyAnalysis):
    """Lightweight view over a spanning-tree :class:`CoreTriplesFactory`.

    Returned by :meth:`HierarchyAnalysis.spanning_tree`. Exposes all
    hierarchical properties from :class:`HierarchyAnalysis` over the
    spanning-tree edge set without wrapping a full dataset.
    """

    def __init__(self, training: CoreTriplesFactory) -> None:
        self.training = training

    @property
    def num_entities(self) -> int:  # noqa: D401
        """Number of entities (preserved from the original graph)."""
        return self.training.num_entities
