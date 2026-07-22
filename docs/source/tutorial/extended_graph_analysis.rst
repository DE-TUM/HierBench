Analyzing Graph & Hierarchy Properties
=======================================

The :class:`pykeen.datasets.extended_graph_analysis.ExtendedGraphAnalysis` class provides a
comprehensive suite of graph and hierarchy metrics for any PyKEEN dataset split.
It wraps a `NetworkX <https://networkx.org/>`_ multigraph and exposes over 30 properties
covering hierarchical structure, degree statistics, and graph measures aligned with
`Zloch et al., 2019 <https://arxiv.org/abs/1906.10433>`_.

Getting Started
---------------

Import the class and pass any PyKEEN :class:`pykeen.datasets.Dataset` to it:

.. code-block:: python

    from pykeen.datasets import Nations
    from pykeen.datasets.extended_graph_analysis import ExtendedGraphAnalysis

    dataset = Nations()
    analysis = ExtendedGraphAnalysis(dataset, split="train")

The ``split`` parameter accepts ``"train"``, ``"test"``, ``"validation"``, or
``"full"`` (which merges all three splits).

Hierarchical Metrics
--------------------

These properties characterise the dataset's tree-like structure.

.. code-block:: python

    # Nodes with no incoming edges (root of the hierarchy)
    print(analysis.root_nodes)       # frozenset of entity IDs

    # Nodes with no outgoing edges (leaves)
    print(analysis.leaf_nodes)       # frozenset of entity IDs

    # Is the graph a directed acyclic graph?
    print(analysis.is_dag)           # bool

    # Depth of the longest path from any root to any node
    print(analysis.max_hierarchy_depth)   # int  (alias: .levels)
    print(analysis.avg_hierarchy_depth)   # float

    # Out-degree statistics
    print(analysis.average_fan_out)   # = num_triples / num_entities
    print(analysis.max_fan_out)       # maximum out-degree
    print(analysis.average_branch_out)  # mean out-degree of non-leaf nodes

    # Balance: 1.0 = perfectly uniform depth, 0.0 = highly unbalanced
    print(analysis.balance)          # float in [0, 1]

Ancestry Queries
----------------

.. code-block:: python

    node_id = 3

    ancestors = analysis.get_ancestors(node_id)       # frozenset
    descendants = analysis.get_descendants(node_id)   # frozenset

    # Nearest (lowest) common ancestor of two nodes, or None
    nca = analysis.nearest_common_ancestor(2, 7)

Spanning Tree
-------------

For datasets that contain cycles (i.e. :attr:`~ExtendedGraphAnalysis.is_dag` is
``False``), a spanning tree can be extracted to obtain a cycle-free view:

.. code-block:: python

    bfs_tree = analysis.spanning_tree(mode="bfs")  # breadth-first (shallow)
    dfs_tree = analysis.spanning_tree(mode="dfs")  # depth-first (deep)

    print(bfs_tree.is_dag)              # always True
    print(bfs_tree.max_hierarchy_depth) # usually < dfs_tree.max_hierarchy_depth

Both return a :class:`~pykeen.datasets.extended_graph_analysis._SpanningTreeView`,
which exposes the full :class:`ExtendedGraphAnalysis` API over the cycle-free edges.

Graph Measures (Zloch et al., 2019)
------------------------------------

In addition to hierarchy-specific properties, the class exposes general graph
statistics following the measures described in *Zloch et al., 2019*.

.. code-block:: python

    # Size
    print(analysis.total_vertices)    # number of entities
    print(analysis.total_edges)       # total (multi-)edges
    print(analysis.unique_edges)      # edges after deduplication
    print(analysis.parallel_edges)    # total_edges - unique_edges

    # Degree statistics
    print(analysis.average_degree)
    print(analysis.average_in_degree)
    print(analysis.average_out_degree)
    print(analysis.max_degree)
    print(analysis.max_in_degree)
    print(analysis.max_out_degree)

    # Degree distribution
    print(analysis.degree_variance_in)
    print(analysis.degree_std_out)
    print(analysis.coefficient_of_variation_in)
    print(analysis.power_law_exponent)   # fit to out-degree distribution

    # Centrality & density
    print(analysis.density)              # edge density of the simple digraph
    print(analysis.unique_edge_density)
    print(analysis.graph_centralization)
    print(analysis.degree_centrality_max)
    print(analysis.pagerank_max)

    # Connectivity
    print(analysis.reciprocity)   # fraction of bidirectional edges
    print(analysis.diameter)      # longest shortest path
    print(analysis.h_index)       # directed h-index
    print(analysis.undirected_h_index)

Gromov Hyperbolicity
--------------------

The :meth:`~ExtendedGraphAnalysis.gromov_hyperbolicity` method measures how
*metrically tree-like* the graph is, following the 4-point :math:`\delta`
definition of `Adcock, Sullivan & Mahoney, 2013
<https://doi.org/10.1109/ICDM.2013.77>`_. Small :math:`\delta` means tree-like
(a tree is 0-hyperbolic); a 4-cycle has :math:`\delta = 1`.

.. code-block:: python

    # delta over the giant component of the undirected projection
    print(analysis.gromov_hyperbolicity())            # float, a multiple of 0.5

    # exact when the giant component has <= max_sample_nodes nodes,
    # otherwise a lower-bound estimate over sampled landmark quadruplets
    print(analysis.gromov_hyperbolicity(max_sample_nodes=400, seed=1))

Exact :math:`\delta` is :math:`O(n^4)`, so on large graphs the method samples
``max_sample_nodes`` landmark nodes and returns the exact :math:`\delta` over
that submatrix — a lower bound on the true value, matching the paper's
quadruplet-sampling scheme. Firm up the bound by taking the maximum across
several seeds.

Per-Node Degree Access
----------------------

.. code-block:: python

    print(analysis.total_degree(node_id))
    print(analysis.in_degree(node_id))
    print(analysis.out_degree(node_id))

API Reference
-------------

See :mod:`pykeen.datasets.extended_graph_analysis` for the full API.
