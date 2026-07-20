HPBench
=======

HPBench (the **H**\ eilbronn–**P**\ aris **Bench**\ mark for Hierarchical Embeddings) is a
benchmarking suite for hierarchy-aware knowledge graph embeddings. It is a fork of
`PyKEEN <https://github.com/pykeen/pykeen>`_ that adds hyperbolic baselines
(:class:`pykeen.models.PoincareE`, :class:`pykeen.models.LorentzE`,
:class:`pykeen.models.HyperbolicCones`), hierarchical datasets, and hierarchy-specific
evaluation tasks (pair/subsumption classification, transitive ancestor-descendant prediction, and
LCA-based hierarchical precision/recall/F1). The import name remains ``pykeen``, so existing
PyKEEN code runs unchanged.

.. automodule:: pykeen

.. toctree::
    :caption: Getting Started
    :name: quickstart
    :maxdepth: 2

    installation
    tutorial/first_steps
    tutorial/models
    tutorial/representations
    tutorial/interactions
    tutorial/trackers/index
    tutorial/checkpoints
    tutorial/translational_toy_example
    tutorial/understanding_evaluation
    tutorial/running_hpo
    tutorial/running_ablation
    tutorial/performance
    tutorial/node_piece
    tutorial/metadata_datasets
    tutorial/inductive_lp
    tutorial/splitting
    contrib/lightning
    tutorial/using_resolvers
    tutorial/normalizer_constrainer_regularizer
    tutorial/extended_graph_analysis
    tutorial/cross_validation
    tutorial/troubleshooting

.. toctree::
    :caption: Bring Your Own
    :name: byo
    :maxdepth: 2

    byo/data
    byo/interaction

.. toctree::
    :caption: Extending HPBench
    :name: extending
    :maxdepth: 2

    extending/datasets
    extending/models

.. toctree::
    :caption: Reference
    :name: reference
    :maxdepth: 2

    reference/pipeline
    reference/cross_validation
    reference/models
    reference/datasets
    reference/triples
    reference/training
    reference/stoppers
    reference/losses
    reference/loss_weighting
    reference/regularizers
    reference/trackers
    reference/negative_sampling
    reference/optimizers
    reference/evaluation
    reference/metrics
    reference/hpo
    reference/ablation
    reference/predict
    reference/uncertainty
    reference/sealant
    reference/constants
    reference/checkpoints
    reference/nn/index
    reference/utils

.. toctree::
    :caption: Appendix
    :name: appendix
    :maxdepth: 2

    analysis/index
    references

Indices and Tables
------------------

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
