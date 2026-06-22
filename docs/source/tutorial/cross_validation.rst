K-Fold Cross Validation
=======================

PyKEEN provides a full k-fold cross-validation pipeline through two cooperating
modules:

- :mod:`pykeen.datasets.kfold` — splits any dataset into *k* folds
- :mod:`pykeen.cross_validation` — runs training and evaluation over each fold
  and aggregates the results

Quick Start
-----------

The simplest way to run cross-validation is to pass a regular dataset (by name
or instance) directly to :func:`~pykeen.cross_validation.cross_validation_pipeline`.
It will call :func:`~pykeen.datasets.kfold.base.to_kfold` internally:

.. code-block:: python

    from pykeen.cross_validation import cross_validation_pipeline

    result = cross_validation_pipeline(
        dataset="nations",
        model="TransE",
        k=5,
        epochs=100,
    )

``k`` defaults to 5. The function accepts all the same keyword arguments as the
regular :func:`~pykeen.pipeline.pipeline`, so any model, loss, optimizer,
training loop, or evaluator configuration is valid.

Working with K-Fold Datasets
-----------------------------

To prepare folds manually — for example to inspect them before training — use
:func:`~pykeen.datasets.kfold.base.to_kfold`:

.. code-block:: python

    from pykeen.datasets import Nations
    from pykeen.datasets.kfold import to_kfold
    from pykeen.cross_validation import cross_validation_pipeline

    dataset = Nations()
    kfold = to_kfold(dataset, k=5, validation_ratio=0.1, random_state=42)

    print(f"Folds: {kfold.num_folds}")
    for i, fold in enumerate(kfold):
        print(f"  Fold {i}: {fold.training.num_triples} train triples")

    # Pass the prepared KFoldDataset directly
    result = cross_validation_pipeline(
        dataset=kfold,
        model="TransE",
        epochs=100,
    )

Custom :class:`~pykeen.datasets.kfold.base.KFoldDataset` Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have pre-defined splits (e.g. from a benchmark), implement
:class:`~pykeen.datasets.kfold.base.KFoldDataset`:

.. code-block:: python

    from pykeen.datasets.kfold import KFoldDataset

    class MyBenchmarkFolds(KFoldDataset):
        def _load_folds(self):
            # load and return a list of pykeen.datasets.Dataset objects
            ...

    result = cross_validation_pipeline(dataset=MyBenchmarkFolds(), model="RotatE")

Inspecting Results
------------------

:func:`~pykeen.cross_validation.cross_validation_pipeline` returns a
:class:`~pykeen.cross_validation.CrossValidationPipelineResult`:

.. code-block:: python

    # Mean ± std of a metric across folds
    mean, std = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
    print(f"MRR: {mean:.4f} ± {std:.4f}")

    # Per-fold metrics as a DataFrame
    df = result.to_df()
    print(df)

    # Save everything to disk
    result.save_to_directory("cv_output/")

The output directory will contain:

- ``cv_results.json`` — aggregate means and stds for every metric
- ``fold_metrics.tsv`` — one row per fold with all metric values
- ``fold-000/``, ``fold-001/``, … — full :class:`~pykeen.pipeline.PipelineResult`
  directories per fold (controlled by ``save_fold_results=True``)

API Reference
-------------

- :func:`pykeen.cross_validation.cross_validation_pipeline`
- :class:`pykeen.cross_validation.CrossValidationPipelineResult`
- :func:`pykeen.datasets.kfold.base.to_kfold`
- :class:`pykeen.datasets.kfold.base.KFoldDataset`
- :class:`pykeen.datasets.kfold.base.EagerKFoldDataset`
