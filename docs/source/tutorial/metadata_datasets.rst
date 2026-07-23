Datasets with Entity Metadata
==============================

Several datasets in PyKEEN ship with per-entity metadata alongside the
knowledge graph triples. These datasets inherit from
:class:`~pykeen.datasets.MetadataDataset` and expose the metadata as a
:class:`pandas.DataFrame` via the :attr:`~pykeen.datasets.MetadataDataset.entity_metadata`
attribute.

Available metadata datasets include:

- :class:`~pykeen.datasets.Cora` — bag-of-words feature vector per paper
- :class:`~pykeen.datasets.Crocodile` — Wikipedia page features
- :class:`~pykeen.datasets.Squirrel` — Wikipedia page features and traffic labels
- :class:`~pykeen.datasets.DOID` — disease term attributes
- :class:`~pykeen.datasets.PPI` — protein feature vectors
- :class:`~pykeen.datasets.PubMed` — article word features

Accessing Entity Metadata
--------------------------

Instantiate any metadata-aware dataset and read its
:attr:`~pykeen.datasets.MetadataDataset.entity_metadata` DataFrame:

.. code-block:: python

    from pykeen.datasets import Cora

    dataset = Cora()
    df = dataset.entity_metadata

    print(df.shape)    # (num_entities, num_feature_columns)
    print(df.dtypes)   # inspect column types
    print(df.head())

The DataFrame has one row per entity, ordered by entity ID (i.e. row *i*
corresponds to the entity with integer ID *i* inside the
:class:`~pykeen.triples.TriplesFactory`).

Joining Embeddings with Metadata
----------------------------------

After training, you can align learned embeddings with metadata by entity ID:

.. code-block:: python

    import torch
    import pandas as pd
    from pykeen.datasets import Cora
    from pykeen.pipeline import pipeline

    dataset = Cora()

    result = pipeline(
        dataset=dataset,
        model="TransE",
        epochs=100,
    )

    # Extract entity embeddings as a numpy array (shape: num_entities × dim)
    embeddings = (
        result.model.entity_representations[0]()
        .detach()
        .cpu()
        .numpy()
    )

    # Join with metadata
    meta_df = dataset.entity_metadata.copy()
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    combined = pd.concat([meta_df, emb_df], axis=1)

    print(combined.head())

Using Only Numeric Features
-----------------------------

For downstream tasks (e.g. node classification) you often need only the
numeric columns:

.. code-block:: python

    numeric_cols = dataset.entity_metadata.select_dtypes("number").columns.tolist()
    features = torch.tensor(
        dataset.entity_metadata[numeric_cols].values,
        dtype=torch.float32,
    )
    print(features.shape)  # (num_entities, num_numeric_features)

Implementing Your Own Metadata Dataset
----------------------------------------

To wrap a custom dataset with metadata, subclass
:class:`~pykeen.datasets.SingleFileRemoteMetadataDataset` and set the URL class attributes:

.. code-block:: python

    from pykeen.datasets.metadata import SingleFileRemoteMetadataDataset

    class MyDataset(SingleFileRemoteMetadataDataset):
        triples_url         = "https://example.com/edges.tsv"
        entity_metadata_url = "https://example.com/node_features.tsv"
        ratios              = (0.8, 0.1, 0.1)  # train / test / val

PyKEEN will download and cache the files automatically on first use. Override
:meth:`~pykeen.datasets.SingleFileRemoteMetadataDataset._load_entity_metadata` to support
non-standard file formats (NumPy, HDF5, pickle, etc.).

API Reference
--------------

- :class:`pykeen.datasets.MetadataDataset`
- :class:`pykeen.datasets.SingleFileRemoteMetadataDataset`
