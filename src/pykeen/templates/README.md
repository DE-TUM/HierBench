<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>

<h1 align="center">
  HPBench
</h1>

<p align="center">
  <a href="https://github.com/DE-TUM/HPBench/actions/workflows/common.yml">
    <img src="https://github.com/DE-TUM/HPBench/actions/workflows/common.yml/badge.svg"
         alt="GitHub Actions">
  </a>

  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>

  <a href="https://optuna.org">
    <img src="https://img.shields.io/badge/Optuna-integrated-blue" alt="Optuna integrated" height="20">
  </a>

  <a href="https://pytorchlightning.ai">
    <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
  </a>

  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
  </a>

  <a href=".github/CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant">
  </a>
</p>

<p align="center">
    <b>HPBench</b> (the <b>H</b>eilbronn–<b>P</b>aris <b>Bench</b>mark for Hierarchical Embeddings) is a benchmarking suite for
    hierarchy-aware knowledge graph embeddings. It is a fork of
    <a href="https://github.com/pykeen/pykeen">PyKEEN</a> that adds hyperbolic baselines,
    hierarchical datasets, and hierarchy-specific evaluation tasks.
</p>

<p align="center">
  <a href="#what-hpbench-adds">What HPBench Adds</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#datasets">Datasets ({{ n_datasets }})</a> •
  <a href="#inductive-datasets">Inductive Datasets ({{ n_inductive_datasets }})</a> •
  <a href="#models">Models ({{ n_models }})</a> •
  <a href="#citation">Citation</a>
</p>

## What HPBench Adds

<p align="center">
  <img src="figures/hpbench_overview.png" alt="HPBench overview: PyKEEN components in grey, HPBench contributions in blue, across datasets, preprocessing, learning, and evaluation" width="900">
</p>

On top of everything PyKEEN provides, HPBench contributes:

- **Hyperbolic baselines** — `PoincareE`, `LorentzE`, and `HyperbolicCones`
  (`pykeen.models`), built on hyperbolic representations and interactions in
  `pykeen.nn.hyperbolic`, plus Riemannian optimizers in `pykeen.optimizers`.
- **Hierarchical datasets** — taxonomy-style graphs (WordNet noun hierarchy, ACM-CCS, DOID, MeSH,
  EuroSciVoc, NASA taxonomy, …) with transitive-closure and metadata utilities in
  `pykeen.datasets.metadata` and `pykeen.datasets.extended_graph_analysis`.
- **Hierarchy evaluation tasks** — subsumption / pair classification metrics
  (`pykeen.evaluation.pair_classification_evaluator`), transitive ancestor–descendant prediction
  (`pykeen.pipeline.transitive_ancestor_descendant_prediction_pipeline`), and LCA-based
  hierarchical precision/recall/F1 (`pykeen.evaluation.LCAClassificationEvaluator`).
- **Hierarchy-aware training** — a hierarchy negative sampler
  (`pykeen.sampling.HierarchyNegativeSampler`) and a hierarchy pipeline (`pykeen.pipeline.hierarchy`).
- **k-fold cross-validation** — `pykeen.cross_validation` with k-fold dataset splits.

The import name stays `pykeen`, so existing PyKEEN code runs unchanged.

## Installation

HPBench requires Python 3.9+. Install it from source:

```shell
pip install git+https://github.com/DE-TUM/HPBench.git
```

For development:

```shell
git clone https://github.com/DE-TUM/HPBench.git
cd HPBench
pip install -e .
```

HPBench is not published on PyPI — `pip install pykeen` installs upstream PyKEEN, not this fork.
More information about installation (e.g., Windows installation, Colab, Kaggle, extras)
can be found in the [installation documentation](https://pykeen.readthedocs.io/en/latest/installation.html).

## Quickstart

This example shows how to train a model on a dataset and test on another dataset.

The fastest way to get up and running is to use the pipeline function. It
provides a high-level entry into the extensible functionality of this package.
The following example shows how to train and evaluate the [TransE](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransE.html#pykeen.models.TransE)
model on the [Nations](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Nations.html#pykeen.datasets.Nations)
dataset. By default, the training loop uses the [stochastic local closed world assumption (sLCWA)](https://pykeen.readthedocs.io/en/latest/reference/training.html#pykeen.training.SLCWATrainingLoop)
training approach and evaluates with [rank-based evaluation](https://pykeen.readthedocs.io/en/latest/reference/evaluation/rank_based.html#pykeen.evaluation.RankBasedEvaluator).

```python
from pykeen.pipeline import pipeline

result = pipeline(
    model='TransE',
    dataset='nations',
)
```

The results are returned in an instance of the [PipelineResult](https://pykeen.readthedocs.io/en/latest/reference/pipeline.html#pykeen.pipeline.PipelineResult)
dataclass that has attributes for the trained model, the training loop, the evaluation, and more. See the tutorials
on [using your own dataset](https://pykeen.readthedocs.io/en/latest/byo/data.html),
[understanding the evaluation](https://pykeen.readthedocs.io/en/latest/tutorial/understanding_evaluation.html),
and [making novel link predictions](https://pykeen.readthedocs.io/en/latest/tutorial/making_predictions.html).

HPBench inherits PyKEEN's extensibility:

- Each model has the same API, so anything from ``pykeen.models`` can be dropped in
- Each training loop has the same API, so ``pykeen.training.LCWATrainingLoop`` can be dropped in
- Triples factories can be generated by the user with ``from pykeen.triples.TriplesFactory``

Documentation for the inherited functionality can be found at https://pykeen.readthedocs.io;
links in the tables below point there.

## Implementation

Below are the models, datasets, training modes, evaluators, and metrics available
in HPBench (inherited from PyKEEN unless noted in [What HPBench Adds](#what-hpbench-adds)).

### Datasets 

The following {{ n_datasets }} datasets are built in. The citation for each dataset corresponds to either the paper
describing the dataset, the first paper published using the dataset with knowledge graph embedding models,
or the URL for the dataset if neither of the first two are available. If you want to use a custom dataset,
see the [Bring Your Own Dataset](https://pykeen.readthedocs.io/en/latest/byo/data.html) tutorial. If you
have a suggestion for another dataset to include in PyKEEN, please let us know
[here](https://github.com/pykeen/pykeen/issues/new?assignees=cthoyt&labels=New+Dataset&template=dataset-request.md&title=Add+%5BDATASET+NAME%5D).

{{ datasets }}

### Inductive Datasets

The following {{ n_inductive_datasets }} inductive datasets are built in to PyKEEN.

{{ inductive_datasets }}

### Representations

The following {{ n_representations }} representations are implemented by PyKEEN.

{{ representations }}

### Interactions

The following {{ n_interactions }} interactions are implemented by PyKEEN.

{{ interactions }}

### Models

The following {{ n_models }} models are implemented by PyKEEN.

{{ models }}

### Losses

The following {{ n_losses }} losses are implemented by PyKEEN.

{{ losses }}

### Regularizers

The following {{ n_regularizers }} regularizers are implemented by PyKEEN.

{{ regularizers }}

### Training Loops

The following {{ n_training_loops }} training loops are implemented in PyKEEN.

{{ training_loops }}

### Negative Samplers

The following {{ n_negative_samplers }} negative samplers are implemented in PyKEEN.

{{ negative_samplers }}

### Stoppers

The following {{ n_stoppers }} stoppers are implemented in PyKEEN.

{{ stoppers }}

### Evaluators

The following {{ n_evaluators }} evaluators are implemented in PyKEEN.

{{ evaluators }}

### Metrics

The following {{ n_metrics }} metrics are implemented in PyKEEN.

{{ metrics }}

### Trackers

The following {{ n_trackers }} trackers are implemented in PyKEEN.

{{ trackers }}

## Experimentation

### Reproduction

PyKEEN includes a set of curated experimental settings for reproducing past landmark
experiments. They can be accessed and run like:

```shell
pykeen experiments reproduce tucker balazevic2019 fb15k
```

Where the three arguments are the model name, the reference, and the dataset.
The output directory can be optionally set with `-d`.

### Ablation

PyKEEN includes the ability to specify ablation studies using the
hyper-parameter optimization module. They can be run like:

```shell
pykeen experiments ablation ~/path/to/config.json
```

### Large-scale Reproducibility and Benchmarking Study

We used PyKEEN to perform a large-scale reproducibility and benchmarking study which are described in
[our article](https://doi.org/10.1109/TPAMI.2021.3124805):

```bibtex
@article{ali2020benchmarking,
  author={Ali, Mehdi and Berrendorf, Max and Hoyt, Charles Tapley and Vermue, Laurent and Galkin, Mikhail and Sharifzadeh, Sahand and Fischer, Asja and Tresp, Volker and Lehmann, Jens},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models under a Unified Framework},
  year={2021},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3124805}}
}
```

We have made all code, experimental configurations, results, and analyses that lead to our interpretations available
at https://github.com/pykeen/benchmarking.

## Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated.
See [CONTRIBUTING.md](/CONTRIBUTING.md) for more information on getting involved.

If you have questions about HPBench, please open an issue at
https://github.com/DE-TUM/HPBench/issues. For questions about the upstream library, use the
PyKEEN GitHub discussions at https://github.com/pykeen/pykeen/discussions/new.

## Acknowledgements

HPBench is built on [PyKEEN](https://github.com/pykeen/pykeen); the acknowledgements below
are inherited from that project.

### Supporters

This project has been supported by several organizations (in alphabetical order):

- [Bayer](https://www.bayer.com/)
- [CoronaWhy](https://www.coronawhy.org/)
- [Enveda Biosciences](https://www.envedabio.com/)
- [Fraunhofer Institute for Algorithms and Scientific Computing](https://www.scai.fraunhofer.de)
- [Fraunhofer Institute for Intelligent Analysis and Information Systems](https://www.iais.fraunhofer.de)
- [Fraunhofer Center for Machine Learning](https://www.cit.fraunhofer.de/de/zentren/maschinelles-lernen.html)
- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)
- [Ludwig-Maximilians-Universität München](https://www.en.uni-muenchen.de/index.html)
- [Munich Center for Machine Learning (MCML)](https://mcml.ai/)
- [Siemens](https://new.siemens.com/global/en.html)
- [Smart Data Analytics Research Group (University of Bonn & Fraunhofer IAIS)](https://sda.tech)
- [Technical University of Denmark - DTU Compute - Section for Cognitive Systems](https://www.compute.dtu.dk/english/research/research-sections/cogsys)
- [Technical University of Denmark - DTU Compute - Section for Statistics and Data Analysis](https://www.compute.dtu.dk/english/research/research-sections/stat)
- [University of Bonn](https://www.uni-bonn.de/)

### Funding

The development of PyKEEN has been funded by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Young Faculty Award (PI: Benjamin Gyori)](https://indralab.github.io/#projects)                                              | W911NF2010255   |
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
| German Federal Ministry of Education and Research (BMBF) | [Maschinelles Lernen mit Wissensgraphen (MLWin)](https://mlwin.de)                                                            | 01IS18050D      |
| German Federal Ministry of Education and Research (BMBF) | [Munich Center for Machine Learning (MCML)](https://mcml.ai)                                                                  | 01IS18036A      |
| Innovation Fund Denmark (Innovationsfonden)              | [Danish Center for Big Data Analytics driven Innovation (DABAI)](https://dabai.dk)                                            | Grand Solutions |

### Logo

The PyKEEN logo was designed by [Carina Steinborn](https://www.xing.com/profile/Carina_Steinborn2)

## Citation

If you have found HPBench useful in your work, please cite the underlying PyKEEN
[article](http://jmlr.org/papers/v22/20-825.html) and link to this repository:

```bibtex
@article{ali2021pykeen,
    author = {Ali, Mehdi and Berrendorf, Max and Hoyt, Charles Tapley and Vermue, Laurent and Sharifzadeh, Sahand and Tresp, Volker and Lehmann, Jens},
    journal = {Journal of Machine Learning Research},
    number = {82},
    pages = {1--6},
    title = {% raw %}{{PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings}}{% endraw %},
    url = {http://jmlr.org/papers/v22/20-825.html},
    volume = {22},
    year = {2021}
}
```
