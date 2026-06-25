"""Test the PyKEEN pipeline function."""

import itertools
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest
import torch

import pykeen.regularizers
from pykeen.datasets import EagerDataset, Nations
from pykeen.models import ERModel, FixedModel, Model
from pykeen.models.resolve import DimensionError, make_model, make_model_cls
from pykeen.nn.modules import TransEInteraction
from pykeen.nn.representation import Embedding
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.pipeline.api import replicate_pipeline_from_config
from pykeen.regularizers import NoRegularizer
from pykeen.sampling.negative_sampler import NegativeSampler
from pykeen.training import SLCWATrainingLoop
from pykeen.triples.generation import generate_triples_factory
from pykeen.triples.triples_factory import CoreTriplesFactory, TriplesFactory
from pykeen.utils import resolve_device

from .utils import needs_packages


class TestPipelineTriples(unittest.TestCase):
    """Test applying the pipeline to triples factories."""

    def setUp(self) -> None:
        """Prepare the training, testing, and validation triples factories."""
        self.base_tf = generate_triples_factory(
            num_entities=50,
            num_relations=9,
            num_triples=500,
        )
        self.training, self.testing, self.validation = self.base_tf.split([0.8, 0.1, 0.1])

    def test_unlabeled_triples(self):
        """Test running the pipeline on unlabeled triples factories."""
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model="TransE",
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
        )

    def test_eager_unlabeled_dataset(self):
        """Test running the pipeline on unlabeled triples factories in a dataset."""
        dataset = EagerDataset(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
        )
        _ = pipeline(
            dataset=dataset,
            model="TransE",
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
        )

    def test_interaction_instance_missing_dimensions(self):
        """Test when a dimension is missing."""
        with pytest.raises(DimensionError) as exc:
            make_model_cls(
                dimensions={},  # missing "d"
                interaction=TransEInteraction(p=2),
            )
        assert isinstance(exc.value, DimensionError)
        assert {"d"} == exc.value.expected
        assert set() == exc.value.given
        assert str(exc.value) == "Expected dimensions dictionary with keys {'d'} but got keys set()"

    def test_interaction_instance_builder(self):
        """Test resolving an interaction model instance."""
        model = make_model(
            dimensions={"d": 3},
            interaction=TransEInteraction,
            interaction_kwargs={"p": 2},
            triples_factory=self.training,
        )
        assert isinstance(model, ERModel)
        assert isinstance(model.interaction, TransEInteraction)
        assert model.interaction.p == 2
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model=model,
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
            random_seed=0,
        )

    def test_interaction_builder(self):
        """Test resolving an interaction model."""
        model_cls = make_model_cls({"d": 3}, TransEInteraction(p=2))
        self._help_test_interaction_resolver(model_cls)

    def test_interaction_resolver_cls(self):
        """Test resolving the interaction function."""
        model_cls = make_model_cls({"d": 3}, TransEInteraction, {"p": 2})
        self._help_test_interaction_resolver(model_cls)

    def test_interaction_resolver_lookup(self):
        """Test resolving the interaction function."""
        model_cls = make_model_cls({"d": 3}, "TransE", {"p": 2})
        self._help_test_interaction_resolver(model_cls)

    def _help_test_interaction_resolver(self, model_cls):
        assert issubclass(model_cls, ERModel)
        assert isinstance(model_cls._interaction, TransEInteraction)
        assert model_cls._interaction.p == 2
        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            model=model_cls,
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
            random_seed=0,
        )

    def test_custom_training_loop(self):
        """Test providing a custom training loop."""
        losses = []

        class ModifiedTrainingLoop(SLCWATrainingLoop):
            """A wrapper around SLCWA training loop which remembers batch losses."""

            def _forward_pass(self, *args, **kwargs):  # noqa: D102
                loss = super()._forward_pass(*args, **kwargs)
                losses.append(loss)
                return loss

        _ = pipeline(
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            training_loop=ModifiedTrainingLoop,
            model="TransE",
            training_kwargs={"num_epochs": 1, "use_tqdm": False},
            evaluation_kwargs={"use_tqdm": False},
            random_seed=0,
        )

        # empty lists are falsy
        assert losses

    @needs_packages("matplotlib", "seaborn")
    def test_plot(self):
        """Test plotting."""
        result = pipeline(dataset="nations", model="transe", training_kwargs={"num_epochs": 0})
        fig, axes = result.plot()
        assert fig is not None
        assert axes is not None

    def test_with_evaluation_loop_callback(self):
        """Smoke-Test for running pipeline with evaluation loop callback."""
        dataset = Nations()
        result = pipeline(
            dataset=dataset,
            model="mure",
            training_kwargs={
                "num_epochs": 2,
                "callbacks": "evaluation-loop",
                "callbacks_kwargs": {
                    "frequency": 1,
                    "prefix": "validation",
                    "factory": dataset.validation,
                    "additional_filter_triples": dataset.training,
                },
            },
        )
        assert result is not None


class TestPipelineReplicate(unittest.TestCase):
    """Test the replication with pipeline."""

    def setUp(self) -> None:  # noqa: D102
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = pathlib.Path(self.tmp_dir.name)

    def tearDown(self) -> None:  # noqa: D102
        self.tmp_dir.cleanup()

    def test_replicate_pipeline_from_config(self):
        """Test replication from config."""
        replicate_pipeline_from_config(
            config={
                "metadata": {},
                "pipeline": {
                    "dataset": "nations",
                    "model": "transe",
                },
                "results": {
                    "best": {"hits_at_k": {"10": 0.538}, "mean_rank": 163},
                },
            },
            directory=self.tmp_dir_path,
            replicates=1,
        )


class TestPipelineCheckpoints(unittest.TestCase):
    """Test the pipeline with checkpoints."""

    def setUp(self) -> None:
        """Set up a shared result as standard to compare to."""
        self.random_seed = 123
        self.model = "TransE"
        self.dataset = "nations"
        self.checkpoint_name = "PyKEEN_training_loop_test_checkpoint.pt"
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.temporary_directory.cleanup()

    def test_pipeline_resumption(self):
        """Test whether the resumed LCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type="LCWA")

    def test_pipeline_slcwa_resumption(self):
        """Test whether the resumed sLCWA pipeline creates the same results as the one shot pipeline."""
        self._test_pipeline_x_resumption(training_loop_type="sLCWA")

    def _test_pipeline_x_resumption(self, training_loop_type: str):
        """Test whether the resumed pipeline creates the same results as the one shot pipeline."""
        # As the resumption capability currently is a function of the training loop, more thorough tests can be found
        # in the test_training.py unit tests. In the tests below the handling of training loop checkpoints by the
        # pipeline is checked.

        result_standard = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs={"num_epochs": 10, "use_tqdm": False, "use_tqdm_batch": False},
            random_seed=self.random_seed,
        )

        # Set up a shared result that runs two pipelines that should replicate the results of the standard pipeline.
        _ = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs={
                "num_epochs": 5,
                "use_tqdm": False,
                "use_tqdm_batch": False,
                "checkpoint_name": self.checkpoint_name,
                "checkpoint_directory": self.temporary_directory.name,
                "checkpoint_frequency": 0,
            },
            random_seed=self.random_seed,
        )

        # Resume the previous pipeline
        result_split = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop=training_loop_type,
            training_kwargs={
                "num_epochs": 10,
                "use_tqdm": False,
                "use_tqdm_batch": False,
                "checkpoint_name": self.checkpoint_name,
                "checkpoint_directory": self.temporary_directory.name,
                "checkpoint_frequency": 0,
            },
        )
        assert result_standard.losses == result_split.losses


class TestAttributes(unittest.TestCase):
    """Test that the keywords given to the pipeline make it through."""

    def test_specify_regularizer(self):
        """Test a pipeline that uses a regularizer."""
        for regularizer, cls in [
            (None, None),
            ("no", pykeen.regularizers.NoRegularizer),
            (NoRegularizer, pykeen.regularizers.NoRegularizer),
            ("powersum", pykeen.regularizers.PowerSumRegularizer),
            ("lp", pykeen.regularizers.LpRegularizer),
        ]:
            with self.subTest(regularizer=regularizer):
                pipeline_result = pipeline(
                    model="TransE",
                    dataset="Nations",
                    regularizer=regularizer,
                    training_kwargs={"num_epochs": 1},
                )
                assert isinstance(pipeline_result, PipelineResult)
                assert isinstance(pipeline_result.model, Model)
                for r in itertools.chain(
                    pipeline_result.model.entity_representations, pipeline_result.model.relation_representations
                ):
                    if isinstance(r, Embedding):
                        if cls is None:
                            assert r.regularizer is None
                        else:
                            assert isinstance(r.regularizer, cls)


class TestPipelineEvaluationFiltering(unittest.TestCase):
    """Test filtering of triples during evaluation using the pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up a shared result."""
        cls.device = resolve_device("cuda")
        cls.dataset = Nations()

        cls.model = FixedModel(triples_factory=cls.dataset.training)

        # The MockModel gives the highest score to the highest entity id
        max_score = cls.dataset.num_entities - 1

        # The test triples are created to yield the third highest score on both head and tail prediction
        cls.dataset.testing.mapped_triples = torch.tensor([[max_score - 2, 0, max_score - 2]])

        # Write new mapped triples to the model, since the model's triples will be used to filter
        # These triples are created to yield the highest score on both head and tail prediction for the
        # test triple at hand
        cls.dataset.training.mapped_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score],
                [max_score, 0, max_score - 2],
            ],
        )

        # The validation triples are created to yield the second highest score on both head and tail prediction for the
        # test triple at hand
        cls.dataset.validation.mapped_triples = torch.tensor(
            [
                [max_score - 2, 0, max_score - 1],
                [max_score - 1, 0, max_score - 2],
            ],
        )

    def test_pipeline_evaluation_filtering_without_validation_triples(self):
        """Test if the evaluator's triple filtering works as expected using the pipeline."""
        results = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop_kwargs={"automatic_memory_optimization": False},
            training_kwargs={"num_epochs": 0, "use_tqdm": False},
            evaluator_kwargs={"filtered": True},
            evaluation_kwargs={"use_tqdm": False},
            device=self.device,
            random_seed=42,
            filter_validation_when_testing=False,
        )
        assert results.metric_results.get_metric("mr") == 2, "The rank should equal 2"

    def test_pipeline_evaluation_filtering_with_validation_triples(self):
        """Test if the evaluator's triple filtering with validation triples works as expected using the pipeline."""
        results = pipeline(
            model=self.model,
            dataset=self.dataset,
            training_loop_kwargs={"automatic_memory_optimization": False},
            training_kwargs={"num_epochs": 0, "use_tqdm": False},
            evaluator_kwargs={"filtered": True},
            evaluation_kwargs={"use_tqdm": False},
            device=self.device,
            random_seed=42,
            filter_validation_when_testing=True,
        )
        assert results.metric_results.get_metric("mr") == 1, "The rank should equal 1"


def test_negative_sampler_kwargs():
    """Test whether negative sampler kwargs are correctly passed through."""
    # cf. https://github.com/pykeen/pykeen/issues/1118

    _num_neg_per_pos = 100

    # save a reference to the old init *before* mocking
    old_init = NegativeSampler.__init__

    def mock_init(*args, **kwargs):
        """Mock init method to check if kwarg arrives."""
        assert kwargs.get("num_negs_per_pos") == _num_neg_per_pos
        old_init(*args, **kwargs)

    # run a small pipline
    with mock.patch.object(NegativeSampler, "__init__", mock_init):
        pipeline(
            # use sampled training loop ...
            training_loop="slcwa",
            # ... without explicitly selecting a negative sampler ...
            negative_sampler=None,
            # ... but providing custom kwargs
            negative_sampler_kwargs={"num_negs_per_pos": _num_neg_per_pos},
            # other parameters for fast test
            dataset="nations",
            model="distmult",
            epochs=0,
        )


@pytest.mark.parametrize("tf_cls", [CoreTriplesFactory, TriplesFactory])
def test_loading_training_triples_factory(tf_cls: type[CoreTriplesFactory]):
    """Test re-loading the training triples factory."""
    result = pipeline(model="rescal", dataset="nations", training_kwargs={"num_epochs": 0})
    with tempfile.TemporaryDirectory() as directory:
        result.save_to_directory(directory)
        tf_cls.from_path_binary(pathlib.Path(directory, "training_triples"))


def _make_chain_dataset() -> EagerDataset:
    """Return a tiny 4-node chain dataset: 0→1→2→3.

    Multi-hop ancestors: 0 is a 2-hop ancestor of 2, and a 3-hop ancestor of 3;
    1 is a 2-hop ancestor of 3.
    """
    from pykeen.triples import CoreTriplesFactory

    # train: direct edges 0→1, 1→2, 2→3
    train_triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=torch.long)
    train = CoreTriplesFactory(mapped_triples=train_triples, num_entities=4, num_relations=1)
    # validation/test: use the same factory as placeholder; pipeline overrides testing anyway
    return EagerDataset(training=train, testing=train, validation=train)


def _make_diamond_dataset() -> EagerDataset:
    """Return a 4-node diamond dataset with a direct shortcut edge 0→3.

    Edges: 0→1, 0→2, 1→3, 2→3, 0→3.
    All five edges are direct edges and always in training; TC-only pairs form the holdout.
    """
    train_triples = torch.tensor(
        [[0, 0, 1], [0, 0, 2], [1, 0, 3], [2, 0, 3], [0, 0, 3]],
        dtype=torch.long,
    )
    train = CoreTriplesFactory(mapped_triples=train_triples, num_entities=4, num_relations=1)
    return EagerDataset(training=train, testing=train, validation=train)


def _make_balanced_tree_dataset() -> EagerDataset:
    """Return a depth-3 complete binary tree (15 nodes, no shortcuts).

    Node ``i`` has children ``2i+1`` and ``2i+2`` for ``i`` in 0..6; nodes 7..14 are leaves.
    Because the graph is a tree, the hop distance of any ancestor-descendant pair equals its
    unique shortest-path length, which makes per-hop sampling balance verifiable.
    """
    rows = [[i, 0, child] for i in range(7) for child in (2 * i + 1, 2 * i + 2)]
    train_triples = torch.tensor(rows, dtype=torch.long)
    train = CoreTriplesFactory(mapped_triples=train_triples, num_entities=15, num_relations=1)
    return EagerDataset(training=train, testing=train, validation=train)


class TestAncestorDescendantPipeline(unittest.TestCase):
    """Tests for the ancestor-descendant benchmark pipeline."""

    def test_pipeline_runs(self):
        """Pipeline completes and returns a valid MRR score on a small chain dataset."""
        from pykeen.pipeline.hierarchy import ancestor_descendant_pipeline

        dataset = _make_chain_dataset()
        result = ancestor_descendant_pipeline(dataset, epochs=1)
        assert result.metric_results is not None
        mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
        assert 0.0 <= mrr <= 1.0

    def test_sampled_pairs_are_valid_multihop(self):
        """Every sampled val/test pair is a reachable, non-direct ancestor-descendant pair."""
        import networkx as nx

        from pykeen.pipeline.hierarchy import ancestor_descendant_split

        dataset = _make_chain_dataset()
        direct = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(dataset.num_entities))
        graph.add_edges_from(direct)

        _train, val, test = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=10, seed=0)
        eval_pairs = {(h, t) for h, _, t in (val.mapped_triples.tolist() + test.mapped_triples.tolist())}
        # On a 4-node chain the only multi-hop pairs that exist are these three.
        assert eval_pairs <= {(0, 2), (0, 3), (1, 3)}
        for h, t in eval_pairs:
            assert h != t
            assert (h, t) not in direct
            assert nx.has_path(graph, h, t)

    def test_poincare_scores_are_non_positive(self):
        """Poincaré interaction scores are always ≤ 0 (negative Poincaré distance).

        Self-pairs score exactly 0; all other pairs score strictly below 0.
        """
        from pykeen.nn.modules import PoincareEInteraction

        interaction = PoincareEInteraction(curvature=1.0)
        # Embeddings strictly inside the unit ball
        embeddings = torch.tensor(
            [[0.0, 0.0], [0.1, 0.1], [0.3, -0.2], [-0.5, 0.4]],
            dtype=torch.float32,
        )

        def score(h_idx: int, t_idx: int) -> float:
            """Score a single (head, tail) pair using Poincaré interaction."""
            return interaction.forward(embeddings[h_idx].unsqueeze(0), (), embeddings[t_idx].unsqueeze(0)).item()

        # Self-pairs: distance to itself is 0 → score = 0
        for i in range(len(embeddings)):
            assert score(i, i) == pytest.approx(0.0, abs=1e-5), f"Self-score for entity {i} should be 0"

        # Cross-pairs: all other scores must be strictly negative
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:
                    assert score(i, j) < 0.0, f"score({i}, {j}) should be negative"

    def test_train_is_exactly_direct_edges(self):
        """Training contains exactly the direct graph edges, regardless of the pair budget."""
        from pykeen.pipeline.hierarchy import ancestor_descendant_split

        dataset = _make_diamond_dataset()
        direct_edges = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}

        for num_pairs in (0, 4, 50):
            train, _val, _test = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=num_pairs, seed=42)
            train_pairs = {(h, t) for h, _, t in train.mapped_triples.tolist()}
            assert train_pairs == direct_edges, f"train != direct edges at num_pairs={num_pairs}"

    def test_hierarchy_splits_no_leakage(self):
        """No sampled eval pair appears in training, and validation/test are disjoint."""
        from pykeen.pipeline.hierarchy import ancestor_descendant_split

        dataset = _make_balanced_tree_dataset()
        train, val, test = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=12, seed=42)
        train_pairs = {(h, t) for h, _, t in train.mapped_triples.tolist()}
        val_pairs = {(h, t) for h, _, t in val.mapped_triples.tolist()}
        test_pairs = {(h, t) for h, _, t in test.mapped_triples.tolist()}
        assert train_pairs.isdisjoint(val_pairs | test_pairs)
        assert val_pairs.isdisjoint(test_pairs)

    def test_hop_balance(self):
        """Sampled pairs match the requested hop distances and are balanced across hops."""
        from collections import Counter

        import networkx as nx

        from pykeen.pipeline.hierarchy import ancestor_descendant_split

        dataset = _make_balanced_tree_dataset()
        graph = nx.DiGraph()
        graph.add_nodes_from(range(dataset.num_entities))
        graph.add_edges_from((h, t) for h, _, t in dataset.training.mapped_triples.tolist())

        # The tree supplies 12 two-hop and 8 three-hop pairs, so a budget of 12 (6 per hop)
        # is fully satisfiable and must come out exactly balanced.
        _train, val, test = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=12, seed=42)
        eval_pairs = [(h, t) for h, _, t in (val.mapped_triples.tolist() + test.mapped_triples.tolist())]
        hop_counts = Counter(nx.shortest_path_length(graph, h, t) for h, t in eval_pairs)
        assert set(hop_counts) <= {2, 3}
        assert hop_counts[2] == 6
        assert hop_counts[3] == 6

    def test_determinism(self):
        """Identical seeds yield identical splits; the total sampled never exceeds the budget."""
        from pykeen.pipeline.hierarchy import ancestor_descendant_split

        dataset = _make_balanced_tree_dataset()
        first = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=12, seed=7)
        second = ancestor_descendant_split(dataset, hops=(2, 3), num_pairs=12, seed=7)
        for factory_a, factory_b in zip(first, second, strict=True):
            assert torch.equal(factory_a.mapped_triples, factory_b.mapped_triples)
        _train, val, test = first
        assert val.num_triples + test.num_triples <= 12

    def test_pipeline_runs_with_hops(self):
        """Pipeline with custom hops/num_pairs completes and returns a valid MRR score."""
        from pykeen.pipeline.hierarchy import ancestor_descendant_pipeline

        dataset = _make_balanced_tree_dataset()
        result = ancestor_descendant_pipeline(dataset, epochs=1, hops=[2, 3], num_pairs=8, seed=0)
        assert result.metric_results is not None
        mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
        assert 0.0 <= mrr <= 1.0


def _make_bipartite_dataset() -> EagerDataset:
    """Return a 2-parent x 4-child bipartite hierarchy (relation 0, 8 edges).

    Parents {0, 1} each link to children {2, 3, 4, 5}, so parents have degree 4 and every child has
    degree 2. Each child can lose exactly one of its two edges without being isolated, giving a clean,
    plentiful pool of removable edges.
    """
    rows = [[p, 0, c] for p in (0, 1) for c in (2, 3, 4, 5)]
    train = CoreTriplesFactory(mapped_triples=torch.tensor(rows, dtype=torch.long), num_entities=6, num_relations=1)
    return EagerDataset(training=train, testing=train, validation=train)


class TestHierarchyCompletionPipeline(unittest.TestCase):
    """Tests for the hierarchy-completion (removed-edge) benchmark pipeline."""

    def test_no_isolated_nodes(self):
        """Removal never orphans a node: every node with a hierarchy edge keeps one in training."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_split

        dataset = _make_bipartite_dataset()
        original = dataset.training.mapped_triples.tolist()
        nodes_with_edge = {h for h, _, _ in original} | {t for _, _, t in original}

        train, _val, _test = hierarchy_completion_split(dataset, test_ratio=0.5, seed=42)
        train_rows = train.mapped_triples.tolist()
        train_nodes = {h for h, _, _ in train_rows} | {t for _, _, t in train_rows}
        assert nodes_with_edge <= train_nodes

    def test_leaves_never_removed(self):
        """On a tree, every leaf's single parent edge is retained (degree-1 nodes are protected)."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_split

        dataset = _make_balanced_tree_dataset()
        train, _val, _test = hierarchy_completion_split(dataset, test_ratio=0.5, seed=42)
        train_edges = {(h, t) for h, _, t in train.mapped_triples.tolist()}
        for leaf in range(7, 15):  # leaves 7..14, parent = (leaf - 1) // 2
            assert ((leaf - 1) // 2, leaf) in train_edges

    def test_held_out_edges_are_removed_originals(self):
        """Val/test edges are original hierarchy edges absent from train; the partition is exact."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_split

        dataset = _make_bipartite_dataset()
        original = {tuple(row) for row in dataset.training.mapped_triples.tolist()}

        train, val, test = hierarchy_completion_split(dataset, test_ratio=0.5, seed=42)
        train_edges = [tuple(row) for row in train.mapped_triples.tolist()]
        held = [tuple(row) for row in (val.mapped_triples.tolist() + test.mapped_triples.tolist())]

        assert set(held) <= original
        assert set(held).isdisjoint(train_edges)
        # exact partition, no duplicates: train + held reconstructs the original edge multiset
        assert sorted(train_edges + held) == sorted(original)

    def test_ratio_and_balanced_split(self):
        """The number removed matches round(test_ratio*|H|) and is split 50/50 within one edge."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_split

        dataset = _make_bipartite_dataset()  # |H| = 8, all four children supply a removable edge
        _train, val, test = hierarchy_completion_split(dataset, test_ratio=0.5, seed=42)
        assert val.num_triples + test.num_triples == 4  # round(0.5 * 8)
        assert abs(val.num_triples - test.num_triples) <= 1

    def test_original_relation_preserved(self):
        """Held-out edges keep their relation id; non-hierarchy edges stay in training."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_split

        # relation 0 = bipartite hierarchy; relation 1 = context edges that must never be held out
        hierarchy = [[p, 0, c] for p in (0, 1) for c in (2, 3, 4, 5)]
        context = [[2, 1, 3], [4, 1, 5]]
        rows = torch.tensor(hierarchy + context, dtype=torch.long)
        train_tf = CoreTriplesFactory(mapped_triples=rows, num_entities=6, num_relations=2)
        dataset = EagerDataset(training=train_tf, testing=train_tf, validation=train_tf)

        train, val, test = hierarchy_completion_split(dataset, test_ratio=0.5, seed=42, hierarchy_relation=0)
        held = val.mapped_triples.tolist() + test.mapped_triples.tolist()
        assert held, "expected some removed edges"
        assert all(r == 0 for _, r, _ in held)
        train_edges = [tuple(row) for row in train.mapped_triples.tolist()]
        for ctx in context:
            assert tuple(ctx) in train_edges

    def test_pipeline_runs(self):
        """Pipeline completes, returns a valid MRR, and reports hierarchical metrics."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_pipeline

        dataset = _make_bipartite_dataset()
        result = hierarchy_completion_pipeline(dataset, epochs=1, test_ratio=0.5, seed=0)
        assert result.metric_results is not None
        mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
        assert 0.0 <= mrr <= 1.0
        assert result.hierarchical_metric_results is not None

    def test_negative_sampler_selectable(self):
        """The hierarchy default is a setdefault: pseudotyped/basic can be chosen instead."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_pipeline

        dataset = _make_bipartite_dataset()
        for sampler in ("pseudotyped", "basic"):
            result = hierarchy_completion_pipeline(
                dataset, epochs=1, test_ratio=0.5, seed=0, negative_sampler=sampler
            )
            mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
            assert 0.0 <= mrr <= 1.0


def test_build_ancestor_paths_chain():
    """build_ancestor_paths returns a total inclusive ancestor map for a chain."""
    from pykeen.pipeline.hierarchy import build_ancestor_paths

    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=torch.long)
    ancestors = build_ancestor_paths(triples, num_entities=4)
    assert ancestors[0] == frozenset({0})
    assert ancestors[2] == frozenset({0, 1, 2})
    assert ancestors[3] == frozenset({0, 1, 2, 3})


def test_build_ancestor_paths_relation_filter():
    """build_ancestor_paths only follows edges of the given hierarchy relation."""
    from pykeen.pipeline.hierarchy import build_ancestor_paths

    # relation 0 = hierarchy chain 0→1→2; relation 1 = a non-hierarchy edge 3→0
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [3, 1, 0]], dtype=torch.long)
    filtered = build_ancestor_paths(triples, num_entities=4, hierarchy_relation=0)
    assert filtered[2] == frozenset({0, 1, 2})
    assert 3 not in filtered[0]
    # without the filter, the relation-1 edge would make 3 an ancestor of 0
    unfiltered = build_ancestor_paths(triples, num_entities=4)
    assert 3 in unfiltered[0]


def test_hierarchical_scores_perfect_and_sibling():
    """Hierarchical scores are 1.0 for a perfect hit and reflect shared root-path for a sibling."""
    from pykeen.evaluation.hierarchical_classification_evaluator import _hierarchical_scores

    # tree R→A, A→B, A→C  (0=R, 1=A, 2=B, 3=C)
    ancestors = {0: frozenset({0}), 1: frozenset({0, 1}), 2: frozenset({0, 1, 2}), 3: frozenset({0, 1, 3})}
    y_true = np.array([0, 0, 1, 0])  # true target is node 2

    # perfect prediction: node 2 ranked top
    perfect = _hierarchical_scores(y_true=y_true, y_score=np.array([0.1, 0.2, 0.9, 0.3]), ancestors=ancestors)
    assert perfect == pytest.approx((1.0, 1.0, 1.0))

    # sibling prediction: node 3 ranked top → shares root-path {0, 1}
    sibling = _hierarchical_scores(y_true=y_true, y_score=np.array([0.1, 0.2, 0.3, 0.9]), ancestors=ancestors)
    assert sibling == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_hierarchical_scores_disjoint_and_empty():
    """Disjoint branches score 0.0; an empty positive mask yields None."""
    from pykeen.evaluation.hierarchical_classification_evaluator import _hierarchical_scores

    # two disjoint chains 0→1 and 2→3
    ancestors = {0: frozenset({0}), 1: frozenset({0, 1}), 2: frozenset({2}), 3: frozenset({2, 3})}
    disjoint = _hierarchical_scores(
        y_true=np.array([0, 1, 0, 0]), y_score=np.array([0.0, 0.0, 0.0, 1.0]), ancestors=ancestors
    )
    assert disjoint == pytest.approx((0.0, 0.0, 0.0))

    empty = _hierarchical_scores(
        y_true=np.array([0, 0, 0, 0]), y_score=np.array([0.1, 0.2, 0.3, 0.4]), ancestors=ancestors
    )
    assert empty is None


def test_hierarchical_evaluator_requires_ancestors():
    """The hierarchical evaluator raises when no ancestors map is given."""
    from pykeen.evaluation import HierarchicalClassificationEvaluator

    with pytest.raises(ValueError, match="ancestors"):
        HierarchicalClassificationEvaluator()


def test_pipeline_reports_hierarchical_metrics():
    """The ancestor-descendant pipeline attaches hierarchical metrics in [0, 1] when enabled."""
    from pykeen.pipeline.hierarchy import ancestor_descendant_pipeline

    dataset = _make_balanced_tree_dataset()
    result = ancestor_descendant_pipeline(dataset, epochs=1, hops=[2, 3], num_pairs=8, seed=0)
    assert result.hierarchical_metric_results is not None
    h_f1 = result.hierarchical_metric_results.get_metric("both.hierarchical_f1")
    assert 0.0 <= h_f1 <= 1.0

    disabled = ancestor_descendant_pipeline(dataset, epochs=1, hops=[2, 3], num_pairs=8, seed=0, hierarchical=False)
    assert disabled.hierarchical_metric_results is None


def test_pipeline_persists_hierarchical_metrics():
    """save_to_directory writes the hierarchical metrics into results.json."""
    import json

    from pykeen.pipeline.hierarchy import ancestor_descendant_pipeline

    dataset = _make_balanced_tree_dataset()
    result = ancestor_descendant_pipeline(dataset, epochs=1, hops=[2, 3], num_pairs=8, seed=0)
    with tempfile.TemporaryDirectory() as directory:
        result.save_to_directory(directory)
        with pathlib.Path(directory, "results.json").open() as file:
            saved = json.load(file)
    assert "hierarchical_metrics" in saved


def test_hpo_pipeline_refits_best_trial():
    """The ancestor-descendant HPO pipeline runs a study and re-fits the best trial's hierarchical metrics."""
    from pykeen.pipeline.hierarchy import hpo_ancestor_descendant_pipeline

    dataset = _make_balanced_tree_dataset()
    outcome = hpo_ancestor_descendant_pipeline(
        dataset, model="PoincareE", n_trials=1, epochs=1, hops=(2, 3), num_pairs=8, seed=0
    )
    assert outcome.hpo_result.study is not None
    assert outcome.result is not None
    h_f1 = outcome.result.hierarchical_metric_results.get_metric("both.hierarchical_f1")
    assert 0.0 <= h_f1 <= 1.0
