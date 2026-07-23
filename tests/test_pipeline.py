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


def test_poincare_scores_are_non_positive():
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
        """Pipeline completes and returns a valid MRR."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_pipeline

        dataset = _make_bipartite_dataset()
        result = hierarchy_completion_pipeline(dataset, epochs=1, test_ratio=0.5, seed=0)
        assert result.metric_results is not None
        mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
        assert 0.0 <= mrr <= 1.0

    def test_negative_sampler_selectable(self):
        """The hierarchy default is a setdefault: pseudotyped/basic can be chosen instead."""
        from pykeen.pipeline.hierarchy import hierarchy_completion_pipeline

        dataset = _make_bipartite_dataset()
        for sampler in ("pseudotyped", "basic"):
            result = hierarchy_completion_pipeline(dataset, epochs=1, test_ratio=0.5, seed=0, negative_sampler=sampler)
            mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
            assert 0.0 <= mrr <= 1.0


def _is_tree_ancestor(ancestor: int, node: int) -> bool:
    """Return whether ``ancestor`` lies on ``node``'s root path in the balanced binary tree."""
    while node > 0:
        node = (node - 1) // 2
        if node == ancestor:
            return True
    return False


class TestTransitiveAncestorDescendantPredictionPipeline(unittest.TestCase):
    """Tests for multi-hop transitive ancestor-descendant prediction (Ganea et al. 2018; He et al. 2024)."""

    def test_split_train_direct_and_heldout_indirect(self):
        """Training keeps all direct edges; val/test are disjoint non-direct closure pairs."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_split

        dataset = _make_balanced_tree_dataset()
        direct = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}

        # the balanced tree has 20 non-direct closure pairs -> 5 val + 5 test
        train, val, test = transitive_ancestor_descendant_prediction_split(dataset, eval_ratio=0.25, seed=42)
        assert sorted(train.mapped_triples.tolist()) == sorted(dataset.training.mapped_triples.tolist())
        val_rows = [tuple(row) for row in val.mapped_triples.tolist()]
        test_rows = [tuple(row) for row in test.mapped_triples.tolist()]
        assert len(val_rows) == len(test_rows) == 5
        assert set(val_rows).isdisjoint(test_rows)
        for h, r, t in val_rows + test_rows:
            assert r == 0
            assert (h, t) not in direct
            assert _is_tree_ancestor(h, t)

    def test_closure_ratio_adds_pairs_to_train(self):
        """``closure_ratio`` moves indirect closure pairs into training, disjoint from val/test."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_split

        dataset = _make_balanced_tree_dataset()
        direct = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}

        # 20 indirect pairs: 5 val + 5 test + 25% of 20 = 5 extra training pairs
        train, val, test = transitive_ancestor_descendant_prediction_split(
            dataset, closure_ratio=0.25, eval_ratio=0.25, seed=42
        )
        extra = [(h, t) for h, _, t in train.mapped_triples.tolist() if (h, t) not in direct]
        assert len(extra) == 5
        held = {(h, t) for h, _, t in val.mapped_triples.tolist() + test.mapped_triples.tolist()}
        assert held.isdisjoint(extra)
        for h, t in extra:
            assert _is_tree_ancestor(h, t)

    def test_invalid_ratios_raise(self):
        """Ratios that overrun the closure pool are rejected."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_split

        dataset = _make_balanced_tree_dataset()
        with pytest.raises(ValueError, match="closure_ratio"):
            transitive_ancestor_descendant_prediction_split(dataset, closure_ratio=0.5, eval_ratio=0.3, seed=42)

    def test_determinism(self):
        """The same seed yields identical splits."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_split

        dataset = _make_balanced_tree_dataset()
        first = transitive_ancestor_descendant_prediction_split(dataset, eval_ratio=0.25, seed=7)
        second = transitive_ancestor_descendant_prediction_split(dataset, eval_ratio=0.25, seed=7)
        for factory_a, factory_b in zip(first, second, strict=True):
            assert factory_a.mapped_triples.tolist() == factory_b.mapped_triples.tolist()

    def test_hierarchy_inverted_canonicalizes_orientation(self):
        """Assert an inverted dataset (child->parent edges, e.g. WN18RR) yields the canonical closure."""
        from pykeen.pipeline.hierarchical_helper import _closure_pool

        dataset = _make_balanced_tree_dataset()
        inverted_rows = [[t, r, h] for h, r, t in dataset.training.mapped_triples.tolist()]
        train = CoreTriplesFactory(
            mapped_triples=torch.tensor(inverted_rows, dtype=torch.long), num_entities=15, num_relations=1
        )
        inverted = EagerDataset(training=train, testing=train, validation=train)
        inverted.hierarchy_inverted = True

        for expected, actual in zip(_closure_pool(dataset, 0), _closure_pool(inverted, 0), strict=True):
            assert expected == actual

    def test_predefined_closure_split_uses_dataset_files(self):
        """A predefined-split dataset keeps its own train/val/test rows; nothing is re-derived."""
        from pykeen.pipeline.transitive_ancestor_descendant import _transitive_ancestor_descendant_split

        # training: chain 0->1->2 plus the in-training closure shortcut 0->2 (as in Ganea's maxn
        # 50% files); held-out closure pairs live only in the fixed val/test factories.
        train_rows = [[0, 0, 1], [1, 0, 2], [0, 0, 2], [2, 0, 3]]
        train = CoreTriplesFactory(
            mapped_triples=torch.tensor(train_rows, dtype=torch.long), num_entities=4, num_relations=1
        )
        val = CoreTriplesFactory(
            mapped_triples=torch.tensor([[1, 0, 3]], dtype=torch.long), num_entities=4, num_relations=1
        )
        test = CoreTriplesFactory(
            mapped_triples=torch.tensor([[0, 0, 3]], dtype=torch.long), num_entities=4, num_relations=1
        )
        dataset = EagerDataset(training=train, testing=test, validation=val)
        dataset.predefined_closure_split = True

        out_train, out_val, out_test, paths, direct = _transitive_ancestor_descendant_split(
            dataset, closure_ratio=0.9, eval_ratio=0.4, seed=42, hierarchy_relation=0
        )
        # training rows are kept verbatim (incl. the shortcut - no transitive reduction, no
        # re-held-out closure edges) and eval rows come from the dataset's own files.
        assert out_train == train_rows
        assert out_val == [[1, 0, 3]]
        assert out_test == [[0, 0, 3]]
        assert direct == {(0, 1), (1, 2), (0, 2), (2, 3)}
        # ancestor paths span the full closure of the training graph (false-negative filtering)
        assert paths[3] == frozenset({0, 1, 2, 3})

    def test_basic_edges_are_transitive_reduction(self):
        """Assert a redundant shortcut edge is demoted from basic (training) to the eval pool."""
        from pykeen.pipeline.hierarchical_helper import _closure_pool

        # 0->1->2 plus the redundant shortcut 0->2; the reduction drops 0->2.
        rows = torch.tensor([[0, 0, 1], [1, 0, 2], [0, 0, 2]], dtype=torch.long)
        train = CoreTriplesFactory(mapped_triples=rows, num_entities=3, num_relations=1)
        dataset = EagerDataset(training=train, testing=train, validation=train)

        paths, _all_rows, direct, pool = _closure_pool(dataset, 0)
        assert direct == {(0, 1), (1, 2)}
        assert (0, 2) in pool
        assert paths[2] == frozenset({0, 1, 2})

    def test_clean_tree_reduction_is_asserted_edges(self):
        """Assert a shortcut-free tree is unchanged: its reduction equals the asserted edges."""
        from pykeen.pipeline.hierarchical_helper import _closure_pool

        dataset = _make_balanced_tree_dataset()
        asserted = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}
        _paths, _all_rows, direct, _pool = _closure_pool(dataset, 0)
        assert direct == asserted

    def test_hard_negatives_prefer_siblings(self):
        """Hard negatives pair an entity with its own sibling first, then top up with random."""
        from pykeen.pipeline.hierarchical_helper import _sample_negatives, build_ancestor_paths
        from pykeen.sampling import sibling_groups

        dataset = _make_balanced_tree_dataset()
        paths = build_ancestor_paths(dataset.training.mapped_triples, num_entities=15)
        direct = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}
        siblings = sibling_groups(direct)
        assert siblings[1] == [2]
        assert siblings[7] == [8]

        # heads' and tails' siblings give two hard candidates ((1,2) and (8,7)); one random top-up
        negatives = _sample_negatives(
            [[1, 0, 7]], paths, num_entities=15, rng=np.random.default_rng(0), num_negatives=3, siblings=siblings
        )
        assert len(negatives) == 3
        assert [1, 0, 2] in negatives
        assert [8, 0, 7] in negatives
        for h, r, t in negatives:
            assert r == 0
            assert h not in paths[t]

    def test_num_negatives_per_positive(self):
        """Random sampling yields exactly ``num_negatives`` valid negatives per positive."""
        from pykeen.pipeline.hierarchical_helper import _sample_negatives, build_ancestor_paths

        dataset = _make_balanced_tree_dataset()
        paths = build_ancestor_paths(dataset.training.mapped_triples, num_entities=15)
        positives = [[1, 0, 7], [2, 0, 11]]
        negatives = _sample_negatives(positives, paths, num_entities=15, rng=np.random.default_rng(0), num_negatives=4)
        assert len(negatives) == 8
        for h, _r, t in negatives:
            assert h not in paths[t]

    def test_two_sided_negatives(self):
        """Two-sided sampling corrupts both slots evenly and stays outside the closure (Ganea et al. 2018 §5)."""
        from pykeen.pipeline.hierarchical_helper import _sample_negatives, build_ancestor_paths

        dataset = _make_balanced_tree_dataset()
        paths = build_ancestor_paths(dataset.training.mapped_triples, num_entities=15)
        positives = [[1, 0, 7], [2, 0, 11]]
        negatives = _sample_negatives(
            positives, paths, num_entities=15, rng=np.random.default_rng(0), num_negatives=10, two_sided=True
        )
        assert len(negatives) == 20
        for h, r, t in negatives:
            assert r == 0
            assert h not in paths[t]
        # negatives come back grouped per positive; a valid corruption never reproduces the
        # original slot (the positive is a closure pair), so the 5/5 counts are exact
        for i, (head, _, tail) in enumerate(positives):
            rows = negatives[10 * i : 10 * (i + 1)]
            assert sum(1 for h, _r, t in rows if h == head) == 5  # tail-corrupted
            assert sum(1 for h, _r, t in rows if t == tail) == 5  # head-corrupted

    def test_negatives_deduped_per_positive(self):
        """Negatives are drawn without replacement: no pair repeats for one positive (issues.md §6)."""
        from pykeen.pipeline.hierarchical_helper import _sample_negatives, build_ancestor_paths
        from pykeen.sampling import sibling_groups

        dataset = _make_balanced_tree_dataset()
        paths = build_ancestor_paths(dataset.training.mapped_triples, num_entities=15)
        direct = {(h, t) for h, _, t in dataset.training.mapped_triples.tolist()}
        siblings = sibling_groups(direct)

        # eight valid tail-corruptions exist for (1, 7); requesting all eight forces collisions
        # under with-replacement sampling, so distinctness proves the dedupe
        for kwargs in ({}, {"two_sided": True}, {"siblings": siblings}):
            negatives = _sample_negatives(
                [[1, 0, 7]], paths, num_entities=15, rng=np.random.default_rng(0), num_negatives=8, **kwargs
            )
            assert len({tuple(n) for n in negatives}) == len(negatives)

    def test_pipeline_runs(self):
        """Pipeline completes and reports thresholded P/R/F1 plus mAP/AUROC."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_pipeline

        dataset = _make_balanced_tree_dataset()
        result = transitive_ancestor_descendant_prediction_pipeline(
            dataset, epochs=1, eval_ratio=0.25, num_negatives=3, seed=0
        )
        mrr = result.get_metric("both.realistic.inverse_harmonic_mean_rank")
        assert 0.0 <= mrr <= 1.0
        metrics = result.ancestor_descendant_metric_results
        assert metrics is not None
        for key in ("precision", "recall", "f1", "average_precision", "roc_auc"):
            assert 0.0 <= metrics.get_metric(key) <= 1.0

    def test_metrics_reevaluate_trained_model(self):
        """``transitive_ancestor_descendant_prediction_metrics`` re-scores a trained model, e.g. with hard negatives."""
        from pykeen.pipeline.transitive_ancestor_descendant import (
            transitive_ancestor_descendant_prediction_metrics,
            transitive_ancestor_descendant_prediction_pipeline,
        )

        dataset = _make_balanced_tree_dataset()
        result = transitive_ancestor_descendant_prediction_pipeline(
            dataset, epochs=1, eval_ratio=0.25, num_negatives=3, seed=0
        )
        metrics = transitive_ancestor_descendant_prediction_metrics(
            result.model, dataset, eval_ratio=0.25, num_negatives=3, hard_negatives=True, seed=0
        )
        assert metrics is not None
        for key in ("precision", "recall", "f1", "average_precision", "roc_auc"):
            assert 0.0 <= metrics.get_metric(key) <= 1.0

    def test_perfect_scorer_yields_perfect_f1(self):
        """A scorer that recognises true ancestor pairs gives F1 = mAP = AUROC = 1."""
        from pykeen.pipeline.transitive_ancestor_descendant import transitive_ancestor_descendant_prediction_pipeline

        dataset = _make_balanced_tree_dataset()

        def perfect_scorer(model, batch):
            """Score 1 for true ancestor-descendant pairs, 0 otherwise (scale-consistent val/test)."""
            return torch.tensor([float(_is_tree_ancestor(h, t)) for h, _r, t in batch.tolist()])

        result = transitive_ancestor_descendant_prediction_pipeline(
            dataset, epochs=1, eval_ratio=0.25, num_negatives=3, seed=0, eval_score_fn=perfect_scorer
        )
        metrics = result.ancestor_descendant_metric_results
        assert metrics is not None
        assert metrics.get_metric("f1") == 1.0
        assert metrics.get_metric("average_precision") == 1.0
        assert metrics.get_metric("roc_auc") == 1.0

    def test_lca_metrics(self):
        """``lca=True`` reports LCA-based P/R/F1 per test descendant, matching the standalone function."""
        from pykeen.pipeline.transitive_ancestor_descendant import (
            transitive_ancestor_descendant_lca_metrics,
            transitive_ancestor_descendant_prediction_pipeline,
        )

        dataset = _make_balanced_tree_dataset()
        result = transitive_ancestor_descendant_prediction_pipeline(
            dataset, epochs=1, eval_ratio=0.25, num_negatives=3, seed=0, lca=True
        )
        metrics = result.lca_metric_results
        assert metrics is not None
        for key in ("lca_precision", "lca_recall", "lca_f1"):
            assert 0.0 <= metrics.get_metric(f"both.{key}") <= 1.0
        standalone = transitive_ancestor_descendant_lca_metrics(result.model, dataset, eval_ratio=0.25, seed=0)
        assert standalone is not None
        assert standalone.get_metric("both.lca_f1") == metrics.get_metric("both.lca_f1")


def test_build_ancestor_paths_chain():
    """build_ancestor_paths returns a total inclusive ancestor map for a chain."""
    from pykeen.pipeline.hierarchical_helper import build_ancestor_paths

    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=torch.long)
    ancestors = build_ancestor_paths(triples, num_entities=4)
    assert ancestors[0] == frozenset({0})
    assert ancestors[2] == frozenset({0, 1, 2})
    assert ancestors[3] == frozenset({0, 1, 2, 3})


def test_build_ancestor_paths_relation_filter():
    """build_ancestor_paths only follows edges of the given hierarchy relation."""
    from pykeen.pipeline.hierarchical_helper import build_ancestor_paths

    # relation 0 = hierarchy chain 0→1→2; relation 1 = a non-hierarchy edge 3→0
    triples = torch.tensor([[0, 0, 1], [1, 0, 2], [3, 1, 0]], dtype=torch.long)
    filtered = build_ancestor_paths(triples, num_entities=4, hierarchy_relation=0)
    assert filtered[2] == frozenset({0, 1, 2})
    assert 3 not in filtered[0]
    # without the filter, the relation-1 edge would make 3 an ancestor of 0
    unfiltered = build_ancestor_paths(triples, num_entities=4)
    assert 3 in unfiltered[0]


def test_hpo_pipeline_refits_best_trial():
    """The hierarchy-completion HPO pipeline runs a study and re-fits the best trial."""
    from pykeen.pipeline.hierarchy import hpo_hierarchy_completion_pipeline

    dataset = _make_balanced_tree_dataset()
    outcome = hpo_hierarchy_completion_pipeline(
        dataset, model="PoincareE", n_trials=1, epochs=1, test_ratio=0.5, seed=0
    )
    assert outcome.hpo_result.study is not None
    assert outcome.result is not None
    mrr = outcome.result.get_metric("both.realistic.inverse_harmonic_mean_rank")
    assert 0.0 <= mrr <= 1.0


def _make_labeled_chain_dataset() -> EagerDataset:
    """Return a labeled hierarchy ``a->b->c`` with the redundant shortcut ``a->c`` and branch ``a->d``.

    The shortcut is non-basic, so the non-direct closure pool is exactly ``{(a, c)}`` — enough to
    exercise both the transitive reduction and the closure-file round-trip.
    """
    labels = np.array(
        [("a", "narrower", "b"), ("b", "narrower", "c"), ("a", "narrower", "c"), ("a", "narrower", "d")],
        dtype=str,
    )
    triples = TriplesFactory.from_labeled_triples(labels)
    return EagerDataset(training=triples, testing=triples, validation=triples)


def test_closure_file_reproduces_computed_pool(tmp_path, monkeypatch):
    """A cached ``closure.tsv`` reproduces exactly the computed paths, direct edges, and pool."""
    from pykeen.datasets import metadata as metadata_module
    from pykeen.pipeline.hierarchical_helper import _closure_pool

    dataset = _make_labeled_chain_dataset()
    computed = _closure_pool(dataset, None)

    monkeypatch.setattr(metadata_module, "PYKEEN_DATASETS", tmp_path)
    cache_dir = tmp_path / type(dataset).__name__.lower()
    cache_dir.mkdir()
    entity_label = dataset.training.entity_id_to_label
    (cache_dir / "closure.tsv").write_text(
        "".join(f"{entity_label[h]}\tnarrower\t{entity_label[t]}\n" for h, t in computed[3]),
        encoding="utf-8",
    )
    dataset.closure_url = "https://example.invalid/closure.tsv"  # cached, so never downloaded

    assert _closure_pool(dataset, None) == computed


def test_closure_download_failure_falls_back_to_computing(tmp_path, monkeypatch):
    """An unreachable ``closure_url`` warns and falls back to the on-demand computation."""
    from pykeen.datasets import metadata as metadata_module
    from pykeen.pipeline.hierarchical_helper import _closure_pool

    dataset = _make_labeled_chain_dataset()
    expected = _closure_pool(dataset, None)

    monkeypatch.setattr(metadata_module, "PYKEEN_DATASETS", tmp_path)
    dataset.closure_url = "https://example.invalid/closure.tsv"

    with pytest.warns(UserWarning, match="could not download closure file"):
        actual = _closure_pool(dataset, None)
    assert actual == expected
