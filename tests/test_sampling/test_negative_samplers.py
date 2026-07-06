"""Test that samplers can be executed."""

import numpy.testing
import torch
import unittest_templates

from pykeen.sampling import (
    BasicNegativeSampler,
    BernoulliNegativeSampler,
    HierarchyNegativeSampler,
    NegativeSampler,
    PseudoTypedNegativeSampler,
)
from pykeen.sampling.pseudo_type import create_index
from tests.test_sampling import cases


def _verify_entity_corruption(instance: NegativeSampler, positive_batch: torch.LongTensor):
    """Verify that at most one entity is corrupted."""
    negative_batch = instance.corrupt_batch(positive_batch=positive_batch)
    positive_batch = positive_batch.unsqueeze(dim=1).repeat(1, instance.num_negs_per_pos, 1)
    # same relation
    numpy.testing.assert_array_equal(negative_batch[..., 1], positive_batch[..., 1])
    # only corruption of a single entity (note: we do not check for exactly 2, since we do not filter).
    numpy.testing.assert_array_less(1, (negative_batch == positive_batch).sum(dim=-1))


class BasicNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the basic negative sampler."""

    cls = BasicNegativeSampler

    def test_sample_basic(self):
        """Test if relations and half of heads and tails are not corrupted."""
        negative_batch = self.instance.sample(positive_batch=self.positive_batch)[0]

        # Test that half of the subjects and half of the objects are corrupted
        positive_batch = self.positive_batch.unsqueeze(dim=1)
        num_triples = negative_batch[..., 0].numel()
        half_size = num_triples // 2
        num_subj_corrupted = (positive_batch[..., 0] != negative_batch[..., 0]).sum()
        num_obj_corrupted = (positive_batch[..., 2] != negative_batch[..., 2]).sum()
        assert num_obj_corrupted - 1 <= num_subj_corrupted
        assert num_subj_corrupted - 1 <= num_obj_corrupted
        assert num_subj_corrupted - 1 <= num_triples
        assert half_size - 1 <= num_subj_corrupted

    def test_entity_corruption(self):
        """Verify entity corruption."""
        _verify_entity_corruption(instance=self.instance, positive_batch=self.positive_batch)


class BernoulliNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the Bernoulli negative sampler."""

    cls = BernoulliNegativeSampler

    def test_entity_corruption(self):
        """Verify entity corruption."""
        _verify_entity_corruption(instance=self.instance, positive_batch=self.positive_batch)


class PseudoTypedNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the pseudo-type negative sampler."""

    cls = PseudoTypedNegativeSampler

    def test_corrupt_batch(self):
        """Additional test for corrupt_batch."""
        negative_batch = self.instance.corrupt_batch(positive_batch=self.positive_batch)
        # check that corrupted entities co-occur with the relation in training data
        for entity_pos in (0, 2):
            er_training = {(r, e) for r, e in self.triples_factory.mapped_triples[:, [1, entity_pos]].tolist()}
            er_negative = {(r, e) for r, e in negative_batch.view(-1, 3)[:, [1, entity_pos]].tolist()}
            assert er_negative.issubset(er_training)

    def test_index_structure(self):
        """Test the index structure."""
        data, offsets = create_index(
            mapped_triples=self.triples_factory.mapped_triples, num_relations=self.triples_factory.num_relations
        )
        triples = self.triples_factory.mapped_triples
        for r in range(self.triples_factory.num_relations):
            triples_with_r = triples[triples[:, 1] == r]
            for i, entity_pos in enumerate((0, 2)):
                index_entities = set(data[offsets[2 * r + i] : offsets[2 * r + i + 1]].tolist())
                triple_entities = set(triples_with_r[:, entity_pos].tolist())
                assert index_entities == triple_entities

    def test_entity_corruption(self):
        """Verify entity corruption."""
        _verify_entity_corruption(instance=self.instance, positive_batch=self.positive_batch)


class HierarchyNegativeSamplerTest(cases.NegativeSamplerGenericTestCase):
    """Test the hierarchy (same-depth) negative sampler."""

    cls = HierarchyNegativeSampler

    def test_entity_corruption(self):
        """Verify entity corruption."""
        _verify_entity_corruption(instance=self.instance, positive_batch=self.positive_batch)


def test_hierarchy_sampler_same_depth():
    """Hard corruptions land at the same depth as the entity they replace."""
    # parent->child tree: depths {0: 0, 1,2: 1, 3,4,5,6: 2}; every non-root depth has >=2 nodes
    mapped = torch.as_tensor([[0, 0, 1], [0, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 5], [2, 0, 6]])
    sampler = HierarchyNegativeSampler(
        mapped_triples=mapped,
        num_entities=7,
        num_relations=1,
        hard_ratio=1.0,  # never fall back to uniform
        head_corruption_prob=0.0,  # always corrupt the (child) tail, whose depth bucket has >=2 nodes
        num_negs_per_pos=20,
    )
    negative_batch = sampler.corrupt_batch(positive_batch=mapped)

    original_tail_depth = sampler.node_depth[mapped[:, 2]].unsqueeze(dim=-1)
    negative_tail_depth = sampler.node_depth[negative_batch[..., 2]]
    assert (negative_tail_depth == original_tail_depth).all()
    # heads are left untouched and tails actually change
    assert (negative_batch[..., 0] == mapped[:, 0].unsqueeze(dim=-1)).all()
    assert (negative_batch[..., 2] != mapped[:, 2].unsqueeze(dim=-1)).all()


def test_hierarchy_sampler_star_fallback():
    """A singleton depth level (e.g. a star's root) falls back to uniform without error."""
    # star: root 0 -> leaves 1..4; depth 0 = {0} is a singleton, so head corruption has no peer
    mapped = torch.as_tensor([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])
    sampler = HierarchyNegativeSampler(
        mapped_triples=mapped,
        num_entities=5,
        num_relations=1,
        hard_ratio=1.0,
        head_corruption_prob=1.0,  # always corrupt the root, whose depth bucket is a singleton
        num_negs_per_pos=15,
    )
    negative_batch = sampler.corrupt_batch(positive_batch=mapped)
    heads = negative_batch[..., 0]
    assert (heads != 0).all()  # uniform fallback never reproduces the root itself
    assert (heads >= 0).all()
    assert (heads < 5).all()


class NegativeSamplerMetaTestCase(unittest_templates.MetaTestCase):
    """Meta test case for testing all negative samplers."""

    base_cls = NegativeSampler
    base_test = cases.NegativeSamplerGenericTestCase
