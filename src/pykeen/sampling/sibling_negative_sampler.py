"""Sibling-based ("hard") negative sampling for hierarchies (He et al. 2024, §4.1)."""

from collections.abc import Iterable

import torch

from .hierarchy_negative_sampler import _warn_if_multi_relational
from .negative_sampler import NegativeSampler
from ..typing import LongTensor, MappedTriples

__all__ = [
    "SiblingNegativeSampler",
    "sibling_groups",
]

#: triple slot indices for head / tail corruption
_HEAD, _TAIL = 0, 2

#: per-entity cap on the sibling list; see ``SiblingNegativeSampler.max_siblings``
_DEFAULT_MAX_SIBLINGS = 100


def sibling_groups(edges: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    """Map each entity to its siblings — entities sharing a direct neighbour on the same edge side.

    Orientation-agnostic: hierarchy relations may point parent->child (e.g. NASA ``has_subclass``)
    or child->parent (e.g. WN18RR ``_hypernym``), so two entities count as siblings when they share
    a direct predecessor *or* a direct successor. On a tree this reduces to the papers' definition
    (same parent) regardless of edge direction.

    :param edges: the hierarchy edges as ``(head, tail)`` pairs.

    :returns: a mapping from entity to its sorted sibling list; entities without siblings are absent.
    """
    by_pred: dict[int, set[int]] = {}
    by_succ: dict[int, set[int]] = {}
    for h, t in edges:
        by_pred.setdefault(h, set()).add(t)
        by_succ.setdefault(t, set()).add(h)
    siblings: dict[int, set[int]] = {}
    for group in (*by_pred.values(), *by_succ.values()):
        for member in group:
            siblings.setdefault(member, set()).update(group - {member})
    return {node: sorted(sibs) for node, sibs in siblings.items()}


def _build_sibling_index(
    mapped_triples: MappedTriples,
    num_entities: int,
    hierarchy_relation: int | None,
    max_siblings: int,
) -> tuple[LongTensor, LongTensor]:
    """CSR-style index of each entity's siblings (cf. :func:`pykeen.sampling.pseudo_type.create_index`).

    :returns: ``(data, offsets)`` where ``data[offsets[e]:offsets[e + 1]]`` are the siblings of
        entity ``e``. ``data`` holds a single padding entry when no entity has a sibling, so that
        the gather in :meth:`SiblingNegativeSampler.corrupt_batch` stays well-defined.
    """
    groups = sibling_groups(
        (h, t)
        for h, r, t in mapped_triples.tolist()
        if (hierarchy_relation is None or r == hierarchy_relation) and h != t
    )

    data: list[int] = []
    offsets = torch.empty(num_entities + 1, dtype=torch.long)
    offsets[0] = 0
    for entity in range(num_entities):
        # ponytail: truncating by ascending id bounds memory on high-branching nodes (a WordNet
        # synset may have thousands of hyponyms) and keeps the index deterministic. Raise
        # max_siblings, or sample the group instead of truncating, if the bias ever shows up.
        data.extend(groups.get(entity, ())[:max_siblings])
        offsets[entity + 1] = len(data)

    return torch.as_tensor(data or [0], dtype=torch.long), offsets


class SiblingNegativeSampler(NegativeSampler):
    r"""A sampler that corrupts a hierarchy edge by pairing an entity with its own sibling.

    For a positive edge $(h, r, t)$, tail corruption yields $(h, r, s)$ where $s$ is a *sibling of
    $h$*, and head corruption yields $(s, r, t)$ where $s$ is a *sibling of $t$* — the hard negative
    setting of He et al. (2024, §4.1), and the same rule the transitive ancestor-descendant
    evaluation uses for its ``F1·hrd`` column.

    Pairing an entity with its own sibling (rather than replacing an endpoint by a sibling *of that
    endpoint*) is what makes the negative sound: two siblings are never ancestor and descendant of
    one another, so the corrupted pair is guaranteed to lie outside the transitive closure. The
    opposite convention — replacing $t$ by a sibling of $t$ — would keep $h$ a plausible ancestor
    and manufacture false negatives.

    Siblings are much harder to separate than uniformly random entities, yet drawing them is a
    vectorized CSR lookup just like :class:`pykeen.sampling.HierarchyNegativeSampler`.

    Three knobs guard the failure modes of pure sibling restriction:

    * ``head_corruption_prob`` fixes the head/tail split, so parent prediction keeps getting trained.
    * ``hard_ratio`` blends in uniform negatives; an entity whose partner has no sibling (or whose
      only sibling *is* the entity being replaced) also falls back to uniform. This keeps the
      sampler well-defined on chains and star hierarchies.
    * ``filtered`` masks the rare sibling draw that happens to be a real training edge.
    """

    #: siblings grouped by entity, shape: (num_sibling_pairs,)
    data: LongTensor
    #: CSR offsets into ``data``, shape: (num_entities + 1,)
    offsets: LongTensor

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        hierarchy_relation: int | None = None,
        hard_ratio: float = 0.5,
        head_corruption_prob: float = 0.5,
        max_siblings: int = _DEFAULT_MAX_SIBLINGS,
        **kwargs,
    ) -> None:
        """Instantiate the sibling negative sampler.

        :param mapped_triples: the positive training triples; their edges define the sibling groups.
        :param hierarchy_relation: if given, only edges with this relation id define the hierarchy.
            Leaving it ``None`` treats *every* relation as a hierarchy edge, so on a
            multi-relational dataset "siblings" degenerates into "entities sharing any neighbour"
            and the hard negatives become nearly uniform; a warning is emitted in that case.
        :param hard_ratio: probability that a negative is a sibling ("hard") draw rather than
            uniform. ``0.0`` reproduces uniform sampling, ``1.0`` is fully sibling-based.
        :param head_corruption_prob: probability of corrupting the head (vs. the tail) of a positive.
        :param max_siblings: per-entity cap on the sibling list, bounding the index on
            high-branching hierarchies.
        :param kwargs: Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        self.hard_ratio = hard_ratio
        self.head_corruption_prob = head_corruption_prob
        self.max_siblings = max_siblings
        _warn_if_multi_relational(mapped_triples, hierarchy_relation)
        data, offsets = _build_sibling_index(mapped_triples, self.num_entities, hierarchy_relation, max_siblings)
        self.register_buffer("data", data)
        self.register_buffer("offsets", offsets)

    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:  # noqa: D102
        device = positive_batch.device
        batch_size = positive_batch.shape[0]
        size = (batch_size, self.num_negs_per_pos)

        # shape: (batch_size, num_negs_per_pos, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

        # choose the slot to corrupt (head=0 / tail=2); the entity in the *other* slot is kept, and
        # supplies the siblings — _HEAD and _TAIL are 0 and 2, so the partner slot is ``2 - slot``
        slot = torch.where(torch.rand(size, device=device) < self.head_corruption_prob, _HEAD, _TAIL)
        original = negative_batch.gather(2, slot.unsqueeze(-1)).squeeze(-1)
        kept = negative_batch.gather(2, (_TAIL - slot).unsqueeze(-1)).squeeze(-1)

        # --- hard path: a sibling of the kept entity, so the pair cannot be an ancestor-descendant one ---
        start = self.offsets[kept]
        num_siblings = self.offsets[kept + 1] - start
        local = (torch.rand(size, device=device) * num_siblings.clamp(min=1)).long()
        hard = self.data[(start + local).clamp(max=self.data.numel() - 1)]

        # --- uniform path: any other entity (also the fallback when the kept entity has no sibling) ---
        uniform = torch.randint(high=self.num_entities - 1, size=size, device=device)
        uniform = uniform + (uniform >= original).long()

        # ``hard == original`` would reproduce the positive verbatim (the kept entity's only sibling
        # happens to be the entity we are replacing), so fall back to uniform there too
        use_uniform = (torch.rand(size, device=device) >= self.hard_ratio) | (num_siblings == 0) | (hard == original)
        replacement = torch.where(use_uniform, uniform, hard)

        negative_batch.scatter_(2, slot.unsqueeze(-1), replacement.unsqueeze(-1))
        return negative_batch
