"""Hierarchy-aware negative sampling via same-depth ("near-miss") corruption."""

from collections import defaultdict, deque

import torch

from .negative_sampler import NegativeSampler
from ..typing import LongTensor, MappedTriples

__all__ = [
    "HierarchyNegativeSampler",
]

#: triple slot indices for head / tail corruption
_HEAD, _TAIL = 0, 2


def _compute_depths(mapped_triples: MappedTriples, num_entities: int, hierarchy_relation: int | None) -> list[int]:
    """Depth (distance from the nearest root) of every entity over the parent→child hierarchy edges.

    Roots are nodes with an incident hierarchy edge but no incoming one; depth is the shortest-path
    BFS distance from any root. Nodes without a finite depth (isolated, or stuck in a cycle) are
    assigned a single extra depth bucket so they never share a level with real hierarchy nodes — the
    sampler's uniform fallback then covers them.
    """
    children: dict[int, list[int]] = defaultdict(list)
    indeg = [0] * num_entities
    has_edge = [False] * num_entities
    for h, r, t in mapped_triples.tolist():
        if (hierarchy_relation is not None and r != hierarchy_relation) or h == t:
            continue
        children[h].append(t)
        indeg[t] += 1
        has_edge[h] = has_edge[t] = True

    depth = [-1] * num_entities
    queue = deque(n for n in range(num_entities) if has_edge[n] and indeg[n] == 0)
    for n in queue:
        depth[n] = 0
    while queue:
        u = queue.popleft()
        for v in children[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)

    # isolated nodes and any cycle-only nodes (no root to reach them) get their own bucket
    isolated_depth = max((d for d in depth if d >= 0), default=0) + 1
    return [d if d >= 0 else isolated_depth for d in depth]


def _build_depth_index(depth: list[int], num_entities: int) -> tuple[LongTensor, LongTensor, LongTensor]:
    """CSR-style index over depth buckets (cf. :func:`pykeen.sampling.pseudo_type.create_index`).

    :returns: ``(data, offsets, entity_pos)`` where ``data[offsets[d]:offsets[d+1]]`` are the entities
        at depth ``d`` and ``entity_pos`` is the inverse permutation (each entity's index in ``data``).
    """
    num_depths = max(depth) + 1
    buckets: list[list[int]] = [[] for _ in range(num_depths)]
    for n, d in enumerate(depth):
        buckets[d].append(n)

    data: list[int] = []
    offsets = torch.empty(num_depths + 1, dtype=torch.long)
    offsets[0] = 0
    for d in range(num_depths):
        data.extend(buckets[d])  # entities stay in ascending id order within a bucket
        offsets[d + 1] = len(data)

    data_t = torch.as_tensor(data, dtype=torch.long)
    entity_pos = torch.empty(num_entities, dtype=torch.long)
    entity_pos[data_t] = torch.arange(num_entities)
    return data_t, offsets, entity_pos


class HierarchyNegativeSampler(NegativeSampler):
    r"""A sampler that corrupts a hierarchy edge with a node at the *same depth* ("near-miss").

    For a positive edge $(h, r, t)$ (parent→child), the tail is replaced by another node at $t$'s depth
    (a plausible alternative child) and the head by another node at $h$'s depth (a plausible alternative
    parent). Same-level nodes are much harder negatives than uniformly random entities, yet drawing
    them is a vectorized CSR lookup just like :class:`pykeen.sampling.PseudoTypedNegativeSampler`.

    Three knobs guard the failure modes of pure type-restriction:

    * ``head_corruption_prob`` fixes the head/tail split (a coin flip, *not* dictated by how many
      parents vs. children exist), so parent prediction keeps getting trained.
    * ``hard_ratio`` blends in uniform negatives; an entity whose depth level has no alternative also
      falls back to uniform. This keeps the sampler well-defined on shallow/star hierarchies and
      dilutes false-negative bias.
    * ``filtered`` (recommended ``True``) masks the same-depth draws that happen to be real edges.

    .. note::

        ``ponytail:`` depth is a coarse proxy for locality — same-depth nodes can sit in far-away
        subtrees. If that proves too easy on very wide trees, restrict the bucket to same-depth nodes
        within a small graph distance (cousins); that is the upgrade path, not built speculatively.
    """

    #: entity ids grouped by depth, shape: (num_entities,)
    data: LongTensor
    #: CSR offsets into ``data``, shape: (num_depths + 1,)
    offsets: LongTensor
    #: each entity's index within ``data``, shape: (num_entities,)
    entity_pos: LongTensor
    #: each entity's depth, shape: (num_entities,)
    node_depth: LongTensor

    def __init__(
        self,
        *,
        mapped_triples: MappedTriples,
        hierarchy_relation: int | None = None,
        hard_ratio: float = 0.5,
        head_corruption_prob: float = 0.5,
        **kwargs,
    ) -> None:
        """Instantiate the hierarchy negative sampler.

        :param mapped_triples: the positive training triples; their depths define the buckets.
        :param hierarchy_relation: if given, only edges with this relation id define the hierarchy.
        :param hard_ratio: probability that a negative is a same-depth ("hard") draw rather than
            uniform. ``0.0`` reproduces uniform sampling, ``1.0`` is fully same-depth.
        :param head_corruption_prob: probability of corrupting the head (vs. the tail) of a positive.
        :param kwargs: Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        self.hard_ratio = hard_ratio
        self.head_corruption_prob = head_corruption_prob
        depth = _compute_depths(mapped_triples, self.num_entities, hierarchy_relation)
        data, offsets, entity_pos = _build_depth_index(depth, self.num_entities)
        self.register_buffer("data", data)
        self.register_buffer("offsets", offsets)
        self.register_buffer("entity_pos", entity_pos)
        self.register_buffer("node_depth", torch.as_tensor(depth, dtype=torch.long))

    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:  # noqa: D102
        device = positive_batch.device
        batch_size = positive_batch.shape[0]
        size = (batch_size, self.num_negs_per_pos)

        # shape: (batch_size, num_negs_per_pos, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

        # choose the slot to corrupt (head=0 / tail=2) and read the entity currently sitting there
        slot = torch.where(torch.rand(size, device=device) < self.head_corruption_prob, _HEAD, _TAIL)
        original = negative_batch.gather(2, slot.unsqueeze(-1)).squeeze(-1)

        # --- hard path: a different entity at the same depth as ``original`` ---
        depth = self.node_depth[original]
        start = self.offsets[depth]
        bucket_size = self.offsets[depth + 1] - start
        local_original = self.entity_pos[original] - start
        # draw a local index in [0, bucket_size - 1) then shift past ``original`` -> excludes itself
        local = (torch.rand(size, device=device) * (bucket_size - 1).clamp(min=1)).long()
        local = local + (local >= local_original).long()
        hard = self.data[(start + local).clamp(max=self.num_entities - 1)]

        # --- uniform path: any other entity (also the fallback when a depth level is a singleton) ---
        uniform = torch.randint(high=self.num_entities - 1, size=size, device=device)
        uniform = uniform + (uniform >= original).long()

        use_uniform = (torch.rand(size, device=device) >= self.hard_ratio) | (bucket_size <= 1)
        replacement = torch.where(use_uniform, uniform, hard)

        negative_batch.scatter_(2, slot.unsqueeze(-1), replacement.unsqueeze(-1))
        return negative_batch
