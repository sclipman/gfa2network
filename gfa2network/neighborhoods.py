"""Shared blocks that connect labeled groups of genome paths.

Given a label for each path (for example a taxonomic group supplied from
outside the graph), a *bridge block* is a run of consecutive steps that paths
from at least ``min_groups`` groups traverse contiguously. A run extends while
every step-to-step adjacency inside it is traversed (in either strand) by
exactly the same set of paths, so each member path reads the whole block as
contiguous sequence; a single step forms a block when its segment is visited by
paths of enough groups. Paths that only visit some of a block's segments, or
visit them in another arrangement, are not members. Each segment belongs to at
most one block; paths are walked in the given order, so the first path to reach
a segment fixes the block's reference direction.

For every path traversing a block the result gives a witness (steps and
coordinates, reversed if the path reads the block on the other strand) and the
local context: the oriented segments immediately before and after the block in
the block's direction, or ``None`` at a path end, a jump or a transition without
link evidence. Traversals are grouped by context. A block shared across groups
whose contexts are all group-specific is embedded in different neighborhoods;
a context class spanning groups shows the shared region extends beyond the
block's flanks.

These are graph observations. A bridge block shows that the graph construction
placed sequence from different groups on the same segments; it is not by itself
evidence of recombination, common ancestry or homology. Check the block
sequence against the source genomes and against construction choices before
interpreting it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .genome_paths import _raw, _text
from .model import reverse_complement
from .path_compare import Witness

_FLIP = {"+": "-", "-": "+"}


@dataclass(frozen=True)
class BlockTraversal:
    path: str
    group: str
    witness: Optional[Witness]  # None when the path's visits are not one run
    left: Optional[str]
    right: Optional[str]
    contiguous: bool


@dataclass(frozen=True)
class BridgeBlock:
    """A block, its contiguous member paths, and those that only visit it.

    ``groups`` and ``paths`` count contiguous traversals only. ``partial``
    lists paths that support every adjacency of the block but do not read it
    as one contiguous run (for example by reusing short segments elsewhere).
    """

    segments: Tuple[str, ...]  # oriented, in the reference direction
    bases: Optional[int]
    sequence: Optional[bytes]
    groups: Tuple[str, ...]
    paths: Tuple[str, ...]
    traversals: Tuple[BlockTraversal, ...]
    contexts: Tuple[dict, ...]
    partial: Tuple[str, ...] = ()

    @property
    def group_specific_contexts(self):
        """Every context class holds paths from a single group."""
        return all(len(c["groups"]) == 1 for c in self.contexts)

    def to_dict(self):
        result = asdict(self)
        result["sequence"] = (
            self.sequence.decode("ascii") if self.sequence is not None else None
        )
        result["group_specific_contexts"] = self.group_specific_contexts
        return result


def _token(segment, orientation):
    return f"{_text(segment)}{orientation}"


def bridge_blocks(
    paths,
    groups,
    *,
    min_groups=2,
    min_bases=0,
    unknown_overlap="strict",
):
    """Find blocks shared by paths from at least ``min_groups`` groups.

    ``groups`` maps path names to labels; unlabeled paths are ignored. Blocks
    shorter than ``min_bases`` (or of unknown length when ``min_bases`` > 0)
    are dropped. Returns blocks ordered by group count, then bases.
    """
    names = [name for name in paths if name in groups]
    layouts = {
        name: paths.layout(name, unknown_overlap=unknown_overlap) for name in names
    }
    tokens = {
        name: [(_raw(s.segment), s.orientation) for s in paths[name].steps]
        for name in names
    }
    visits = defaultdict(set)
    adjacent = defaultdict(set)
    for name in names:
        walk, layout = tokens[name], layouts[name]
        for i, (segment, orientation) in enumerate(walk):
            visits[segment].add(name)
            if i and layout.contiguous[i]:
                adjacent[_adjacency(walk[i - 1], walk[i])].add(name)
    frozen = {}

    def support(members):
        key = frozenset(members)
        return frozen.setdefault(key, key)

    def enough(members):
        return len({groups[n] for n in members}) >= min_groups

    assigned = set()
    blocks = []
    for name in names:
        walk, layout = tokens[name], layouts[name]
        i = 0
        while i < len(walk):
            if walk[i][0] in assigned:
                i += 1
                continue
            run, members = [i], None
            j = i + 1
            while (
                j < len(walk)
                and layout.contiguous[j]
                and walk[j][0] not in assigned
                and all(walk[j][0] != walk[k][0] for k in run)
            ):
                shared = support(adjacent[_adjacency(walk[j - 1], walk[j])])
                if members is None:
                    if not enough(shared):
                        break
                    members = shared
                elif shared != members:
                    break
                run.append(j)
                j += 1
            if members is None:
                members = support(visits[walk[i][0]])
            if enough(members):
                blocks.append((name, run, members))
                assigned.update(walk[k][0] for k in run)
                i = run[-1] + 1
            else:
                i += 1
    index = {name: _positions(tokens[name]) for name in names}
    results = []
    for reference, run, members in blocks:
        block = _describe(paths, layouts, groups, reference, run, members, index)
        if min_bases and (block.bases is None or block.bases < min_bases):
            continue
        if len(block.groups) < min_groups:
            continue  # enough groups only among non-contiguous visitors
        results.append(block)
    results.sort(key=lambda b: (-len(b.groups), -(b.bases or 0), b.segments))
    return results


def _positions(tokens):
    found = defaultdict(list)
    for position, token in enumerate(tokens):
        found[token].append(position)
    return tokens, found


def _adjacency(first, second):
    (a, oa), (b, ob) = first, second
    return min((a, oa, b, ob), (b, _FLIP[ob], a, _FLIP[oa]))


def _describe(paths, layouts, groups, reference, run, members, indexed):
    ref_steps = paths[reference].steps
    tokens = [(_raw(ref_steps[i].segment), ref_steps[i].orientation) for i in run]
    layout = layouts[reference]
    bases = 0
    for position, index in enumerate(run):
        length = layout.lengths[index]
        overlap = layout.overlaps[index] if position else 0
        if bases is None or length is None or overlap is None:
            bases = None
        else:
            bases += length - overlap
    sequence = _sequence(paths, reference, run, layout)
    members = sorted(members)
    found = [
        _traverse(paths, layouts[name], name, groups[name], tokens, indexed[name])
        for name in members
    ]
    traversals = [item for item in found if item.contiguous]
    partial = tuple(item.path for item in found if not item.contiguous)
    contexts = defaultdict(list)
    for item in traversals:
        if item.contiguous:
            contexts[(item.left, item.right)].append(item)
    context_list = tuple(
        {
            "left": left,
            "right": right,
            "paths": [t.path for t in items],
            "groups": sorted({t.group for t in items}),
        }
        for (left, right), items in sorted(
            contexts.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))
        )
    )
    return BridgeBlock(
        tuple(_token(segment, ori) for segment, ori in tokens),
        bases,
        sequence,
        tuple(sorted({item.group for item in traversals})),
        tuple(item.path for item in traversals),
        tuple(traversals),
        context_list,
        partial,
    )


def _sequence(paths, name, run, layout):
    parts = bytearray()
    for position, index in enumerate(run):
        step = paths[name].steps[index]
        sequence = (
            paths._sequences.get(_raw(step.segment)) if paths._sequences else None
        )
        if sequence is None or (position and layout.overlaps[index] is None):
            return None
        if step.orientation == "-":
            sequence = reverse_complement(sequence)
        parts.extend(sequence[(layout.overlaps[index] or 0) if position else 0 :])
    return bytes(parts)


def _traverse(paths, layout, name, group, tokens, indexed):
    path_tokens, positions = indexed
    n, k = len(path_tokens), len(tokens)
    flipped = [(segment, _FLIP[ori]) for segment, ori in reversed(tokens)]
    candidates = sorted(
        [(start, False) for start in positions.get(tokens[0], ())]
        + [(start, True) for start in positions.get(flipped[0], ())]
    )
    label = _text(paths[name].name)
    for start, reverse in candidates:
        pattern = flipped if reverse else tokens
        if path_tokens[start : start + k] != pattern:
            continue
        if not all(layout.contiguous[start + j] for j in range(1, k)):
            continue
        first, last = start, start + k - 1
        before = path_tokens[first - 1] if first and layout.contiguous[first] else None
        after = (
            path_tokens[last + 1]
            if last + 1 < n and layout.contiguous[last + 1]
            else None
        )
        if reverse:
            before, after = (
                (after[0], _FLIP[after[1]]) if after else None,
                (before[0], _FLIP[before[1]]) if before else None,
            )
        starts, lengths = layout.starts, layout.lengths
        end = (
            starts[last] + lengths[last]
            if starts[last] is not None and lengths[last] is not None
            else None
        )
        witness = Witness(
            label, first, last, starts[first], end, layout.coordinates, reverse
        )
        return BlockTraversal(
            label,
            group,
            witness,
            _token(*before) if before else None,
            _token(*after) if after else None,
            True,
        )
    return BlockTraversal(label, group, None, None, None, False)
