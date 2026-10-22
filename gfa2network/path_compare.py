"""Base-aware shared content, ordered shared blocks and path witnesses.

These measures compare genome paths through the segments they traverse,
counted in bases rather than vertices, so a sequence-preserving change of
segmentation (or renaming) leaves them unchanged.

Content
    Each step contributes the bases of its segment that the path does not
    already spell through a neighbouring step. The bases of an exact overlap
    are removed from one of the two segments, chosen by the strand-canonical
    form of the link, so a path and its reverse-complement traversal (or a
    rotation of a declared circular path, whose closing link then counts too)
    lose the same forward-strand bases. A complete path's content totals its
    spelled length unless overlaps on both sides of a segment meet. Content is
    strand-independent. ``copies='unique'`` counts each segment base once per
    path (set semantics); ``copies='count'`` counts repeated visits
    (multiset: shared = sum of the smaller copy number). Steps with an unknown
    segment length are excluded and reported; steps whose incoming overlap is
    unknown contribute their whole segment and are reported as uncertain.
    Shared content means the paths traverse the same graph segments. It is not
    an alignment identity (ANI) and implies nothing about homology beyond what
    the graph construction asserted.

Blocks
    A shared block is a maximal run of consecutive steps visiting the same
    segments with the same orientations in both paths (``forward``), or in
    reverse order and opposite orientation (``reverse``). Blocks never cross a
    jump, a W chunk gap, or a transition without supported link evidence.

Collinear chain
    The heaviest set of non-overlapping blocks in the same order in both paths,
    weighted by each block's shared bases: for every pair of matched steps, the
    overlap of the segment bases each step contributes. A chain therefore never
    exceeds the copy-counted shared content of the two paths. Blocks with unknown segment lengths
    weigh their known bases and are marked incomplete. It measures conserved order; reordered content lowers it
    while leaving shared content unchanged. Blocks are chained whole, not
    trimmed. ``circular=True`` additionally lets the chain start anywhere in
    both paths; wrapping across a path's end requires supported closing-link
    evidence.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    _HAS_PANDAS = False

from .genome_paths import _incoming, _raw, _text

COPIES = ("unique", "count")
ORIENTATIONS = ("forward", "both")
MEASURES = ("jaccard", "shared", "containment")
_FLIP = {"+": "-", "-": "+"}


@dataclass(frozen=True)
class PathContent:
    """Segment bases one path traverses, with evidence counts."""

    name: str
    bases: int  # unique segment bases
    copy_bases: int  # bases counting repeated visits; spelled length if complete
    steps: int
    unresolved_steps: Tuple[int, ...]  # unknown segment length
    uncertain_overlap_steps: Tuple[int, ...]  # whole segment counted

    @property
    def evidence(self):
        if self.unresolved_steps or self.uncertain_overlap_steps:
            return "partial"
        return "complete"


@dataclass(frozen=True)
class ContentComparison:
    copies: str
    a_bases: int
    b_bases: int
    shared_bases: int
    union_bases: int
    jaccard: Optional[float]
    containment_a: Optional[float]  # fraction of A's content shared with B
    containment_b: Optional[float]
    a: PathContent
    b: PathContent
    note: str = ""


@dataclass(frozen=True)
class Witness:
    """Where a block lies in one path: steps and coordinates.

    ``reverse`` marks a block read against the path's direction. ``wraps``
    marks a block crossing the end of a circular path; ``end`` is then below
    ``start``. Coordinates are ``None`` when lengths or overlaps are unknown.
    """

    path: str
    first_step: int
    last_step: int
    start: Optional[int]
    end: Optional[int]
    coordinates: str
    reverse: bool = False
    wraps: bool = False


@dataclass(frozen=True)
class SharedBlock:
    """``bases`` sums, over matched step pairs, the shared contributed bases.

    ``complete`` is false when a segment length in the block is unknown; the
    known bases are still counted.
    """

    orientation: str  # "forward" or "reverse"
    steps: int
    bases: int
    first_segment: str
    last_segment: str
    a: Witness
    b: Witness
    complete: bool = True


@dataclass(frozen=True)
class CollinearChain:
    orientation: str
    bases: int
    fraction_a: Optional[float]
    fraction_b: Optional[float]
    blocks: Tuple[SharedBlock, ...]
    rotation: Optional[Tuple[int, int]] = None
    exhaustive: bool = True
    complete: bool = True


@dataclass
class PathComparison:
    a: str
    b: str
    content: ContentComparison
    blocks: Tuple[SharedBlock, ...]
    collinear: CollinearChain
    reverse_collinear: Optional[CollinearChain]
    policy: dict

    def to_dict(self):
        result = asdict(self)
        result["content"]["a"]["evidence"] = self.content.a.evidence
        result["content"]["b"]["evidence"] = self.content.b.evidence
        return result


# ---------------------------------------------------------------- content


def _link_key(first, second):
    (a, oa), (b, ob) = first, second
    return min((a, oa, b, ob), (b, _FLIP[ob], a, _FLIP[oa]))


def _attribution(paths, name, layout, closing=None):
    """Per-step forward-strand intervals after removing overlaps canonically.

    Returns ``(intervals, contributions, unresolved, uncertain)``: one
    ``(segment, start, end)`` or ``None`` per step, the bases each step
    contributes (``None`` if its length is unknown), and the steps with an
    unknown length or an unknown incoming overlap. ``closing`` is
    ``(contiguous, overlap)`` for the link from the last step back to the
    first, when the path is declared circular.
    """
    steps = paths[name].steps
    n = len(steps)
    tokens = [(_raw(step.segment), step.orientation) for step in steps]
    prefix, suffix = [0] * n, [0] * n
    uncertain = []

    def remove(before, after, overlap):
        if (
            tokens[before][0],
            tokens[before][1],
            tokens[after][0],
            tokens[after][1],
        ) == _link_key(tokens[before], tokens[after]):
            prefix[after] += overlap  # the later segment's leading bases
        else:
            suffix[before] += overlap  # the earlier segment's trailing bases

    for i in range(1, n):
        overlap = layout.overlaps[i]
        if overlap is None:
            uncertain.append(i)
        elif overlap and layout.contiguous[i]:
            remove(i - 1, i, overlap)
    if closing is not None and closing[0] and closing[1] and n:
        remove(n - 1, 0, closing[1])
    intervals, contributions, unresolved = [], [], []
    for i, (segment, orientation) in enumerate(tokens):
        length = layout.lengths[i]
        if length is None:
            unresolved.append(i)
            intervals.append(None)
            contributions.append(None)
            continue
        lead, trail = min(prefix[i], length), min(suffix[i], length)
        if lead + trail >= length:
            interval = (0, 0)
        elif orientation == "+":
            interval = (lead, length - trail)
        else:
            interval = (trail, length - lead)
        intervals.append((segment,) + interval)
        contributions.append(interval[1] - interval[0])
    return intervals, contributions, tuple(unresolved), tuple(uncertain)


class _Content:
    def __init__(self, paths, name, layout, closing=None):
        intervals, contributions, unresolved, uncertain = _attribution(
            paths, name, layout, closing
        )
        self.contributions = contributions
        self.intervals = [None if item is None else item[1:] for item in intervals]
        self.by_segment = defaultdict(list)
        for item in intervals:
            if item is not None and item[2] > item[1]:
                self.by_segment[item[0]].append((item[1], item[2]))
        copy_bases = sum(c for c in contributions if c is not None)
        unique = sum(_merged_length(parts) for parts in self.by_segment.values())
        self.summary = PathContent(
            _text(paths[name].name),
            unique,
            copy_bases,
            len(paths[name].steps),
            unresolved,
            uncertain,
        )


def _merged_length(parts):
    total, current_start, current_end = 0, None, None
    for start, end in sorted(parts):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def _sweep(a_parts, b_parts):
    """``(shared_unique, shared_copies)`` for two interval lists on one segment."""
    events = defaultdict(lambda: [0, 0])
    for start, end in a_parts:
        events[start][0] += 1
        events[end][0] -= 1
    for start, end in b_parts:
        events[start][1] += 1
        events[end][1] -= 1
    shared_unique = shared_copies = 0
    depth_a = depth_b = 0
    previous = None
    for position in sorted(events):
        if previous is not None and depth_a and depth_b:
            span = position - previous
            shared_unique += span
            shared_copies += span * min(depth_a, depth_b)
        delta = events[position]
        depth_a += delta[0]
        depth_b += delta[1]
        previous = position
    return shared_unique, shared_copies


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _compare_content(a, b, copies):
    shared_unique = shared_copies = 0
    for segment, parts in a.by_segment.items():
        other = b.by_segment.get(segment)
        if other:
            unique, counted = _sweep(parts, other)
            shared_unique += unique
            shared_copies += counted
    if copies == "unique":
        a_total, b_total, shared = a.summary.bases, b.summary.bases, shared_unique
    else:
        a_total, b_total, shared = (
            a.summary.copy_bases,
            b.summary.copy_bases,
            shared_copies,
        )
    union = a_total + b_total - shared
    note = "" if union else "no resolved bases in either path"
    return ContentComparison(
        copies,
        a_total,
        b_total,
        shared,
        union,
        _ratio(shared, union),
        _ratio(shared, a_total),
        _ratio(shared, b_total),
        a.summary,
        b.summary,
        note,
    )


# ----------------------------------------------------------------- blocks


class _Track:
    """A path as tokens with contiguity, optionally rotated or reversed."""

    def __init__(self, paths, name, layout, closing, intervals):
        path = paths[name]
        self.name = _text(path.name)
        self.layout = layout
        self.intervals = intervals  # contributed (start, end) per original step
        self.tokens = [(_raw(s.segment), s.orientation) for s in path.steps]
        self.labels = [_text(s.segment) for s in path.steps]
        self.contiguous = list(layout.contiguous)
        self.overlaps = list(layout.overlaps)
        self.origin = list(range(len(self.tokens)))
        self.direction = 1
        self.closing = closing  # (contiguous, overlap) back to step 0

    def rotated(self, offset):
        track = _Track.__new__(_Track)
        n = len(self.tokens)
        track.__dict__.update(self.__dict__)
        order = [(offset + t) % n for t in range(n)]
        track.tokens = [self.tokens[i] for i in order]
        track.labels = [self.labels[i] for i in order]
        track.origin = [self.origin[i] for i in order]
        contiguous, overlaps = [False], [0]
        for t in range(1, n):
            i = order[t]
            if i == 0:
                # The original end-to-start junction now lies inside.
                contiguous.append(self.closing[0])
                overlaps.append(self.closing[1] if self.closing[0] else 0)
            else:
                contiguous.append(self.contiguous[i])
                overlaps.append(self.overlaps[i])
        track.contiguous, track.overlaps = contiguous, overlaps
        return track

    def reversed(self):
        track = _Track.__new__(_Track)
        n = len(self.tokens)
        track.__dict__.update(self.__dict__)
        track.tokens = [(seg, _FLIP[ori]) for seg, ori in reversed(self.tokens)]
        track.labels = self.labels[::-1]
        track.origin = self.origin[::-1]
        track.direction = -self.direction
        track.contiguous = [False] + [self.contiguous[n - t] for t in range(1, n)]
        track.overlaps = [0] + [self.overlaps[n - t] for t in range(1, n)]
        return track

    def length(self, t):
        return self.layout.lengths[self.origin[t]]

    def interval(self, t):
        return self.intervals[self.origin[t]]

    def witness(self, t0, t1):
        first, last = self.origin[t0], self.origin[t1]
        if self.direction == -1:
            first, last = last, first
        starts, lengths = self.layout.starts, self.layout.lengths
        start = starts[first]
        end = (
            starts[last] + lengths[last]
            if starts[last] is not None and lengths[last] is not None
            else None
        )
        return Witness(
            self.name,
            first,
            last,
            start,
            end,
            self.layout.coordinates,
            self.direction == -1,
            first > last,
        )


def _maximal_blocks(a, b, max_matches):
    positions = defaultdict(list)
    for j, token in enumerate(b.tokens):
        positions[token].append(j)
    pairs = sum(len(positions.get(token, ())) for token in a.tokens)
    if pairs > max_matches:
        raise ValueError(
            f"{pairs} step matches exceed max_matches={max_matches}; "
            "repetitive paths need a larger limit"
        )
    n, m = len(a.tokens), len(b.tokens)
    blocks = []
    for i, token in enumerate(a.tokens):
        for j in positions.get(token, ()):
            if (
                i
                and j
                and a.contiguous[i]
                and b.contiguous[j]
                and a.tokens[i - 1] == b.tokens[j - 1]
            ):
                continue  # not left-maximal
            k = 0
            while (
                i + k + 1 < n
                and j + k + 1 < m
                and a.contiguous[i + k + 1]
                and b.contiguous[j + k + 1]
                and a.tokens[i + k + 1] == b.tokens[j + k + 1]
            ):
                k += 1
            blocks.append((i, i + k, j, j + k))
    return blocks


def _block(a, b, span, orientation):
    i0, i1, j0, j1 = span
    bases, complete = 0, True
    for offset in range(i1 - i0 + 1):
        first, second = a.interval(i0 + offset), b.interval(j0 + offset)
        if first is None or second is None:
            complete = False
            continue
        bases += max(0, min(first[1], second[1]) - max(first[0], second[0]))
    return SharedBlock(
        orientation,
        i1 - i0 + 1,
        bases,
        a.labels[i0],
        a.labels[i1],
        a.witness(i0, i1),
        b.witness(j0, j1),
        complete,
    )


class _MaxTree:
    """Fenwick tree for prefix maxima of (weight, index)."""

    def __init__(self, size):
        self.tree = [(0, -1)] * (size + 1)

    def update(self, position, value):
        position += 1
        while position < len(self.tree):
            if value > self.tree[position]:
                self.tree[position] = value
            position += position & -position

    def query(self, position):
        best = (0, -1)
        while position > 0:
            if self.tree[position] > best:
                best = self.tree[position]
            position -= position & -position
        return best


def _chain(spans, weights):
    """Heaviest chain with strictly increasing, non-overlapping A and B spans."""
    if not spans:
        return 0, []
    ends = sorted({span[3] for span in spans})
    tree = _MaxTree(len(ends))
    best, previous = [0] * len(spans), [-1] * len(spans)
    by_start = sorted(range(len(spans)), key=lambda k: (spans[k][0], spans[k][2]))
    by_end = sorted(range(len(spans)), key=lambda k: spans[k][1])
    pointer = 0
    for k in by_start:
        while pointer < len(by_end) and spans[by_end[pointer]][1] < spans[k][0]:
            done = by_end[pointer]
            tree.update(bisect_left(ends, spans[done][3]), (best[done], done))
            pointer += 1
        weight, index = tree.query(bisect_left(ends, spans[k][2]))
        best[k], previous[k] = weight + weights[k], index
    k = max(range(len(spans)), key=lambda x: (best[x], -x))
    total, chain = best[k], []
    while k != -1:
        chain.append(k)
        k = previous[k]
    return total, chain[::-1]


def _collinear(a, b, orientation, max_matches):
    spans = _maximal_blocks(a, b, max_matches)
    blocks = [_block(a, b, span, orientation) for span in spans]
    weights = [block.bases for block in blocks]
    total, chain = _chain(spans, weights)
    return total, [blocks[k] for k in chain], spans, weights


def _closing(paths, name, unknown_overlap):
    path = paths[name]
    transition = paths.closing_transition(name)
    overlap, joined = _incoming(transition, path, path.steps[0], unknown_overlap)
    return joined, overlap


def compare_paths(
    paths,
    a,
    b,
    *,
    copies="unique",
    orientation="forward",
    circular=False,
    unknown_overlap="strict",
    max_matches=5_000_000,
    max_rotations=64,
):
    """Compare two paths of a :class:`~gfa2network.genome_paths.GenomePathSet`.

    Returns a :class:`PathComparison` with base-aware shared content, every
    maximal shared block (with witnesses in both paths), and the collinear
    chain. ``orientation='both'`` also finds reverse blocks and a reverse
    chain, for a genome stored on the opposite strand. ``circular=True``
    declares both genomes circular; the chain search tries a rotation at the
    start of each of the ``max_rotations`` heaviest maximal blocks, and
    ``exhaustive`` reports whether every such start was tried. Each rotation's
    best chain is exact; with repeated segments a better rotation can start
    inside a block, so the circular optimum is then a lower bound.
    """
    if copies not in COPIES:
        raise ValueError("copies must be unique or count")
    if orientation not in ORIENTATIONS:
        raise ValueError("orientation must be forward or both")
    if not isinstance(max_rotations, int) or max_rotations < 0:
        raise ValueError("max_rotations must be a nonnegative integer")
    layouts = {
        name: paths.layout(name, unknown_overlap=unknown_overlap) for name in (a, b)
    }
    closings = {
        name: _closing(paths, name, unknown_overlap) if circular else (False, 0)
        for name in (a, b)
    }
    contents = {
        name: _Content(paths, name, layouts[name], closings[name] if circular else None)
        for name in (a, b)
    }
    content = _compare_content(contents[a], contents[b], copies)
    tracks = {
        name: _Track(
            paths,
            name,
            layouts[name],
            closings[name],
            contents[name].intervals,
        )
        for name in (a, b)
    }
    first, second = tracks[a], tracks[b]
    a_len, b_len = content.a.copy_bases, content.b.copy_bases

    def best_chain(target, label):
        total, chain, spans, weights = _collinear(first, target, label, max_matches)
        rotation, exhaustive = None, True
        if circular and spans:
            order = sorted(range(len(spans)), key=lambda k: (-weights[k], k))
            candidates, seen = [], set()
            for k in order:
                pair = (spans[k][0], spans[k][2])
                if pair not in seen:
                    seen.add(pair)
                    candidates.append(pair)
                    if len(candidates) > max_rotations:
                        break
            exhaustive = len(candidates) <= max_rotations
            for i0, j0 in candidates[:max_rotations]:
                rotated = _collinear(
                    first.rotated(i0), target.rotated(j0), label, max_matches
                )
                if rotated[0] > total:
                    total, chain = rotated[0], rotated[1]
                    rotation = (
                        first.origin[i0],
                        target.origin[j0],
                    )
        return CollinearChain(
            label,
            total,
            _ratio(total, a_len),
            _ratio(total, b_len),
            tuple(chain),
            rotation,
            exhaustive,
            all(block.complete for block in chain),
        )

    forward_blocks = [
        _block(first, second, span, "forward")
        for span in _maximal_blocks(first, second, max_matches)
    ]
    collinear = best_chain(second, "forward")
    reverse_collinear, reverse_blocks = None, []
    if orientation == "both":
        flipped = second.reversed()
        reverse_blocks = [
            _block(first, flipped, span, "reverse")
            for span in _maximal_blocks(first, flipped, max_matches)
        ]
        reverse_collinear = best_chain(flipped, "reverse")
    policy = {
        "copies": copies,
        "orientation": orientation,
        "circular": circular,
        "unknown_overlap": unknown_overlap,
        "content": "segment bases traversed; overlap bases removed from the strand-canonical side",
        "blocks": "maximal runs of identical oriented steps across supported links",
    }
    return PathComparison(
        _text(paths[a].name),
        _text(paths[b].name),
        content,
        tuple(forward_blocks + reverse_blocks),
        collinear,
        reverse_collinear,
        policy,
    )


def shared_content_matrix(
    paths,
    names=None,
    *,
    measure="jaccard",
    copies="unique",
    unknown_overlap="strict",
    circular=(),
):
    """Pairwise base-aware shared content for many paths.

    ``measure`` selects the returned matrix: ``jaccard`` (shared / union),
    ``shared`` (bases) or ``containment`` (row path's fraction shared with the
    column path). Returns ``(matrix, details)``; ``details`` has every matrix,
    per-path :class:`PathContent` and the policy. Pairs with no resolved bases
    are ``NaN`` rather than 0, so missing evidence is not reported as absence.
    Segments every path traverses whole are summed with sparse products;
    partially covered segments use an interval sweep. ``circular`` names
    paths (or ``True`` for all) whose closing-link overlap is attributed, as in
    :func:`compare_paths`.
    """
    if measure not in MEASURES:
        raise ValueError("measure must be jaccard, shared or containment")
    if copies not in COPIES:
        raise ValueError("copies must be unique or count")
    names = list(paths) if names is None else list(names)
    if circular is True:
        ring = set(map(_text, names))
    else:
        ring = set(map(_text, circular or ()))
    contents = [
        _Content(
            paths,
            name,
            paths.layout(name, unknown_overlap=unknown_overlap),
            _closing(paths, name, unknown_overlap) if _text(name) in ring else None,
        )
        for name in names
    ]
    n = len(names)
    shared = np.zeros((n, n), dtype=np.int64)
    segments = {}
    partial = defaultdict(list)
    whole = defaultdict(lambda: defaultdict(int))
    for p, content in enumerate(contents):
        for segment, parts in content.by_segment.items():
            length = paths.segment_length(segment)
            if all(part == (0, length) for part in parts):
                whole[segment][p] = len(parts)
            else:
                partial[segment].append(p)
    # Any partial coverer routes the whole segment through the sweep.
    for segment in partial:
        for p in whole.pop(segment, {}):
            partial[segment].append(p)
    columns = {segment: k for k, segment in enumerate(whole)}
    if columns:
        layers = max(c for counts in whole.values() for c in counts.values())
        layers = 1 if copies == "unique" else layers
        lengths = np.asarray(
            [paths.segment_length(segment) for segment in columns], dtype=np.int64
        )
        for layer in range(1, layers + 1):
            rows, cols = [], []
            for segment, counts in whole.items():
                for p, count in counts.items():
                    if count >= layer:
                        rows.append(p)
                        cols.append(columns[segment])
            incidence = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.int64), (rows, cols)),
                shape=(n, len(columns)),
            )
            shared += (incidence.multiply(lengths) @ incidence.T).toarray()
    for segment, members in partial.items():
        for x, p in enumerate(members):
            for q in members[x:]:
                unique, counted = _sweep(
                    contents[p].by_segment[segment], contents[q].by_segment[segment]
                )
                value = unique if copies == "unique" else counted
                shared[p, q] += value
                if p != q:
                    shared[q, p] += value
    totals = np.asarray(
        [
            c.summary.bases if copies == "unique" else c.summary.copy_bases
            for c in contents
        ],
        dtype=np.int64,
    )
    union = totals[:, None] + totals[None, :] - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, shared / np.maximum(union, 1), np.nan)
        containment = np.where(
            totals[:, None] > 0, shared / np.maximum(totals[:, None], 1), np.nan
        )
    chosen = {"jaccard": jaccard, "shared": shared, "containment": containment}[measure]
    labels = [c.summary.name for c in contents]
    output = (
        pd.DataFrame(chosen, index=labels, columns=labels) if _HAS_PANDAS else chosen
    )
    details = {
        "names": labels,
        "measure": measure,
        "copies": copies,
        "unknown_overlap": unknown_overlap,
        "shared": shared,
        "union": union,
        "jaccard": jaccard,
        "containment": containment,
        "paths": [c.summary for c in contents],
        "note": "graph-traversed segment bases; not alignment identity",
    }
    return output, details


__all__ = [
    "compare_paths",
    "shared_content_matrix",
    "PathComparison",
    "ContentComparison",
    "PathContent",
    "SharedBlock",
    "CollinearChain",
    "Witness",
]
