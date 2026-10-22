"""Optional rich genome paths: ordered steps, transition evidence and spelling.

``parse_gfa(..., return_paths=True)`` returns each path as a list of vertex
identifiers. This module keeps what a path declares in the input instead:
ordered oriented steps including repeated visits, P overlaps and link/jump
separators, W chunk coordinates and sample identity. Each consecutive transition
is checked against the declared relations. Sequence is spelled only where
segment sequences, overlaps and transitions make it unambiguous; every other
case is reported with a reason rather than filled in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import gzip
from pathlib import Path
import re
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from .model import reverse_complement
from .projection import classify_edge
from .parser import (
    GFAParser,
    Segment,
    Link,
    EdgeRecord,
    GapRecord,
    PathRecord,
    WalkRecord,
)
from .paths import PathCollector

Name = Union[str, bytes]

OVERLAP_POLICIES = ("strict", "zero")
# Issues describing sequence the input itself declares unknown. They split a
# spelling into pieces but are not fidelity failures.
DECLARED_GAPS = frozenset({"jump", "chunk_gap"})
MATCHES = frozenset(
    {
        "identical",
        "reverse_complement",
        "rotation",
        "rotation_reverse_complement",
        "coordinates_match",
        "pieces_located",
    }
)

_FLIP = {"+": "-", "-": "+"}
AMBIGUITY = ("exact", "as-n")
# Every byte other than an unambiguous base becomes N.
_AS_N = bytes(value if value in b"ACGTacgt" else ord("N") for value in range(256))
_CIGAR = re.compile(rb"(\d+)([MIDNSHPX=])")


def _is_cigar(value):
    operations = _CIGAR.findall(value)
    return bool(operations) and b"".join(n + op for n, op in operations) == value


def overlap_length(cigar):
    """Classify an overlap CIGAR as ``(state, length)``.

    ``None`` and ``*`` are ``unspecified``. A CIGAR made only of ``M``/``=``
    operations is ``exact``; spelling still verifies the overlapping bases.
    Insertions, deletions, mismatches, clipping and trace strings describe two
    different versions of the overlap, so spelling would have to choose one;
    they are ``unsupported``.
    """
    if cigar is None or cigar == b"*":
        return "unspecified", None
    if not _is_cigar(cigar):
        return "unsupported", None
    operations = _CIGAR.findall(cigar)
    if any(op not in (b"M", b"=") for _, op in operations):
        return "unsupported", None
    return "exact", sum(int(n) for n, _ in operations)


def _canonical(a, oa, b, ob):
    """A transition and its reverse-complement traversal share one key."""
    return min((a, oa, b, ob), (b, _FLIP[ob], a, _FLIP[oa]))


def _raw(value):
    return value if isinstance(value, bytes) else value.encode("ascii")


def _text(value):
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


class Relation(NamedTuple):
    """One declared traversal relation between two oriented segments."""

    kind: str  # "link" (L, dovetail E) or "jump" (J, G)
    record: str  # source record letter
    line: Optional[int]
    overlap_state: Optional[str] = None
    overlap_length: Optional[int] = None
    distance: Optional[int] = None  # an estimate; never sequence


class Interval(NamedTuple):
    """A subdivided vertex and its coordinates in the parent segment."""

    vertex: Name
    start: int
    end: int


@dataclass(frozen=True)
class PathStep:
    """One visit to a segment; repeated visits are separate steps.

    ``intervals`` lists the subdivided vertices in traversal order when the
    graph was built with ``split_on_alignment``.
    """

    segment: Name
    orientation: str
    ordinal: int
    chunk: int = 0
    intervals: Tuple[Interval, ...] = ()


@dataclass(frozen=True)
class PathTransition:
    """Evidence for the move from step ``index`` to step ``index + 1``.

    ``declared`` is what the path record states: ``link`` (P comma, W walk),
    ``jump`` (P semicolon), ``group`` (GFA2 O; an edge or a gap) or
    ``chunk-boundary`` (between W chunks, which declares no traversal).
    ``status`` is ``supported`` when one kind of matching relation exists with a
    consistent overlap, ``missing``, ``ambiguous`` when relations disagree,
    ``conflict`` when the P overlap disagrees with every relation, or
    ``boundary``. Relations match in either strand of the traversal.
    """

    index: int
    declared: str
    status: str
    relation: Optional[str] = None
    overlap_state: Optional[str] = None
    overlap_length: Optional[int] = None
    distance: Optional[int] = None
    evidence: Tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class PathChunk:
    """Steps from one P/O record or one W chunk, with W coordinates if given."""

    first_step: int
    last_step: int
    start: Optional[int] = None
    end: Optional[int] = None
    line: Optional[int] = None


@dataclass(frozen=True)
class PathIssue:
    code: str
    step: Optional[int] = None
    transition: Optional[int] = None
    detail: str = ""


@dataclass
class GenomePath:
    """A genome path as declared: identity, steps, transitions and chunks."""

    name: Name
    record_type: str
    naming: str
    steps: Tuple[PathStep, ...]
    transitions: Tuple[PathTransition, ...]
    chunks: Tuple[PathChunk, ...]
    sample: Optional[Name] = None
    haplotype: Optional[int] = None
    sequence_id: Optional[Name] = None
    tags: Tuple[Optional[dict], ...] = ()
    issues: Tuple[PathIssue, ...] = ()

    def vertices(self, oriented=False):
        """Graph vertex identifiers in traversal order, as the list API returns.

        Repeated visits are retained. ``oriented=True`` appends ``:+``/``:-`` for
        bidirected graphs; the step orientation itself never depends on it.
        """
        result = []
        for step in self.steps:
            ids = [item.vertex for item in step.intervals] or [step.segment]
            for vertex in ids:
                if oriented:
                    suffix = ":" + step.orientation
                    vertex = vertex + (
                        suffix.encode() if isinstance(vertex, bytes) else suffix
                    )
                result.append(vertex)
        return result

    @property
    def transition_counts(self):
        return dict(Counter(t.status for t in self.transitions))

    @property
    def validated(self):
        """Every declared traversal has exactly supported relation evidence."""
        return all(t.status in ("supported", "boundary") for t in self.transitions)


@dataclass(frozen=True)
class SpelledPiece:
    first_step: int
    last_step: int
    sequence: bytes


@dataclass(frozen=True)
class SpelledPath:
    """Sequence spelled from a path, split wherever evidence stops.

    ``complete`` means one contiguous sequence covers every step. ``partial``
    means at least one piece was spelled; ``unavailable`` means none was.
    ``chunk_spans`` gives ``(piece, start, end)`` for each fully spelled chunk.
    """

    name: Name
    status: str
    pieces: Tuple[SpelledPiece, ...]
    issues: Tuple[PathIssue, ...]
    unknown_overlap: str
    assumed_overlaps: Tuple[int, ...] = ()
    chunk_spans: Tuple[Optional[Tuple[int, int, int]], ...] = ()

    @property
    def sequence(self):
        return self.pieces[0].sequence if self.status == "complete" else None

    @property
    def failures(self):
        return tuple(i for i in self.issues if i.code not in DECLARED_GAPS)


class _Relations:
    def __init__(self, relations):
        self.relations = relations

    def resolve(self, index, source, target, declared, path_overlap=None):
        (a, oa), (b, ob) = source, target
        candidates = self.relations.get(_canonical(a, oa, b, ob), ())
        wanted = {"link": ("link",), "jump": ("jump",), "group": ("link", "jump")}[
            declared
        ]
        matches = [r for r in candidates if r.kind in wanted]
        evidence = tuple(
            r.record if r.line is None else f"{r.record}:{r.line}" for r in matches
        )
        if not matches:
            others = sorted({r.kind for r in candidates})
            detail = "no " + " or ".join(wanted) + " relation"
            if others:
                detail += "; found " + ", ".join(others)
            return PathTransition(index, declared, "missing", detail=detail)
        kinds = {r.kind for r in matches}
        if len(kinds) > 1:
            return PathTransition(
                index,
                declared,
                "ambiguous",
                evidence=evidence,
                detail="both link and jump relations",
            )
        if kinds.pop() == "jump":
            distances = {r.distance for r in matches}
            distance = next(iter(distances)) if len(distances) == 1 else None
            detail = "" if len(distances) == 1 else "jump estimates disagree"
            return PathTransition(
                index,
                declared,
                "supported",
                "jump",
                distance=distance,
                evidence=evidence,
                detail=detail,
            )
        exact = {r.overlap_length for r in matches if r.overlap_state == "exact"}
        unsupported = any(r.overlap_state == "unsupported" for r in matches)
        state, length = overlap_length(path_overlap)
        if state == "exact":
            if exact and length not in exact:
                return PathTransition(
                    index,
                    declared,
                    "conflict",
                    "link",
                    "exact",
                    length,
                    evidence=evidence,
                    detail=f"path overlap {length} disagrees with relation overlaps {sorted(exact)}",
                )
            return PathTransition(
                index, declared, "supported", "link", "exact", length, evidence=evidence
            )
        if state == "unsupported":
            return PathTransition(
                index,
                declared,
                "supported",
                "link",
                "unsupported",
                evidence=evidence,
                detail="path overlap " + path_overlap.decode("ascii", "replace"),
            )
        if len(exact) > 1 or (exact and unsupported):
            return PathTransition(
                index,
                declared,
                "ambiguous",
                "link",
                evidence=evidence,
                detail="relations declare different overlaps",
            )
        if exact:
            return PathTransition(
                index,
                declared,
                "supported",
                "link",
                "exact",
                exact.pop(),
                evidence=evidence,
            )
        state = "unsupported" if unsupported else "unspecified"
        return PathTransition(
            index, declared, "supported", "link", state, evidence=evidence
        )


class GenomePathBuilder:
    """Collect segments, traversal relations and paths in one record pass.

    Containments and non-dovetail GFA2 alignments are not traversal relations
    and are ignored here; they remain graph connectivity in ``parse_gfa``.
    """

    def __init__(self, *, sequences=True, raw_bytes=False, bidirected=False):
        self.keep_sequences = sequences
        self.raw_bytes = raw_bytes
        self.bidirected = bidirected
        self.lengths = {}
        self.sequences = {}
        self.relations = {}
        self.pending_edges = []
        self.intervals = {}
        self.collector = PathCollector()

    def observe(self, records, label="<stream>"):
        """Record each item, then pass it on unchanged (single-pass composition)."""
        for count, record in enumerate(records, 1):
            try:
                self.add(record)
            except ValueError as exc:
                line = getattr(record, "line_number", count)
                raise ValueError(f"{label}:{line}: {exc}") from exc
            yield record

    def add(self, record):
        line = getattr(record, "line_number", None)
        if isinstance(record, Segment):
            if record.parent is not None:
                self.intervals.setdefault(record.parent, []).append(
                    Interval(self._name(record.id), record.start, record.end)
                )
                return
            if record.id in self.lengths:
                raise ValueError(f"duplicate segment {record.id!r}")
            length = record.length
            if length is None and record.sequence is not None:
                length = len(record.sequence)
            self.lengths[record.id] = length
            if self.keep_sequences and record.sequence is not None:
                self.sequences[record.id] = record.sequence
        elif isinstance(record, Link):
            state, length = overlap_length(record.overlap)
            self._relate(record, Relation("link", "L", line, state, length))
        elif isinstance(record, GapRecord):
            letter = record.line_type or "J"
            self._relate(
                record, Relation("jump", letter, line, distance=record.distance)
            )
        elif isinstance(record, EdgeRecord):
            # Dovetail classification needs both segment lengths.
            self.pending_edges.append(record)
        elif isinstance(record, (PathRecord, WalkRecord)):
            self.collector.add(record)

    def _relate(self, record, relation, swap=False):
        a, oa, b, ob = (
            record.from_segment,
            record.orientation_from,
            record.to_segment,
            record.orientation_to,
        )
        if swap:
            a, oa, b, ob = b, ob, a, oa
        self.relations.setdefault(_canonical(a, oa, b, ob), []).append(relation)

    def _add_edge(self, edge):
        line = getattr(edge, "line_number", None)
        relation, swapped = classify_edge(edge, self.lengths)
        if relation != "adjacency":
            # Containments, internal alignments and unclassifiable edges are
            # not traversals.
            return
        coords = (edge.from_start, edge.from_end, edge.to_start, edge.to_end)
        if any(value is None for value in coords):
            # Historical orientation-only E records are links without overlap.
            self._relate(edge, Relation("link", "E", line, "unspecified"))
            return
        nu, nv = edge.from_end - edge.from_start, edge.to_end - edge.to_start
        if edge.cigar is None or edge.cigar == b"*" or not _is_cigar(edge.cigar):
            # Coordinates define the overlap; spelling verifies its bases.
            state, length = ("exact", nu) if nu == nv else ("unsupported", None)
        else:
            state, length = overlap_length(edge.cigar)
            if state == "exact" and not length == nu == nv:
                state, length = "unsupported", None
        self._relate(edge, Relation("link", "E", line, state, length), swap=swapped)

    def _name(self, value):
        return value if self.raw_bytes else value.decode("ascii")

    def finish(self, unsupported_records=()):
        if unsupported_records:
            raise ValueError(
                "path analysis cannot ignore unsupported records: "
                + ", ".join(sorted(unsupported_records))
            )
        for edge in self.pending_edges:
            self._add_edge(edge)
        self.pending_edges = []
        for parts in self.intervals.values():
            parts.sort(key=lambda item: item.start)
        relations = _Relations(self.relations)
        paths = {}
        for name, records in self.collector.grouped():
            path = self._path(name, records, relations)
            paths[path.name] = path
        return GenomePathSet(
            paths,
            lengths=self.lengths,
            sequences=self.sequences if self.keep_sequences else None,
            relations=relations,
            bidirected=self.bidirected,
            subdivided=bool(self.intervals),
        )

    def _path(self, name, records, relations):
        first = records[0]
        if isinstance(first, PathRecord):
            kind, naming = "P", "P record name"
        elif first.is_chunk:
            kind = "W"
            naming = "W sample#haplotype#sequence, percent-escaped"
        else:
            kind, naming = "O", "O record name"
            if first.name == b"*":
                naming = "assigned ordinal for an unnamed O group"
        steps, transitions, chunks, issues = [], [], [], []
        for chunk, record in enumerate(records):
            first_step = len(steps)
            overlaps = getattr(record, "overlaps", None)
            separators = getattr(record, "separators", None)
            if overlaps is not None and len(overlaps) != len(record.segments) - 1:
                issues.append(
                    PathIssue(
                        "overlap_count_mismatch",
                        detail=f"{len(overlaps)} overlaps for {len(record.segments)} steps; using relation overlaps",
                    )
                )
                overlaps = None
            previous = None
            for position, (segment, orientation) in enumerate(record.segments):
                if segment not in self.lengths:
                    raise ValueError(
                        f"path {name!r} references undefined segment/group {segment!r}"
                    )
                ordinal = len(steps)
                if position:
                    if kind == "P":
                        declared = (
                            "jump"
                            if separators and separators[position - 1] == ";"
                            else "link"
                        )
                    else:
                        declared = "link" if kind == "W" else "group"
                    transitions.append(
                        relations.resolve(
                            ordinal - 1,
                            previous,
                            (segment, orientation),
                            declared,
                            overlaps[position - 1] if overlaps else None,
                        )
                    )
                elif chunk:
                    transitions.append(
                        PathTransition(ordinal - 1, "chunk-boundary", "boundary")
                    )
                parts = self.intervals.get(segment, ())
                if orientation == "-":
                    parts = parts[::-1]
                steps.append(
                    PathStep(
                        self._name(segment), orientation, ordinal, chunk, tuple(parts)
                    )
                )
                previous = (segment, orientation)
            chunks.append(
                PathChunk(
                    first_step,
                    len(steps) - 1,
                    getattr(record, "start", None),
                    getattr(record, "end", None),
                    getattr(record, "line_number", None),
                )
            )
        public = self._name(name)
        return GenomePath(
            public,
            kind,
            naming,
            tuple(steps),
            tuple(transitions),
            tuple(chunks),
            self._name(first.sample) if kind == "W" else None,
            first.haplotype if kind == "W" else None,
            self._name(first.sequence_id) if kind == "W" else None,
            tuple(record.tags for record in records),
            tuple(issues),
        )


@dataclass
class PathFidelity:
    """Per-path receipt: transition evidence, spelling and source comparison."""

    name: str
    record_type: str
    steps: int
    transitions: Dict[str, int]
    spelling: str
    spelled_length: Optional[int]
    pieces: int
    assumed_overlaps: int
    issues: List[dict]
    reference: Optional[str] = None
    reference_length: Optional[int] = None
    comparison: str = "no_reference"
    offset: Optional[int] = None
    first_difference: Optional[int] = None
    circular: bool = False
    closing: Optional[dict] = None
    chunks: List[dict] = None

    @property
    def passed(self):
        failures = [i for i in self.issues if i["code"] not in DECLARED_GAPS]
        if failures or self.spelling == "unavailable":
            return False
        if self.comparison == "no_reference":
            return True
        return self.comparison in MATCHES


@dataclass
class FidelityReport:
    paths: List[PathFidelity]
    policy: dict

    @property
    def ok(self):
        return all(path.passed for path in self.paths)

    @property
    def summary(self):
        return {
            "paths": len(self.paths),
            "passed": sum(path.passed for path in self.paths),
            "spelling": dict(Counter(path.spelling for path in self.paths)),
            "comparison": dict(Counter(path.comparison for path in self.paths)),
            "transitions": dict(
                sum((Counter(path.transitions) for path in self.paths), Counter())
            ),
        }

    def to_dict(self):
        return {
            "policy": self.policy,
            "summary": self.summary,
            "ok": self.ok,
            "paths": [dict(asdict(path), passed=path.passed) for path in self.paths],
        }


@dataclass(frozen=True)
class PathLayout:
    """Base accounting for each step of one path.

    ``lengths[i]`` is the segment length (``None`` if unknown). ``overlaps[i]``
    is the overlap with step ``i - 1`` in bases: 0 at the start and after a
    jump or chunk boundary, ``None`` when the evidence does not determine it.
    ``contiguous[i]`` says whether step ``i`` continues the sequence of step
    ``i - 1`` (always false for step 0). ``starts[i]`` is where step ``i``'s
    oriented sequence begins in ``coordinates`` (``source`` for W paths whose
    chunks all have coordinates, otherwise ``path``), or ``None`` once unknown
    lengths, overlaps or gaps make it unknown.
    """

    name: Name
    coordinates: str
    lengths: Tuple[Optional[int], ...]
    overlaps: Tuple[Optional[int], ...]
    contiguous: Tuple[bool, ...]
    starts: Tuple[Optional[int], ...]
    unknown_overlap: str

    def contributes(self, index):
        """Bases step ``index`` adds beyond its predecessor, or ``None``."""
        length, overlap = self.lengths[index], self.overlaps[index]
        if length is None:
            return None
        return length - (overlap or 0)


class GenomePathSet(Mapping):
    """Read-only mapping of path name to :class:`GenomePath`.

    Holds segment lengths, optional segment sequences and the traversal
    relation index needed to spell and audit paths.
    """

    def __init__(
        self,
        paths,
        *,
        lengths,
        sequences,
        relations,
        bidirected=False,
        subdivided=False,
    ):
        self._paths = paths
        self._lengths = lengths
        self._sequences = sequences
        self._relations = relations
        self.bidirected = bidirected
        self.subdivided = subdivided

    def __getitem__(self, name):
        return self._paths[name]

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def vertex_lists(self, oriented=None):
        """The legacy ``{name: [vertex, ...]}`` result for the same graph."""
        oriented = self.bidirected if oriented is None else oriented
        return {name: path.vertices(oriented) for name, path in self.items()}

    def segment_length(self, segment):
        """Length of a segment from its S record (``None`` if unknown)."""
        return self._lengths.get(_raw(segment))

    def layout(self, name, *, unknown_overlap="strict"):
        """Per-step lengths, overlaps, contiguity and start coordinates.

        Uses the same transition evidence as :meth:`spell`: a step continues
        its predecessor only across a supported link whose overlap is exact
        (or unspecified under ``unknown_overlap='zero'``), or across adjacent W
        chunk coordinates.
        """
        if unknown_overlap not in OVERLAP_POLICIES:
            raise ValueError("unknown_overlap must be strict or zero")
        path = self[name]
        coordinates = (
            "source"
            if path.record_type == "W"
            and all(chunk.start is not None for chunk in path.chunks)
            else "path"
        )
        lengths, overlaps, contiguous, starts = [], [], [], []
        end = None
        for step in path.steps:
            chunk = path.chunks[step.chunk]
            length = self.segment_length(step.segment)
            if step.ordinal == 0:
                overlap, joined = 0, False
                start = chunk.start if coordinates == "source" else 0
            else:
                transition = path.transitions[step.ordinal - 1]
                overlap, joined = _incoming(transition, path, step, unknown_overlap)
                previous = lengths[-1]
                if (
                    overlap
                    and length is not None
                    and previous is not None
                    and overlap > min(length, previous)
                ):
                    # Spelling reports overlap_exceeds_sequence here.
                    overlap, joined = None, False
                if coordinates == "source" and transition.status == "boundary":
                    start = chunk.start
                elif joined and end is not None:
                    start = end - overlap
                else:
                    start = None
            lengths.append(length)
            overlaps.append(overlap)
            contiguous.append(joined)
            starts.append(start)
            end = start + length if start is not None and length is not None else None
        return PathLayout(
            path.name,
            coordinates,
            tuple(lengths),
            tuple(overlaps),
            tuple(contiguous),
            tuple(starts),
            unknown_overlap,
        )

    def _sequence(self, step):
        sequence = self._sequences.get(_raw(step.segment))
        if sequence is None or step.orientation == "+":
            return sequence
        return reverse_complement(sequence)

    def spell(self, name, *, unknown_overlap="strict"):
        """Spell a path's sequence where evidence permits; see :class:`SpelledPath`.

        ``unknown_overlap='zero'`` treats unspecified link overlaps as blunt and
        lists those transitions in ``assumed_overlaps``; the default abstains.
        """
        if unknown_overlap not in OVERLAP_POLICIES:
            raise ValueError("unknown_overlap must be strict or zero")
        if self._sequences is None:
            raise ValueError("spelling requires paths loaded with sequences=True")
        path = self[name]
        issues, assumed, pieces = list(path.issues), [], []
        current, previous_length = None, 0
        lengths = [0] * len(path.chunks)
        broken = [False] * len(path.chunks)
        spans = [None] * len(path.chunks)

        def close(last):
            nonlocal current
            if current is not None:
                pieces.append(SpelledPiece(current[0], last, bytes(current[1])))
                current = None

        for step in path.steps:
            index, overlap = step.ordinal, 0
            chunk = path.chunks[step.chunk]
            if index:
                transition = path.transitions[index - 1]
                joined, overlap = self._join(
                    path, transition, step, issues, assumed, unknown_overlap
                )
                if not joined:
                    close(index - 1)
                    if transition.status != "boundary":
                        broken[step.chunk] = True
            sequence = self._sequence(step)
            if sequence is None:
                issues.append(
                    PathIssue(
                        "missing_sequence", step=index, detail=_text(step.segment)
                    )
                )
                close(index - 1)
                broken[step.chunk] = True
                continue
            if current is not None and (
                overlap > len(sequence) or overlap > previous_length
            ):
                issues.append(
                    PathIssue(
                        "overlap_exceeds_sequence",
                        transition=index - 1,
                        detail=f"overlap {overlap}",
                    )
                )
                close(index - 1)
                broken[step.chunk] = True
            elif current is not None and overlap:
                previous = bytes(current[1][-overlap:]).upper()
                if previous != sequence[:overlap].upper():
                    issues.append(
                        PathIssue(
                            "overlap_mismatch",
                            transition=index - 1,
                            detail=f"overlap {overlap}",
                        )
                    )
                    close(index - 1)
                    broken[step.chunk] = True
            if index == chunk.first_step:
                start = 0 if current is None else len(current[1])
                spans[step.chunk] = [len(pieces), start, None]
            previous_length = len(sequence)
            if current is None:
                current = [index, bytearray(sequence)]
                lengths[step.chunk] += len(sequence)
            else:
                current[1].extend(sequence[overlap:])
                lengths[step.chunk] += len(sequence) - overlap
            if index == chunk.last_step and spans[step.chunk] is not None:
                spans[step.chunk][2] = len(current[1])
        close(len(path.steps) - 1)
        for number, chunk in enumerate(path.chunks):
            if broken[number]:
                spans[number] = None
            elif chunk.start is not None and chunk.end is not None:
                if lengths[number] != chunk.end - chunk.start:
                    issues.append(
                        PathIssue(
                            "coordinate_length_mismatch",
                            step=chunk.first_step,
                            detail=f"chunk {number} spells {lengths[number]} bases for coordinates {chunk.start}-{chunk.end}",
                        )
                    )
        complete = len(pieces) == 1 and (
            pieces[0].first_step == 0 and pieces[0].last_step == len(path.steps) - 1
        )
        status = "complete" if complete else ("partial" if pieces else "unavailable")
        return SpelledPath(
            path.name,
            status,
            tuple(pieces),
            tuple(issues),
            unknown_overlap,
            tuple(assumed),
            tuple(tuple(span) if span else None for span in spans),
        )

    def _join(self, path, transition, step, issues, assumed, policy):
        index = transition.index
        if transition.status == "boundary":
            before, after = path.chunks[step.chunk - 1], path.chunks[step.chunk]
            if (
                before.end is not None
                and after.start is not None
                and before.end == after.start
            ):
                return True, 0
            gap = (
                after.start - before.end
                if before.end is not None and after.start is not None
                else None
            )
            issues.append(
                PathIssue(
                    "chunk_gap",
                    transition=index,
                    detail="unknown length" if gap is None else f"{gap} bases",
                )
            )
            return False, 0
        if transition.status != "supported":
            issues.append(
                PathIssue(
                    transition.status + "_transition",
                    transition=index,
                    detail=transition.detail,
                )
            )
            return False, 0
        if transition.relation == "jump":
            detail = (
                "unknown distance"
                if transition.distance is None
                else f"estimated {transition.distance} bases"
            )
            issues.append(PathIssue("jump", transition=index, detail=detail))
            return False, 0
        if transition.overlap_state == "exact":
            return True, transition.overlap_length
        if transition.overlap_state == "unspecified" and policy == "zero":
            assumed.append(index)
            return True, 0
        code = (
            "unknown_overlap"
            if transition.overlap_state == "unspecified"
            else "unsupported_overlap"
        )
        issues.append(PathIssue(code, transition=index, detail=transition.detail))
        return False, 0

    def closing_transition(self, name):
        """Relation evidence for returning from the last step to the first."""
        path = self[name]
        last, first = path.steps[-1], path.steps[0]
        return self._relations.resolve(
            len(path.steps) - 1,
            (_raw(last.segment), last.orientation),
            (_raw(first.segment), first.orientation),
            "link",
        )

    def audit(
        self,
        references=None,
        *,
        names=None,
        circular=(),
        reverse_complement=False,
        ignore_case=True,
        unknown_overlap="strict",
        reference_names=None,
        ambiguity="exact",
    ):
        """Compare spelled paths with source sequences and return a receipt.

        ``references`` is a FASTA path or a ``{name: sequence}`` mapping. Paths
        match references by name unless ``reference_names`` maps them. Accepted
        equivalences are declared, never inferred: ``reverse_complement=True``
        permits strand reversal, and ``circular`` names paths (or ``True`` for
        all) whose comparison permits rotation. W chunks with coordinates are
        compared with ``reference[start:end]``. ``ambiguity='as-n'`` declares
        that IUPAC ambiguity codes (any non-ACGT byte) compare equal to ``N``,
        for graphs built from sequences with normalized ambiguity codes. A
        FASTA match establishes path fidelity to that source, not biological
        validity of the source.
        """
        if ambiguity not in AMBIGUITY:
            raise ValueError("ambiguity must be exact or as-n")
        if isinstance(references, (str, Path)):
            references = read_fasta(references)
        references = {_text(k): _bytes(v) for k, v in (references or {}).items()}
        reference_names = reference_names or {}
        selected = list(self) if names is None else list(names)
        if circular is True:
            declared_circular = set(map(_text, self))
        elif not circular:
            declared_circular = set()
        else:
            declared_circular = set(map(_text, circular))

        def norm(value):
            value = value.upper() if ignore_case else value
            return value.translate(_AS_N) if ambiguity == "as-n" else value

        results = []
        for name in selected:
            path = self[name]
            label = _text(path.name)
            spelled = self.spell(name, unknown_overlap=unknown_overlap)
            is_circular = label in declared_circular
            result = PathFidelity(
                label,
                path.record_type,
                len(path.steps),
                path.transition_counts,
                spelled.status,
                len(spelled.sequence) if spelled.sequence is not None else None,
                len(spelled.pieces),
                len(spelled.assumed_overlaps),
                [asdict(issue) for issue in spelled.issues],
                circular=is_circular,
                chunks=[],
            )
            sequence = spelled.sequence
            if is_circular and sequence is not None:
                closing = self.closing_transition(name)
                result.closing = {
                    "status": closing.status,
                    "overlap_length": closing.overlap_length,
                }
                trim = closing.overlap_length or 0
                if (
                    closing.status == "supported"
                    and closing.overlap_state == "exact"
                    and 0 < trim < len(sequence)
                    and norm(sequence[-trim:]) == norm(sequence[:trim])
                ):
                    sequence = sequence[:-trim]
                    result.closing["trimmed"] = trim
            key = reference_names.get(label, label)
            reference = references.get(key)
            if reference is None:
                if references:
                    result.comparison = "reference_missing"
                elif spelled.status == "unavailable":
                    result.comparison = "not_spelled"
                results.append(result)
                continue
            result.reference, result.reference_length = key, len(reference)
            ref = norm(reference)
            located = path.record_type == "W" and all(
                chunk.start is not None and chunk.end is not None
                for chunk in path.chunks
            )
            if located:
                matched = True
                for number, chunk in enumerate(path.chunks):
                    span = spelled.chunk_spans[number]
                    entry = {"start": chunk.start, "end": chunk.end, "matches": False}
                    if span is not None:
                        piece, start, end = span
                        text = spelled.pieces[piece].sequence[start:end]
                        entry["spelled_length"] = len(text)
                        entry["matches"] = (
                            chunk.end <= len(ref)
                            and norm(text) == ref[chunk.start : chunk.end]
                        )
                    matched = matched and entry["matches"]
                    result.chunks.append(entry)
                result.comparison = (
                    "coordinates_match" if matched else "coordinates_differ"
                )
            elif sequence is not None:
                self._compare(
                    result, norm(sequence), ref, is_circular, reverse_complement
                )
            elif spelled.pieces:
                self._locate(
                    result,
                    [norm(p.sequence) for p in spelled.pieces],
                    ref,
                    is_circular,
                    reverse_complement,
                )
            else:
                result.comparison = "not_spelled"
            results.append(result)
        policy = {
            "unknown_overlap": unknown_overlap,
            "reverse_complement": reverse_complement,
            "circular": "all" if circular is True else sorted(declared_circular),
            "ignore_case": ignore_case,
            "ambiguity": ambiguity,
            "subdivided": self.subdivided,
        }
        return FidelityReport(results, policy)

    @staticmethod
    def _compare(result, sequence, reference, circular, allow_rc):
        candidates = [("", sequence)]
        if allow_rc:
            candidates.append(("reverse_complement", reverse_complement(sequence)))
        for label, value in candidates:
            if value == reference:
                result.comparison, result.offset = label or "identical", 0
                return
        if circular and len(sequence) == len(reference):
            doubled = reference + reference
            for label, value in candidates:
                offset = doubled.find(value)
                if offset >= 0:
                    result.comparison = "rotation" + ("_" + label if label else "")
                    result.offset = offset
                    return
        result.comparison = "different"
        shared = min(len(sequence), len(reference))
        result.first_difference = next(
            (i for i in range(shared) if sequence[i] != reference[i]), shared
        )

    @staticmethod
    def _locate(result, pieces, reference, circular, allow_rc):
        targets = [reference]
        if allow_rc:
            targets.append(reverse_complement(reference))
        for target in targets:
            size = len(target)
            if circular:
                target = target + target
            position, offsets = 0, []
            for piece in pieces:
                found = target.find(piece, position)
                if found < 0:
                    break
                offsets.append(found)
                position = found + len(piece)
            else:
                # After wrap-around, pieces must not reuse circle positions.
                if position - offsets[0] <= size:
                    result.comparison, result.offset = "pieces_located", offsets[0]
                    return
        result.comparison = "pieces_unlocated"


def _incoming(transition, path, step, policy):
    """``(overlap, contiguous)`` for the transition into ``step``."""
    if transition.status == "boundary":
        before, after = path.chunks[step.chunk - 1], path.chunks[step.chunk]
        adjacent = (
            before.end is not None
            and after.start is not None
            and before.end == after.start
        )
        return 0, adjacent
    if transition.status != "supported":
        return None, False
    if transition.relation == "jump":
        return 0, False
    if transition.overlap_state == "exact":
        return transition.overlap_length, True
    if transition.overlap_state == "unspecified" and policy == "zero":
        return 0, True
    return None, False


def _bytes(value):
    return value.encode("ascii") if isinstance(value, str) else bytes(value)


def read_fasta(path):
    """Read FASTA (optionally gzip) into ``{first-word-of-header: bytes}``."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    records, name, chunks = {}, None, []
    with opener(path, "rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                if name is not None:
                    records[name] = b"".join(chunks)
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"{path}: FASTA header without a name")
                name, chunks = fields[0].decode("ascii"), []
                if name in records:
                    raise ValueError(f"{path}: duplicate FASTA name {name!r}")
            elif name is None:
                raise ValueError(f"{path}: sequence before the first FASTA header")
            else:
                chunks.append(line)
    if name is not None:
        records[name] = b"".join(chunks)
    return records


def load_genome_paths(path, *, sequences=True, raw_bytes=False, bidirected=False):
    """Read rich P/O/W paths, relations and segment sequences in one pass.

    Works on stdin (``"-"``) and gzip input. ``sequences=False`` keeps only
    structure and lengths; spelling and auditing then raise. ``bidirected``
    only selects the default for :meth:`GenomePathSet.vertex_lists`.
    """
    parser = GFAParser(path)
    builder = GenomePathBuilder(
        sequences=sequences, raw_bytes=raw_bytes, bidirected=bidirected
    )
    for _ in builder.observe(parser, str(path)):
        pass
    return builder.finish(parser.unsupported_records)
