"""Split coordinate intervals once, then reuse the ordinary graph builder."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import re
import warnings

from .projection import classify_edge
from .parser import (
    Segment,
    Link,
    EdgeRecord,
    ContainmentRecord,
    GapRecord,
    PathRecord,
    WalkRecord,
)


def cigar_operations(cigar):
    if cigar in (None, b"*"):
        return None
    operations = [(int(n), op) for n, op in re.findall(rb"(\d+)([MIDNSHPX=])", cigar)]
    if (
        not operations
        or b"".join(str(n).encode() + op for n, op in operations) != cigar
    ):
        raise ValueError("alignment splitting requires a CIGAR, not a trace string")
    return operations


def split_records(records, *, max_nodes=1_000_000):
    """Propagate CIGAR match-block boundaries across aligned segments.

    Missing alignment detail is allowed only when no interior cut must be mapped.
    Inserted/deleted sequence remains on its original segment chain. Zero-span
    alignments and L/J/G links attach to the correct oriented boundaries.
    """
    if not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_split_nodes must be a positive integer")
    records = list(records)
    segments = {}
    for rec in records:
        if isinstance(rec, Segment):
            if rec.id in segments:
                raise ValueError(f"duplicate segment {rec.id!r}")
            if rec.length is None:
                raise ValueError(
                    f"splitting requires a sequence or length for segment {rec.id!r}"
                )
            if rec.sequence is not None and len(rec.sequence) != rec.length:
                raise ValueError(
                    f"splitting requires sequence/length agreement for {rec.id!r}"
                )
            segments[rec.id] = rec
    cuts = {node: {0, seg.length} for node, seg in segments.items()}
    blocks = []
    edge_blocks = {}
    unaligned = {}

    def require_segment(node):
        if node not in segments:
            raise ValueError(f"alignment/link references undefined segment {node!r}")
        return segments[node]

    for index, rec in enumerate(records):
        if not isinstance(rec, (EdgeRecord, ContainmentRecord)):
            continue
        u, v = require_segment(rec.from_segment), require_segment(rec.to_segment)
        operations = cigar_operations(rec.cigar)
        start_u, end_u, start_v, end_v = (
            rec.from_start,
            rec.from_end,
            rec.to_start,
            rec.to_end,
        )
        if isinstance(rec, ContainmentRecord) and start_u is not None:
            if operations is None:
                raise ValueError("splitting a containment requires a CIGAR")
            length_u = sum(
                n for n, op in operations if op in (b"M", b"D", b"N", b"=", b"X")
            )
            length_v = sum(
                n for n, op in operations if op in (b"M", b"I", b"S", b"=", b"X")
            )
            end_u, start_v, end_v = start_u + length_u, 0, length_v
        if all(x is None for x in (start_u, end_u, start_v, end_v)):
            continue
        if any(x is None for x in (start_u, end_u, start_v, end_v)):
            raise ValueError(
                "alignment splitting requires complete coordinate intervals"
            )
        if not (
            0 <= start_u <= end_u <= u.length and 0 <= start_v <= end_v <= v.length
        ):
            raise ValueError(
                f"alignment coordinates outside segment bounds at record {getattr(rec, 'line_number', index + 1)}"
            )
        cuts[u.id].update((start_u, end_u))
        cuts[v.id].update((start_v, end_v))
        if operations is None or (start_u == end_u and start_v == end_v):
            unaligned[index] = (start_u, end_u, start_v, end_v)
            continue
        du = 1 if rec.orientation_from == "+" else -1
        dv = 1 if rec.orientation_to == "+" else -1
        pu = start_u if du == 1 else end_u
        pv = start_v if dv == 1 else end_v
        matched = []
        for count, operation in operations:
            consume_u = count if operation in (b"M", b"D", b"N", b"=", b"X") else 0
            consume_v = count if operation in (b"M", b"I", b"S", b"=", b"X") else 0
            qu, qv = pu + du * consume_u, pv + dv * consume_v
            cuts[u.id].update((pu, qu))
            cuts[v.id].update((pv, qv))
            if consume_u and consume_v:
                block = (
                    u.id,
                    min(pu, qu),
                    max(pu, qu),
                    v.id,
                    min(pv, qv),
                    max(pv, qv),
                    du,
                    dv,
                )
                blocks.append(block)
                matched.append(block)
            pu, pv = qu, qv
        if pu != (end_u if du == 1 else start_u) or pv != (
            end_v if dv == 1 else start_v
        ):
            raise ValueError(
                f"CIGAR consumption disagrees with alignment coordinates at record {getattr(rec, 'line_number', index + 1)}"
            )
        if not matched:
            raise ValueError(
                "alignment has no paired sequence; use an explicit link/gap"
            )
        edge_blocks[index] = matched

    projections = defaultdict(list)
    for u, a, b, v, c, d, du, dv in blocks:
        projections[u].append((a, b, v, c, d, du, dv))
        projections[v].append((c, d, u, a, b, dv, du))
    node_count = sum(max(1, len(points) - 1) for points in cuts.values())
    if node_count > max_nodes:
        raise MemoryError("alignment splitting exceeds max_split_nodes")
    pending = deque((node, point) for node, points in cuts.items() for point in points)
    while pending:
        node, point = pending.popleft()
        for a, b, other, c, d, source_sign, target_sign in projections[node]:
            if a <= point <= b:
                offset = point - a if source_sign == 1 else b - point
                mapped = c + offset if target_sign == 1 else d - offset
                if mapped not in cuts[other]:
                    node_count += 1
                    if node_count > max_nodes:
                        raise MemoryError("alignment splitting exceeds max_split_nodes")
                    cuts[other].add(mapped)
                    pending.append((other, mapped))

    pieces = {}
    generated = []
    for node, seg in segments.items():
        points = sorted(cuts[node])
        if len(points) == 1:
            points.append(points[0])
        intervals = []
        for start, end in zip(points, points[1:]):
            name = node + f":{start}-{end}".encode()
            intervals.append((start, end, name))
            generated.append(
                Segment(
                    name,
                    end - start,
                    seg.sequence[start:end] if seg.sequence is not None else None,
                    seg.tags,
                    node,
                    start,
                    end,
                )
            )
        pieces[node] = intervals
        for first, second in zip(intervals, intervals[1:]):
            # Adjacent intervals of one segment abut exactly.
            generated.append(Link(first[2], second[2], "+", "+", b"0M"))
    if len([rec for rec in generated if isinstance(rec, Segment)]) > 10 * len(segments):
        warnings.warn("split-on-alignment created >10x more nodes", RuntimeWarning)

    def interval_parts(node, start, end, orientation):
        selected = [
            item for item in pieces[node] if start <= item[0] and item[1] <= end
        ]
        return selected if orientation == "+" else selected[::-1]

    def boundary(node, position, orientation, source):
        intervals = pieces[node]
        ending = (orientation == "+") == source
        matches = [
            piece
            for piece in intervals
            if (piece[1] if ending else piece[0]) == position
        ]
        if matches:
            return matches[-1 if ending else 0][2]
        raise ValueError(f"no interval at boundary {node!r}:{position}")

    def terminal(node, orientation, source):
        require_segment(node)
        if (orientation == "+") == source:
            return pieces[node][-1][2]
        return pieces[node][0][2]

    lengths = {node: seg.length for node, seg in segments.items()}
    for index, rec in enumerate(records):
        if isinstance(rec, Segment):
            continue
        if isinstance(rec, EdgeRecord) and rec.relation_class is None:
            # Generated intervals no longer match the original coordinates.
            rec = replace(rec, relation_class=classify_edge(rec, lengths)[0])
        if isinstance(rec, (PathRecord, WalkRecord)):
            expanded = []
            separators, overlaps = [], []
            declared = getattr(rec, "separators", None)
            raw_overlaps = getattr(rec, "overlaps", None)
            mapped = (
                raw_overlaps is not None and len(raw_overlaps) == len(rec.segments) - 1
            )
            for position, (node, orientation) in enumerate(rec.segments):
                require_segment(node)
                items = pieces[node] if orientation == "+" else pieces[node][::-1]
                for offset, item in enumerate(items):
                    if expanded and declared is not None:
                        inner = offset > 0
                        separators.append("," if inner else declared[position - 1])
                        if mapped:
                            overlaps.append(
                                b"0M" if inner else raw_overlaps[position - 1]
                            )
                    expanded.append((item[2], orientation))
            changes = {"segments": expanded}
            if declared is not None:
                changes["separators"] = separators
            if mapped:
                # A malformed overlap count is kept raw rather than re-indexed.
                changes["overlaps"] = overlaps
            generated.append(replace(rec, **changes))
        elif index in edge_blocks:
            for u, a, b, v, c, d, _, _ in edge_blocks[index]:
                left = interval_parts(u, a, b, rec.orientation_from)
                right = interval_parts(v, c, d, rec.orientation_to)
                if len(left) != len(right) or any(
                    x[1] - x[0] != y[1] - y[0] for x, y in zip(left, right)
                ):
                    raise ValueError("alignment boundaries could not be paired exactly")
                for x, y in zip(left, right):
                    generated.append(replace(rec, from_segment=x[2], to_segment=y[2]))
        elif index in unaligned:
            a, b, c, d = unaligned[index]
            if a == b and c == d:
                u = boundary(rec.from_segment, a, rec.orientation_from, True)
                v = boundary(rec.to_segment, c, rec.orientation_to, False)
            else:
                left = interval_parts(rec.from_segment, a, b, rec.orientation_from)
                right = interval_parts(rec.to_segment, c, d, rec.orientation_to)
                if len(left) != 1 or len(right) != 1:
                    raise ValueError(
                        "an alignment without a CIGAR spans internal cuts; an exact split cannot be inferred"
                    )
                u, v = left[0][2], right[0][2]
            generated.append(replace(rec, from_segment=u, to_segment=v))
        elif isinstance(rec, (Link, EdgeRecord, ContainmentRecord, GapRecord)):
            generated.append(
                replace(
                    rec,
                    from_segment=terminal(rec.from_segment, rec.orientation_from, True),
                    to_segment=terminal(rec.to_segment, rec.orientation_to, False),
                )
            )
    return generated
