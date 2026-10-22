"""Named graph projections over GFA relation classes, and their manifests.

A GFA file declares several different relationships between segments. Each
record is assigned one relation class:

``adjacency``
    Sequence adjacency usable by a traversal: L links, GFA2 dovetail E edges and
    coordinate-free (historical) E edges.
``containment``
    One segment lies within another: C records and GFA2 E edges whose interval
    spans a whole segment.
``alignment``
    Other GFA2 E alignments between segment interiors.
``gap``
    J jumps and GFA2 G gaps: an estimated distance, not observed sequence.
``unclassified``
    GFA2 E edges whose class cannot be decided, because a segment length is
    unknown or a coordinate lies outside its segment.

A projection selects the classes that become graph edges. ``all`` (the default)
keeps every record, as in earlier releases. Under alignment subdivision, links
between abutting intervals of one segment have the class ``subdivision`` and are
kept by every projection, because they restore the segment itself.
"""

from __future__ import annotations

from collections import Counter

from .parser import ContainmentRecord, EdgeRecord, GapRecord, Link, Segment

CLASSES = ("adjacency", "containment", "alignment", "gap", "unclassified")
PROJECTIONS = {
    "all": frozenset(CLASSES),
    "traversal": frozenset({"adjacency"}),
    "gap-inclusive": frozenset({"adjacency", "gap"}),
    "alignment": frozenset({"containment", "alignment"}),
}


def resolve_projection(projection):
    """Return ``(name, classes)`` for a preset name or an iterable of classes.

    A comma-separated string of classes is accepted, so command-line values
    and Python values share one syntax.
    """
    if projection is None:
        projection = "all"
    if isinstance(projection, str):
        if projection in PROJECTIONS:
            return projection, PROJECTIONS[projection]
        projection = [part.strip() for part in projection.split(",") if part.strip()]
    classes = frozenset(projection)
    unknown = sorted(classes - set(CLASSES))
    if unknown or not classes:
        raise ValueError(
            "projection must be one of "
            + ", ".join(PROJECTIONS)
            + " or a set of relation classes from "
            + ", ".join(CLASSES)
            + (f"; unknown: {', '.join(unknown)}" if unknown else "")
        )
    for name, preset in PROJECTIONS.items():
        if classes == preset:
            return name, preset
    return "custom", classes


def _suffix(length, orientation, n):
    return (length - n, length) if orientation == "+" else (0, n)


def _prefix(length, orientation, n):
    return (0, n) if orientation == "+" else (length - n, length)


def classify_edge(edge, lengths):
    """Classify a GFA2 E record as ``(relation_class, swapped)``.

    ``swapped`` is true for a dovetail in which the second segment precedes the
    first, so the traversal is ``to -> from``. Coordinates are on the forward
    strand of each segment, as in the GFA2 specification.
    """
    coords = (edge.from_start, edge.from_end, edge.to_start, edge.to_end)
    if any(value is None for value in coords):
        return "adjacency", False
    lu = lengths.get(edge.from_segment)
    lv = lengths.get(edge.to_segment)
    if lu is None or lv is None:
        return "unclassified", False
    fs, fe, ts, te = coords
    if not (0 <= fs <= fe <= lu and 0 <= ts <= te <= lv):
        return "unclassified", False
    nu, nv = fe - fs, te - ts
    if (nu and nu == lu) or (nv and nv == lv):
        return "containment", False
    ou, ov = edge.orientation_from, edge.orientation_to
    if (fs, fe) == _suffix(lu, ou, nu) and (ts, te) == _prefix(lv, ov, nv):
        return "adjacency", False
    if (ts, te) == _suffix(lv, ov, nv) and (fs, fe) == _prefix(lu, ou, nu):
        return "adjacency", True
    return "alignment", False


def record_letter(record):
    if isinstance(record, Link):
        return "L"
    if isinstance(record, EdgeRecord):
        return "E"
    if isinstance(record, ContainmentRecord):
        return "C"
    return record.line_type or "J"


def relation_class(record, lengths):
    """Relation class of a record, or ``None`` until E segment lengths are known."""
    if isinstance(record, Link):
        return record.relation_class or "adjacency"
    if isinstance(record, ContainmentRecord):
        return "containment"
    if isinstance(record, GapRecord):
        return "gap"
    if record.relation_class is not None:
        return record.relation_class
    if any(
        value is None
        for value in (
            record.from_start,
            record.from_end,
            record.to_start,
            record.to_end,
        )
    ):
        return "adjacency"
    if record.from_segment not in lengths or record.to_segment not in lengths:
        return None
    return classify_edge(record, lengths)[0]


class RelationCounts:
    """Included and excluded relation records by class and source record type."""

    def __init__(self):
        self.counts = Counter()

    def add(self, relation, letter, included):
        self.counts[(relation, letter, included)] += 1

    def summary(self, classes):
        result = {}
        extra = sorted({name for name, _, _ in self.counts} - set(CLASSES))
        for relation in CLASSES + tuple(extra):
            by_record = Counter()
            for (name, letter, _), count in self.counts.items():
                if name == relation:
                    by_record[letter] += count
            result[relation] = {
                "included": relation in classes or relation == "subdivision",
                "records": sum(by_record.values()),
                "by_record": dict(sorted(by_record.items())),
            }
        return result


class SourceRelations:
    """Count input relation records by class before any subdivision."""

    def __init__(self, classes):
        self.classes = classes
        self.lengths = {}
        self.pending = []
        self.counts = RelationCounts()

    def observe(self, records):
        for record in records:
            if isinstance(record, Segment):
                length = record.length
                if length is None and record.sequence is not None:
                    length = len(record.sequence)
                if length is not None:
                    self.lengths[record.id] = length
            elif isinstance(record, (Link, EdgeRecord, ContainmentRecord, GapRecord)):
                self._count(record, relation_class(record, self.lengths))
            yield record

    def _count(self, record, relation):
        if relation is None:
            self.pending.append(record)
        else:
            self.counts.add(relation, record_letter(record), relation in self.classes)

    def summary(self):
        pending, self.pending = self.pending, []
        for record in pending:
            self._count(record, relation_class(record, self.lengths) or "unclassified")
        return self.counts.summary(self.classes)


EXPORT_FORMATS = {
    "pickle": (
        ["all vertex and edge attributes"],
        [],
    ),
    "matrix": (
        ["summed edge weight per vertex pair", "vertex order (nodes.tsv sidecar)"],
        [
            "multiplicity",
            "record_type and relation provenance",
            "orientation attributes",
            "gap estimates",
            "tags and sequences",
            "parallel edge identity",
        ],
    ),
    "edge-list": (
        ["edge endpoints", "one line per graph edge (per record in parallel mode)"],
        [
            "multiplicity",
            "record_type and relation provenance",
            "orientation attributes",
            "gap estimates",
            "tags and sequences",
        ],
    ),
    "graphml": (
        ["scalar attributes", "nested attributes as JSON strings"],
        ["attributes whose value is None"],
    ),
    "gexf": (
        ["scalar attributes", "nested attributes as JSON strings"],
        ["attributes whose value is None"],
    ),
    "json": (
        ["scalar attributes", "nested attributes as JSON strings"],
        ["attributes whose value is None"],
    ),
}


def export_manifest(manifest, formats, *, with_weights=False):
    """Copy of ``manifest`` declaring what each export format keeps and drops."""
    exports = []
    for fmt in formats:
        retains, drops = (list(items) for items in EXPORT_FORMATS[fmt])
        if fmt == "edge-list":
            (retains if with_weights else drops).append("summed edge weight")
        exports.append({"format": fmt, "retains": retains, "drops": drops})
    result = dict(manifest)
    result["exports"] = exports
    return result
