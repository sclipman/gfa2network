from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Iterable, Tuple
import sys
import warnings
import gzip
import re
from urllib.parse import quote


@dataclass
class Segment:
    """A segment (node) record."""

    id: bytes
    length: int | None = None
    sequence: bytes | None = None
    tags: dict[str, Any] | None = None
    parent: bytes | None = None
    start: int | None = None
    end: int | None = None


@dataclass
class Link:
    """A link (edge) record with orientation preserved.

    ``relation_class`` is ``"subdivision"`` for links that alignment
    subdivision creates between abutting intervals of one segment.
    """

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    overlap: bytes | None = None
    tags: dict[str, Any] | None = None
    relation_class: str | None = None


@dataclass
class PathRecord:
    """A path consisting of ordered oriented segments.

    ``separators`` holds the delimiter between consecutive steps: ``,`` for a
    link and ``;`` for a GFA1.2 jump. ``overlaps`` holds the raw overlap field
    split on commas, or ``None`` when the whole field is ``*``.
    """

    name: bytes
    segments: list[Tuple[bytes, str]]
    tags: dict[str, Any] | None = None
    overlaps: list[bytes] | None = None
    separators: list[str] | None = None


@dataclass
class EdgeRecord:
    """GFA2 edge/alignment record.

    ``relation_class`` is set when the class was decided before the record's
    coordinates stopped describing its endpoints (alignment subdivision).
    """

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    from_start: int | None = None
    from_end: int | None = None
    to_start: int | None = None
    to_end: int | None = None
    cigar: bytes | None = None
    tags: dict[str, Any] | None = None
    relation_class: str | None = None


@dataclass
class ContainmentRecord:
    """GFA1 containment record."""

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    from_start: int | None = None
    from_end: int | None = None
    to_start: int | None = None
    to_end: int | None = None
    cigar: bytes | None = None
    tags: dict[str, Any] | None = None


@dataclass
class WalkRecord:
    """GFA2 ordered group or GFA1.1 walk chunk."""

    name: bytes
    segments: list[Tuple[bytes, str]]
    tags: dict[str, Any] | None = None

    start: int | None = None
    end: int | None = None
    is_chunk: bool = False
    sample: bytes | None = None
    haplotype: int | None = None
    sequence_id: bytes | None = None


@dataclass
class GapRecord:
    """GFA1.2 jump or GFA2 estimated gap; distance is metadata, not an edge cost."""

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    distance: int | None = None
    variance: int | None = None
    tags: dict[str, Any] | None = None
    line_type: str | None = None


class GFAParser:
    """Streaming parser yielding record dataclasses."""

    def __init__(self, source: str | Path | BinaryIO):
        if isinstance(source, (str, Path)):
            self.path = str(source)
            self.file: BinaryIO | None = None
        else:
            self.path = None
            self.file = source
        self._warned_unknown = False

    def __iter__(self) -> Iterator[object]:
        if self.file is not None:
            fh, close = self.file, False
        else:
            path = self.path or "-"
            if path == "-":
                fh, close = sys.stdin.buffer, False
            else:
                fh = (
                    gzip.open(path, "rb")
                    if path.endswith(".gz")
                    else open(path, "rb", buffering=1 << 20)
                )
                close = True
        warned = set()
        self.unsupported_records = set()
        readers = {
            b"S": self._parse_segment,
            b"L": self._parse_link,
            b"E": self._parse_edge,
            b"C": self._parse_containment,
            b"P": self._parse_path,
            b"O": self._parse_walk,
            b"W": self._parse_gfa1_walk,
            b"J": self._parse_gap,
            b"G": self._parse_gap,
        }
        try:
            for number, line in enumerate(fh, 1):
                if not line.strip() or line.startswith(b"#"):
                    continue
                fields = line.rstrip(b"\r\n").split(b"\t")
                kind = fields[0]
                try:
                    if kind == b"H":
                        for field in fields[1:]:
                            if field.startswith(b"VN:Z:") and field[5:] not in (
                                b"1.0",
                                b"1.1",
                                b"1.2",
                                b"2.0",
                            ):
                                raise ValueError(
                                    "unsupported GFA version " + field[5:].decode()
                                )
                        continue
                    if kind in (b"F", b"U"):
                        # Fragment alignments and unordered groups do not define
                        # connectivity or ordered segment paths.
                        continue
                    if kind not in readers:
                        self.unsupported_records.add(
                            kind.decode("ascii", errors="replace")
                        )
                        if kind not in warned:
                            warnings.warn(
                                f"Skipping unsupported record: {kind.decode('ascii', errors='replace')}",
                                RuntimeWarning,
                            )
                            warned.add(kind)
                        continue
                    record = readers[kind](fields)
                    if isinstance(record, (EdgeRecord, ContainmentRecord)):
                        for start, end in (
                            (record.from_start, record.from_end),
                            (record.to_start, record.to_end),
                        ):
                            if (start is not None and start < 0) or (
                                end is not None
                                and (end < 0 or (start is not None and end < start))
                            ):
                                raise ValueError(
                                    "invalid alignment coordinate interval"
                                )
                    record.line_number = number
                    yield record
                except (ValueError, IndexError, UnicodeError) as exc:
                    source = self.path or "<stream>"
                    raise ValueError(
                        f"{source}:{number}: {kind.decode('ascii', errors='replace')} record: {exc}"
                    ) from exc
        finally:
            if close:
                fh.close()

    @staticmethod
    def _parse_segment(fields):
        if len(fields) < 3 or not fields[1]:
            raise ValueError("Malformed S record")
        seq, length, tag_start = None, None, 3
        try:
            length = int(fields[2])
            if length < 0:
                raise ValueError("negative segment length")
            if len(fields) > 3 and not re.match(
                rb"^[A-Za-z0-9]{2}:[A-Za-z]:", fields[3]
            ):
                seq, tag_start = fields[3], 4
        except ValueError:
            if fields[2].startswith(b"-") and fields[2][1:].isdigit():
                raise ValueError("negative segment length")
            seq = fields[2]
        tags = GFAParser._parse_tags(fields[tag_start:])
        if seq in (b"*", b""):
            seq = None
        if length is None:
            length = (tags or {}).get("LN", len(seq) if seq is not None else None)
        if length is not None and (not isinstance(length, int) or length < 0):
            raise ValueError("invalid segment length")
        return Segment(fields[1], length, seq, tags)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tags(fields: Iterable[bytes]) -> dict[str, Any] | None:
        tags: dict[str, Any] = {}
        for f in fields:
            try:
                tag, typ, value = f.decode().split(":", 2)
            except ValueError:
                continue
            if typ == "i":
                try:
                    tags[tag] = int(value)
                except ValueError:
                    pass
            elif typ == "f":
                try:
                    tags[tag] = float(value)
                except ValueError:
                    pass
            elif typ == "B":
                try:
                    subtype, *values = value.split(",")
                    if subtype not in "cCsSiIf" or len(subtype) != 1:
                        continue
                    cast = float if subtype == "f" else int
                    tags[tag] = [cast(x) for x in values]
                except ValueError:
                    continue
            else:
                tags[tag] = value
        return tags or None

    @staticmethod
    def _parse_link(fields: list[bytes]) -> Link:
        if len(fields) < 5:
            raise ValueError("Malformed L record")
        if fields[2] in (b"+", b"-"):
            u = fields[1]
            ori_from = fields[2].decode()
            v = fields[3]
            ori_to = fields[4].decode()
            overlap = fields[5] if len(fields) > 5 else None
            tags = fields[6:]
        else:
            u_field = fields[1]
            v_field = fields[2]
            ori_from = chr(u_field[-1]) if u_field[-1] in (43, 45) else "+"
            ori_to = chr(v_field[-1]) if v_field[-1] in (43, 45) else "+"
            u = u_field.rstrip(b"+-")
            v = v_field.rstrip(b"+-")
            overlap = fields[3] if len(fields) > 3 else None
            tags = fields[4:]
        tag_dict = GFAParser._parse_tags(tags)
        return Link(u, v, ori_from, ori_to, overlap, tag_dict)

    @staticmethod
    def _parse_path(fields: list[bytes]) -> PathRecord:
        if len(fields) < 3:
            raise ValueError("Malformed P record")
        name = fields[1]
        segments: list[Tuple[bytes, str]] = []
        parts = re.split(rb"(?<=[+-])([,;])", fields[2])
        separators = [part.decode() for part in parts[1::2]]
        for entry in parts[0::2]:
            if len(entry) < 2 or not entry.endswith((b"+", b"-")):
                raise ValueError("path references must have an orientation")
            orientation = "+"
            if entry.endswith(b"+"):
                seg = entry[:-1]
                orientation = "+"
            elif entry.endswith(b"-"):
                seg = entry[:-1]
                orientation = "-"
            else:
                seg = entry
            segments.append((seg, orientation))
        tags = GFAParser._parse_tags(fields[4:]) if len(fields) > 4 else None
        overlaps = None
        if len(fields) > 3 and fields[3] not in (b"*", b""):
            overlaps = fields[3].split(b",")
        return PathRecord(name, segments, tags, overlaps, separators)

    @staticmethod
    def _parse_edge(fields: list[bytes]) -> EdgeRecord:
        if len(fields) < 6:
            raise ValueError("Malformed E record")
        # GFA2 puts both oriented references before the four coordinates.
        if (
            len(fields) >= 9
            and fields[2].endswith((b"+", b"-"))
            and fields[3].endswith((b"+", b"-"))
        ):

            def position(value: bytes) -> int:
                return int(value[:-1] if value.endswith(b"$") else value)

            return EdgeRecord(
                fields[2][:-1],
                fields[3][:-1],
                chr(fields[2][-1]),
                chr(fields[3][-1]),
                *(position(value) for value in fields[4:8]),
                fields[8],
                GFAParser._parse_tags(fields[9:]),
            )
        # Retain the older, nonstandard coordinate layout for compatibility.
        if len(fields) >= 9:
            try:
                int(fields[3])
                int(fields[4])
                int(fields[6])
                int(fields[7])
            except ValueError:
                pass
            else:
                u_field = fields[2]
                v_field = fields[5]
                ori_from = "-" if u_field.endswith(b"-") else "+"
                ori_to = "-" if v_field.endswith(b"-") else "+"
                u = u_field[:-1] if u_field.endswith((b"+", b"-")) else u_field
                v = v_field[:-1] if v_field.endswith((b"+", b"-")) else v_field
                from_start = int(fields[3])
                from_end = int(fields[4])
                to_start = int(fields[6])
                to_end = int(fields[7])
                cigar = fields[8]
                tags = GFAParser._parse_tags(fields[9:]) if len(fields) > 9 else None
                return EdgeRecord(
                    u,
                    v,
                    ori_from,
                    ori_to,
                    from_start,
                    from_end,
                    to_start,
                    to_end,
                    cigar,
                    tags,
                )
        # Historical orientation-only edges are also accepted.
        if fields[3] not in (b"+", b"-") or fields[5] not in (b"+", b"-"):
            raise ValueError(
                "Malformed E record: invalid oriented references or coordinates"
            )
        u = fields[2]
        ori_from = fields[3].decode()
        v = fields[4]
        ori_to = fields[5].decode()
        tags = GFAParser._parse_tags(fields[6:]) if len(fields) > 6 else None
        return EdgeRecord(u, v, ori_from, ori_to, None, None, None, None, None, tags)

    @staticmethod
    def _parse_containment(fields: list[bytes]) -> ContainmentRecord:
        if len(fields) < 5:
            raise ValueError("Malformed C record")
        if len(fields) >= 7 and fields[2] in (b"+", b"-") and fields[4] in (b"+", b"-"):
            return ContainmentRecord(
                fields[1],
                fields[3],
                fields[2].decode(),
                fields[4].decode(),
                int(fields[5]),
                None,
                0,
                None,
                fields[6],
                GFAParser._parse_tags(fields[7:]),
            )
        if len(fields) >= 9:
            try:
                int(fields[3])
                int(fields[4])
                int(fields[6])
                int(fields[7])
            except ValueError:
                pass
            else:
                u_field = fields[2]
                v_field = fields[5]
                ori_from = "-" if u_field.endswith(b"-") else "+"
                ori_to = "-" if v_field.endswith(b"-") else "+"
                u = u_field.rstrip(b"+-")
                v = v_field.rstrip(b"+-")
                from_start = int(fields[3])
                from_end = int(fields[4])
                to_start = int(fields[6])
                to_end = int(fields[7])
                cigar = fields[8]
                tags = GFAParser._parse_tags(fields[9:]) if len(fields) > 9 else None
                return ContainmentRecord(
                    u,
                    v,
                    ori_from,
                    ori_to,
                    from_start,
                    from_end,
                    to_start,
                    to_end,
                    cigar,
                    tags,
                )
        u = fields[1]
        ori_from = fields[2].decode()
        v = fields[3]
        ori_to = fields[4].decode()
        tags = GFAParser._parse_tags(fields[5:]) if len(fields) > 5 else None
        return ContainmentRecord(
            u, v, ori_from, ori_to, None, None, None, None, None, tags
        )

    @staticmethod
    def _parse_walk(fields: list[bytes]) -> WalkRecord:
        if len(fields) < 3:
            raise ValueError("Malformed O record")
        name = fields[1]
        segments: list[Tuple[bytes, str]] = []
        # GFA2 groups use spaces. Commas are legal within a segment identifier.
        if not fields[2].split():
            raise ValueError("empty ordered group")
        for entry in fields[2].split():
            if len(entry) < 2 or not entry.endswith((b"+", b"-")):
                raise ValueError("group references must have an orientation")
            orientation = "+"
            if entry.endswith(b"+"):
                seg = entry[:-1]
                orientation = "+"
            elif entry.endswith(b"-"):
                seg = entry[:-1]
                orientation = "-"
            else:
                seg = entry
            segments.append((seg, orientation))
        tags = GFAParser._parse_tags(fields[3:]) if len(fields) > 3 else None
        return WalkRecord(name, segments, tags)

    @staticmethod
    def _parse_gfa1_walk(fields):
        if len(fields) < 7:
            raise ValueError("Malformed W record")
        haplotype = int(fields[2])
        if haplotype < 0:
            raise ValueError("negative haplotype index")
        start = None if fields[4] == b"*" else int(fields[4])
        end = None if fields[5] == b"*" else int(fields[5])
        if (start is not None and start < 0) or (
            end is not None and (end < 0 or (start is not None and end < start))
        ):
            raise ValueError("invalid walk interval")
        tokens = re.findall(rb"([><])([^><]+)", fields[6])
        if not tokens or b"".join(sign + node for sign, node in tokens) != fields[6]:
            raise ValueError("invalid oriented walk")
        name = "#".join(
            quote(value.decode("ascii"), safe="")
            for value in (fields[1], str(haplotype).encode(), fields[3])
        ).encode()
        return WalkRecord(
            name,
            [(node, "+" if sign == b">" else "-") for sign, node in tokens],
            GFAParser._parse_tags(fields[7:]),
            start,
            end,
            True,
            fields[1],
            haplotype,
            fields[3],
        )

    @staticmethod
    def _parse_gap(fields):
        if fields[0] == b"J":
            if (
                len(fields) < 6
                or fields[2] not in (b"+", b"-")
                or fields[4] not in (b"+", b"-")
            ):
                raise ValueError("Malformed J record")
            return GapRecord(
                fields[1],
                fields[3],
                fields[2].decode(),
                fields[4].decode(),
                None if fields[5] == b"*" else int(fields[5]),
                None,
                GFAParser._parse_tags(fields[6:]),
                "J",
            )
        if (
            len(fields) < 6
            or not fields[2].endswith((b"+", b"-"))
            or not fields[3].endswith((b"+", b"-"))
        ):
            raise ValueError("Malformed G record")
        variance = None if fields[5] == b"*" else int(fields[5])
        if variance is not None and variance < 0:
            raise ValueError("negative gap variance")
        return GapRecord(
            fields[2][:-1],
            fields[3][:-1],
            chr(fields[2][-1]),
            chr(fields[3][-1]),
            None if fields[4] == b"*" else int(fields[4]),
            variance,
            GFAParser._parse_tags(fields[6:]),
            "G",
        )
