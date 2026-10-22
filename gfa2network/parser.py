from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Iterable, Tuple
import sys
import warnings
import gzip


@dataclass
class Segment:
    """A segment (node) record."""

    id: bytes
    length: int | None = None
    sequence: bytes | None = None
    tags: dict[str, Any] | None = None


@dataclass
class Link:
    """A link (edge) record with orientation preserved."""

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    overlap: bytes | None = None
    tags: dict[str, Any] | None = None


@dataclass
class PathRecord:
    """A path consisting of ordered oriented segments."""

    name: bytes
    segments: list[Tuple[bytes, str]]
    tags: dict[str, Any] | None = None


@dataclass
class EdgeRecord:
    """GFA2 edge/alignment record."""

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
    """GFA2 ordered walk record (O)."""

    name: bytes
    segments: list[Tuple[bytes, str]]
    tags: dict[str, Any] | None = None


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

    def __iter__(
        self,
    ) -> Iterator[
        Segment | Link | EdgeRecord | ContainmentRecord | PathRecord | WalkRecord
    ]:
        if self.file is not None:
            fh = self.file
            close = False
        else:
            path = self.path or "-"
            if path == "-":
                fh = sys.stdin.buffer
            else:
                if str(path).endswith(".gz"):
                    fh = gzip.open(path, "rb")
                else:
                    fh = open(path, "rb", buffering=1 << 20)
            close = path != "-"
        try:
            for line in fh:
                if not line.strip() or line.startswith(b"#"):
                    continue
                if line[0] not in (
                    ord("S"),
                    ord("L"),
                    ord("P"),
                    ord("E"),
                    ord("C"),
                    ord("O"),
                ):
                    if line[0] not in (ord("H"), ord("F")) and not self._warned_unknown:
                        warnings.warn(
                            f"Skipping unsupported record: {line[:1].decode()}",
                            RuntimeWarning,
                            stacklevel=1,
                        )
                        self._warned_unknown = True
                    continue
                fields = line.rstrip(b"\r\n").split(b"\t")
                rec_type = fields[0]
                if rec_type == b"S":
                    if len(fields) < 3:
                        raise ValueError("Malformed S record")
                    seq: bytes | None = None
                    length: int | None = None
                    tag_start = 3
                    if len(fields) > 2:
                        third = fields[2]
                        try:
                            length = int(third)
                            if len(fields) > 3:
                                fourth = fields[3]
                                # Heuristically detect if field 4 is a tag
                                parts = fourth.split(b":", 2)
                                if (
                                    len(parts) == 3
                                    and len(parts[0]) == 2
                                    and len(parts[1]) == 1
                                ):
                                    tag_start = 3
                                else:
                                    seq = fourth
                                    tag_start = 4
                        except ValueError:
                            seq = third
                    tags = (
                        self._parse_tags(fields[tag_start:])
                        if len(fields) > tag_start
                        else None
                    )
                    if seq == b"*":
                        seq = None
                    if length is None:
                        tagged_length = (tags or {}).get("LN")
                        if isinstance(tagged_length, int):
                            length = tagged_length
                        elif seq is not None:
                            length = len(seq)
                    yield Segment(fields[1], length, seq, tags)
                elif rec_type == b"L":
                    yield self._parse_link(fields)
                elif rec_type == b"E":
                    yield self._parse_edge(fields)
                elif rec_type == b"C":
                    yield self._parse_containment(fields)
                elif rec_type == b"P":
                    yield self._parse_path(fields)
                elif rec_type == b"O":
                    yield self._parse_walk(fields)
        finally:
            if close:
                fh.close()

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
        for entry in fields[2].split(b","):
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
        return PathRecord(name, segments, tags)

    @staticmethod
    def _parse_edge(fields: list[bytes]) -> EdgeRecord:
        if len(fields) < 6:
            raise ValueError("Malformed E record")
        # GFA2 puts both oriented references before the four coordinates.
        if (len(fields) >= 9 and fields[2].endswith((b"+", b"-"))
                and fields[3].endswith((b"+", b"-"))):
            def position(value: bytes) -> int:
                return int(value[:-1] if value.endswith(b"$") else value)

            return EdgeRecord(
                fields[2][:-1], fields[3][:-1],
                chr(fields[2][-1]), chr(fields[3][-1]),
                *(position(value) for value in fields[4:8]),
                fields[8], GFAParser._parse_tags(fields[9:]),
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
                tags = (
                    GFAParser._parse_tags(fields[9:]) if len(fields) > 9 else None
                )
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
            raise ValueError("Malformed E record: invalid oriented references or coordinates")
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
        if (len(fields) >= 7 and fields[2] in (b"+", b"-")
                and fields[4] in (b"+", b"-")):
            return ContainmentRecord(
                fields[1], fields[3], fields[2].decode(), fields[4].decode(),
                int(fields[5]), None, 0, None, fields[6],
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
                tags = (
                    GFAParser._parse_tags(fields[9:]) if len(fields) > 9 else None
                )
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
        return ContainmentRecord(u, v, ori_from, ori_to, None, None, None, None, None, tags)

    @staticmethod
    def _parse_walk(fields: list[bytes]) -> WalkRecord:
        if len(fields) < 3:
            raise ValueError("Malformed O record")
        name = fields[1]
        segments: list[Tuple[bytes, str]] = []
        # GFA2 groups use spaces. Commas are legal within a segment identifier.
        for entry in fields[2].split():
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
