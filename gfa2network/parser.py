from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Iterable, Tuple
import sys
import warnings


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
    tags: dict[str, Any] | None = None


@dataclass
class ContainmentRecord:
    """GFA1 containment record."""

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
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

    def __iter__(self) -> Iterator[
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
                fh = open(path, "rb", buffering=1 << 20)
            close = path != "-"
        try:
            for line in fh:
                if not line:
                    continue
                if line[0] not in (ord("S"), ord("L"), ord("P"), ord("E"), ord("C"), ord("O")):
                    if line[0] not in (ord("H"), ord("F")):
                        warnings.warn(f"Skipping unsupported record: {line[:1].decode()}", RuntimeWarning, stacklevel=1)
                    continue
                fields = line.rstrip(b"\n").split(b"\t")
                rec_type = fields[0]
                if rec_type == b"S":
                    seq = fields[2] if len(fields) > 2 else None
                    length = None
                    if seq is None and len(fields) > 2:
                        try:
                            length = int(fields[2])
                        except ValueError:
                            pass
                    tags = self._parse_tags(fields[3:]) if len(fields) > 3 else None
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
                tag, typ, value = f.decode().split(':', 2)
            except ValueError:
                continue
            if typ == 'i':
                try:
                    tags[tag] = int(value)
                except ValueError:
                    pass
            elif typ == 'f':
                try:
                    tags[tag] = float(value)
                except ValueError:
                    pass
            elif typ == 'B':
                try:
                    tags[tag] = [int(x) for x in value.split(',') if x]
                except ValueError:
                    tags[tag] = value.split(',')
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
        u = fields[2]
        ori_from = fields[3].decode()
        v = fields[4]
        ori_to = fields[5].decode()
        tags = GFAParser._parse_tags(fields[6:]) if len(fields) > 6 else None
        return EdgeRecord(u, v, ori_from, ori_to, tags)

    @staticmethod
    def _parse_containment(fields: list[bytes]) -> ContainmentRecord:
        if len(fields) < 5:
            raise ValueError("Malformed C record")
        u = fields[1]
        ori_from = fields[2].decode()
        v = fields[3]
        ori_to = fields[4].decode()
        tags = GFAParser._parse_tags(fields[5:]) if len(fields) > 5 else None
        return ContainmentRecord(u, v, ori_from, ori_to, tags)

    @staticmethod
    def _parse_walk(fields: list[bytes]) -> WalkRecord:
        if len(fields) < 3:
            raise ValueError("Malformed O record")
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
        return WalkRecord(name, segments, tags)
