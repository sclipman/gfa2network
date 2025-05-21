from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Iterable, Tuple
import sys


@dataclass
class Segment:
    """A segment (node) record."""

    id: bytes
    sequence: bytes | None = None


@dataclass
class Link:
    """A link (edge) record with orientation preserved."""

    from_segment: bytes
    to_segment: bytes
    orientation_from: str
    orientation_to: str
    tags: list[bytes] | None = None


@dataclass
class PathRecord:
    """A path consisting of ordered oriented segments."""

    name: bytes
    segments: list[Tuple[bytes, str]]


class GFAParser:
    """Streaming parser yielding :class:`Segment`, :class:`Link`, or :class:`PathRecord`."""

    def __init__(self, source: str | Path | BinaryIO):
        if isinstance(source, (str, Path)):
            self.path = str(source)
            self.file: BinaryIO | None = None
        else:
            self.path = None
            self.file = source

    def __iter__(self) -> Iterator[Segment | Link | PathRecord]:
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
                if line[0] not in (ord("S"), ord("L"), ord("P")):
                    continue
                fields = line.rstrip(b"\n").split(b"\t")
                rec_type = fields[0]
                if rec_type == b"S":
                    seq = fields[2] if len(fields) > 2 else None
                    yield Segment(fields[1], seq)
                elif rec_type == b"L":
                    yield self._parse_link(fields)
                elif rec_type == b"P":
                    yield self._parse_path(fields)
        finally:
            if close:
                fh.close()

    @staticmethod
    def _parse_link(fields: list[bytes]) -> Link:
        if len(fields) < 3:
            raise ValueError("Malformed L record")
        if fields[2] in (b"+", b"-"):
            u = fields[1]
            ori_from = fields[2].decode()
            v = fields[3]
            ori_to = fields[4].decode()
            tags = fields[5:]
        else:
            u_field = fields[1]
            v_field = fields[2]
            ori_from = chr(u_field[-1]) if u_field[-1] in (43, 45) else "+"
            ori_to = chr(v_field[-1]) if v_field[-1] in (43, 45) else "+"
            u = u_field.rstrip(b"+-")
            v = v_field.rstrip(b"+-")
            tags = fields[3:]
        return Link(u, v, ori_from, ori_to, list(tags))

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
        return PathRecord(name, segments)
