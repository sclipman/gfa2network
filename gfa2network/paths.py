"""Ordered path collection, including coordinate-sorted GFA1.1 walk chunks."""

from __future__ import annotations

from .parser import WalkRecord


class PathCollector:
    def __init__(self):
        self.records = {}
        self.unnamed = 0

    def add(self, record):
        name = record.name
        if name == b"*":
            self.unnamed += 1
            name = f"@O:{self.unnamed}".encode()
        entries = self.records.setdefault(name, [])
        chunk = isinstance(record, WalkRecord) and record.is_chunk
        if entries and (
            not chunk
            or not all(isinstance(x, WalkRecord) and x.is_chunk for x in entries)
        ):
            raise ValueError(f"duplicate path name {name!r}")
        entries.append(record)

    def finish(self, *, raw_bytes=False, oriented=False, nodes=None):
        result = {}
        for name, records in self.records.items():
            if len(records) > 1:
                if any(rec.start is None or rec.end is None for rec in records):
                    raise ValueError(
                        f"multiple W chunks for {name!r} require coordinates"
                    )
                records = sorted(records, key=lambda rec: rec.start)
                if any(a.end > b.start for a, b in zip(records, records[1:])):
                    raise ValueError(f"overlapping W chunks for {name!r}")
            values = []
            for record in records:
                for node, orientation in record.segments:
                    key = node + b":" + orientation.encode() if oriented else node
                    if nodes is not None and key not in nodes:
                        raise ValueError(
                            f"path {name!r} references undefined segment/group {node!r}"
                        )
                    values.append(key if raw_bytes else key.decode("ascii"))
            result[name if raw_bytes else name.decode("ascii")] = values
        return result
