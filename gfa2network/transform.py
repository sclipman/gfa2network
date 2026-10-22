"""Sequence-preserving rewrites of GFA1 files, for representation checks.

``chop`` splits long segments into pieces and ``rename`` replaces segment
identifiers. Neither changes the sequence any path spells, so measures that
should describe sequence (shared content, collinear blocks, spelled paths) must
be unchanged afterwards, while vertex-count statistics such as hop distances
may change. Both read the input twice and stream the output; they accept GFA1
``H``/``S``/``L``/``J``/``P``/``W`` records (``rename`` also ``C``) and reject
anything else, rather than silently dropping it.
"""

from __future__ import annotations

import gzip
import re

_WALK = re.compile(rb"([><])([^><]+)")


def _open(path, mode):
    path = str(path)
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def _records(path):
    with _open(path, "rb") as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip(b"\r\n")
            if line and not line.startswith(b"#"):
                yield number, line.split(b"\t")


def _check(kind, allowed, number):
    if kind not in allowed:
        raise ValueError(
            f"line {number}: {kind.decode('ascii', 'replace')} records are not "
            "supported by this transform"
        )


def chop(source, destination, max_length):
    """Split every segment longer than ``max_length`` into ordered pieces.

    Pieces are named ``<id>.<k>`` (1-based); a name that already exists is an
    error. Links attach to the terminal piece on each side according to
    orientation, abutting pieces are joined by ``0M`` links, and P/W paths
    expand in traversal order. Links must be blunt (``0M``) or unspecified
    (``*``): a real overlap could not be kept on a shorter piece. Segments
    without a sequence are left whole. Returns the number of segments split.
    """
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    allowed = {b"H", b"S", b"L", b"J", b"P", b"W"}
    names, pieces = set(), {}
    for number, fields in _records(source):
        _check(fields[0], allowed, number)
        if fields[0] == b"S":
            names.add(fields[1])
            sequence = fields[2]
            if sequence != b"*" and len(sequence) > max_length:
                pieces[fields[1]] = (len(sequence) + max_length - 1) // max_length
    for segment, count in pieces.items():
        for k in range(1, count + 1):
            if segment + b".%d" % k in names:
                raise ValueError(f"piece name {segment!r}.{k} already exists")

    def parts(segment, orientation):
        count = pieces.get(segment)
        if count is None:
            return [segment]
        ids = [segment + b".%d" % k for k in range(1, count + 1)]
        return ids if orientation == b"+" else ids[::-1]

    def exit_piece(segment, orientation):
        return parts(segment, orientation)[-1]

    def entry_piece(segment, orientation):
        return parts(segment, orientation)[0]

    with _open(destination, "wb") as out:
        for number, fields in _records(source):
            kind = fields[0]
            if kind == b"S" and fields[1] in pieces:
                segment, sequence, tags = fields[1], fields[2], fields[3:]
                tags = [t for t in tags if not t.startswith(b"LN:i:")]
                ids = parts(segment, b"+")
                for k, piece in enumerate(ids):
                    chunk = sequence[k * max_length : (k + 1) * max_length]
                    out.write(b"\t".join([b"S", piece, chunk] + tags) + b"\n")
                for u, v in zip(ids, ids[1:]):
                    out.write(b"\t".join([b"L", u, b"+", v, b"+", b"0M"]) + b"\n")
            elif kind in (b"L", b"J"):
                if kind == b"L" and len(fields) > 5 and fields[5] not in (b"0M", b"*"):
                    raise ValueError(f"line {number}: chop requires blunt links")
                fields = list(fields)
                fields[1] = exit_piece(fields[1], fields[2])
                fields[3] = entry_piece(fields[3], fields[4])
                out.write(b"\t".join(fields) + b"\n")
            elif kind == b"P":
                fields = list(fields)
                tokens = re.split(rb"(?<=[+-])([,;])", fields[2])
                steps, separators = tokens[0::2], tokens[1::2]
                overlaps = fields[3] if len(fields) > 3 else b"*"
                entries = None if overlaps == b"*" else overlaps.split(b",")
                if entries is not None:
                    if any(o not in (b"0M", b"*") for o in entries):
                        raise ValueError(f"line {number}: chop requires blunt overlaps")
                    if len(entries) != len(steps) - 1:
                        raise ValueError(
                            f"line {number}: chop requires one overlap per transition"
                        )
                expanded, joins, written = [], [], []
                for index, step in enumerate(steps):
                    ids = parts(step[:-1], step[-1:])
                    if index:
                        joins.append(separators[index - 1])
                        # Each original transition keeps its own declaration.
                        written.append(entries[index - 1] if entries else None)
                    joins.extend([b","] * (len(ids) - 1))
                    written.extend([b"0M"] * (len(ids) - 1))
                    expanded.extend(piece + step[-1:] for piece in ids)
                text = expanded[0] + b"".join(
                    join + piece for join, piece in zip(joins, expanded[1:])
                )
                fields[2] = text
                if entries is not None:
                    fields[3] = b",".join(written) or b"*"
                out.write(b"\t".join(fields) + b"\n")
            elif kind == b"W":
                fields = list(fields)
                walk = b""
                for sign, segment in _WALK.findall(fields[6]):
                    orientation = b"+" if sign == b">" else b"-"
                    walk += b"".join(
                        sign + piece for piece in parts(segment, orientation)
                    )
                fields[6] = walk
                out.write(b"\t".join(fields) + b"\n")
            else:
                out.write(b"\t".join(fields) + b"\n")
    return len(pieces)


def rename(source, destination, prefix="s"):
    """Rename segments to ``<prefix><n>`` in order of their S records.

    Returns the mapping from old to new identifiers. A generated name that
    collides with an old one is harmless because every reference is rewritten.
    """
    allowed = {b"H", b"S", b"L", b"J", b"C", b"P", b"W"}
    mapping = {}
    for number, fields in _records(source):
        _check(fields[0], allowed, number)
        if fields[0] == b"S":
            mapping[fields[1]] = prefix.encode() + b"%d" % (len(mapping) + 1)

    def new(segment, number):
        try:
            return mapping[segment]
        except KeyError:
            raise ValueError(f"line {number}: undefined segment {segment!r}") from None

    with _open(destination, "wb") as out:
        for number, fields in _records(source):
            fields = list(fields)
            kind = fields[0]
            if kind == b"S":
                fields[1] = new(fields[1], number)
            elif kind in (b"L", b"J", b"C"):
                fields[1] = new(fields[1], number)
                fields[3] = new(fields[3], number)
            elif kind == b"P":
                tokens = re.split(rb"(?<=[+-])([,;])", fields[2])
                fields[2] = b"".join(
                    token if i % 2 else new(token[:-1], number) + token[-1:]
                    for i, token in enumerate(tokens)
                )
            elif kind == b"W":
                fields[6] = b"".join(
                    sign + new(segment, number)
                    for sign, segment in _WALK.findall(fields[6])
                )
            out.write(b"\t".join(fields) + b"\n")
    return {
        old.decode("ascii"): value.decode("ascii") for old, value in mapping.items()
    }
