from __future__ import annotations

try:
    import igraph as ig

    _HAS_IGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    ig = None  # type: ignore
    _HAS_IGRAPH = False

try:
    import scipy.sparse as sp

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    sp = None  # type: ignore
    _HAS_SCIPY = False

from .parser import GFAParser, Segment, Link, EdgeRecord, ContainmentRecord


class IGraphBuilder:
    """Build an :class:`igraph.Graph` from GFA records."""

    def __init__(
        self,
        *,
        directed: bool = True,
        weight_tag: str | None = None,
        store_seq: bool = False,
        strip_orientation: bool = False,
        bidirected: bool = False,
        keep_directed_bidir: bool = False,
    ) -> None:
        if bidirected:
            self.directed = True if keep_directed_bidir else False
        else:
            self.directed = directed
        self.weight_tag = weight_tag
        self.store_seq = store_seq
        self.strip_orientation = strip_orientation
        self.bidirected = bidirected
        self.keep_directed_bidir = keep_directed_bidir
        self.graph = ig.Graph(directed=self.directed) if _HAS_IGRAPH else None
        self._node_index: dict[bytes, int] = {}

    # ------------------------------------------------------------------
    def _add_vertex(self, node: bytes, seg: Segment | None = None) -> int:
        idx = self._node_index.get(node)
        if idx is None:
            name = node.decode()
            self.graph.add_vertex(name=name)  # type: ignore[union-attr]
            idx = self.graph.vcount() - 1  # type: ignore[union-attr]
            self._node_index[node] = idx
            if seg is not None:
                if seg.length is not None:
                    self.graph.vs[idx]["length"] = seg.length  # type: ignore[union-attr]
                if self.store_seq and seg.sequence is not None:
                    self.graph.vs[idx]["sequence"] = seg.sequence  # type: ignore[union-attr]
                if seg.tags:
                    self.graph.vs[idx]["tags"] = seg.tags  # type: ignore[union-attr]
        return idx

    # ------------------------------------------------------------------
    def add_segment(self, seg: Segment) -> None:
        if self.bidirected:
            for ori in ("+", "-"):
                node = seg.id + b":" + ori.encode()
                self._add_vertex(node, seg)
        else:
            self._add_vertex(seg.id, seg)

    # ------------------------------------------------------------------
    def add_edge_record(self, rec: Link | EdgeRecord | ContainmentRecord) -> None:
        u = rec.from_segment
        v = rec.to_segment
        if self.strip_orientation:
            u = u.rstrip(b"+-")
            v = v.rstrip(b"+-")
        if self.bidirected:
            u = u + b":" + rec.orientation_from.encode()
            v = v + b":" + rec.orientation_to.encode()
        idx_u = self._add_vertex(u)
        idx_v = self._add_vertex(v)
        attrs: dict[str, object] = {}
        if not self.strip_orientation and not self.bidirected:
            attrs["orientation_from"] = rec.orientation_from
            attrs["orientation_to"] = rec.orientation_to
        if rec.tags is not None:
            attrs["tags"] = rec.tags
        w = None
        if self.weight_tag and rec.tags and self.weight_tag in rec.tags:
            val = rec.tags[self.weight_tag]
            if isinstance(val, (int, float)):
                w = float(val)
        if w is not None:
            attrs["weight"] = w
        self.graph.add_edge(idx_u, idx_v, **attrs)  # type: ignore[union-attr]
        if self.bidirected and not self.keep_directed_bidir:
            rev_from = "-" if rec.orientation_from == "+" else "+"
            rev_to = "-" if rec.orientation_to == "+" else "+"
            u2 = v + b":" + rev_to.encode()
            v2 = u + b":" + rev_from.encode()
            idx_u2 = self._add_vertex(u2)
            idx_v2 = self._add_vertex(v2)
            self.graph.add_edge(idx_u2, idx_v2, **attrs)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    def to_matrix(self) -> "sp.spmatrix":
        """Return the adjacency matrix of the built graph."""
        if not _HAS_SCIPY:
            raise RuntimeError("Matrix output requires SciPy")
        return self.graph.get_adjacency_sparse(attribute="weight", default=1.0)  # type: ignore[union-attr]


# ----------------------------------------------------------------------


def parse_gfa_igraph(
    path: str | bytes,
    *,
    build_graph: bool,
    build_matrix: bool,
    directed: bool = True,
    weight_tag: str | None = None,
    store_seq: bool = False,
    strip_orientation: bool = False,
    verbose: bool = False,
    bidirected: bool = False,
    keep_directed_bidir: bool = False,
) -> ig.Graph | "sp.spmatrix" | tuple[ig.Graph, "sp.spmatrix"] | None:
    """Parse *path* and return an igraph graph and/or a sparse matrix."""
    if not _HAS_IGRAPH:
        raise RuntimeError("python-igraph is not available")
    if build_matrix and not _HAS_SCIPY:
        raise RuntimeError("Matrix output requires SciPy")

    builder = IGraphBuilder(
        directed=directed,
        weight_tag=weight_tag,
        store_seq=store_seq,
        strip_orientation=strip_orientation,
        bidirected=bidirected,
        keep_directed_bidir=keep_directed_bidir,
    )
    parser = GFAParser(path)
    for lineno, record in enumerate(parser, 1):
        if isinstance(record, Segment):
            builder.add_segment(record)
        elif isinstance(record, (Link, EdgeRecord, ContainmentRecord)):
            builder.add_edge_record(record)
        if verbose and lineno % 500_000 == 0:
            print(f"\r[{lineno:,} lines]", end="")
    if verbose:
        print("\r[parse_gfa_igraph] done")

    G = builder.graph if build_graph else None
    A = builder.to_matrix() if build_matrix else None
    if build_graph and build_matrix:
        return G, A
    if build_graph:
        return G
    return A
