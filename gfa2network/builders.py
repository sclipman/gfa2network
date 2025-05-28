from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import sys
import warnings
import pickle

from .parser import (
    GFAParser,
    Segment,
    Link,
    EdgeRecord,
    ContainmentRecord,
)
from .utils import available_memory, convert_format, save_matrix
from .igraph_builder import parse_gfa_igraph, _HAS_IGRAPH

try:
    import scipy.sparse as sp

    _HAS_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAS_SCIPY = False


def parse_gfa(
    path: str | Path,
    *,
    build_graph: bool,
    build_matrix: bool,
    directed: bool = True,
    weight_tag: str | None = None,
    store_seq: bool = False,
    store_tags: bool = False,
    strip_orientation: bool = False,
    verbose: bool = False,
    bidirected: bool = False,
    keep_directed_bidir: bool = False,
    backend: str = "networkx",
    dtype: str | object = "float64",
    asymmetric: bool = False,
    raw_bytes_id: bool = False,
    return_node_list: bool = False,
):
    """Stream-parse *path* and return the requested artefacts.

    Parameters
    ----------
    path : str | Path
        Input GFA file (``-`` for ``stdin``).
    build_graph : bool
        Return a NetworkX/igraph graph.
    build_matrix : bool
        Return a SciPy sparse adjacency matrix.
    directed : bool, optional
        Treat the graph as directed, by default ``True``.
    weight_tag : str | None, optional
        Numeric GFA tag to use as edge weight.
    store_seq : bool, optional
        Attach sequences from ``S`` records to nodes.
    store_tags : bool, optional
        Keep tag dictionaries and segment lengths.
    strip_orientation : bool, optional
        Remove ``+/-`` from segment identifiers.
    verbose : bool, optional
        Emit progress information.
    bidirected : bool, optional
        Duplicate nodes as ``<id>:+`` and ``<id>:-``.
    keep_directed_bidir : bool, optional
        Keep directed edges when ``bidirected`` is ``True``.
    backend : str, optional
        ``"networkx"`` (default) or ``"igraph"``.
    dtype : str | object, optional
        Data type for the adjacency matrix.
    asymmetric : bool, optional
        Do not mirror the upper triangle when directed.
    raw_bytes_id : bool, optional
        Use raw byte strings for node identifiers.
    return_node_list : bool, optional
        Return a node list alongside the matrix.

    Notes
    -----
    When the bidirected representation is kept directed, distances should be
    computed on ``G.to_undirected()`` as orientation is ignored.
    """
    if backend == "igraph":
        return parse_gfa_igraph(
            path,
            build_graph=build_graph,
            build_matrix=build_matrix,
            directed=directed,
            weight_tag=weight_tag,
            store_seq=store_seq,
            store_tags=store_tags,
            strip_orientation=strip_orientation,
            verbose=verbose,
            bidirected=bidirected,
            keep_directed_bidir=keep_directed_bidir,
            return_node_list=return_node_list,
        )
    if return_node_list and not build_matrix:
        raise ValueError("return_node_list requires build_matrix=True")
    if build_matrix and not _HAS_SCIPY:
        raise RuntimeError("Matrix output requires SciPy")
    if store_seq and not build_graph:
        store_seq = False
    if store_tags and not build_graph:
        store_tags = False

    if bidirected:
        graph_cls = nx.MultiDiGraph if keep_directed_bidir else nx.MultiGraph
    else:
        graph_cls = nx.DiGraph if directed else nx.Graph
    G = graph_cls() if build_graph else None
    graph_directed = keep_directed_bidir or (not bidirected and directed)

    node2idx: dict[bytes, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    seq_bytes_total = 0
    tags_bytes_total = 0

    parser = GFAParser(path)
    node_str: dict[bytes, str] = {}

    def _id(n: bytes) -> bytes | str:
        if raw_bytes_id:
            return n
        s = node_str.get(n)
        if s is None:
            s = n.decode("ascii")
            node_str[n] = s
        return s
    for lineno, record in enumerate(parser, 1):
        if isinstance(record, Segment):
            seg = record.id
            if build_graph:
                if bidirected:
                    for ori in ("+", "-"):
                        node = seg + b":" + ori.encode()
                        attrs = {}
                        if store_seq and record.sequence is not None:
                            attrs["sequence"] = record.sequence
                        if store_tags and record.length is not None:
                            attrs["length"] = record.length
                        if store_tags and record.tags is not None:
                            attrs["tags"] = record.tags
                            tags_bytes_total += len(pickle.dumps(record.tags))
                        G.add_node(_id(node), **attrs)
                else:
                    attrs = {}
                    if store_seq and record.sequence is not None:
                        attrs["sequence"] = record.sequence
                        seq_bytes_total += len(record.sequence)
                    if store_tags and record.length is not None:
                        attrs["length"] = record.length
                    if store_tags and record.tags is not None:
                        attrs["tags"] = record.tags
                        tags_bytes_total += len(pickle.dumps(record.tags))
                    G.add_node(_id(seg), **attrs)
            if build_matrix:
                if bidirected:
                    for ori in ("+", "-"):
                        node = seg + b":" + ori.encode()
                        if node not in node2idx:
                            node2idx[node] = len(node2idx)
                else:
                    if seg not in node2idx:
                        node2idx[seg] = len(node2idx)
        elif isinstance(record, (Link, EdgeRecord, ContainmentRecord)):
            u = record.from_segment
            v = record.to_segment
            if strip_orientation:
                u = u.rstrip(b"+-")
                v = v.rstrip(b"+-")
            w: float | None = None
            if weight_tag and record.tags and weight_tag in record.tags:
                val = record.tags[weight_tag]
                if isinstance(val, (int, float)):
                    w = float(val)
            if bidirected:
                u_node = u + b":" + record.orientation_from.encode()
                v_node = v + b":" + record.orientation_to.encode()
            else:
                u_node = u
                v_node = v
            if build_matrix:

                def add_mat_edge(a: bytes, b: bytes) -> None:
                    for n in (a, b):
                        if n not in node2idx:
                            node2idx[n] = len(node2idx)
                    rows.append(node2idx[a])
                    cols.append(node2idx[b])
                    data.append(1.0 if w is None else w)
                    if not graph_directed:
                        rows.append(node2idx[b])
                        cols.append(node2idx[a])
                        data.append(1.0 if w is None else w)

                add_mat_edge(u_node, v_node)
                if bidirected and not keep_directed_bidir:
                    rev_from = b"-" if record.orientation_from == "+" else b"+"
                    rev_to = b"-" if record.orientation_to == "+" else b"+"
                    add_mat_edge(v + b":" + rev_to, u + b":" + rev_from)
            if build_graph:
                attrs = {}
                if not strip_orientation and not bidirected:
                    attrs = {
                        "orientation_from": record.orientation_from,
                        "orientation_to": record.orientation_to,
                    }
                if store_tags and record.tags is not None:
                    attrs["tags"] = record.tags
                    tags_bytes_total += len(pickle.dumps(record.tags))

                def add_graph_edge(a: bytes, b: bytes) -> None:
                    if w is None:
                        G.add_edge(_id(a), _id(b), **attrs)
                    else:
                        G.add_edge(_id(a), _id(b), weight=w, **attrs)

                add_graph_edge(u_node, v_node)
                if bidirected and not keep_directed_bidir:
                    rev_from = b"-" if record.orientation_from == "+" else b"+"
                    rev_to = b"-" if record.orientation_to == "+" else b"+"
                    add_graph_edge(v + b":" + rev_to, u + b":" + rev_from)
        if verbose and lineno % 500_000 == 0:
            print(f"\r[{lineno:,} lines]", end="", file=sys.stderr)

    if verbose:
        print("\r[parse_gfa] done")
        if store_seq and build_graph:
            avail = available_memory()
            if avail and seq_bytes_total > 0.5 * avail:
                extra_gb = seq_bytes_total / 1e9
                print(
                    f"[warning] stored sequences use {extra_gb:.1f} GB (>50% of available memory)",
                )
    if store_tags and build_graph and tags_bytes_total > 100_000_000:
        warnings.warn(
            f"stored tag dictionaries use {tags_bytes_total/1e6:.1f} MB",
            RuntimeWarning,
        )

    out_graph = G
    out_mat = None
    node_list = None
    if build_matrix:
        n = len(node2idx)
        dt = np.dtype(dtype)
        out_mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=dt)
        if not asymmetric and graph_directed:
            out_mat = out_mat.maximum(out_mat.T)
        if return_node_list:
            node_list = [None] * n
            for node, idx in node2idx.items():
                val = node if raw_bytes_id else node.decode()
                node_list[idx] = val

    if build_graph and build_matrix:
        if return_node_list:
            return out_graph, out_mat, node_list
        return out_graph, out_mat
    if build_graph:
        return out_graph
    if build_matrix:
        if return_node_list:
            return out_mat, node_list
        return out_mat
