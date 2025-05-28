from __future__ import annotations

import networkx as nx
import warnings
import os

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _HAS_PANDAS = False

from .builders import parse_gfa
from .parser import GFAParser, PathRecord, WalkRecord


def _warn_directed_bidirected(G: nx.Graph) -> None:
    """Issue a warning if *G* is a directed bidirected graph."""
    if G.is_directed():
        for n in G.nodes:
            s = n.decode() if isinstance(n, (bytes, bytearray)) else str(n)
            if s.endswith(":+") or s.endswith(":-"):
                warnings.warn(
                    "distance functions ignore orientation; use G.to_undirected()",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break


def compute_stats(
    path: str,
    *,
    directed: bool = True,
    strip_orientation: bool = False,
    raw_bytes_id: bool = False,
) -> dict[str, float | int]:
    """Return simple graph statistics for *path*."""
    G = parse_gfa(
        path,
        build_graph=True,
        build_matrix=False,
        directed=directed,
        strip_orientation=strip_orientation,
        raw_bytes_id=raw_bytes_id,
    )
    path_count = sum(
        1
        for rec in GFAParser(path)
        if isinstance(rec, (PathRecord, WalkRecord))
    )
    components = nx.number_connected_components(G.to_undirected())
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 0
    density = nx.density(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "paths": path_count,
        "components": components,
        "max_degree": max_degree,
        "density": density,
    }


def sequence_distance(G: nx.Graph, seq_a: str | bytes, seq_b: str | bytes) -> float:
    """Return the shortest path length between two sequences.

    Orientation information is ignored.  If ``G`` was built using the
    bidirected representation, convert it to an undirected graph with
    ``G.to_undirected()`` before calling this function.

    Parameters
    ----------
    G : nx.Graph
        Graph with sequences stored on nodes via the ``sequence`` attribute.
    seq_a, seq_b : str | bytes
        Sequence strings or bytes to locate in ``G``.

    Raises
    ------
    KeyError
        If either sequence is not present on any node.
    """

    _warn_directed_bidirected(G)

    def _to_bytes(s: str | bytes) -> bytes:
        return s if isinstance(s, bytes) else s.encode()

    s1 = _to_bytes(seq_a)
    s2 = _to_bytes(seq_b)

    seq2nodes: dict[bytes, list[bytes]] = {}
    for node, data in G.nodes(data=True):
        seq = data.get("sequence")
        if isinstance(seq, (bytes, bytearray)):
            key = bytes(seq)
            seq2nodes.setdefault(key, []).append(node)

    if s1 not in seq2nodes or s2 not in seq2nodes:
        missing = [repr(x) for x in (seq_a, seq_b) if _to_bytes(x) not in seq2nodes]
        raise KeyError(f"sequence(s) {', '.join(missing)} not found")

    nodes_a = seq2nodes[s1]
    nodes_b = seq2nodes[s2]
    lengths = nx.multi_source_dijkstra_path_length(G, nodes_a, weight="weight")
    dists = [lengths[n] for n in nodes_b if n in lengths]
    if not dists:
        raise nx.NetworkXNoPath("no path between sequences")
    return min(dists)


def genome_distance(
    G: nx.Graph,
    nodes_a: list[bytes] | tuple[bytes, ...] | set[bytes],
    nodes_b: list[bytes] | tuple[bytes, ...] | set[bytes],
    *,
    method: str = "min",
) -> float:
    """Calculate distance between two sets of nodes.

    Orientation of nodes is ignored.  ``method`` can be ``"min"`` (default)
    to return the minimal distance or ``"mean"`` to average all pairwise
    distances between reachable nodes.  Convert bidirected graphs to
    undirected before calling this function.
    """

    _warn_directed_bidirected(G)

    nodes_a = list(nodes_a)
    nodes_b = list(nodes_b)

    if method == "min":
        lengths = nx.multi_source_dijkstra_path_length(G, nodes_a, weight="weight")
        dists = [lengths[n] for n in nodes_b if n in lengths]
        if not dists:
            raise nx.NetworkXNoPath("no path between node sets")
        return min(dists)
    elif method == "mean":
        if len(nodes_a) * len(nodes_b) > 1000 and os.getenv("GFANET_DISABLE_WARNINGS") != "1":
            warnings.warn(
                "Mean distance scales quadratically; this may be very slow on large sets",
                RuntimeWarning,
            )
        total = 0.0
        count = 0
        for u in nodes_a:
            for v in nodes_b:
                try:
                    total += nx.shortest_path_length(G, u, v, weight="weight")
                    count += 1
                except nx.NetworkXNoPath:
                    continue
        if count == 0:
            raise nx.NetworkXNoPath("no path between node sets")
        return total / count
    else:
        raise ValueError(f"unknown method: {method}")


def load_paths(
    path: str, *, raw_bytes: bool = False
) -> dict[str | bytes, list[str | bytes]]:
    """Return mapping of path/walk names to their node lists."""

    paths: dict[str | bytes, list[str | bytes]] = {}
    for rec in GFAParser(path):
        if isinstance(rec, (PathRecord, WalkRecord)):
            key = rec.name if raw_bytes else rec.name.decode("ascii")
            segs = [
                seg if raw_bytes else seg.decode("ascii") for seg, _ in rec.segments
            ]
            paths[key] = segs
    return paths


def genome_distance_matrix(
    gfa_path: str,
    method: str = "min",
    *,
    raw_bytes_id: bool = False,
    backend: str = "networkx",
    verbose: bool = False,
):
    """Return pairwise distances between all paths in *gfa_path*.

    The function loads path definitions, builds the graph and computes
    distances between every pair of paths using :func:`genome_distance`.

    Parameters
    ----------
    gfa_path : str
        Path to the input GFA file.
    method : str, optional
        Distance aggregation method (``"min"`` or ``"mean"``), by default
        ``"min"``.
    backend : str, optional
        ``"networkx"`` (default) or ``"igraph"``.
    verbose : bool, optional
        Emit progress information while parsing.

    Returns
    -------
    numpy.ndarray | pandas.DataFrame
        Matrix of pairwise distances.  A ``pandas.DataFrame`` is returned
        when pandas is installed, otherwise a ``numpy.ndarray``.

    Notes
    -----
    Orientation is ignored when computing distances.  Convert bidirected
    graphs to undirected first to avoid unexpected results.
    """

    paths = load_paths(gfa_path, raw_bytes=raw_bytes_id)
    names = list(paths)

    G = parse_gfa(
        gfa_path,
        build_graph=True,
        build_matrix=False,
        raw_bytes_id=raw_bytes_id,
        backend=backend,
        verbose=verbose,
    )
    _warn_directed_bidirected(G)

    import numpy as np

    n = len(names)
    M = np.zeros((n, n), dtype=float)

    # cache one multi-source Dijkstra per path
    cache = {
        name: nx.multi_source_dijkstra_path_length(G, nodes, weight="weight")
        for name, nodes in paths.items()
    }

    for i, name_a in enumerate(names):
        nodes_a = paths[name_a]
        lengths_a = cache[name_a]
        for j in range(i, n):
            if i == j:
                dist = 0.0
            else:
                nodes_b = paths[names[j]]
                lengths_b = cache[names[j]]
                if method == "min":
                    dists = [lengths_a[n] for n in nodes_b if n in lengths_a]
                    dist = min(dists) if dists else float("inf")
                else:  # mean of node-to-path distances
                    total = 0.0
                    count = 0
                    for u in nodes_a:
                        if u in lengths_b:
                            total += lengths_b[u]
                            count += 1
                    for v in nodes_b:
                        if v in lengths_a:
                            total += lengths_a[v]
                            count += 1
                    dist = total / count if count else float("inf")
            M[i, j] = dist
            M[j, i] = dist

    if _HAS_PANDAS:
        labels = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        return pd.DataFrame(M, index=labels, columns=labels)

    return M
