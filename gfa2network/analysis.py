from __future__ import annotations

import networkx as nx

from .builders import parse_gfa
from .parser import GFAParser, PathRecord


def compute_stats(
    path: str, *, directed: bool = True, strip_orientation: bool = False
) -> dict[str, float | int]:
    """Return simple graph statistics for *path*."""
    G = parse_gfa(
        path,
        build_graph=True,
        build_matrix=False,
        directed=directed,
        strip_orientation=strip_orientation,
    )
    path_count = sum(1 for rec in GFAParser(path) if isinstance(rec, PathRecord))
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
