"""Graph connectivity distances with explicit aggregation and missing-path rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    _HAS_PANDAS = False

from .builders import parse_gfa
from .parser import GFAParser, Segment, PathRecord, WalkRecord
from .paths import PathCollector

METHODS = ("min", "mean", "mean-pairwise", "mean-nearest")
POLICIES = ("infinite", "ignore", "raise")
COSTS = ("weight", "hops")


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    method: str
    reachable: int
    total: int

    @property
    def coverage(self):
        return self.reachable / self.total if self.total else 0.0


def _method(method):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    return "mean-pairwise" if method == "mean" else method


def _policy(policy):
    if policy not in POLICIES:
        raise ValueError("unreachable must be infinite, ignore, or raise")


def _cost(cost):
    if cost not in COSTS:
        raise ValueError("cost must be weight or hops")


class DistanceEngine:
    """One CSR index per graph; searches retain one O(V) distance vector.

    Parallel edges use the cheapest cost for shortest paths, matching standard
    multigraph shortest-path semantics. Adjacency export separately sums weights.
    ``cost='weight'`` uses edge weights (summed record weights in the default
    edge mode); ``cost='hops'`` charges 1 per edge, independent of record
    multiplicity, support or weight tags.
    """

    def __init__(self, graph, *, warn=True, cost="weight"):
        _cost(cost)
        self.cost = cost
        if isinstance(graph, nx.Graph):
            self.nodes = list(graph.nodes)
            self.directed = graph.is_directed()
            edges = (
                (u, v, attrs.get("weight", 1.0))
                for u, v, attrs in graph.edges(data=True)
            )
        elif hasattr(graph, "vs") and hasattr(graph, "es"):
            self.nodes = (
                list(graph.vs["name"])
                if "name" in graph.vertex_attributes()
                else list(range(graph.vcount()))
            )
            self.directed = graph.is_directed()
            edges = (
                (
                    self.nodes[e.source],
                    self.nodes[e.target],
                    e.attributes().get("weight", 1.0),
                )
                for e in graph.es
            )
        else:
            raise TypeError("expected a NetworkX or igraph graph")
        self.index = {node: i for i, node in enumerate(self.nodes)}
        if len(self.index) != len(self.nodes):
            raise ValueError("distance analysis requires unique vertex names")
        if (
            warn
            and self.directed
            and any(str(node).endswith((":+", ":-")) for node in self.nodes)
        ):
            warnings.warn(
                "the supplied graph is directed; use an undirected graph for orientation-free distances",
                RuntimeWarning,
            )
        weights = {}
        for u, v, weight in edges:
            weight = 1.0 if weight is None or cost == "hops" else weight
            if (
                not isinstance(weight, (int, float, np.number))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("Dijkstra edge costs must be finite and nonnegative")
            key = (self.index[u], self.index[v])
            weights[key] = min(weights.get(key, math.inf), float(weight))
            if not self.directed:
                other = key[::-1]
                weights[other] = min(weights.get(other, math.inf), float(weight))
        rows, cols = zip(*weights) if weights else ((), ())
        self.matrix = sp.csr_matrix(
            (list(weights.values()), (rows, cols)),
            shape=(len(self.nodes), len(self.nodes)),
        )
        self.reverse = None

    def indices(self, nodes):
        unique = list(dict.fromkeys(nodes))
        if not unique:
            raise ValueError("distance node sets must be nonempty")
        try:
            return np.asarray([self.index[node] for node in unique], dtype=np.int64)
        except KeyError as exc:
            raise nx.NodeNotFound(
                f"node {exc.args[0]!r} is absent from the graph"
            ) from exc

    def search(self, sources, *, reverse=False):
        matrix = self.matrix
        if reverse and self.directed:
            if self.reverse is None:
                self.reverse = self.matrix.T.tocsr()
            matrix = self.reverse
        return dijkstra(matrix, directed=True, indices=sources, min_only=True)


def _summarize(total_sum, reachable, total, policy):
    if reachable < total:
        if policy == "raise":
            raise nx.NetworkXNoPath(
                f"only {reachable}/{total} distance contributions are reachable"
            )
        if policy == "infinite":
            return math.inf
    return total_sum / reachable if reachable else math.inf


def _distance(engine, a, b, method, policy):
    if method == "mean-pairwise":
        if len(a) * len(b) > 1000 and os.getenv("GFANET_DISABLE_WARNINGS") != "1":
            warnings.warn(
                "Mean pairwise distance examines a quadratic number of node pairs",
                RuntimeWarning,
            )
        total_sum, reachable = 0.0, 0
        for node in a:
            values = engine.search([node])[b]
            finite = np.isfinite(values)
            total_sum += float(values[finite].sum())
            reachable += int(finite.sum())
        total = len(a) * len(b)
        value = _summarize(total_sum, reachable, total, policy)
    else:
        to_b = engine.search(a)[b]
        from_a = engine.search(b, reverse=True)[a]
        values = np.concatenate((from_a, to_b))
        finite = np.isfinite(values)
        reachable, total = int(finite.sum()), len(values)
        if method == "min":
            value = float(to_b.min())
            if policy == "raise" and not np.isfinite(value):
                raise nx.NetworkXNoPath("no path between node sets")
        else:
            value = _summarize(float(values[finite].sum()), reachable, total, policy)
    return DistanceResult(value, method, reachable, total)


def genome_distance(
    graph,
    nodes_a,
    nodes_b,
    *,
    method="min",
    unreachable="infinite",
    return_details=False,
    cost="weight",
):
    """Distance between unique node sets, respecting the supplied direction.

    ``mean`` aliases ``mean-pairwise`` everywhere. ``mean-nearest`` averages
    nearest-target distances for source nodes and nearest-source distances for
    target nodes. Means include infinity by default; ignoring unreachable
    contributions requires ``unreachable='ignore'``. ``min`` is the ordinary
    minimum, so any shared node yields zero. No method estimates sequence
    divergence. With ``return_details=True`` return a DistanceResult including
    contribution coverage. Pairwise coverage counts pairs; min/nearest coverage
    counts source/target nodes that have at least one reachable counterpart.
    ``cost='hops'`` charges 1 per edge instead of the (summed) edge weight.
    """
    method = _method(method)
    _policy(unreachable)
    engine = DistanceEngine(graph, cost=cost)
    result = _distance(
        engine, engine.indices(nodes_a), engine.indices(nodes_b), method, unreachable
    )
    return result if return_details else result.distance


def sequence_distance(graph, seq_a, seq_b, *, return_details=False, cost="weight"):
    """Minimum graph distance between nodes with exact stored sequence strings."""
    first = seq_a.encode() if isinstance(seq_a, str) else bytes(seq_a)
    second = seq_b.encode() if isinstance(seq_b, str) else bytes(seq_b)
    if isinstance(graph, nx.Graph):
        entries = (
            (node, attrs.get("sequence")) for node, attrs in graph.nodes(data=True)
        )
    elif hasattr(graph, "vs"):
        entries = (
            (vertex["name"], vertex.attributes().get("sequence")) for vertex in graph.vs
        )
    else:
        raise TypeError("expected a NetworkX or igraph graph")
    a, b = [], []
    for node, sequence in entries:
        if sequence == first:
            a.append(node)
        if sequence == second:
            b.append(node)
    if not a or not b:
        raise KeyError(
            "one or both sequences were not found; build the graph with store_seq=True"
        )
    return genome_distance(
        graph, a, b, unreachable="raise", return_details=return_details, cost=cost
    )


def load_paths(path, *, raw_bytes=False, oriented=False):
    """Load P/O/W paths, sorting nonoverlapping W chunks by coordinate.

    W names use escaped ``sample#haplotype#sequence`` components. Chunk gaps do
    not create graph edges. ``oriented=True`` appends :+ / :- for oriented graphs.
    """
    parser, collector = GFAParser(path), PathCollector()
    declared = set()
    for record in parser:
        if isinstance(record, Segment):
            if oriented:
                declared.update((record.id + b":+", record.id + b":-"))
            else:
                declared.add(record.id)
        if isinstance(record, (PathRecord, WalkRecord)):
            collector.add(record)
    if parser.unsupported_records:
        raise ValueError(
            "path analysis cannot ignore unsupported records: "
            + ", ".join(sorted(parser.unsupported_records))
        )
    return collector.finish(raw_bytes=raw_bytes, oriented=oriented, nodes=declared)


def compute_stats(
    path,
    *,
    directed=True,
    strip_orientation=False,
    raw_bytes_id=False,
    backend="networkx",
    projection="all",
):
    """Compute graph and logical path counts in one input pass, including stdin."""
    graph, paths = parse_gfa(
        path,
        build_graph=True,
        build_matrix=False,
        directed=directed,
        strip_orientation=strip_orientation,
        raw_bytes_id=raw_bytes_id,
        backend=backend,
        return_paths=True,
        projection=projection,
    )
    if isinstance(graph, nx.Graph):
        nodes, edges = graph.number_of_nodes(), graph.number_of_edges()
        components = nx.number_connected_components(graph.to_undirected(as_view=True))
        maximum = max((degree for _, degree in graph.degree()), default=0)
        density = nx.density(graph)
    else:
        nodes, edges = graph.vcount(), graph.ecount()
        components = len(graph.connected_components(mode="weak"))
        maximum = max(graph.degree(), default=0)
        density = graph.density(loops=False) if nodes > 1 else 0.0
    return dict(
        nodes=nodes,
        edges=edges,
        paths=len(paths),
        components=components,
        max_degree=maximum,
        density=density,
    )


def genome_distance_matrix(
    gfa_path,
    method="min",
    *,
    raw_bytes_id=False,
    backend="networkx",
    verbose=False,
    directed=False,
    unreachable="infinite",
    weight_tag=None,
    return_details=False,
    max_dense_gb=5.0,
    projection="all",
    cost="weight",
    names=None,
):
    """Pairwise path-set distances using the same definitions as genome_distance.

    Input is consumed once. Search workspace is O(V+E); output and aggregation
    require O(P²) memory. Pairwise mean searches once per unique source node,
    while minimum/nearest methods search once per path (twice if directed).
    Mean pairwise self-distances are computed, not artificially set to zero.
    ``return_details=True`` additionally returns names, coverage, reachable and
    total contribution matrices, plus the graph projection manifest and cost.
    ``names`` restricts the matrix to those paths (in that order); the graph
    still contains every record. Output allocation is guarded before creation.
    """
    method = _method(method)
    _policy(unreachable)
    _cost(cost)
    if not math.isfinite(max_dense_gb) or max_dense_gb < 0:
        raise ValueError("max_dense_gb must be finite and nonnegative")
    graph, paths = parse_gfa(
        gfa_path,
        build_graph=True,
        build_matrix=False,
        backend=backend,
        directed=directed,
        weight_tag=weight_tag,
        raw_bytes_id=raw_bytes_id,
        verbose=verbose,
        return_paths=True,
        projection=projection,
    )
    if names is None:
        names = list(paths)
    else:
        names = [
            name.encode() if raw_bytes_id and isinstance(name, str) else name
            for name in names
        ]
        missing = [name for name in names if name not in paths]
        if missing:
            raise KeyError(f"unknown path: {missing[0]!r}")
    if not names:
        raise ValueError("input contains no named P/O/W paths")
    n = len(names)
    allocation = n * n * 8 * 7
    if allocation > max_dense_gb * 1e9:
        raise MemoryError(
            f"distance matrix workspace would allocate about {allocation / 1e9:.3g} GB; increase max_dense_gb or select fewer paths"
        )
    engine = DistanceEngine(graph, warn=False, cost=cost)
    sets = [engine.indices(paths[name]) for name in names]
    lengths = np.asarray([len(nodes) for nodes in sets], dtype=np.int64)
    sums = np.zeros((n, n), dtype=float)
    reachable = np.zeros((n, n), dtype=np.int64)
    matrix = np.full((n, n), np.inf)
    if method == "mean-pairwise":
        total = lengths[:, None] * lengths[None, :]
        memberships = {}
        for i, nodes in enumerate(sets):
            for node in nodes:
                memberships.setdefault(int(node), []).append(i)
        for node, members in memberships.items():
            distances = engine.search([node])
            row_sums = np.asarray(
                [
                    distances[nodes][np.isfinite(distances[nodes])].sum()
                    for nodes in sets
                ]
            )
            row_counts = np.asarray(
                [np.isfinite(distances[nodes]).sum() for nodes in sets]
            )
            for i in members:
                sums[i] += row_sums
                reachable[i] += row_counts
    else:
        total = lengths[:, None] + lengths[None, :]
        for i, nodes in enumerate(sets):
            forward = engine.search(nodes)
            reverse = engine.search(nodes, reverse=True) if engine.directed else forward
            for j, targets in enumerate(sets):
                values = forward[targets]
                finite = np.isfinite(values)
                if method == "min":
                    matrix[i, j] = values.min()
                sums[i, j] += values[finite].sum()
                reachable[i, j] += finite.sum()
                # Reverse search from B gives A-to-B nearest target costs.
                values = reverse[targets]
                finite = np.isfinite(values)
                sums[j, i] += values[finite].sum()
                reachable[j, i] += finite.sum()
    if method != "min":
        np.divide(sums, reachable, out=matrix, where=reachable > 0)
        if unreachable == "infinite":
            matrix[reachable < total] = np.inf
        elif unreachable == "raise" and np.any(reachable < total):
            raise nx.NetworkXNoPath("some path distance contributions are unreachable")
    elif unreachable == "raise" and np.any(~np.isfinite(matrix)):
        raise nx.NetworkXNoPath("some paths have no connection")
    labels = [
        name.decode("ascii") if isinstance(name, bytes) else str(name) for name in names
    ]
    output = (
        pd.DataFrame(matrix, index=labels, columns=labels) if _HAS_PANDAS else matrix
    )
    if return_details:
        return output, {
            "names": labels,
            "coverage": reachable / total,
            "reachable": reachable,
            "total": total,
            "method": method,
            "unreachable": unreachable,
            "directed": directed,
            "cost": cost,
            "projection": (
                graph.graph["projection"]
                if isinstance(graph, nx.Graph)
                else graph["projection"]
            ),
        }
    return output
