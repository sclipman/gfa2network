"""Streaming GFA conversion with a common graph and sparse-matrix contract."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

from .alignment import split_records
from .model import GraphAccumulator
from .parser import (
    GFAParser,
    Segment,
    Link,
    EdgeRecord,
    ContainmentRecord,
    GapRecord,
    PathRecord,
    WalkRecord,
)
from .paths import PathCollector
from .utils import available_memory


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
    asymmetric: bool | None = None,
    raw_bytes_id: bool = False,
    return_node_list: bool = False,
    max_tag_mb: float = 100.0,
    split_on_alignment: bool = False,
    return_paths: bool = False,
    edge_mode: str = "sum",
    progress_step: int = 500_000,
    max_split_nodes: int = 1_000_000,
):
    """Convert supported GFA records into graphs and/or sparse adjacency.

    Repeated edges have summed weights by default. ``edge_mode='parallel'``
    retains individual edges in a multigraph; its adjacency also sums weights.
    Self-loops count once per record. Matrices respect graph direction unless
    ``asymmetric=False`` explicitly requests an undirected projection.

    Bidirected mode creates two oriented vertices per segment, and both the
    forward and reverse-complement connection. Stored reverse sequences are
    reverse-complemented. ``strip_orientation`` suppresses orientation metadata;
    it never strips characters from a base identifier.

    ``return_paths=True`` returns ``(ordinary_result, paths)``. This collects P/O/W
    paths in the same input pass, works on stdin, and validates path references.
    Split paths refer to the generated intervals in traversal order.
    """
    if backend not in {"networkx", "igraph"}:
        raise ValueError(f"unknown backend: {backend}")
    if backend == "igraph":
        from .igraph_builder import _HAS_IGRAPH

        if not _HAS_IGRAPH:
            raise RuntimeError(
                "python-igraph is not available; install gfa2network[igraph]"
            )
    if return_node_list and not build_matrix:
        raise ValueError("return_node_list requires build_matrix=True")
    if progress_step <= 0:
        raise ValueError("progress_step must be positive")
    builder = GraphAccumulator(
        build_graph=build_graph,
        build_matrix=build_matrix,
        backend=backend,
        directed=directed,
        weight_tag=weight_tag,
        store_seq=store_seq,
        store_tags=store_tags,
        strip_orientation=strip_orientation,
        bidirected=bidirected,
        keep_directed_bidir=keep_directed_bidir,
        raw_bytes_id=raw_bytes_id,
        edge_mode=edge_mode,
        max_tag_mb=max_tag_mb,
    )
    parser = GFAParser(path)
    records = (
        split_records(parser, max_nodes=max_split_nodes)
        if split_on_alignment
        else parser
    )
    paths = PathCollector() if return_paths else None
    for count, record in enumerate(records, 1):
        try:
            if isinstance(record, Segment):
                builder.add_segment(record)
            elif isinstance(record, (Link, EdgeRecord, ContainmentRecord, GapRecord)):
                builder.add_edge_record(record)
            elif paths is not None and isinstance(record, (PathRecord, WalkRecord)):
                paths.add(record)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{getattr(record, 'line_number', count)}: {exc}"
            ) from exc
        if verbose and count % progress_step == 0:
            print(f"[{count:,} lines]", file=sys.stderr, flush=True)
    if return_paths and parser.unsupported_records:
        raise ValueError(
            "path analysis cannot ignore unsupported records: "
            + ", ".join(sorted(parser.unsupported_records))
        )
    if verbose:
        label = "parse_gfa_igraph" if backend == "igraph" else "parse_gfa"
        print(f"[{label}] done", file=sys.stderr)
        available = available_memory()
        if available and builder.sequence_bytes > 0.5 * available:
            warnings.warn(
                "stored sequences exceed 50% of available memory", RuntimeWarning
            )
    graph = builder.finish_graph()
    matrix = builder.finish_matrix(dtype, asymmetric)
    if graph is not None:
        if backend == "networkx":
            graph.graph["edge_mode"] = edge_mode
        else:
            graph["edge_mode"] = edge_mode
    node_list = (
        [builder.name(node) for node in builder.node_index]
        if return_node_list
        else None
    )
    if build_graph and build_matrix:
        result = (graph, matrix, node_list) if return_node_list else (graph, matrix)
    elif build_graph:
        result = graph
    elif build_matrix:
        result = (matrix, node_list) if return_node_list else matrix
    else:
        result = None
    if return_paths:
        return result, paths.finish(
            raw_bytes=raw_bytes_id,
            oriented=bidirected,
            nodes={
                node + b":" + orientation
                for node in builder.defined_segments
                for orientation in (b"+", b"-")
            }
            if bidirected
            else builder.defined_segments,
        )
    return result
