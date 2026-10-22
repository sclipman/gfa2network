"""Streaming GFA conversion with a common graph and sparse-matrix contract."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

from .alignment import split_records
from .genome_paths import GenomePathBuilder
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
    rich_paths: bool = False,
    edge_mode: str = "sum",
    projection="all",
    return_manifest: bool = False,
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

    ``projection`` selects which relation classes become edges: ``all`` (the
    default and the 1.x/2.0 behavior), ``traversal``, ``gap-inclusive``,
    ``alignment``, or an iterable of classes; see :mod:`gfa2network.projection`.
    A manifest describing the projection and policies is stored as the graph
    attribute ``projection``. ``return_manifest=True`` additionally returns it:
    ``(result, manifest)``, where ``result`` is whatever would otherwise return.

    ``rich_paths=True`` (with ``return_paths=True``) returns a
    :class:`~gfa2network.genome_paths.GenomePathSet` instead of the list mapping.
    It keeps ordered oriented steps, repeated visits, P overlaps and link/jump
    separators, W chunk coordinates, per-transition relation evidence, and
    segment sequences for spelling. Its ``vertex_lists()`` equals the list
    result. Steps describe the input segments; under ``split_on_alignment`` each
    step also lists its interval vertices with parent coordinates.
    """
    if backend not in {"networkx", "igraph"}:
        raise ValueError(f"unknown backend: {backend}")
    if backend == "igraph":
        from .igraph_builder import _HAS_IGRAPH

        if not _HAS_IGRAPH:
            raise RuntimeError(
                "python-igraph is not available; install gfa2network[igraph]"
            )
    if rich_paths and not return_paths:
        raise ValueError("rich_paths requires return_paths=True")
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
        projection=projection,
    )
    parser = GFAParser(path)
    rich = (
        GenomePathBuilder(raw_bytes=raw_bytes_id, bidirected=bidirected)
        if rich_paths
        else None
    )
    source = parser if rich is None else rich.observe(parser, str(path))
    records = (
        split_records(source, max_nodes=max_split_nodes)
        if split_on_alignment
        else source
    )
    paths = PathCollector() if return_paths and rich is None else None
    for count, record in enumerate(records, 1):
        try:
            if isinstance(record, Segment):
                builder.add_segment(record)
                if rich is not None and record.parent is not None:
                    rich.add(record)
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
    manifest = builder.manifest(subdivided=split_on_alignment)
    if graph is not None:
        if backend == "networkx":
            manifest["graph_edges"] = graph.number_of_edges()
            graph.graph["edge_mode"] = edge_mode
            graph.graph["projection"] = manifest
        else:
            manifest["graph_edges"] = graph.ecount()
            graph["edge_mode"] = edge_mode
            graph["projection"] = manifest
    if matrix is not None:
        manifest["matrix_nonzeros"] = int(matrix.nnz)
        manifest["matrix_symmetric_projection"] = asymmetric is False
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
    if rich is not None:
        result = (result, rich.finish())
    elif paths is not None:
        result = (
            result,
            paths.finish(
                raw_bytes=raw_bytes_id,
                oriented=bidirected,
                nodes={
                    node + b":" + orientation
                    for node in builder.defined_segments
                    for orientation in (b"+", b"-")
                }
                if bidirected
                else builder.defined_segments,
            ),
        )
    return (result, manifest) if return_manifest else result
