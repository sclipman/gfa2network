"""Command line conversion, graph export, statistics, and path analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np

from .analysis import (
    METHODS,
    POLICIES,
    compute_stats,
    sequence_distance,
    genome_distance,
    genome_distance_matrix,
)
from .builders import parse_gfa
from .utils import convert_format, save_matrix, save_node_map
from .version import __version__


def _shared(parser, *, root=False):
    default = {} if root else {"default": argparse.SUPPRESS}
    parser.add_argument("--raw-bytes-id", action="store_true", **default)
    parser.add_argument(
        "--max-dense-gb", type=float, **({"default": 5.0} if root else default)
    )
    parser.add_argument(
        "--max-tag-mb", type=float, **({"default": 100.0} if root else default)
    )


def _backend(parser):
    parser.add_argument("--backend", choices=["networkx", "igraph"], default="networkx")
    parser.add_argument("--verbose", action="store_true")


def _direction(parser, *, default=True):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--directed", dest="directed", action="store_true", default=default
    )
    group.add_argument("--undirected", dest="directed", action="store_false")


def _distance_options(parser):
    parser.add_argument("--method", choices=METHODS, default="min")
    parser.add_argument("--unreachable", choices=POLICIES, default="infinite")
    parser.add_argument("--weight-tag")


def _portable(graph):
    # XML formats accept scalar attributes. Nested provenance remains available
    # as JSON strings rather than being dropped or passed to unsupported writers.
    result = nx.relabel_nodes(
        graph, lambda node: str(node) if isinstance(node, bytes) else node, copy=True
    )

    def clean(attrs):
        for key, value in list(attrs.items()):
            if value is None:
                del attrs[key]
            elif isinstance(value, bytes):
                attrs[key] = value.decode("ascii")
            elif not isinstance(value, (str, int, float, bool)):
                attrs[key] = json.dumps(
                    value,
                    default=lambda obj: (
                        obj.decode("ascii") if isinstance(obj, bytes) else str(obj)
                    ),
                )

    clean(result.graph)
    for _, attrs in result.nodes(data=True):
        clean(attrs)
    for _, _, attrs in result.edges(data=True):
        clean(attrs)
    return result


def _run(argv):
    parser = argparse.ArgumentParser(prog="gfa2network")
    parser.add_argument(
        "--version", action="version", version=f"gfa2network {__version__}"
    )
    _shared(parser, root=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    convert = sub.add_parser(
        "convert", help="Convert GFA into a graph and/or adjacency matrix"
    )
    convert.add_argument("gfa")
    _backend(convert)
    _direction(convert)
    convert.add_argument("--graph", action="store_true")
    convert.add_argument("--matrix", "--save-matrix", metavar="PATH")
    convert.add_argument("-o", "--output", help="Graph pickle path")
    convert.add_argument(
        "--matrix-format", choices=["csr", "csc", "coo", "dok"], default="csr"
    )
    convert.add_argument(
        "--dtype",
        choices=["bool", "int8", "int32", "float32", "float64"],
        default="float64",
    )
    projection = convert.add_mutually_exclusive_group()
    projection.add_argument(
        "--asymmetric",
        action="store_true",
        default=None,
        help="Keep graph direction in the adjacency",
    )
    projection.add_argument(
        "--symmetric",
        dest="asymmetric",
        action="store_false",
        help="Explicitly project directed adjacency to undirected",
    )
    convert.add_argument("--no-node-map", action="store_true")
    convert.add_argument("--weight-tag")
    convert.add_argument("--store-seq", action="store_true")
    convert.add_argument("--store-tags", action="store_true")
    convert.add_argument("--split-on-alignment", action="store_true")
    convert.add_argument("--max-split-nodes", type=int, default=1_000_000)
    convert.add_argument("--strip-orientation", action="store_true")
    convert.add_argument("--bidirected", action="store_true")
    convert.add_argument("--keep-directed-bidir", action="store_true")
    convert.add_argument("--edge-mode", choices=["sum", "parallel"], default="sum")

    export = sub.add_parser("export", help="Export normalized graph connectivity")
    export.add_argument("gfa")
    export.add_argument(
        "--format",
        choices=["edge-list", "graphml", "gexf", "json"],
        default="edge-list",
    )
    export.add_argument("--output", default="-")
    export.add_argument("--bidirected", action="store_true")
    export.add_argument("--keep-directed-bidir", action="store_true")
    export.add_argument("--edge-mode", choices=["sum", "parallel"], default="sum")
    export.add_argument("--weight-tag")
    export.add_argument(
        "--with-weights",
        action="store_true",
        help="Add summed edge weights as a third edge-list column",
    )
    _direction(export)

    stats = sub.add_parser(
        "stats", aliases=["stat"], help="Graph statistics in one input pass"
    )
    stats.add_argument("gfa")
    stats.add_argument("--strip-orientation", action="store_true")
    _direction(stats)
    _backend(stats)

    distance = sub.add_parser(
        "distance", help="Distances between stored sequences or path node sets"
    )
    distance.add_argument("gfa")
    query = distance.add_mutually_exclusive_group(required=True)
    query.add_argument("--seq", nargs=2)
    query.add_argument("--path", nargs=2)
    distance.add_argument(
        "--details",
        action="store_true",
        help="Print distance and reachable coverage as JSON",
    )
    _direction(distance)
    _backend(distance)
    _distance_options(distance)

    matrix = sub.add_parser(
        "distance-matrix",
        help="Pairwise path node-set distances with labels and coverage",
    )
    matrix.add_argument("gfa")
    matrix.add_argument("-o", "--output", required=True)
    _direction(matrix, default=False)
    _backend(matrix)
    _distance_options(matrix)
    for command in (convert, export, stats, distance, matrix):
        _shared(command)
    args = parser.parse_args(argv)

    if args.cmd == "convert":
        if not args.graph and not args.matrix:
            parser.error("convert requires --graph or --matrix")
        result = parse_gfa(
            args.gfa,
            build_graph=args.graph,
            build_matrix=bool(args.matrix),
            directed=args.directed,
            backend=args.backend,
            weight_tag=args.weight_tag,
            store_seq=args.store_seq,
            store_tags=args.store_tags,
            strip_orientation=args.strip_orientation,
            verbose=args.verbose,
            bidirected=args.bidirected,
            keep_directed_bidir=args.keep_directed_bidir,
            dtype=args.dtype,
            asymmetric=args.asymmetric,
            raw_bytes_id=args.raw_bytes_id,
            return_node_list=bool(args.matrix) and not args.no_node_map,
            max_tag_mb=args.max_tag_mb,
            split_on_alignment=args.split_on_alignment,
            max_split_nodes=args.max_split_nodes,
            edge_mode=args.edge_mode,
        )
        graph = adjacency = nodes = None
        if args.graph and args.matrix:
            if args.no_node_map:
                graph, adjacency = result
            else:
                graph, adjacency, nodes = result
        elif args.graph:
            graph = result
        elif args.no_node_map:
            adjacency = result
        else:
            adjacency, nodes = result
        if args.matrix:
            save_matrix(
                convert_format(adjacency, args.matrix_format, verbose=args.verbose),
                Path(args.matrix),
                verbose=args.verbose,
                max_dense_gb=args.max_dense_gb,
            )
            if nodes is not None:
                save_node_map(nodes, Path(args.matrix + ".nodes.tsv"))
        if args.graph:
            globals()["G"] = graph
            if args.output:
                if args.backend == "igraph":
                    graph.write_pickle(args.output)
                else:
                    with open(args.output, "wb") as file:
                        pickle.dump(graph, file)
    elif args.cmd == "export":
        graph = parse_gfa(
            args.gfa,
            build_graph=True,
            build_matrix=False,
            directed=args.directed,
            bidirected=args.bidirected,
            keep_directed_bidir=args.keep_directed_bidir,
            raw_bytes_id=args.raw_bytes_id,
            max_tag_mb=args.max_tag_mb,
            weight_tag=args.weight_tag,
            edge_mode=args.edge_mode,
        )
        if args.format == "edge-list":
            stream = (
                sys.stdout
                if args.output == "-"
                else open(args.output, "w", encoding="utf-8")
            )
            try:
                for u, v, attrs in graph.edges(data=True):
                    u = u.decode("ascii") if isinstance(u, bytes) else u
                    v = v.decode("ascii") if isinstance(v, bytes) else v
                    suffix = f"\t{attrs['weight']:.17g}" if args.with_weights else ""
                    stream.write(f"{u}\t{v}{suffix}\n")
            finally:
                if stream is not sys.stdout:
                    stream.close()
        else:
            graph = _portable(graph)
            if args.format in {"graphml", "gexf"}:
                if args.output == "-":
                    lines = (
                        nx.generate_graphml(graph)
                        if args.format == "graphml"
                        else nx.generate_gexf(graph)
                    )
                    print("\n".join(lines))
                elif args.format == "graphml":
                    nx.write_graphml(graph, args.output)
                else:
                    nx.write_gexf(graph, args.output)
            else:
                payload = nx.node_link_data(graph)
                stream = (
                    sys.stdout
                    if args.output == "-"
                    else open(args.output, "w", encoding="utf-8")
                )
                try:
                    json.dump(payload, stream)
                    stream.write("\n")
                finally:
                    if stream is not sys.stdout:
                        stream.close()
    elif args.cmd == "distance":
        if args.seq and (args.method != "min" or args.unreachable != "infinite"):
            parser.error(
                "--seq uses minimum distance; method/unreachable options apply to --path"
            )
        graph, paths = parse_gfa(
            args.gfa,
            build_graph=True,
            build_matrix=False,
            directed=args.directed,
            store_seq=bool(args.seq),
            raw_bytes_id=args.raw_bytes_id,
            max_tag_mb=args.max_tag_mb,
            backend=args.backend,
            verbose=args.verbose,
            weight_tag=args.weight_tag,
            return_paths=True,
        )
        if args.seq:
            result = sequence_distance(graph, *args.seq, return_details=args.details)
        else:
            keys = [name.encode() if args.raw_bytes_id else name for name in args.path]
            if any(key not in paths for key in keys):
                raise ValueError(
                    "unknown path: "
                    + ", ".join(
                        name for key, name in zip(keys, args.path) if key not in paths
                    )
                )
            result = genome_distance(
                graph,
                paths[keys[0]],
                paths[keys[1]],
                method=args.method,
                unreachable=args.unreachable,
                return_details=args.details,
            )
        if args.details:
            payload = asdict(result)
            payload["coverage"] = result.coverage
            payload["distance"] = (
                result.distance if np.isfinite(result.distance) else "inf"
            )
            print(json.dumps(payload, allow_nan=False))
        else:
            print(f"{result:g}")
    elif args.cmd == "distance-matrix":
        result, details = genome_distance_matrix(
            args.gfa,
            method=args.method,
            backend=args.backend,
            raw_bytes_id=args.raw_bytes_id,
            verbose=args.verbose,
            directed=args.directed,
            unreachable=args.unreachable,
            weight_tag=args.weight_tag,
            return_details=True,
            max_dense_gb=args.max_dense_gb,
        )
        save_matrix(
            result,
            Path(args.output),
            verbose=args.verbose,
            max_dense_gb=args.max_dense_gb,
        )
        save_node_map(details["names"], Path(args.output + ".paths.tsv"))
        np.save(args.output + ".coverage.npy", details["coverage"])
        metadata = {key: details[key] for key in ("method", "unreachable", "directed")}
        metadata["coverage_definition"] = (
            "reachable node pairs / all node pairs"
            if details["method"] == "mean-pairwise"
            else "nodes with a reachable counterpart / all source and target nodes"
        )
        Path(args.output + ".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    else:
        result = compute_stats(
            args.gfa,
            directed=args.directed,
            strip_orientation=args.strip_orientation,
            raw_bytes_id=args.raw_bytes_id,
            backend=args.backend,
        )
        for key, value in result.items():
            print(
                f"{key}\t {value:.6g}"
                if isinstance(value, float)
                else f"{key}\t {value}"
            )


def main(argv=None):
    try:
        _run(argv)
    except (
        ValueError,
        RuntimeError,
        MemoryError,
        KeyError,
        OSError,
        nx.NetworkXException,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
