"""Command line conversion, graph export, statistics, and path analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np

from .analysis import (
    COSTS,
    METHODS,
    POLICIES,
    compute_stats,
    sequence_distance,
    genome_distance,
    genome_distance_matrix,
)
from .builders import parse_gfa
from .genome_paths import OVERLAP_POLICIES, load_genome_paths
from .path_compare import COPIES, MEASURES, compare_paths, shared_content_matrix
from .projection import CLASSES, PROJECTIONS, export_manifest, resolve_projection
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
    parser.add_argument(
        "--cost",
        choices=COSTS,
        default="weight",
        help="Edge cost: summed weight (default) or 1 per edge (hops)",
    )


def _checked_projection(value):
    try:
        resolve_projection(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _projection(parser):
    parser.add_argument(
        "--projection",
        type=_checked_projection,
        default="all",
        help=(
            "Relations that become edges: "
            + ", ".join(PROJECTIONS)
            + ", or comma-separated classes from "
            + ", ".join(CLASSES)
        ),
    )


def _write_manifest(path, manifest, formats, *, with_weights=False):
    payload = export_manifest(manifest, formats, with_weights=with_weights)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


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


def _sha256(path):
    if path == "-":
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    convert.add_argument(
        "--manifest", metavar="PATH", help="Write the projection manifest"
    )

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
    export.add_argument(
        "--manifest", metavar="PATH", help="Write the projection manifest"
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
    paths = sub.add_parser(
        "paths",
        help="Validate path transitions, spell sequences and audit them against FASTA",
    )
    paths.add_argument("gfa")
    paths.add_argument("--reference", metavar="FASTA", help="Source sequences")
    paths.add_argument("--path", nargs="+", help="Audit only these paths")
    circular = paths.add_mutually_exclusive_group()
    circular.add_argument(
        "--circular",
        nargs="+",
        default=(),
        help="Paths whose comparison permits rotation",
    )
    circular.add_argument("--all-circular", action="store_true")
    paths.add_argument(
        "--reverse-complement",
        action="store_true",
        help="Accept a reverse-complement match",
    )
    paths.add_argument("--case-sensitive", action="store_true")
    paths.add_argument(
        "--unknown-overlap",
        choices=OVERLAP_POLICIES,
        default="strict",
        help="Abstain on unspecified overlaps (strict) or treat them as blunt (zero)",
    )
    paths.add_argument("-o", "--output", default="-", help="JSON receipt path")
    paths.add_argument("--fasta", help="Write completely spelled paths as FASTA")
    paths.add_argument(
        "--strict", action="store_true", help="Exit nonzero unless every path passes"
    )
    compare = sub.add_parser(
        "compare",
        help="Shared content, ordered shared blocks and witnesses for two paths",
    )
    compare.add_argument("gfa")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument(
        "--orientation", choices=["forward", "both"], default="forward"
    )
    compare.add_argument("--circular", action="store_true")
    compare.add_argument("--copies", choices=COPIES, default="unique")
    compare.add_argument(
        "--unknown-overlap", choices=OVERLAP_POLICIES, default="strict"
    )
    compare.add_argument("--max-rotations", type=int, default=64)
    compare.add_argument("-o", "--output", default="-", help="JSON output path")

    content = sub.add_parser(
        "shared-content", help="Pairwise base-aware shared content between paths"
    )
    content.add_argument("gfa")
    content.add_argument("-o", "--output", required=True)
    content.add_argument("--measure", choices=MEASURES, default="jaccard")
    content.add_argument("--copies", choices=COPIES, default="unique")
    content.add_argument(
        "--unknown-overlap", choices=OVERLAP_POLICIES, default="strict"
    )
    content.add_argument("--path", nargs="+", help="Compare only these paths")
    for command in (convert, export, stats, distance, matrix, paths, compare, content):
        _shared(command)
    for command in (convert, export, stats, distance, matrix):
        _projection(command)
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
            projection=args.projection,
            return_manifest=True,
        )
        result, manifest = result
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
        if args.manifest:
            formats = (["pickle"] if args.graph and args.output else []) + (
                ["matrix"] if args.matrix else []
            )
            _write_manifest(args.manifest, manifest, formats)
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
            projection=args.projection,
        )
        if args.manifest:
            _write_manifest(
                args.manifest,
                graph.graph["projection"],
                [args.format],
                with_weights=args.with_weights,
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
            projection=args.projection,
        )
        if args.seq:
            result = sequence_distance(
                graph, *args.seq, return_details=args.details, cost=args.cost
            )
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
                cost=args.cost,
            )
        if args.details:
            payload = asdict(result)
            payload["cost"] = args.cost
            payload["projection"] = resolve_projection(args.projection)[0]
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
            projection=args.projection,
            cost=args.cost,
        )
        save_matrix(
            result,
            Path(args.output),
            verbose=args.verbose,
            max_dense_gb=args.max_dense_gb,
        )
        save_node_map(details["names"], Path(args.output + ".paths.tsv"))
        np.save(args.output + ".coverage.npy", details["coverage"])
        metadata = {
            key: details[key]
            for key in ("method", "unreachable", "directed", "cost", "projection")
        }
        metadata["coverage_definition"] = (
            "reachable node pairs / all node pairs"
            if details["method"] == "mean-pairwise"
            else "nodes with a reachable counterpart / all source and target nodes"
        )
        Path(args.output + ".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    elif args.cmd == "compare":
        collection = load_genome_paths(args.gfa)
        unknown = [name for name in (args.a, args.b) if name not in collection]
        if unknown:
            raise ValueError("unknown path: " + ", ".join(unknown))
        result = compare_paths(
            collection,
            args.a,
            args.b,
            copies=args.copies,
            orientation=args.orientation,
            circular=args.circular,
            unknown_overlap=args.unknown_overlap,
            max_rotations=args.max_rotations,
        )
        payload = {
            "format": "gfa2network-path-comparison/1",
            "version": __version__,
            "input": {"path": args.gfa, "sha256": _sha256(args.gfa)},
        }
        payload.update(result.to_dict())
        text = json.dumps(payload, indent=2) + "\n"
        if args.output == "-":
            sys.stdout.write(text)
        else:
            Path(args.output).write_text(text)
    elif args.cmd == "shared-content":
        collection = load_genome_paths(args.gfa)
        unknown = [name for name in args.path or () if name not in collection]
        if unknown:
            raise ValueError("unknown path: " + ", ".join(unknown))
        result, details = shared_content_matrix(
            collection,
            args.path,
            measure=args.measure,
            copies=args.copies,
            unknown_overlap=args.unknown_overlap,
        )
        save_matrix(
            result,
            Path(args.output),
            verbose=False,
            max_dense_gb=args.max_dense_gb,
        )
        save_node_map(details["names"], Path(args.output + ".paths.tsv"))
        metadata = {
            "format": "gfa2network-shared-content/1",
            "version": __version__,
            "input": {"path": args.gfa, "sha256": _sha256(args.gfa)},
            "measure": details["measure"],
            "copies": details["copies"],
            "unknown_overlap": details["unknown_overlap"],
            "note": details["note"],
            "missing_values": "NaN where neither path has resolved bases",
            "paths": [
                dict(asdict(item), evidence=item.evidence) for item in details["paths"]
            ],
        }
        Path(args.output + ".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    elif args.cmd == "paths":
        collection = load_genome_paths(args.gfa)
        unknown = [name for name in args.path or () if name not in collection]
        if unknown:
            raise ValueError("unknown path: " + ", ".join(unknown))
        report = collection.audit(
            args.reference,
            names=args.path,
            circular=True if args.all_circular else args.circular,
            reverse_complement=args.reverse_complement,
            ignore_case=not args.case_sensitive,
            unknown_overlap=args.unknown_overlap,
        )
        payload = {
            "format": "gfa2network-path-fidelity/1",
            "version": __version__,
            "input": {"path": args.gfa, "sha256": _sha256(args.gfa)},
            "reference": (
                {"path": args.reference, "sha256": _sha256(args.reference)}
                if args.reference
                else None
            ),
        }
        payload.update(report.to_dict())
        text = json.dumps(payload, indent=2) + "\n"
        if args.output == "-":
            sys.stdout.write(text)
        else:
            Path(args.output).write_text(text)
        if args.fasta:
            with open(args.fasta, "wb") as handle:
                for name in args.path or list(collection):
                    spelled = collection.spell(
                        name, unknown_overlap=args.unknown_overlap
                    )
                    if spelled.sequence is not None:
                        handle.write(b">" + name.encode() + b"\n")
                        sequence = spelled.sequence
                        for start in range(0, len(sequence), 80):
                            handle.write(sequence[start : start + 80] + b"\n")
        if args.strict and not report.ok:
            failed = len(report.paths) - report.summary["passed"]
            raise SystemExit(
                f"{failed} of {len(report.paths)} paths did not pass the fidelity audit"
            )
    else:
        result = compute_stats(
            args.gfa,
            directed=args.directed,
            strip_orientation=args.strip_orientation,
            raw_bytes_id=args.raw_bytes_id,
            backend=args.backend,
            projection=args.projection,
        )
        name = resolve_projection(args.projection)[0]
        if name != "all":
            print(f"projection\t {name}")
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
