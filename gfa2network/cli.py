from __future__ import annotations

import argparse
from pathlib import Path
import sys

import networkx as nx
from .builders import parse_gfa
from .igraph_builder import parse_gfa_igraph, _HAS_IGRAPH
from .parser import GFAParser, Link, EdgeRecord, ContainmentRecord
from .utils import convert_format, save_matrix, save_node_map
from .analysis import (
    compute_stats,
    sequence_distance,
    genome_distance,
    genome_distance_matrix,
    load_paths,
)
from importlib.metadata import version, PackageNotFoundError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gfa2network")
    try:
        pkg_version = version("gfa2network")
    except PackageNotFoundError:  # pragma: no cover - fallback when not installed
        from .version import __version__ as pkg_version

    parser.add_argument(
        "--version",
        action="version",
        version=f"gfa2network {pkg_version}",
    )
    parser.add_argument(
        "--raw-bytes-id",
        action="store_true",
        help="Use raw bytes for node identifiers (legacy)",
    )
    parser.add_argument(
        "--max-dense-gb",
        type=float,
        default=5.0,
        help="Abort dense matrix saves over N GB (default 5)",
    )
    parser.add_argument(
        "--max-tag-mb",
        type=float,
        default=100.0,
        help="Warn when stored tags exceed N MB (default 100)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert GFA to graph or matrix")
    p_conv.add_argument("gfa", help="Input *.gfa* file or - for stdin")
    p_conv.add_argument(
        "--backend",
        choices=["networkx", "igraph"],
        default="networkx",
        help="Graph backend to use",
    )
    g = p_conv.add_mutually_exclusive_group()
    g.add_argument(
        "--directed",
        dest="directed",
        action="store_true",
        default=True,
        help="Treat graph as directed",
    )
    g.add_argument(
        "--undirected",
        dest="directed",
        action="store_false",
        help="Treat graph as undirected",
    )
    p_conv.add_argument("--graph", action="store_true", help="Build a NetworkX object")
    p_conv.add_argument(
        "--matrix",
        metavar="PATH",
        help="Write adjacency matrix to PATH (.npz|.npy|.csv)",
    )
    p_conv.add_argument(
        "--save-matrix",
        dest="matrix",
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    p_conv.add_argument(
        "--matrix-format",
        default="csr",
        help="Sparse format for .npz (csr|csc|coo|dok)",
    )
    p_conv.add_argument(
        "--dtype",
        choices=["bool", "int8", "int32", "float32", "float64"],
        default="float64",
        help="Data type for adjacency matrix",
    )
    p_conv.add_argument(
        "--asymmetric",
        action="store_true",
        help="Do not mirror upper triangle",
    )
    p_conv.add_argument(
        "--no-node-map",
        action="store_true",
        help="Do not write <matrix>.nodes.tsv sidecar",
    )
    p_conv.add_argument("--weight-tag")
    p_conv.add_argument("--store-seq", action="store_true")
    p_conv.add_argument("--store-tags", action="store_true")
    p_conv.add_argument(
        "--split-on-alignment",
        action="store_true",
        help="Split segments at alignment boundaries",
    )
    p_conv.add_argument(
        "--strip-orientation",
        action="store_true",
        help="Strip +/- from IDs (v0.1 behaviour)",
    )
    p_conv.add_argument(
        "--bidirected", action="store_true", help="Use bidirected representation"
    )
    p_conv.add_argument(
        "--keep-directed-bidir",
        action="store_true",
        help="Keep original directed bidirected behaviour",
    )
    p_conv.add_argument("--verbose", action="store_true")
    p_conv.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write graph pickle to PATH",
    )

    p_exp = sub.add_parser("export", help="Stream edges in simple formats")
    p_exp.add_argument("gfa")
    p_exp.add_argument(
        "--format",
        default="edge-list",
        choices=["edge-list", "graphml", "gexf", "json"],
    )
    p_exp.add_argument("--bidirected", action="store_true")
    p_exp.add_argument(
        "--keep-directed-bidir",
        action="store_true",
        help="Keep original directed bidirected behaviour",
    )
    p_exp.add_argument("--output", help="Output path", default="-")

    p_stats = sub.add_parser(
        "stats", help="Print basic graph statistics", aliases=["stat"]
    )
    p_stats.add_argument("gfa", help="Input *.gfa* file or - for stdin")
    g2 = p_stats.add_mutually_exclusive_group()
    g2.add_argument("--directed", dest="directed", action="store_true", default=True)
    g2.add_argument("--undirected", dest="directed", action="store_false")
    p_stats.add_argument("--strip-orientation", action="store_true")

    p_dist = sub.add_parser("distance", help="Compute distances")
    p_dist.add_argument("gfa", help="Input *.gfa* file")
    g3 = p_dist.add_mutually_exclusive_group(required=True)
    g3.add_argument("--seq", nargs=2, metavar=("SEQ_A", "SEQ_B"))
    g3.add_argument("--path", nargs=2, metavar=("PATH_A", "PATH_B"))
    g4 = p_dist.add_mutually_exclusive_group()
    g4.add_argument("--directed", dest="directed", action="store_true", default=True)
    g4.add_argument("--undirected", dest="directed", action="store_false")

    p_dm = sub.add_parser("distance-matrix", help="Pairwise distances between paths")
    p_dm.add_argument("gfa", help="Input *.gfa* file")
    p_dm.add_argument(
        "-o", "--output", required=True, help="Write matrix to PATH (.csv|.npy|.npz)"
    )
    p_dm.add_argument("--method", choices=["min", "mean"], default="min")

    args = parser.parse_args(argv)

    if args.cmd == "convert":
        if not args.graph and not args.matrix:
            parser.error("convert requires --graph or --matrix")
        build_mat = bool(args.matrix)
        build_g = args.graph
        print(f"Using backend: {args.backend}")
        if args.backend == "igraph" and not _HAS_IGRAPH:
            print(
                "Error: python-igraph is required for --backend igraph. Install with `pip install python-igraph`.",
                file=sys.stderr,
            )
            sys.exit(1)

        result = parse_gfa(
            args.gfa,
            build_graph=build_g,
            build_matrix=build_mat,
            directed=args.directed,
            weight_tag=args.weight_tag,
            store_seq=args.store_seq,
            store_tags=args.store_tags,
            strip_orientation=args.strip_orientation,
            verbose=args.verbose,
            bidirected=args.bidirected,
            keep_directed_bidir=args.keep_directed_bidir,
            backend=args.backend,
            dtype=args.dtype,
            asymmetric=args.asymmetric,
            raw_bytes_id=args.raw_bytes_id,
            return_node_list=build_mat and not args.no_node_map,
            max_tag_mb=args.max_tag_mb,
            split_on_alignment=args.split_on_alignment,
        )
        if build_g and build_mat:
            if build_mat and not args.no_node_map:
                G, A, nodes = result  # type: ignore[misc]
            else:
                G, A = result  # type: ignore[misc]
        elif build_g:
            G = result  # type: ignore[assignment]
        else:
            if build_mat and not args.no_node_map:
                A, nodes = result  # type: ignore[assignment]
            else:
                A = result  # type: ignore[assignment]
        if build_mat:
            A = convert_format(A, args.matrix_format, verbose=args.verbose)
            try:
                save_matrix(
                    A,
                    Path(args.matrix),
                    verbose=args.verbose,
                    max_dense_gb=args.max_dense_gb,
                )
            except MemoryError as exc:
                raise SystemExit(str(exc)) from exc
            if not args.no_node_map:
                save_node_map(nodes, Path(str(args.matrix) + ".nodes.tsv"))
        if build_g:
            globals().update({"G": G})
            if args.output:
                if args.backend == "networkx":
                    if hasattr(nx, "write_gpickle"):
                        nx.write_gpickle(G, args.output)
                    else:  # pragma: no cover - legacy NetworkX
                        import pickle

                        with open(args.output, "wb") as fh:
                            pickle.dump(G, fh)
                else:
                    G.write_pickle(args.output)
    elif args.cmd == "export":
        parser_format = args.format
        out_path = Path(args.output) if args.output != "-" else None
        if parser_format == "edge-list":
            fh = open(out_path, "w") if out_path else sys.stdout
            try:
                for rec in GFAParser(args.gfa):
                    if isinstance(rec, (Link, EdgeRecord, ContainmentRecord)):
                        u = rec.from_segment
                        v = rec.to_segment
                        if args.bidirected:
                            u = u + b":" + rec.orientation_from.encode()
                            v = v + b":" + rec.orientation_to.encode()
                        line = f"{u.decode()}\t{v.decode()}\n"
                        fh.write(line)
            finally:
                if out_path:
                    fh.close()
        else:
            G = parse_gfa(
                args.gfa,
                build_graph=True,
                build_matrix=False,
                directed=True,
                strip_orientation=False,
                bidirected=args.bidirected,
                keep_directed_bidir=args.keep_directed_bidir,
                raw_bytes_id=args.raw_bytes_id,
                max_tag_mb=args.max_tag_mb,
            )
            if parser_format == "graphml":
                nx.write_graphml(G, args.output)
            elif parser_format == "gexf":
                nx.write_gexf(G, args.output)
            elif parser_format == "json":
                import json

                data = nx.readwrite.json_graph.node_link_data(G)
                if args.output == "-":
                    json.dump(data, sys.stdout)
                else:
                    with open(args.output, "w") as fh:
                        json.dump(data, fh)
    elif args.cmd == "distance":
        if args.seq:
            seq_a, seq_b = args.seq
            G = parse_gfa(
                args.gfa,
                build_graph=True,
                build_matrix=False,
                directed=args.directed,
                store_seq=True,
                raw_bytes_id=args.raw_bytes_id,
                max_tag_mb=args.max_tag_mb,
            )
            dist = sequence_distance(G, seq_a, seq_b)
        else:
            paths = load_paths(args.gfa, raw_bytes=args.raw_bytes_id)
            name_a, name_b = args.path
            try:
                key_a = name_a if not args.raw_bytes_id else name_a.encode()
                key_b = name_b if not args.raw_bytes_id else name_b.encode()
                nodes_a = paths[key_a]
                nodes_b = paths[key_b]
            except KeyError as exc:
                msg = exc.args[0]
                if isinstance(msg, bytes):
                    msg = msg.decode()
                raise SystemExit(f"unknown path: {msg}") from exc
            G = parse_gfa(
                args.gfa,
                build_graph=True,
                build_matrix=False,
                directed=args.directed,
                raw_bytes_id=args.raw_bytes_id,
                max_tag_mb=args.max_tag_mb,
            )
            dist = genome_distance(G, nodes_a, nodes_b)
        print(dist)
    elif args.cmd == "distance-matrix":
        M = genome_distance_matrix(
            args.gfa, method=args.method, raw_bytes_id=args.raw_bytes_id
        )
        try:
            save_matrix(
                M,
                Path(args.output),
                max_dense_gb=args.max_dense_gb,
            )
        except MemoryError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.cmd == "stats":
        stats = compute_stats(
            args.gfa,
            directed=args.directed,
            strip_orientation=args.strip_orientation,
            raw_bytes_id=args.raw_bytes_id,
        )
        print("nodes\t", stats["nodes"])
        print("edges\t", stats["edges"])
        print("paths\t", stats["paths"])
        print("components\t", stats["components"])
        print("max_degree\t", stats["max_degree"])
        print("density\t", f"{stats['density']:.6g}")


if __name__ == "__main__":
    main()
