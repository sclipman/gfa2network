from __future__ import annotations

import argparse
from pathlib import Path
import sys

import networkx as nx
from .builders import parse_gfa
from .igraph_builder import parse_gfa_igraph, _HAS_IGRAPH
from .parser import GFAParser, Link, EdgeRecord, ContainmentRecord
from .utils import convert_format, save_matrix
from .analysis import (
    compute_stats,
    sequence_distance,
    genome_distance,
    genome_distance_matrix,
    load_paths,
)
from .version import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gfa2network")
    parser.add_argument(
        "--version", action="version", version=f"gfa2network {__version__}"
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
        "--matrix-format",
        default="csr",
        help="Sparse format for .npz (csr|csc|coo|dok)",
    )
    p_conv.add_argument("--weight-tag")
    p_conv.add_argument("--store-seq", action="store_true")
    p_conv.add_argument(
        "--strip-orientation",
        action="store_true",
        help="Strip +/- from IDs (v0.1 behaviour)",
    )
    p_conv.add_argument(
        "--bidirected", action="store_true", help="Use bidirected representation"
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
    p_exp.add_argument("--output", help="Output path", default="-")

    p_stats = sub.add_parser("stats", help="Print basic graph statistics")
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
        if args.backend == "igraph":
            if not _HAS_IGRAPH:
                print(
                    "Error: python-igraph is required for --backend igraph. Please install with `pip install python-igraph`.",
                    file=sys.stderr,
                )
                sys.exit(1)
            result = parse_gfa_igraph(
                args.gfa,
                build_graph=build_g,
                build_matrix=build_mat,
                directed=args.directed,
                weight_tag=args.weight_tag,
                store_seq=args.store_seq,
                strip_orientation=args.strip_orientation,
                verbose=args.verbose,
                bidirected=args.bidirected,
            )
        else:
            result = parse_gfa(
                args.gfa,
                build_graph=build_g,
                build_matrix=build_mat,
                directed=args.directed,
                weight_tag=args.weight_tag,
                store_seq=args.store_seq,
                strip_orientation=args.strip_orientation,
                verbose=args.verbose,
                bidirected=args.bidirected,
            )
        if build_g and build_mat:
            G, A = result  # type: ignore[misc]
        elif build_g:
            G = result  # type: ignore[assignment]
        else:
            A = result  # type: ignore[assignment]
        if build_mat:
            A = convert_format(A, args.matrix_format, verbose=args.verbose)
            save_matrix(A, Path(args.matrix), verbose=args.verbose)
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
            for rec in GFAParser(args.gfa):
                if isinstance(rec, (Link, EdgeRecord, ContainmentRecord)):
                    u = rec.from_segment
                    v = rec.to_segment
                    if args.bidirected:
                        u = u + b":" + rec.orientation_from.encode()
                        v = v + b":" + rec.orientation_to.encode()
                    line = f"{u.decode()}\t{v.decode()}\n"
                    if out_path:
                        with open(out_path, "a") as fh:
                            fh.write(line)
                    else:
                        sys.stdout.write(line)
        else:
            G = parse_gfa(
                args.gfa,
                build_graph=True,
                build_matrix=False,
                directed=True,
                strip_orientation=False,
                bidirected=args.bidirected,
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
            )
            dist = sequence_distance(G, seq_a, seq_b)
        else:
            paths = load_paths(args.gfa)
            name_a, name_b = args.path
            try:
                nodes_a = paths[name_a.encode()]
                nodes_b = paths[name_b.encode()]
            except KeyError as exc:
                raise SystemExit(f"unknown path: {exc.args[0].decode()}") from exc
            G = parse_gfa(
                args.gfa, build_graph=True, build_matrix=False, directed=args.directed
            )
            dist = genome_distance(G, nodes_a, nodes_b)
        print(dist)
    elif args.cmd == "distance-matrix":
        M = genome_distance_matrix(args.gfa, method=args.method)
        save_matrix(M, Path(args.output))
    elif args.cmd == "stats":
        stats = compute_stats(
            args.gfa, directed=args.directed, strip_orientation=args.strip_orientation
        )
        print("nodes\t", stats["nodes"])
        print("edges\t", stats["edges"])
        print("paths\t", stats["paths"])
        print("components\t", stats["components"])
        print("max_degree\t", stats["max_degree"])
        print("density\t", f"{stats['density']:.6g}")


if __name__ == "__main__":
    main()
