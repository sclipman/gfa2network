from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .builders import parse_gfa
from .utils import convert_format, save_matrix
from .analysis import compute_stats
from .version import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gfa2network")
    parser.add_argument(
        "--version", action="version", version=f"gfa2network {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert GFA to graph or matrix")
    p_conv.add_argument("gfa", help="Input *.gfa* file or - for stdin")
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
    p_conv.add_argument("--verbose", action="store_true")

    p_stats = sub.add_parser("stats", help="Print basic graph statistics")
    p_stats.add_argument("gfa", help="Input *.gfa* file or - for stdin")
    g2 = p_stats.add_mutually_exclusive_group()
    g2.add_argument("--directed", dest="directed", action="store_true", default=True)
    g2.add_argument("--undirected", dest="directed", action="store_false")
    p_stats.add_argument("--strip-orientation", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "convert":
        if not args.graph and not args.matrix:
            parser.error("convert requires --graph or --matrix")
        build_mat = bool(args.matrix)
        build_g = args.graph
        result = parse_gfa(
            args.gfa,
            build_graph=build_g,
            build_matrix=build_mat,
            directed=args.directed,
            weight_tag=args.weight_tag,
            store_seq=args.store_seq,
            strip_orientation=args.strip_orientation,
            verbose=args.verbose,
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
