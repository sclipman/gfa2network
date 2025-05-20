#!/usr/bin/env python3
"""
GFA2Network.py
============================
Load a (potentially huge) pangenome variation graph in **GFA-1** or **GFA-2**
format and convert it into

  • a `networkx.Graph` / `DiGraph`   (optional `--graph`)
  • a SciPy sparse adjacency matrix (optional `--matrix PATH`)

The reader is one-pass, allocation-aware and stays within ≈ (#edges × 16 bytes)
RAM, so even a 2 M-node / 3 M-edge graph fits comfortably on a 16 GB laptop.

----------
Quick usage
------------
Build *both* outputs (default: undirected graph, CSR matrix):

    python GFA2Network.py input.gfa \
        --graph \
        --matrix adj.npz

Matrix-only run (lowest RAM):

    python GFA2Network.py input.gfa \
        --matrix adj.npz \
        --matrix-format coo

Directed graph only, verbose progress:

    python GFA2Network.py input.gfa \
        --directed --graph --verbose

-------------
Positional argument
-------------------
input.gfa            Path to a *.gfa* file (GFA-1 or GFA-2).

----------------
Optional arguments
------------------
--graph              Build a NetworkX object.  Omit to skip it.
--matrix PATH        Write an adjacency matrix to *PATH*.
                     Extensions: *.npz* (sparse), *.csv* or *.npy* (dense).
--matrix-format FMT  Sparse format for *.npz* (**csr**, csc, coo, dok). Default *csr*.
--directed           Treat the graph as directed.
--undirected         Treat the graph as undirected (**default**).
--weight-tag TAG     Use numeric value of GFA tag *TAG* (e.g. *RC*) as edge
                     weight; otherwise every edge weight = 1.
--verbose            Emit progress to *stderr* (every 500 k lines + tqdm bars).

---------------
Implementation notes
--------------------
* Lines are read as **bytes**, split on `b'\t'`; no giant Python strings.
* Only **S** (segment) and **L** (link) records are used; paths/overlaps ignored.
* Supports both GFA-1 and GFA-2 *L* syntax.
* Orientation symbols ‘+’/‘-’ are stripped from node IDs; store them as an
  attribute or use a bidirected model if needed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Iterable, Tuple

import networkx as nx
import numpy as np

# ── optional SciPy ───────────────────────────────────────────────────────────
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:                # missing wheel / ABI mismatch
    sp = None                    # type: ignore[assignment]
    _HAS_SCIPY = False

# ── optional tqdm (pretty progress bars) ─────────────────────────────────────
try:
    from tqdm.auto import tqdm                         # noqa: WPS433
    _HAS_TQDM = True
except Exception:
    tqdm = None                                        # type: ignore
    _HAS_TQDM = False

# ───────────────────────────── helpers ───────────────────────────────────────
def _parse_link(fields: list[bytes]) -> Tuple[bytes, bytes]:
    """
    Return (u, v) segment IDs with orientation removed.
    Supports both GFA-1 and GFA-2 'L' records.
    """
    if len(fields) < 5:
        raise ValueError("Malformed L record – need ≥ 5 tab-fields")

    # GFA-1: L from + to - ov
    # GFA-2: L from to type dist …
    if fields[2] in (b"+", b"-"):           # GFA-1
        u, v = fields[1], fields[3]
    else:                                   # GFA-2
        u, v = fields[1], fields[2]

    return u.rstrip(b"+-"), v.rstrip(b"+-")

# ─────────────────────────── core parser ─────────────────────────────────────
def parse_gfa(
    path: str | pathlib.Path,
    *,
    build_graph: bool,
    build_matrix: bool,
    directed: bool = True,
    weight_tag: str | None = None,
    verbose: bool = False,
):
    """
    Stream-parse *path* and return the requested artefacts.

    Returns
    -------
    Graph only          → nx.Graph / nx.DiGraph
    Matrix only         → scipy.sparse.coo_matrix
    Both requested      → (Graph, coo_matrix)
    """
    if build_matrix and not _HAS_SCIPY:
        raise RuntimeError("Matrix output requires SciPy")

    graph_cls = nx.DiGraph if directed else nx.Graph
    G = graph_cls() if build_graph else None

    # compact integer indexing for the matrix
    node2idx: dict[bytes, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    tag_prefix = (weight_tag + ":").encode() if weight_tag else None

    path = pathlib.Path(path)
    with path.open("rb", buffering=1 << 20) as fh:       # 1 MiB buffer
        for lineno, line in enumerate(fh, 1):
            if not line or line[0] not in (ord("S"), ord("L")):
                continue
            fields = line.rstrip(b"\n").split(b"\t")

            if fields[0] == b"S":                        # segment
                seg = fields[1]
                if build_graph:
                    G.add_node(seg)                      # type: ignore[arg-type]
                if build_matrix and seg not in node2idx:
                    node2idx[seg] = len(node2idx)

            else:                                        # link
                u, v = _parse_link(fields)

                w: float | None = None
                if tag_prefix:
                    for f in fields[5:]:
                        if f.startswith(tag_prefix):
                            try:
                                w = float(f.split(b":", 2)[-1])
                            except ValueError:
                                pass
                            break

                if build_matrix:
                    for n in (u, v):
                        if n not in node2idx:
                            node2idx[n] = len(node2idx)
                    rows.append(node2idx[u])
                    cols.append(node2idx[v])
                    data.append(1.0 if w is None else w)

                if build_graph:
                    if w is None:
                        G.add_edge(u, v)                 # type: ignore[arg-type]
                    else:
                        G.add_edge(u, v, weight=w)       # type: ignore[arg-type]

            if verbose and lineno % 500_000 == 0:
                print(f"\r[{lineno:,} lines]", end="", file=sys.stderr)

    if verbose:
        print("\r[parse_gfa] done", file=sys.stderr)

    out_graph = G
    out_mat = None
    if build_matrix:
        n = len(node2idx)
        out_mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float)

        # build graph from sparse matrix if not already built
        if build_graph and G is None:
            out_graph = nx.from_scipy_sparse_array(
                out_mat, create_using=graph_cls, edge_attribute="weight"
            )
            mapping = {i: seg for seg, i in node2idx.items()}
            out_graph = nx.relabel_nodes(out_graph, mapping, copy=False)

    if build_graph and build_matrix:
        return out_graph, out_mat
    if build_graph:
        return out_graph                      # type: ignore[return-value]
    return out_mat                            # type: ignore[return-value]

# ───────────────────── sparse-format helper ──────────────────────────────────
def convert_format(A, fmt: str, *, verbose: bool = False):
    """Convert COO → *fmt* with optional progress indication."""
    fmt = fmt.lower()
    if fmt not in {"csr", "csc", "coo", "dok"}:
        raise ValueError("matrix-format must be csr|csc|coo|dok")
    if fmt == "coo":
        return A

    if verbose:
        if _HAS_TQDM:
            bar = tqdm(total=1, bar_format="{desc} …{elapsed}", desc=f"[convert→{fmt}]")
        else:
            start = time.perf_counter()
            print(f"[convert] -> {fmt} …", end="", file=sys.stderr, flush=True)

    out = A.asformat(fmt)

    if verbose:
        if _HAS_TQDM:
            bar.update(1); bar.close()                   # type: ignore[name-defined]
        else:
            dt = time.perf_counter() - start
            print(f" done in {dt:,.1f}s", file=sys.stderr)
    return out

def _save_matrix(A, dest: pathlib.Path, *, verbose: bool = False):
    """Write *A* to *dest* with progress bar and dense-size guard."""
    _MAX_DENSE_BYTES = 5_000_000_000  # 5 GB

    # ---- guard against accidental dense dump --------------------------------
    if dest.suffix in {".csv", ".npy"}:
        nnz = A.nnz if sp.issparse(A) else A.size
        if nnz * 8 > _MAX_DENSE_BYTES:
            raise MemoryError(
                f"dense export would allocate {nnz*8/1e9:.1f} GB; "
                "choose a sparse .npz or write an edge list instead"
            )

    # ---- progress header ----------------------------------------------------
    if verbose:
        msg = f"[save] {dest.suffix[1:]} → {dest}"
        if _HAS_TQDM:
            bar = tqdm(total=1, bar_format="{desc} …{elapsed}", desc=msg)
        else:
            start = time.perf_counter()
            print(msg, "...", end="", file=sys.stderr, flush=True)

    # ---- actual write -------------------------------------------------------
    if dest.suffix == ".npz":
        sp.save_npz(dest, A)                            # type: ignore[arg-type]
    elif dest.suffix == ".npy":
        np.save(dest, A.toarray() if sp.issparse(A) else A)
    elif dest.suffix == ".csv":
        np.savetxt(
            dest,
            A.toarray() if sp.issparse(A) else A,
            delimiter=",",
            fmt="%.6g",
        )
    else:
        raise ValueError("matrix path must end with .npz, .npy, or .csv")

    # ---- progress footer ----------------------------------------------------
    if verbose:
        if _HAS_TQDM:
            bar.update(1); bar.close()                  # type: ignore[name-defined]
        else:
            dt = time.perf_counter() - start            # type: ignore[name-defined]
            print(f" done in {dt:,.1f}s", file=sys.stderr)

# ─────────────────────────────── CLI ─────────────────────────────────────────
def main(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(
        description="GFA-1/2 → NetworkX and/or SciPy sparse adjacency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("gfa", help="Input *.gfa* file")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--directed", action="store_true", default=False,
                   help="Treat graph as directed")
    g.add_argument("--undirected", action="store_true", default=True,
                   help="Treat graph as undirected (default)")
    p.add_argument("--graph", action="store_true",
                   help="Build a NetworkX object")
    p.add_argument("--matrix", metavar="PATH",
                   help="Write adjacency matrix to PATH (.npz|.npy|.csv)")
    p.add_argument("--matrix-format", default="csr",
                   help="Sparse format for .npz (csr|csc|coo|dok)")
    p.add_argument("--weight-tag",
                   help="Optional GFA tag (e.g. RC) to use as edge weight")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    if not args.graph and not args.matrix:
        p.error("At least one of --graph or --matrix is required")

    build_mat = bool(args.matrix)
    build_g   = args.graph

    result = parse_gfa(
        args.gfa,
        build_graph=build_g,
        build_matrix=build_mat,
        directed=args.directed,
        weight_tag=args.weight_tag,
        verbose=args.verbose,
    )

    if build_g and build_mat:
        G, A = result                                   # type: ignore[misc]
    elif build_g:
        G = result                                      # type: ignore[assignment]
    else:
        A = result                                      # type: ignore[assignment]

    if build_mat:
        A = convert_format(A, args.matrix_format, verbose=args.verbose)  # type: ignore[arg-type]
        _save_matrix(A, pathlib.Path(args.matrix), verbose=args.verbose)
        if args.verbose:
            print(f"[output] adjacency → {args.matrix}", file=sys.stderr)

    # expose objects in interactive sessions
    if build_g:
        globals().update({"G": G})
    if build_mat:
        globals().update({"A": A})

if __name__ == "__main__":
    main()