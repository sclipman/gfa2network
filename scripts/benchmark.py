#!/usr/bin/env python3
"""Fresh-process timings, peak RSS and output fingerprints; no timing assertions."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def worker(args):
    sys.path.insert(0, str(args.checkout.resolve()))
    import numpy as np
    from gfa2network import parse_gfa, __version__

    start = time.perf_counter()
    result = parse_gfa(
        args.input,
        backend=args.backend,
        build_graph=args.mode == "graph",
        build_matrix=args.mode == "matrix",
        asymmetric=True,
        return_node_list=args.mode == "matrix",
    )
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    if args.mode == "matrix":
        matrix, names = result
        matrix = matrix.tocsr()
        matrix.sort_indices()
        digest.update(json.dumps(names, separators=(",", ":")).encode())
        for data in (
            matrix.indptr.astype(np.int64),
            matrix.indices.astype(np.int64),
            matrix.data.astype(np.float64),
        ):
            digest.update(data.tobytes())
        nodes, edges = len(names), matrix.nnz
    else:
        if args.backend == "networkx":
            names = list(result)
            records = (
                (u, v, attrs.get("weight", 1.0))
                for u, v, attrs in result.edges(data=True)
            )
            edges = result.number_of_edges()
        else:
            names = result.vs["name"]
            records = (
                (names[e.source], names[e.target], e.attributes().get("weight", 1.0))
                for e in result.es
            )
            edges = result.ecount()
        adjacency = {}
        for u, v, value in records:
            adjacency[(u, v)] = adjacency.get((u, v), 0.0) + (
                1.0 if value is None else value
            )
        digest.update(json.dumps(sorted(names), separators=(",", ":")).encode())
        digest.update(
            json.dumps(
                sorted((u, v, float(weight)) for (u, v), weight in adjacency.items()),
                separators=(",", ":"),
            ).encode()
        )
        nodes = len(names)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            dict(
                version=__version__,
                seconds=elapsed,
                peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
                nodes=nodes,
                edges=edges,
                adjacency_sha256=digest.hexdigest(),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.nodes < 2 or args.repeats < 1:
        parser.error("nodes must be >=2 and repeats >=1")
    candidate = args.checkout.resolve()
    roots = [("candidate", candidate)]
    if args.baseline:
        roots.insert(0, ("baseline", args.baseline.resolve()))
    report = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={},
        settings=dict(
            nodes=args.nodes,
            repeats=args.repeats,
            directed=True,
            asymmetric=True,
            sequences=False,
            tags=False,
        ),
        scope="Parse/build wall time; peak process RSS includes imports and fingerprint verification. No million-node or biological-validation claim.",
        cases=[],
    )
    for package in ("numpy", "scipy", "networkx", "igraph"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        for topology in ("chain", "star", "bundled-drb1"):
            source = directory / (topology + ".gfa")
            if topology == "bundled-drb1":
                source.write_bytes(
                    (candidate / "tests/data/DRB1-3123_unsorted.gfa").read_bytes()
                )
            else:
                with source.open("w") as file:
                    for i in range(args.nodes):
                        file.write(f"S\t{i}\t*\n")
                    for i in range(1, args.nodes):
                        parent = i - 1 if topology == "chain" else 0
                        file.write(f"L\t{parent}\t+\t{i}\t+\t0M\n")
            input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            for backend in ("networkx", "igraph"):
                if backend == "igraph" and "igraph" not in report["dependencies"]:
                    continue
                for mode in ("graph", "matrix"):
                    case = dict(
                        input=topology,
                        input_sha256=input_hash,
                        backend=backend,
                        mode=mode,
                        runs={},
                    )
                    for label, checkout in roots:
                        results = []
                        for _ in range(args.repeats):
                            command = [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                "--checkout",
                                str(checkout),
                                "--input",
                                str(source),
                                "--backend",
                                backend,
                                "--mode",
                                mode,
                            ]
                            process = subprocess.run(
                                command,
                                cwd=directory,
                                text=True,
                                capture_output=True,
                                timeout=180,
                            )
                            if process.returncode:
                                results.append(
                                    {
                                        "error": process.stderr.splitlines()[-1]
                                        if process.stderr
                                        else str(process.returncode)
                                    }
                                )
                            else:
                                results.append(json.loads(process.stdout))
                        case["runs"][label] = results
                    if args.baseline:
                        runs = case["runs"]
                        a, b = runs["baseline"][0], runs["candidate"][0]
                        case["equivalent_adjacency"] = (
                            "error" not in a
                            and "error" not in b
                            and a["adjacency_sha256"] == b["adjacency_sha256"]
                        )
                    report["cases"].append(case)
                    print(
                        f"{topology} {backend} {mode}: complete",
                        file=sys.stderr,
                        flush=True,
                    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
