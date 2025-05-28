import time
from pathlib import Path

import numpy as np
import networkx as nx

from gfa2network import parse_gfa
from gfa2network.analysis import genome_distance, genome_distance_matrix, load_paths


def make_gfa(num_paths: int) -> bytes:
    lines = []
    for i in range(num_paths * 2):
        lines.append(f"S\ts{i}\t*")
    for i in range(num_paths * 2 - 1):
        lines.append(f"L\ts{i}\t+\ts{i+1}\t+\t0M")
    for i in range(num_paths):
        lines.append(f"P\tp{i}\ts{i}+,s{i + num_paths}+\t*")
    return ("\n".join(lines) + "\n").encode()


def legacy_matrix(path: str) -> np.ndarray:
    paths = load_paths(path)
    names = list(paths)
    G = parse_gfa(path, build_graph=True, build_matrix=False)
    n = len(names)
    M = np.zeros((n, n), dtype=float)
    for i, name_a in enumerate(names):
        nodes_a = paths[name_a]
        for j in range(i, n):
            if i == j:
                dist = 0.0
            else:
                try:
                    dist = genome_distance(G, nodes_a, paths[names[j]], method="min")
                except nx.NetworkXNoPath:
                    dist = float("inf")
            M[i, j] = dist
            M[j, i] = dist
    return M


def test_distance_matrix_benchmark(tmp_path: Path):
    gfa = tmp_path / "big.gfa"
    gfa.write_bytes(make_gfa(60))

    t0 = time.perf_counter()
    ref = legacy_matrix(str(gfa))
    t1 = time.perf_counter()
    fast = genome_distance_matrix(str(gfa))
    t2 = time.perf_counter()

    if hasattr(fast, "values"):
        fast_arr = fast.values
    else:
        fast_arr = fast

    assert np.allclose(ref, fast_arr)
    assert (t1 - t0) / (t2 - t1) >= 4
