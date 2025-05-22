from pathlib import Path
import subprocess
import sys

from gfa2network import parse_gfa
from gfa2network.analysis import (
    sequence_distance,
    genome_distance,
    genome_distance_matrix,
    load_paths,
)

SAMPLE_SEQ_GFA = b"""S\ts1\tACGT\nS\ts2\tTTTT\nL\ts1\t+\ts2\t+\t0M\n"""

SAMPLE_PATH_GFA = b"""S\ts1\t*\nS\ts2\t*\nS\ts3\t*\nL\ts1\t+\ts2\t+\t0M\nL\ts2\t+\ts3\t+\t0M\nP\tp1\ts1+,s2+\t*\nP\tp2\ts3+,s2+\t*\n"""

def write_gfa(tmp_path: Path, content: bytes, name: str) -> Path:
    gfa = tmp_path / name
    gfa.write_bytes(content)
    return gfa

def test_sequence_distance(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_SEQ_GFA, "seq.gfa")
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_seq=True)
    dist = sequence_distance(G, b"ACGT", b"TTTT")
    assert dist == 1

def test_genome_distance(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_PATH_GFA, "paths.gfa")
    paths = load_paths(str(gfa))
    G = parse_gfa(gfa, build_graph=True, build_matrix=False)
    d = genome_distance(G, paths[b"p1"], paths[b"p2"], method="min")
    assert d == 0


def test_genome_distance_matrix(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_PATH_GFA, "matrix.gfa")
    M = genome_distance_matrix(str(gfa))
    import numpy as np

    arr = M.values if hasattr(M, "values") else M
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[0, 0], [0, 0]])

def test_cli_distance_seq(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_SEQ_GFA, "cli_seq.gfa")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "distance",
            str(gfa),
            "--seq",
            "ACGT",
            "TTTT",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "1"

def test_cli_distance_path(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_PATH_GFA, "cli_path.gfa")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "distance",
            str(gfa),
            "--path",
            "p1",
            "p2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0"


def test_cli_distance_matrix(tmp_path: Path):
    gfa = write_gfa(tmp_path, SAMPLE_PATH_GFA, "cli_matrix.gfa")
    out = tmp_path / "dist.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "distance-matrix",
            str(gfa),
            "-o",
            str(out),
        ],
        check=True,
    )
    import numpy as np

    arr = np.loadtxt(out, delimiter=",")
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[0, 0], [0, 0]])
