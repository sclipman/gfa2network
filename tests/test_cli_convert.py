import subprocess
import sys
from pathlib import Path
import networkx as nx
import pytest

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_convert_save_networkx(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "g.pkl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "convert",
            str(gfa),
            "--graph",
            "-o",
            str(out),
        ],
        check=True,
    )
    if hasattr(nx, "read_gpickle"):
        G = nx.read_gpickle(out)
    else:
        import pickle

        with open(out, "rb") as fh:
            G = pickle.load(fh)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_convert_save_igraph(tmp_path: Path):
    ig = pytest.importorskip("igraph")
    gfa = write_gfa(tmp_path)
    out = tmp_path / "g.pkl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "convert",
            str(gfa),
            "--graph",
            "--backend",
            "igraph",
            "-o",
            str(out),
        ],
        check=True,
    )
    G = ig.Graph.Read_Pickle(str(out))
    assert G.vcount() == 2
    assert G.ecount() == 1
