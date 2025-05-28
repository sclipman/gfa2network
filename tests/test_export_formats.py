import subprocess
import sys
from pathlib import Path
import networkx as nx

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""

def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_export_graphml(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "g.graphml"
    subprocess.run([
        sys.executable,
        "-m",
        "gfa2network",
        "export",
        str(gfa),
        "--format",
        "graphml",
        "--output",
        str(out),
    ], check=True)
    G = nx.read_graphml(out)
    assert sorted(G.nodes()) == ["s1", "s2"]


def test_export_gexf(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "g.gexf"
    subprocess.run([
        sys.executable,
        "-m",
        "gfa2network",
        "export",
        str(gfa),
        "--format",
        "gexf",
        "--output",
        str(out),
    ], check=True)
    G = nx.read_gexf(out)
    assert sorted(G.nodes()) == ["s1", "s2"]


def test_export_raw_bytes(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "g.graphml"
    subprocess.run([
        sys.executable,
        "-m",
        "gfa2network",
        "--raw-bytes-id",
        "export",
        str(gfa),
        "--format",
        "graphml",
        "--output",
        str(out),
    ], check=True)
    G = nx.read_graphml(out)
    assert sorted(G.nodes()) == ["b's1'", "b's2'"]

