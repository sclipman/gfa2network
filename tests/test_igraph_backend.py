import pytest
from pathlib import Path

ig = pytest.importorskip("igraph")

from gfa2network import parse_gfa

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_parse_gfa_igraph_backend(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, backend="igraph")
    assert G.vcount() == 2
    assert G.ecount() == 1
    assert sorted(v["name"] for v in G.vs) == ["s1", "s2"]
