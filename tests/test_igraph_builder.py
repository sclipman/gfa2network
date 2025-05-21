import pytest
from pathlib import Path

ig = pytest.importorskip("igraph")

from gfa2network.igraph_builder import parse_gfa_igraph

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\tRC:i:3\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_directed_flag(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa_igraph(gfa, build_graph=True, build_matrix=False)
    assert G.is_directed()
    G2 = parse_gfa_igraph(gfa, build_graph=True, build_matrix=False, directed=False)
    assert not G2.is_directed()


def test_vertex_edge_attributes(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa_igraph(gfa, build_graph=True, build_matrix=False, weight_tag="RC")
    assert G.vcount() == 2
    assert G.ecount() == 1
    v = G.vs.find(name="s1")
    assert v["length"] == 4
    e = G.es[0]
    assert e["weight"] == 3.0
    assert e["orientation_from"] == "+"
    assert e["orientation_to"] == "-"
    assert e["tags"] == {"RC": 3}
