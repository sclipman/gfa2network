import networkx as nx
from pathlib import Path
from gfa2network import parse_gfa

SAMPLE_GFA = b"""S\ts1\t6\nS\ts2\t10\nE\t*\ts1+\t0\t6\ts2+\t0\t6\t6M\n"""
ORIENT_GFA = b"""S\ts1\t6\nS\ts2\t10\nE\t*\ts1\t+\ts2\t+\n"""
LINK_GFA = b"""S\ts1\t6\nS\ts2\t10\nL\ts1\t+\ts2\t-\t0M\nE\t*\ts1+\t0\t3\ts2+\t0\t3\t3M\n"""

def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "a.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_no_split(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_split_on_alignment(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, split_on_alignment=True)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert nx.shortest_path_length(G, "s1:0-6", "s2:6-10") == 2


def test_split_orientation_only_edges(tmp_path: Path):
    gfa = tmp_path / "b.gfa"
    gfa.write_bytes(ORIENT_GFA)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, split_on_alignment=True)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_split_with_link_records(tmp_path: Path):
    gfa = tmp_path / "c.gfa"
    gfa.write_bytes(LINK_GFA)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, split_on_alignment=True)
    assert G.number_of_nodes() == 4
    # verify that the link edge was preserved after splitting
    assert ("s1:0-3", "s2:0-3") in G.edges
    attrs = G.edges[("s1:0-3", "s2:0-3")]
    assert attrs["orientation_from"] == "+"
    assert attrs["orientation_to"] == "-"
