from pathlib import Path
import networkx as nx
from gfa2network import parse_gfa

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_bidirected_edges_and_distance(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, bidirected=True)
    assert not G.is_directed()
    assert G.has_edge("s1:+", "s2:-")
    assert G.has_edge("s2:+", "s1:-")
    d1 = nx.shortest_path_length(G, "s1:+", "s2:-")
    d2 = nx.shortest_path_length(G, "s2:+", "s1:-")
    assert d1 == d2 == 1
