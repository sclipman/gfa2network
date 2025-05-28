from pathlib import Path
from gfa2network import parse_gfa

SAMPLE_GFA = b"""S\ts1\t4\tRC:i:5\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\tRC:i:3\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "test.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_store_tags_on(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_tags=True)
    assert G.nodes["s1"]["tags"] == {"RC": 5}
    assert G.nodes["s1"]["length"] == 4
    assert G.edges[("s1", "s2")]["tags"] == {"RC": 3}


def test_store_tags_off(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_tags=False)
    assert "tags" not in G.nodes["s1"]
    assert "length" not in G.nodes["s1"]
    assert "tags" not in G.edges[("s1", "s2")]
