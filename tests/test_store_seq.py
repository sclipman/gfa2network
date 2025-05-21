from pathlib import Path
from gfa2network import parse_gfa

SAMPLE_GFA = b"""S\ts1\tACGT\nS\ts2\tTTTT\nL\ts1\t+\ts2\t+\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "test.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_store_seq_on(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_seq=True)
    assert G.nodes[b"s1"]["sequence"] == b"ACGT"


def test_store_seq_off(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_seq=False)
    assert "sequence" not in G.nodes[b"s1"]
