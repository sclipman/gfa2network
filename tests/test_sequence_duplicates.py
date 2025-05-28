from pathlib import Path
from gfa2network import parse_gfa
from gfa2network.analysis import sequence_distance

SAMPLE_GFA = b"""S\ts1\tAAAA\nS\ts2\tTTTT\nS\ts3\tAAAA\nL\ts1\t+\ts2\t+\t0M\nL\ts3\t+\ts1\t+\t0M\n"""

def test_sequence_distance_duplicates(tmp_path: Path):
    gfa = tmp_path / "dup.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, store_seq=True)
    dist = sequence_distance(G, b"AAAA", b"TTTT")
    assert dist == 1
