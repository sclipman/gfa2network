from pathlib import Path
from gfa2network.analysis import compute_stats

SAMPLE_GFA = b"""S\ts1\t*\nS\ts2\t*\nO\tw1\ts1+,s2+\n"""

def test_walk_record_count(tmp_path: Path):
    gfa = tmp_path / "walk.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    stats = compute_stats(str(gfa))
    assert stats["paths"] == 1
