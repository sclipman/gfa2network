from pathlib import Path
import subprocess
import sys
import warnings
import gzip

from gfa2network import parse_gfa
from gfa2network.parser import GFAParser, PathRecord
from gfa2network.analysis import compute_stats

SAMPLE_GFA = b"""S\ts1\tACGT\nS\ts2\tTTTT\nL\ts1\t+\ts2\t-\t0M\nP\tp1\ts1+,s2-\t*\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "test.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_orientation_attributes(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False)
    assert G.edges[("s1", "s2")]["orientation_from"] == "+"
    assert G.edges[("s1", "s2")]["orientation_to"] == "-"


def test_strip_orientation(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, strip_orientation=True)
    assert "s1" in G.nodes
    assert "orientation_from" not in G.edges[("s1", "s2")]


def test_path_record(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    parser = GFAParser(gfa)
    paths = [rec for rec in parser if isinstance(rec, PathRecord)]
    assert len(paths) == 1
    assert paths[0].segments == [(b"s1", "+"), (b"s2", "-")]


def test_stats(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    stats = compute_stats(str(gfa))
    assert stats["nodes"] == 2
    assert stats["edges"] == 1
    assert stats["paths"] == 1
    assert stats["components"] == 1
    assert stats["max_degree"] == 1
    assert abs(stats["density"] - 0.5) < 1e-6


def test_stats_cli(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "gfa2network", "stats", str(gfa)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "nodes" in result.stdout


def test_tag_parsing(tmp_path: Path):
    gfa = tmp_path / "t.gfa"
    gfa.write_text("S\ts1\t4\tRC:i:5\nS\ts2\t4\t\nL\ts1\t+\ts2\t+\t0M\tRC:i:2\n")
    parser = GFAParser(gfa)
    recs = list(parser)
    seg = recs[0]
    assert seg.tags == {"RC": 5}
    link = recs[-1]
    assert link.tags == {"RC": 2}


def test_bidirected(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, bidirected=True)
    assert ("s1:+", "s2:-") in G.edges


def test_segment_length(tmp_path: Path):
    gfa = tmp_path / "len.gfa"
    gfa.write_text("S\ts1\t10\n")
    recs = list(GFAParser(gfa))
    assert recs[0].length == 10
    assert recs[0].sequence is None


def test_unknown_record_warning(tmp_path: Path):
    gfa = tmp_path / "bad.gfa"
    gfa.write_text("X\tfoo\nX\tbar\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(GFAParser(gfa))
    assert len(w) == 1


def test_gzip_input(tmp_path: Path):
    gz = tmp_path / "test.gfa.gz"
    gz.write_bytes(gzip.compress(SAMPLE_GFA))
    recs = list(GFAParser(gz))
    assert len([r for r in recs if isinstance(r, PathRecord)]) == 1
    G = parse_gfa(gz, build_graph=True, build_matrix=False)
    assert G.number_of_nodes() == 2


def test_raw_bytes_flag(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(gfa, build_graph=True, build_matrix=False, raw_bytes_id=True)
    assert b"s1" in G.nodes
