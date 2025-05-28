import pytest
from pathlib import Path
import pytest
from gfa2network import parse_gfa
from gfa2network.analysis import genome_distance

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""

def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "warn.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_warning_directed_bidirected(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        bidirected=True,
        keep_directed_bidir=True,
    )
    with pytest.warns(RuntimeWarning):
        dist = genome_distance(G, ["s1:+"], ["s2:-"])
    assert dist == 1


def test_no_warning_after_to_undirected(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    G = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        bidirected=True,
        keep_directed_bidir=True,
    )
    G = G.to_undirected()
    dist = genome_distance(G, ["s1:+"], ["s2:-"])
    assert dist == 1
