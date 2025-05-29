import os
from pathlib import Path

import pytest

from gfa2network import parse_gfa

LARGE_GRAPH_PATH = Path(os.environ.get("LARGE_GRAPH", ""))


@pytest.mark.skipif(not LARGE_GRAPH_PATH.exists(), reason="large graph not available")
def test_parse_large_graph_verbose(capsys):
    parse_gfa(LARGE_GRAPH_PATH, build_graph=False, build_matrix=False, verbose=True)
    captured = capsys.readouterr()
    assert "[parse_gfa] done" in captured.out
    assert "lines]" in captured.err


@pytest.mark.skipif(not LARGE_GRAPH_PATH.exists(), reason="large graph not available")
def test_igraph_backend_verbose(capsys):
    parse_gfa(
        LARGE_GRAPH_PATH,
        build_graph=False,
        build_matrix=False,
        backend="igraph",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "[parse_gfa_igraph] done" in captured.out
    assert "lines]" in captured.err
