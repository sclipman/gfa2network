"""Small, specification-based fixtures for the October 2026 review."""

from io import BytesIO
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from gfa2network import parse_gfa
from gfa2network.cli import main
from gfa2network.parser import GFAParser
from gfa2network.utils import save_matrix
from gfa2network.analysis import genome_distance_matrix


def test_standard_gfa2_edge_coordinates_and_identifier_suffixes():
    rec = next(iter(GFAParser(BytesIO(b"E\te1\ta++\tb--\t1\t4$\t0\t3$\t3M\tRC:i:7\n"))))
    assert (rec.from_segment, rec.to_segment) == (b"a+", b"b-")
    assert (rec.orientation_from, rec.orientation_to) == ("+", "-")
    assert (rec.from_start, rec.from_end, rec.to_start, rec.to_end) == (1, 4, 0, 3)
    assert rec.cigar == b"3M"
    assert rec.tags == {"RC": 7}


def test_standard_gfa2_graph_and_ordered_group(tmp_path):
    gfa = tmp_path / "standard.gfa"
    gfa.write_text(
        "H\tVN:Z:2.0\nS\ta\t4\tAAAA\nS\tb\t4\tTTTT\n"
        "E\t*\ta+\tb-\t0\t4$\t0\t4$\t4M\nO\tp\ta+ b-\n"
    )
    graph = parse_gfa(gfa, build_graph=True, build_matrix=False)
    assert set(graph.nodes) == {"a", "b"}
    assert list(graph.edges) == [("a", "b")]
    assert list(GFAParser(gfa))[-1].segments == [(b"a", "+"), (b"b", "-")]


def test_gfa2_ordered_group_preserves_commas_in_ids():
    rec = next(iter(GFAParser(BytesIO(b"O\tp\ta,b+ c-\n"))))
    assert rec.segments == [(b"a,b", "+"), (b"c", "-")]


@pytest.mark.parametrize(
    "data, length, sequence",
    [
        (b"S\ta\t*\tLN:i:4\r\n", 4, None),
        (b"S\ta\tACGT\r\n", 4, b"ACGT"),
        (b"S\ta\t4\t*\r\n", 4, None),
    ],
)
def test_segment_lengths_and_missing_sequences(data, length, sequence):
    rec = next(iter(GFAParser(BytesIO(data))))
    assert rec.length == length
    assert rec.sequence == sequence


def test_containment_fields_and_numeric_array_tags():
    rec = next(iter(GFAParser(BytesIO(b"C\ta\t+\tb\t-\t2\t3M\tRC:i:5\tBA:B:i,1,2\n"))))
    assert rec.from_start == 2
    assert rec.cigar == b"3M"
    assert rec.tags == {"RC": 5, "BA": [1, 2]}


@pytest.mark.parametrize("suffix", [".npy", ".csv"])
def test_dense_guard_uses_full_shape_before_allocation(tmp_path, monkeypatch, suffix):
    matrix = sp.csr_matrix((10000, 10000), dtype=np.float64)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("dense allocation was attempted before checking the limit")

    monkeypatch.setattr(type(matrix), "toarray", unexpected_allocation)
    out = tmp_path / ("large" + suffix)
    with pytest.raises(MemoryError, match="0.8 GB"):
        save_matrix(matrix, out, max_dense_gb=0.001)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["dense", "dok"])
def test_npz_roundtrip_for_distance_and_dok_matrices(tmp_path, kind):
    expected = np.array([[0.0, 2.0], [2.0, 0.0]])
    matrix = expected if kind == "dense" else sp.dok_matrix(expected)
    out = tmp_path / "matrix.npz"
    save_matrix(matrix, out)
    np.testing.assert_array_equal(sp.load_npz(out).toarray(), expected)


@pytest.fixture
def tiny_gfa(tmp_path):
    gfa = tmp_path / "tiny.gfa"
    gfa.write_text("S\ta\tAAAA\nS\tb\tTTTT\nL\ta\t+\tb\t-\t0M\n")
    return gfa


def test_cli_limit_after_subcommand_is_enforced(tmp_path, tiny_gfa):
    with pytest.raises(SystemExit, match="dense export"):
        main(
            [
                "convert",
                str(tiny_gfa),
                "--matrix",
                str(tmp_path / "x.npy"),
                "--max-dense-gb",
                "0",
            ]
        )


def test_cli_stat_alias_prints_statistics(tiny_gfa, capsys):
    main(["stat", str(tiny_gfa)])
    assert "nodes" in capsys.readouterr().out


@pytest.mark.parametrize("fmt", ["graphml", "gexf"])
def test_xml_exports_use_stdout(tiny_gfa, capsys, monkeypatch, tmp_path, fmt):
    monkeypatch.chdir(tmp_path)
    main(["export", str(tiny_gfa), "--format", fmt])
    text = capsys.readouterr().out
    graph = (
        nx.parse_graphml(text)
        if fmt == "graphml"
        else nx.read_gexf(BytesIO(text.encode()))
    )
    assert set(graph) == {"a", "b"}
    assert not Path("-").exists()


def test_igraph_bidirected_does_not_invent_vertices(tiny_gfa):
    pytest.importorskip("igraph")
    graph = parse_gfa(
        tiny_gfa,
        backend="igraph",
        build_graph=True,
        build_matrix=False,
        bidirected=True,
    )
    assert set(graph.vs["name"]) == {"a:+", "a:-", "b:+", "b:-"}
    assert {
        frozenset((graph.vs[e.source]["name"], graph.vs[e.target]["name"]))
        for e in graph.es
    } == {frozenset(("a:+", "b:-")), frozenset(("b:+", "a:-"))}


def test_igraph_segment_metadata_survives_unsorted_records(tmp_path):
    pytest.importorskip("igraph")
    gfa = tmp_path / "unsorted.gfa"
    gfa.write_text("L\ta\t+\tb\t+\t0M\nS\ta\tACGT\tRC:i:5\nS\tb\tTTTT\n")
    graph = parse_gfa(
        gfa,
        backend="igraph",
        build_graph=True,
        build_matrix=False,
        store_seq=True,
        store_tags=True,
    )
    assert graph.vs.find(name="a")["sequence"] == b"ACGT"
    assert graph.vs.find(name="a")["tags"] == {"RC": 5}


@pytest.mark.parametrize("asymmetric", [False, True])
def test_igraph_matrix_matches_simple_networkx_graph(tiny_gfa, asymmetric):
    pytest.importorskip("igraph")
    opts = dict(
        build_graph=False,
        build_matrix=True,
        return_node_list=True,
        dtype="float32",
        asymmetric=asymmetric,
    )
    expected, expected_nodes = parse_gfa(tiny_gfa, **opts)
    actual, actual_nodes = parse_gfa(tiny_gfa, backend="igraph", **opts)
    assert actual_nodes == expected_nodes
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())


def test_igraph_matrix_defaults_missing_edge_weights(tmp_path):
    pytest.importorskip("igraph")
    gfa = tmp_path / "weights.gfa"
    gfa.write_text(
        "S\ta\t*\nS\tb\t*\nS\tc\t*\nL\ta\t+\tb\t+\t0M\tRC:i:3\nL\tb\t+\tc\t+\t0M\n"
    )
    matrix, nodes = parse_gfa(
        gfa,
        backend="igraph",
        build_graph=False,
        build_matrix=True,
        weight_tag="RC",
        asymmetric=True,
        return_node_list=True,
    )
    assert matrix[nodes.index("a"), nodes.index("b")] == 3
    assert matrix[nodes.index("b"), nodes.index("c")] == 1


def test_igraph_rejects_unsupported_alignment_splitting(tiny_gfa):
    pytest.importorskip("igraph")
    with pytest.raises(ValueError, match="split_on_alignment"):
        parse_gfa(
            tiny_gfa,
            backend="igraph",
            build_graph=True,
            build_matrix=False,
            split_on_alignment=True,
        )


def test_unknown_backend_is_rejected(tiny_gfa):
    with pytest.raises(ValueError, match="backend"):
        parse_gfa(tiny_gfa, backend="typo", build_graph=True, build_matrix=False)


def test_min_distance_matrix_is_independent_of_path_record_order(tmp_path, tiny_gfa):
    for first, second in (("a", "b"), ("b", "a")):
        gfa = tmp_path / (first + ".gfa")
        gfa.write_text(
            tiny_gfa.read_text() + f"P\tp\t{first}+\t*\nP\tq\t{second}+\t*\n"
        )
        matrix = np.asarray(genome_distance_matrix(str(gfa)))
        np.testing.assert_array_equal(matrix, [[0, 1], [1, 0]])


def test_distance_matrix_rejects_unknown_method(tiny_gfa):
    with pytest.raises(ValueError, match="method"):
        genome_distance_matrix(str(tiny_gfa), method="typo")


def test_distance_matrix_rejects_unsupported_backend(tiny_gfa):
    with pytest.raises(ValueError, match="NetworkX"):
        genome_distance_matrix(str(tiny_gfa), backend="igraph")
