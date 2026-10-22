"""Known-answer and independent-reference checks for the 2.0 graph contract."""

from io import BytesIO, TextIOWrapper
import json
import math
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from gfa2network import parse_gfa, __version__
from gfa2network.analysis import (
    genome_distance,
    genome_distance_matrix,
    sequence_distance,
    load_paths,
    compute_stats,
)
from gfa2network.cli import main
from gfa2network.parser import GFAParser, GapRecord


@pytest.fixture(params=["networkx", "igraph"])
def backend(request):
    if request.param == "igraph":
        pytest.importorskip("igraph")
    return request.param


def names_and_edges(graph):
    if isinstance(graph, nx.Graph):
        return set(graph), {
            (u, v): attrs["weight"] for u, v, attrs in graph.edges(data=True)
        }
    return set(graph.vs["name"]), {
        (graph.vs[e.source]["name"], graph.vs[e.target]["name"]): e["weight"]
        for e in graph.es
    }


@pytest.mark.parametrize(
    "bidirected,keep", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("edge_mode", ["sum", "parallel"])
def test_graph_and_matrix_have_the_same_adjacency(
    tmp_path, backend, bidirected, keep, directed, edge_mode
):
    gfa = tmp_path / "duplicates.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tb\tCCCC\nS\tc\tGGGG\n"
        "L\ta\t+\tb\t-\t0M\tWT:i:2\nL\ta\t+\tb\t-\t0M\tWT:i:3\n"
        "L\tb\t-\ta\t+\t0M\tWT:i:4\nL\ta\t+\ta\t+\t0M\tWT:i:5\n"
        "L\tb\t+\tb\t-\t0M\tWT:i:6\n"
    )
    opts = dict(
        build_graph=True,
        build_matrix=True,
        return_node_list=True,
        backend=backend,
        directed=directed,
        bidirected=bidirected,
        keep_directed_bidir=keep,
        weight_tag="WT",
        edge_mode=edge_mode,
    )
    graph, matrix, nodes = parse_gfa(gfa, **opts)
    if backend == "networkx":
        expected = nx.to_scipy_sparse_array(
            graph, nodelist=nodes, weight="weight"
        ).toarray()
    else:
        expected = np.zeros(matrix.shape)
        for e in graph.es:
            expected[e.source, e.target] += e["weight"]
            if not graph.is_directed() and e.source != e.target:
                expected[e.target, e.source] += e["weight"]
    np.testing.assert_array_equal(matrix.toarray(), expected)
    only, only_nodes = parse_gfa(gfa, **dict(opts, build_graph=False))
    assert only_nodes == nodes
    np.testing.assert_array_equal(only.toarray(), expected)
    if not bidirected:
        assert matrix[nodes.index("a"), nodes.index("b")] == (5 if directed else 9)
        assert matrix[nodes.index("a"), nodes.index("a")] == 5
        assert matrix[nodes.index("b"), nodes.index("b")] == 6
    else:
        assert len(nodes) == 6
        assert matrix[nodes.index("b:+"), nodes.index("b:-")] == 6
        assert matrix[nodes.index("a:+"), nodes.index("a:+")] == 5


def test_stripping_metadata_preserves_literal_identifier_suffixes(tmp_path, backend):
    gfa = tmp_path / "literal.gfa"
    gfa.write_text("S\ta+\tAAAA\nS\tb-\tCCCC\nL\ta+\t+\tb-\t-\t0M\n")
    graph, matrix, nodes = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=True,
        strip_orientation=True,
        raw_bytes_id=True,
        return_node_list=True,
    )
    assert set(nodes) == {b"a+", b"b-"}
    assert matrix[nodes.index(b"a+"), nodes.index(b"b-")] == 1


def test_bundled_fixture_has_equivalent_backends():
    pytest.importorskip("igraph")
    from pathlib import Path

    source = Path(__file__).parent / "data/DRB1-3123_unsorted.gfa"
    graph, matrix, nodes = parse_gfa(
        source, build_graph=True, build_matrix=True, return_node_list=True
    )
    other, adjacency, names = parse_gfa(
        source,
        backend="igraph",
        build_graph=True,
        build_matrix=True,
        return_node_list=True,
    )
    assert names == nodes
    assert graph.number_of_nodes() == other.vcount() == 3214
    assert graph.number_of_edges() == other.ecount() == 6236
    assert (matrix != adjacency).nnz == 0


def test_split_full_span_preserves_both_halves_metadata_and_paths(tmp_path, backend):
    gfa = tmp_path / "split.gfa"
    gfa.write_text(
        "S\ta\t10\tACGTACGTAA\tRC:i:7\nS\tb\t10\tTGCATGCATT\nS\tc\t5\tCCCCC\n"
        "E\t*\ta+\tb+\t0\t10$\t0\t10$\t10M\n"
        "E\t*\ta+\tc+\t0\t5\t0\t5$\t5M\n"
        "L\ta\t+\tb\t-\t0M\nP\tp\ta+,b-\t*\n"
    )
    (graph, matrix, nodes), paths = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=True,
        return_node_list=True,
        return_paths=True,
        bidirected=True,
        keep_directed_bidir=True,
        split_on_alignment=True,
        store_seq=True,
        store_tags=True,
    )
    assert len(nodes) == 10
    assert all(n.endswith((":+", ":-")) for n in nodes)
    assert matrix[nodes.index("a:0-5:+"), nodes.index("b:0-5:+")] == 1
    assert matrix[nodes.index("a:5-10:+"), nodes.index("b:5-10:+")] == 1
    assert matrix[nodes.index("a:5-10:+"), nodes.index("b:5-10:-")] == 1
    assert paths["p"] == ["a:0-5:+", "a:5-10:+", "b:5-10:-", "b:0-5:-"]
    attrs = (
        graph.nodes["a:0-5:+"]
        if backend == "networkx"
        else graph.vs.find(name="a:0-5:+").attributes()
    )
    assert attrs["sequence"] == b"ACGTA"
    assert attrs["tags"] == {"RC": 7}
    assert attrs["parent_segment"] == "a"
    assert attrs["length"] == 5
    attrs = (
        graph.nodes["b:5-10:-"]
        if backend == "networkx"
        else graph.vs.find(name="b:5-10:-").attributes()
    )
    assert attrs["sequence"] == b"AATGC"


@pytest.mark.parametrize(
    "cigar,span,expected",
    [
        ("2M1I2M1D1M", 6, [("a:0-2", "b:0-2"), ("a:2-4", "b:3-5"), ("a:5-6", "b:5-6")]),
        ("6M", 6, [("a:0-6", "b:0-6")]),
    ],
)
def test_split_cigar_blocks_have_known_endpoints(
    tmp_path, backend, cigar, span, expected
):
    gfa = tmp_path / "indels.gfa"
    gfa.write_text(
        f"S\ta\t{span}\t{'A' * span}\nS\tb\t{span}\t{'C' * span}\n"
        f"E\t*\ta+\tb+\t0\t{span}$\t0\t{span}$\t{cigar}\n"
    )
    graph = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=False,
        split_on_alignment=True,
    )
    _, edges = names_and_edges(graph)
    assert all(edge in edges for edge in expected)
    assert set(
        edge for edge in edges if edge[0].startswith("a:") and edge[1].startswith("b:")
    ) == set(expected)


def test_reverse_alignment_projects_cut_to_opposite_end(tmp_path, backend):
    gfa = tmp_path / "reverse.gfa"
    gfa.write_text(
        "S\ta\t10\nS\tb\t10\nS\tc\t3\n"
        "E\t*\ta+\tb-\t0\t10$\t0\t10$\t10M\n"
        "E\t*\ta+\tc+\t0\t3\t0\t3$\t3M\n"
    )
    graph = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=False,
        split_on_alignment=True,
    )
    nodes, edges = names_and_edges(graph)
    assert {"b:0-7", "b:7-10"}.issubset(nodes)
    assert ("a:0-3", "b:7-10") in edges
    assert ("a:3-10", "b:0-7") in edges


def test_containment_and_zero_overlap_boundaries(tmp_path, backend):
    gfa = tmp_path / "containment.gfa"
    gfa.write_text(
        "S\ta\t10\nS\tb\t4\nC\ta\t-\tb\t+\t2\t4M\nE\t*\ta+\tb+\t10$\t10$\t0\t0\t*\n"
    )
    graph = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=False,
        bidirected=True,
        keep_directed_bidir=True,
        split_on_alignment=True,
    )
    _, edges = names_and_edges(graph)
    assert ("a:2-6:-", "b:0-4:+") in edges
    assert ("a:6-10:+", "b:0-4:+") in edges


@pytest.mark.parametrize("alignment", ["*", "10", "9M", "11M"])
def test_ambiguous_or_inconsistent_split_fails_loudly(tmp_path, alignment):
    gfa = tmp_path / "bad.gfa"
    gfa.write_text(
        "S\ta\t10\nS\tb\t10\nS\tc\t5\n"
        f"E\t*\ta+\tb+\t0\t10$\t0\t10$\t{alignment}\n"
        "E\t*\ta+\tc+\t0\t5\t0\t5$\t5M\n"
    )
    with pytest.raises(ValueError):
        parse_gfa(gfa, build_graph=True, build_matrix=False, split_on_alignment=True)


def test_walk_chunks_are_sorted_and_gaps_are_preserved(tmp_path, backend):
    gfa = tmp_path / "walks.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tb\tCCCC\nS\tc\tGGGG\n"
        "J\ta\t+\tb\t-\t*\nG\tg\tb-\tc+\t12\t3\n"
        "W\tsample\t0\tchr1\t8\t12\t<c\n"
        "W\tsample\t0\tchr1\t0\t8\t>a<b\n"
        "P\tjump\ta+;b-\t.\n"
    )
    graph, paths = parse_gfa(
        gfa, build_graph=True, build_matrix=False, return_paths=True, backend=backend
    )
    assert paths["sample#0#chr1"] == ["a", "b", "c"]
    assert load_paths(gfa, oriented=True)["sample#0#chr1"] == ["a:+", "b:-", "c:-"]
    assert paths["jump"] == ["a", "b"]
    assert compute_stats(gfa, backend=backend)["paths"] == 2
    gaps = [rec for rec in GFAParser(gfa) if isinstance(rec, GapRecord)]
    assert gaps[0].distance is None
    assert gaps[1].distance == 12 and gaps[1].variance == 3


def test_overlapping_or_unpositioned_walk_chunks_are_rejected(tmp_path):
    for coordinates in [("0", "8", "4", "12"), ("*", "*", "*", "*")]:
        gfa = tmp_path / "overlap.gfa"
        x, y, z, w = coordinates
        gfa.write_text(
            f"S\ta\tAAAA\nW\ts\t0\tc\t{x}\t{y}\t>a\nW\ts\t0\tc\t{z}\t{w}\t>a\n"
        )
        with pytest.raises(ValueError):
            load_paths(gfa)


def test_stats_and_path_distance_matrix_consume_stdin_once(monkeypatch):
    data = b"S\ta\tAAAA\nS\tb\tCCCC\nL\ta\t+\tb\t+\t0M\nP\tp\ta+\t*\nP\tq\tb+\t*\n"
    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(data)))
    assert compute_stats("-")["paths"] == 2
    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(data)))
    np.testing.assert_array_equal(
        np.asarray(genome_distance_matrix("-")), [[0, 1], [1, 0]]
    )


def reference(graph, a, b, method, policy):
    a, b = list(dict.fromkeys(a)), list(dict.fromkeys(b))
    pairs = np.array(
        [
            [
                nx.shortest_path_length(graph, u, v, weight="weight")
                if nx.has_path(graph, u, v)
                else np.inf
                for v in b
            ]
            for u in a
        ]
    )
    if method == "min":
        return pairs.min()
    contributions = (
        pairs.ravel()
        if method in {"mean", "mean-pairwise"}
        else np.concatenate((pairs.min(axis=0), pairs.min(axis=1)))
    )
    if policy == "ignore":
        contributions = contributions[np.isfinite(contributions)]
    return contributions.mean() if len(contributions) else math.inf


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["min", "mean", "mean-pairwise", "mean-nearest"])
@pytest.mark.parametrize("policy", ["infinite", "ignore"])
def test_all_distance_apis_match_independent_pairwise_reference(
    tmp_path, backend, directed, method, policy
):
    gfa = tmp_path / "distances.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tb\tCCCC\nS\tc\tGGGG\nS\td\tTTTT\n"
        "L\ta\t+\tb\t+\t0M\tWT:i:2\nL\tb\t+\tc\t+\t0M\tWT:i:3\n"
        "L\ta\t+\tc\t+\t0M\tWT:i:10\n"
        "P\tx\ta+,b+\t*\nP\ty\tb+,c+\t*\nP\tz\td+\t*\n"
    )
    graph, paths = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        backend=backend,
        directed=directed,
        return_paths=True,
        weight_tag="WT",
    )
    expected = nx.DiGraph() if directed else nx.Graph()
    expected.add_nodes_from("abcd")
    expected.add_weighted_edges_from([("a", "b", 2), ("b", "c", 3), ("a", "c", 10)])
    matrix = np.asarray(
        genome_distance_matrix(
            gfa,
            method=method,
            backend=backend,
            directed=directed,
            weight_tag="WT",
            unreachable=policy,
        )
    )
    for i, a in enumerate(paths.values()):
        for j, b in enumerate(paths.values()):
            answer = reference(expected, a, b, method, policy)
            assert matrix[i, j] == pytest.approx(answer)
            assert genome_distance(
                graph, a, b, method=method, unreachable=policy
            ) == pytest.approx(answer)


def test_distance_coverage_deduplicates_nodes_and_exposes_partial_reachability():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 2), ("b", "c", 3)])
    graph.add_node("d")
    result = genome_distance(graph, ["a", "a", "d"], ["c"], return_details=True)
    assert result.distance == 5 and result.coverage == pytest.approx(2 / 3)
    result = genome_distance(
        graph,
        ["a", "d"],
        ["c"],
        method="mean",
        unreachable="ignore",
        return_details=True,
    )
    assert result.distance == 5 and result.coverage == 0.5
    assert math.isinf(genome_distance(graph, ["a", "d"], ["c"], method="mean"))
    with pytest.raises(nx.NetworkXNoPath):
        genome_distance(graph, ["a", "d"], ["c"], method="mean", unreachable="raise")


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_dijkstra_rejects_invalid_costs(weight):
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=weight)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        genome_distance(graph, ["a"], ["b"])


def test_igraph_sequence_distances_and_raw_ids(tmp_path):
    pytest.importorskip("igraph")
    gfa = tmp_path / "seq.gfa"
    gfa.write_text("S\ta\tAAAA\nS\tb\tCCCC\nL\ta\t+\tb\t+\t0M\n")
    graph = parse_gfa(
        gfa,
        backend="igraph",
        build_graph=True,
        build_matrix=False,
        store_seq=True,
        raw_bytes_id=True,
    )
    assert sequence_distance(graph, "AAAA", "CCCC") == 1
    assert genome_distance(graph, [b"a"], [b"b"]) == 1


@pytest.mark.parametrize("backend_name", ["networkx", "igraph"])
def test_tag_limit_is_honored_in_both_backends(tmp_path, backend_name):
    if backend_name == "igraph":
        pytest.importorskip("igraph")
    gfa = tmp_path / "tags.gfa"
    gfa.write_text("S\ta\tAAAA\tRC:i:7\n")
    with pytest.warns(RuntimeWarning, match="tag dictionaries"):
        parse_gfa(
            gfa,
            backend=backend_name,
            build_graph=True,
            build_matrix=False,
            store_tags=True,
            max_tag_mb=0,
        )


def test_parser_error_has_source_line(tmp_path):
    gfa = tmp_path / "broken.gfa"
    gfa.write_text("S\ta\tAAAA\nE\t*\ta+\tb+\t5\t2\t0\t3\t3M\n")
    with pytest.raises(ValueError, match=r"broken.gfa:2: E record"):
        list(GFAParser(gfa))


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_matrix_dtype_cannot_silently_overflow(tmp_path, dtype):
    gfa = tmp_path / "overflow.gfa"
    value = "200" if dtype == "int8" else "1e40"
    gfa.write_text(f"S\ta\t*\nL\ta\t+\ta\t+\t0M\tWT:f:{value}\n")
    with pytest.raises(ValueError, match="cannot be represented"):
        parse_gfa(
            gfa, build_graph=False, build_matrix=True, weight_tag="WT", dtype=dtype
        )


def test_cli_distance_exports_labels_coverage_and_metadata(tmp_path):
    gfa = tmp_path / "labels.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tb\tCCCC\nL\ta\t+\tb\t+\t0M\nP\tx\ta+\t*\nP\ty\tb+\t*\n"
    )
    for suffix in ("csv", "npy", "npz"):
        out = tmp_path / ("dist." + suffix)
        main(["distance-matrix", str(gfa), "-o", str(out), "--method", "mean"])
        assert (tmp_path / (out.name + ".paths.tsv")).read_text() == "0\tx\n1\ty\n"
        np.testing.assert_array_equal(
            np.load(str(out) + ".coverage.npy"), np.ones((2, 2))
        )
        metadata = json.loads((tmp_path / (out.name + ".metadata.json")).read_text())
        assert metadata["method"] == "mean-pairwise"
        values = (
            sp.load_npz(out).toarray()
            if suffix == "npz"
            else np.load(out)
            if suffix == "npy"
            else np.loadtxt(out, delimiter=",")
        )
        np.testing.assert_array_equal(values, [[0, 1], [1, 0]])


def test_distance_output_allocation_is_guarded_before_creation(tmp_path):
    gfa = tmp_path / "allocation.gfa"
    gfa.write_text("S\ta\tAAAA\nP\tp\ta+\t*\n")
    with pytest.raises(MemoryError, match="workspace"):
        genome_distance_matrix(gfa, max_dense_gb=0)


def test_cli_version_matches_package_version():
    result = subprocess.run(
        [sys.executable, "-m", "gfa2network", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"gfa2network {__version__}"


@pytest.mark.parametrize("bidirected", [False, True])
def test_empty_conversion_and_explicit_projection(tmp_path, backend, bidirected):
    gfa = tmp_path / "empty.gfa"
    gfa.write_text("")
    graph, matrix, nodes = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=True,
        return_node_list=True,
        bidirected=bidirected,
        asymmetric=False,
    )
    assert matrix.shape == (0, 0) and nodes == []
    assert compute_stats(gfa, backend=backend)["components"] == 0


def test_duplicate_segment_definitions_are_rejected(tmp_path, backend):
    gfa = tmp_path / "duplicate.gfa"
    gfa.write_text("S\ta\tAAAA\nS\ta\tCCCC\n")
    with pytest.raises(ValueError, match="duplicate segment"):
        parse_gfa(gfa, backend=backend, build_graph=True, build_matrix=False)


def test_zero_cost_edges_survive_sparse_projection(tmp_path, backend):
    gfa = tmp_path / "zero.gfa"
    gfa.write_text("S\ta\tAAAA\nS\tb\tCCCC\nL\ta\t+\tb\t+\t0M\tWT:i:0\n")
    graph, matrix = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=True,
        weight_tag="WT",
        asymmetric=False,
    )
    assert matrix.nnz == 2
    assert genome_distance(graph, ["a"], ["b"]) == 0


def test_parallel_cost_is_the_cheapest_connection(tmp_path, backend):
    gfa = tmp_path / "parallel.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tb\tCCCC\nL\ta\t+\tb\t+\t0M\tWT:i:2\nL\ta\t+\tb\t+\t0M\tWT:i:3\n"
    )
    graph, matrix = parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=True,
        weight_tag="WT",
        edge_mode="parallel",
    )
    assert matrix[0, 1] == 5
    assert genome_distance(graph, ["a"], ["b"]) == 2


def test_incomplete_ordered_group_is_not_treated_as_a_segment_path(tmp_path):
    gfa = tmp_path / "group.gfa"
    gfa.write_text("S\ta\tAAAA\nO\tp\tedge_reference+\n")
    with pytest.raises(ValueError, match="undefined segment/group"):
        load_paths(gfa)
    with pytest.raises(ValueError, match="undefined segment/group"):
        parse_gfa(gfa, build_graph=True, build_matrix=False, return_paths=True)


def test_cigar_cut_propagation_has_an_expansion_limit(tmp_path):
    gfa = tmp_path / "expansion.gfa"
    gfa.write_text(
        "S\ta\t100\nS\tb\t100\n"
        "E\t*\ta+\tb+\t0\t99\t1\t100$\t99M\n"
        "E\t*\ta+\tb+\t1\t100$\t0\t99\t99M\n"
    )
    with pytest.raises(MemoryError, match="max_split_nodes"):
        parse_gfa(
            gfa,
            build_graph=True,
            build_matrix=False,
            split_on_alignment=True,
            max_split_nodes=10,
        )


def test_verbose_progress_stays_off_stdout(tmp_path, backend, capsys):
    gfa = tmp_path / "progress.gfa"
    gfa.write_text("S\ta\tAAAA\n")
    parse_gfa(
        gfa,
        backend=backend,
        build_graph=True,
        build_matrix=False,
        verbose=True,
        progress_step=1,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[1 lines]" in captured.err and "done" in captured.err


def test_path_analysis_rejects_undefined_segments_even_if_an_edge_mentions_them(
    tmp_path,
):
    gfa = tmp_path / "undefined.gfa"
    gfa.write_text("S\ta\tAAAA\nL\ta\t+\tb\t+\t0M\nP\tp\tb+\t*\n")
    with pytest.raises(ValueError, match="undefined segment/group"):
        parse_gfa(gfa, build_graph=True, build_matrix=False, return_paths=True)
