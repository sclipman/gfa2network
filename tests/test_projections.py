"""Explicit relation projections, edge-cost policy and analysis manifests."""

import json
import math
import pickle

import networkx as nx
import numpy as np
import pytest

from gfa2network import parse_gfa
from gfa2network.analysis import genome_distance, genome_distance_matrix
from gfa2network.cli import main
from gfa2network.projection import PROJECTIONS, resolve_projection

# L adjacency (with a duplicate, a reciprocal and a self-loop), C containment,
# and J jumps, including one J that is the only route from b to d.
MIXED = (
    "H\tVN:Z:1.2\n"
    "S\ta\tACGTAC\nS\tb\tGG\nS\tc\tGTA\nS\td\tTT\n"
    "L\ta\t+\tb\t+\t0M\tWT:i:2\nL\ta\t+\tb\t+\t0M\tWT:i:3\n"
    "L\tb\t-\ta\t-\t0M\nL\tb\t+\tb\t+\t0M\n"
    "C\ta\t+\tc\t+\t2\t3M\n"
    "J\tb\t+\td\t+\t100\nJ\ta\t+\td\t+\t*\n"
    "P\tleft\ta+,b+\t*\nP\tright\td+\t*\nP\tinner\tc+\t*\n"
)

# GFA2: E before S; dovetail, containment, internal alignment, out-of-range
# coordinates, an orientation-only historical layout and a G gap.
GFA2 = (
    "H\tVN:Z:2.0\n"
    "E\te1\ta+\tb+\t3\t5$\t0\t2\t2M\n"
    "S\ta\t5\tACGTA\nS\tb\t5\tTACCC\nS\tc\t2\tCG\nS\td\t3\tGGG\n"
    "E\te2\tb+\tc+\t1\t3\t0\t2$\t2M\n"
    "E\te3\ta+\td+\t1\t3\t1\t3\t2M\n"
    "E\te4\ta-\td+\t0\t9\t0\t9\t9M\n"
    "E\te5\tc\t+\td\t-\n"
    "G\tg1\tb+\td+\t50\t*\n"
)


def write(tmp_path, text, name="graph.gfa"):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture(params=["networkx", "igraph"])
def backend(request):
    if request.param == "igraph":
        pytest.importorskip("igraph")
    return request.param


def edges_of(graph):
    if isinstance(graph, nx.Graph):
        return {(u, v): attrs["weight"] for u, v, attrs in graph.edges(data=True)}
    names = graph.vs["name"]
    return {(names[e.source], names[e.target]): e["weight"] for e in graph.es}


def manifest_of(graph):
    return (
        graph.graph["projection"]
        if isinstance(graph, nx.Graph)
        else graph["projection"]
    )


def test_resolve_projection_names_and_classes():
    assert resolve_projection(None) == ("all", PROJECTIONS["all"])
    assert resolve_projection("traversal")[1] == {"adjacency"}
    assert resolve_projection("adjacency,gap")[0] == "gap-inclusive"
    assert resolve_projection({"gap"}) == ("custom", frozenset({"gap"}))
    for bad in ("everything", "adjacency,links", ""):
        with pytest.raises(ValueError, match="projection must be"):
            resolve_projection(bad)


def test_relation_classes_are_counted_by_record(tmp_path):
    _, manifest = parse_gfa(
        write(tmp_path, GFA2),
        build_graph=True,
        build_matrix=False,
        return_manifest=True,
    )
    records = manifest["records"]
    assert records["adjacency"]["by_record"] == {"E": 2}  # e1 dovetail, e5 legacy
    assert records["containment"]["by_record"] == {"E": 1}  # e2 spans all of c
    assert records["alignment"]["by_record"] == {"E": 1}  # e3 interior
    assert records["unclassified"]["by_record"] == {"E": 1}  # e4 out of range
    assert records["gap"]["by_record"] == {"G": 1}
    assert all(entry["included"] for entry in records.values())
    _, mixed = parse_gfa(
        write(tmp_path, MIXED, "mixed.gfa"),
        build_graph=True,
        build_matrix=False,
        return_manifest=True,
    )
    # Duplicate and reciprocal records are each counted; the graph sums them.
    assert mixed["records"]["adjacency"]["records"] == 4
    assert mixed["records"]["containment"]["by_record"] == {"C": 1}
    assert mixed["records"]["gap"]["by_record"] == {"J": 2}


@pytest.mark.parametrize("projection", list(PROJECTIONS))
def test_e_records_before_segments_are_classified_at_input_end(
    tmp_path, projection, backend
):
    graph = parse_gfa(
        write(tmp_path, GFA2),
        build_graph=True,
        build_matrix=False,
        directed=False,
        projection=projection,
        backend=backend,
    )
    pairs = {frozenset(pair) for pair in edges_of(graph)}
    expected = {
        "all": {"ab", "bc", "ad", "cd", "bd"},
        "traversal": {"ab", "cd"},
        "gap-inclusive": {"ab", "cd", "bd"},
        "alignment": {"bc", "ad"},
    }[projection]
    assert pairs == {frozenset(pair) for pair in expected}


@pytest.mark.parametrize("projection", list(PROJECTIONS) + [("gap",)])
@pytest.mark.parametrize(
    "bidirected,keep", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("edge_mode", ["sum", "parallel"])
def test_backends_and_matrix_agree_within_each_projection(
    tmp_path, projection, bidirected, keep, directed, edge_mode
):
    pytest.importorskip("igraph")
    gfa = write(tmp_path, MIXED)
    results = {}
    for backend in ("networkx", "igraph"):
        graph, matrix, nodes = parse_gfa(
            gfa,
            build_graph=True,
            build_matrix=True,
            return_node_list=True,
            backend=backend,
            directed=directed,
            bidirected=bidirected,
            keep_directed_bidir=keep,
            weight_tag="WT",
            edge_mode=edge_mode,
            projection=projection,
        )
        dense = np.zeros(matrix.shape)
        index = {node: i for i, node in enumerate(nodes)}
        is_directed = graph.is_directed()
        if isinstance(graph, nx.Graph):
            edge_iter = ((u, v, d["weight"]) for u, v, d in graph.edges(data=True))
        else:
            names = graph.vs["name"]
            edge_iter = (
                (names[e.source], names[e.target], e["weight"]) for e in graph.es
            )
        for u, v, weight in edge_iter:
            dense[index[u], index[v]] += weight
            if not is_directed and u != v:
                dense[index[v], index[u]] += weight
        np.testing.assert_array_equal(matrix.toarray(), dense)
        manifest = manifest_of(graph)
        edges = {
            (pair if is_directed else tuple(sorted(pair))): weight
            for pair, weight in edges_of(graph).items()
        }
        results[backend] = (
            nodes,
            edges if edge_mode == "sum" else None,
            matrix.toarray(),
            {k: v for k, v in manifest.items() if k != "graph_edges"},
        )
    assert results["networkx"][0] == results["igraph"][0]
    assert results["networkx"][1] == results["igraph"][1]
    np.testing.assert_array_equal(results["networkx"][2], results["igraph"][2])
    assert results["networkx"][3] == results["igraph"][3]


def test_default_projection_matches_explicit_all(tmp_path, backend):
    gfa = write(tmp_path, MIXED)
    opts = dict(build_graph=True, build_matrix=True, backend=backend, weight_tag="WT")
    default_graph, default_matrix = parse_gfa(gfa, **opts)
    explicit_graph, explicit_matrix = parse_gfa(gfa, projection="all", **opts)
    assert edges_of(default_graph) == edges_of(explicit_graph)
    assert (default_matrix != explicit_matrix).nnz == 0
    assert manifest_of(default_graph)["projection"] == "all"


def test_gaps_and_containment_shortcuts_are_not_contiguous_sequence(tmp_path):
    gfa = write(tmp_path, MIXED)

    def distance(projection, a, b):
        graph, paths = parse_gfa(
            gfa,
            build_graph=True,
            build_matrix=False,
            directed=False,
            return_paths=True,
            projection=projection,
        )
        return genome_distance(graph, paths[a], paths[b])

    # Only a jump connects left and right: no adjacency route exists.
    assert distance("traversal", "left", "right") == math.inf
    assert distance("gap-inclusive", "left", "right") == 1
    # The containment shortcut makes c reachable only outside traversal.
    assert distance("all", "left", "inner") == 1
    assert distance("traversal", "left", "inner") == math.inf
    _, details = genome_distance_matrix(
        str(gfa), projection="traversal", return_details=True
    )
    assert details["projection"]["projection"] == "traversal"
    assert details["projection"]["records"]["gap"]["included"] is False
    assert details["cost"] == "weight"


def test_hop_cost_is_independent_of_record_multiplicity(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tA\nS\tb\tC\nS\tc\tG\n"
        "L\ta\t+\tb\t+\t0M\nL\ta\t+\tb\t+\t0M\nL\tb\t+\tc\t+\t0M\n",
    )
    for edge_mode in ("sum", "parallel"):
        graph = parse_gfa(
            gfa, build_graph=True, build_matrix=False, edge_mode=edge_mode
        )
        weighted = genome_distance(graph, ["a"], ["c"])
        hops = genome_distance(graph, ["a"], ["c"], cost="hops")
        # Summed duplicate records cost 2; the cheapest parallel record costs 1.
        assert weighted == (3 if edge_mode == "sum" else 2)
        assert hops == 2
    with pytest.raises(ValueError, match="weight or hops"):
        genome_distance(graph, ["a"], ["c"], cost="length")


def test_subdivided_relations_keep_their_source_class(tmp_path, backend):
    gfa = write(
        tmp_path,
        "H\tVN:Z:2.0\nS\ta\t5\tACGTA\nS\tb\t5\tTACCC\nS\tc\t2\tCG\n"
        "E\te1\ta+\tb+\t3\t5$\t0\t2\t2M\nE\te2\tb+\tc+\t1\t3\t0\t2$\t2M\n",
    )
    graph, manifest = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        directed=False,
        split_on_alignment=True,
        projection="traversal",
        backend=backend,
        return_manifest=True,
    )
    assert manifest["subdivided"] is True
    # Input records are counted once each; generated relations separately.
    assert manifest["records"]["adjacency"]["by_record"] == {"E": 1}
    assert manifest["records"]["containment"]["by_record"] == {"E": 1}
    generated = manifest["subdivided_relations"]
    assert generated["containment"]["records"] == 2
    assert generated["subdivision"] == {
        "included": True,
        "records": 6,
        "by_record": {"L": 6},
    }
    pairs = {frozenset(pair) for pair in edges_of(graph)}
    # Dovetail blocks (a:3-5 aligned to b:0-2, cut at the propagated b:1) and
    # interval links survive; the blocks aligning b:1-3 with c do not.
    assert {frozenset({"a:3-4", "b:0-1"}), frozenset({"a:4-5", "b:1-2"})} <= pairs
    assert frozenset({"c:0-1", "c:1-2"}) in pairs
    assert not any({v[0] for v in pair} == {"b", "c"} for pair in map(sorted, pairs))
    assert manifest["records"]["containment"]["included"] is False
    # Under the alignment projection, interval links still rebuild each segment.
    aligned = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        directed=False,
        split_on_alignment=True,
        projection="alignment",
        backend=backend,
    )
    aligned_pairs = {frozenset(pair) for pair in edges_of(aligned)}
    assert frozenset({"c:0-1", "c:1-2"}) in aligned_pairs
    assert frozenset({"b:1-2", "c:0-1"}) in aligned_pairs
    assert frozenset({"a:3-4", "b:0-1"}) not in aligned_pairs


def test_held_records_wait_for_the_end_of_input(tmp_path):
    from gfa2network.model import GraphAccumulator
    from gfa2network.parser import GFAParser

    gfa = write(tmp_path, GFA2)
    builder = GraphAccumulator(projection="traversal")
    records = list(GFAParser(str(gfa)))
    builder.add_edge_record(records[0])  # e1 precedes its segments
    builder.finish_graph()  # an early look must not misclassify it
    assert len(builder.pending) == 1
    for record in records[1:5]:
        builder.add_segment(record)
    graph = builder.finish_graph()
    assert graph.has_edge("a", "b") and builder.pending == []
    bad = write(
        tmp_path,
        "H\tVN:Z:2.0\nE\te\ta+\tb+\t3\t5$\t0\t2\t2M\tWT:Z:x\n"
        "S\ta\t5\tACGTA\nS\tb\t5\tTACCC\n",
        "bad.gfa",
    )
    with pytest.raises(ValueError, match=r"bad\.gfa: E record at line 2"):
        parse_gfa(
            bad, build_graph=True, build_matrix=False, weight_tag="WT",
            projection="traversal",
        )  # fmt: skip


def test_return_manifest_wraps_every_result_shape(tmp_path):
    gfa = write(tmp_path, MIXED)
    matrix, manifest = parse_gfa(
        gfa, build_graph=False, build_matrix=True, return_manifest=True
    )
    assert manifest["matrix_nonzeros"] == matrix.nnz and "graph_edges" not in manifest
    (graph, paths), manifest = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        return_paths=True,
        projection="traversal",
        return_manifest=True,
    )
    assert set(paths) == {"left", "right", "inner"}
    assert manifest == graph.graph["projection"]
    assert manifest["graph_edges"] == graph.number_of_edges()
    assert manifest["relations"] == ["adjacency"]


def test_cli_projection_manifests_and_exports(tmp_path, capsys):
    gfa = write(tmp_path, MIXED)
    manifest_path = tmp_path / "convert.json"
    main(
        [
            "convert", str(gfa), "--graph", "-o", str(tmp_path / "g.pkl"),
            "--matrix", str(tmp_path / "m.npz"), "--projection", "traversal",
            "--manifest", str(manifest_path),
        ]
    )  # fmt: skip
    manifest = json.loads(manifest_path.read_text())
    assert manifest["projection"] == "traversal"
    assert [e["format"] for e in manifest["exports"]] == ["pickle", "matrix"]
    assert "multiplicity" in manifest["exports"][1]["drops"]
    # The pickle was written by this test a moment ago.
    with open(tmp_path / "g.pkl", "rb") as handle:
        assert pickle.load(handle).graph["projection"]["projection"] == "traversal"

    edge_manifest = tmp_path / "export.json"
    main(
        [
            "export", str(gfa), "--format", "edge-list", "--projection", "gap",
            "--manifest", str(edge_manifest), "--undirected",
        ]
    )  # fmt: skip
    lines = sorted(capsys.readouterr().out.split())
    assert lines == ["a", "b", "d", "d"]
    exported = json.loads(edge_manifest.read_text())
    assert exported["projection"] == "custom"
    assert "summed edge weight" in exported["exports"][0]["drops"]

    out = tmp_path / "dist.csv"
    main(
        [
            "distance-matrix", str(gfa), "-o", str(out), "--projection",
            "gap-inclusive", "--cost", "hops",
        ]
    )  # fmt: skip
    metadata = json.loads((tmp_path / "dist.csv.metadata.json").read_text())
    assert metadata["cost"] == "hops"
    assert metadata["projection"]["projection"] == "gap-inclusive"

    main(["stats", str(gfa), "--projection", "traversal"])
    assert "projection\t traversal" in capsys.readouterr().out
    main(["distance", str(gfa), "--path", "left", "right", "--undirected",
          "--projection", "gap-inclusive", "--cost", "hops", "--details"])  # fmt: skip
    details = json.loads(capsys.readouterr().out)
    assert (details["distance"], details["cost"], details["projection"]) == (
        1.0,
        "hops",
        "gap-inclusive",
    )
    with pytest.raises(SystemExit):
        main(["stats", str(gfa), "--projection", "links"])
