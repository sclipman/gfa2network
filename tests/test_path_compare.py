"""Predeclared controls for base-aware shared content and ordered shared blocks."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from gfa2network import parse_gfa
from gfa2network.analysis import genome_distance
from gfa2network.cli import main
from gfa2network.genome_paths import load_genome_paths
from gfa2network.path_compare import compare_paths, shared_content_matrix
from gfa2network.transform import rename

DRB1 = Path(__file__).parent / "data" / "DRB1-3123_unsorted.gfa"
SEGMENTS = "S\ta\tAAAA\nS\tc\tCCCC\nS\tg\tGGGG\nS\tt\tTTAT\n"
LINKS = (
    "L\ta\t+\tc\t+\t0M\nL\tc\t+\tg\t+\t0M\nL\ta\t+\tg\t+\t0M\n"
    "L\tg\t+\tc\t+\t0M\nL\tc\t+\ta\t+\t0M\nL\tg\t+\tt\t+\t0M\n"
    "L\tt\t+\ta\t+\t0M\n"
)


def load(tmp_path, paths, extra="", name="graph.gfa", segments=SEGMENTS):
    gfa = tmp_path / name
    gfa.write_text("H\tVN:Z:1.2\n" + segments + LINKS + extra + paths)
    return load_genome_paths(gfa)


def test_identical_paths_share_everything(tmp_path):
    paths = load(tmp_path, "P\tx\ta+,c+,g+\t*\nP\ty\ta+,c+,g+\t*\n")
    result = compare_paths(paths, "x", "y")
    content = result.content
    assert (content.shared_bases, content.union_bases, content.jaccard) == (12, 12, 1)
    assert len(result.blocks) == 1 and result.blocks[0].bases == 12
    assert result.blocks[0].a.start == 0 and result.blocks[0].a.end == 12
    assert result.collinear.bases == 12 and result.collinear.fraction_a == 1


def test_reordered_content_is_shared_but_not_collinear(tmp_path):
    paths = load(tmp_path, "P\tx\ta+,c+,g+\t*\nP\ty\ta+,g+,c+\t*\n")
    result = compare_paths(paths, "x", "y")
    assert result.content.jaccard == 1
    assert sorted(b.first_segment for b in result.blocks) == ["a", "c", "g"]
    assert result.collinear.bases == 8
    assert result.collinear.fraction_a == pytest.approx(2 / 3)
    # Connectivity distances cannot see the order change.
    graph, lists = parse_gfa(
        tmp_path / "graph.gfa",
        build_graph=True,
        build_matrix=False,
        directed=False,
        return_paths=True,
    )
    assert genome_distance(graph, lists["x"], lists["y"]) == 0


def test_repeated_visits_follow_the_declared_copy_policy(tmp_path):
    paths = load(tmp_path, "P\tx\ta+,c+,a+,g+\t*\nP\ty\ta+,c+,g+\t*\n")
    unique = compare_paths(paths, "x", "y").content
    assert (unique.a_bases, unique.shared_bases, unique.jaccard) == (12, 12, 1)
    counted = compare_paths(paths, "x", "y", copies="count").content
    assert (counted.a_bases, counted.b_bases, counted.shared_bases) == (16, 12, 12)
    assert counted.jaccard == 0.75 and counted.containment_b == 1


def test_reverse_complement_traversal_needs_the_reverse_policy(tmp_path):
    paths = load(tmp_path, "P\tx\ta+,c+,g+\t*\nP\ty\tg-,c-,a-\t*\n")
    forward = compare_paths(paths, "x", "y")
    assert forward.content.jaccard == 1  # content is strand-independent
    assert forward.blocks == () and forward.collinear.bases == 0
    both = compare_paths(paths, "x", "y", orientation="both")
    (block,) = both.blocks
    assert (block.orientation, block.steps, block.bases) == ("reverse", 3, 12)
    assert block.b.reverse and (block.b.first_step, block.b.last_step) == (0, 2)
    assert both.reverse_collinear.bases == 12


def test_fragmented_path_breaks_blocks_but_keeps_content(tmp_path):
    paths = load(
        tmp_path,
        "P\tx\ta+,c+,g+\t*\nP\ty\ta+,c+;g+\t*\n",
        extra="J\tc\t+\tg\t+\t10\n",
    )
    result = compare_paths(paths, "x", "y")
    assert result.content.jaccard == 1
    assert [b.steps for b in result.blocks] == [2, 1]
    assert result.collinear.bases == 12 and len(result.collinear.blocks) == 2
    # After a jump the path coordinate is unknown, not invented.
    assert result.blocks[1].b.start is None


def test_distinct_content_shares_nothing(tmp_path):
    paths = load(tmp_path, "P\tx\ta+\t*\nP\ty\tt+\t*\n")
    result = compare_paths(paths, "x", "y")
    assert (result.content.shared_bases, result.content.jaccard) == (0, 0)
    assert result.blocks == () and result.collinear.fraction_a == 0


def test_unresolved_lengths_are_not_absence(tmp_path):
    segments = SEGMENTS + "S\tu\t*\nS\tv\t*\n"
    paths = load(
        tmp_path,
        "P\tx\tu+\t*\nP\ty\tv+\t*\nP\tz\ta+,u+\t*\n",
        extra="L\ta\t+\tu\t+\t0M\n",
        segments=segments,
    )
    result = compare_paths(paths, "x", "y")
    assert result.content.union_bases == 0 and result.content.jaccard is None
    assert result.content.note == "no resolved bases in either path"
    assert result.content.a.unresolved_steps == (0,)
    assert result.content.a.evidence == "partial"
    matrix, details = shared_content_matrix(paths)
    index = {name: i for i, name in enumerate(details["names"])}
    assert math.isnan(details["jaccard"][index["x"], index["y"]])
    assert details["paths"][index["z"]].evidence == "partial"
    assert result.collinear.fraction_a is None


def test_unknown_overlaps_are_uncertain_unless_declared(tmp_path):
    gfa = tmp_path / "unknown.gfa"
    gfa.write_text(
        "S\ta\tACGT\nS\tb\tTTGG\nL\ta\t+\tb\t+\t*\nP\tx\ta+,b+\t*\nP\ty\ta+,b+\t*\n"
    )
    paths = load_genome_paths(gfa)
    strict = compare_paths(paths, "x", "y")
    assert strict.content.a.uncertain_overlap_steps == (1,)
    assert [b.steps for b in strict.blocks] == [1, 1]
    blunt = compare_paths(paths, "x", "y", unknown_overlap="zero")
    assert blunt.content.a.evidence == "complete"
    assert [b.steps for b in blunt.blocks] == [2]


def test_overlap_bases_count_once_per_segment(tmp_path):
    gfa = tmp_path / "overlap.gfa"
    gfa.write_text(
        "S\ta\tACGTA\nS\tb\tTACCC\nL\ta\t+\tb\t+\t2M\nP\tx\ta+,b+\t*\nP\ty\tb+\t*\n"
        "P\tz\tb-,a-\t*\n"
    )
    paths = load_genome_paths(gfa)
    result = compare_paths(paths, "x", "y")
    # Both segments are content in full; the path spells 8 of those 10 bases.
    assert len(paths.spell("x").sequence) == 8
    assert result.content.a.copy_bases == 10
    assert (result.content.shared_bases, result.content.union_bases) == (5, 10)
    assert result.blocks[0].bases == 5
    assert result.collinear.bases <= result.content.shared_bases
    # The reverse-complement traversal has the same content.
    strand = compare_paths(paths, "x", "z", orientation="both")
    assert strand.content.jaccard == 1
    assert strand.reverse_collinear.bases == 10
    assert strand.reverse_collinear.fraction_a == 1
    _, details = shared_content_matrix(paths)
    assert details["shared"].tolist() == [[10, 5, 10], [5, 5, 5], [10, 5, 10]]


def test_content_with_overlaps_ignores_names_and_rotation(tmp_path):
    text = (
        "S\ta\tACGTA\nS\tb\tTACCG\nS\tc\tCGGAC\n"
        "L\ta\t+\tb\t+\t2M\nL\tb\t+\tc\t+\t2M\nL\tc\t+\ta\t+\t2M\n"
        "L\ta\t+\tc\t-\t1M\n"
        "P\tx\ta+,b+,c+\t*\nP\ty\tb+,c+,a+\t*\nP\tw\ta+,c-\t*\n"
    )
    gfa = tmp_path / "ring.gfa"
    gfa.write_text(text)
    paths = load_genome_paths(gfa)
    ring = compare_paths(paths, "x", "y", circular=True)
    assert ring.content.jaccard == 1 and ring.content.a_bases == 15
    assert ring.collinear.bases == 15 and ring.collinear.fraction_a == 1
    linear = compare_paths(paths, "x", "y")
    assert linear.collinear.bases <= linear.content.shared_bases
    # Renaming so that the name order of a and c reverses changes nothing.
    reordered = tmp_path / "reordered.gfa"
    lines = text.splitlines(keepends=True)
    reordered.write_text("".join(lines[2::-1] + lines[3:]))
    renamed = tmp_path / "renamed.gfa"
    mapping = rename(reordered, renamed, prefix="z")
    assert mapping == {"c": "z1", "b": "z2", "a": "z3"}
    _, before = shared_content_matrix(paths)
    _, after = shared_content_matrix(load_genome_paths(renamed))
    assert before["shared"].tolist() == after["shared"].tolist()
    moved = compare_paths(load_genome_paths(renamed), "x", "w")
    assert moved.content.jaccard == compare_paths(paths, "x", "w").content.jaccard


def test_chains_never_exceed_shared_content(tmp_path):
    gfa = tmp_path / "bubble.gfa"
    gfa.write_text(
        "S\ta\tACGTA\nS\tb1\tTACCG\nS\tb2\tTAGCG\nS\tc\tCGGAC\n"
        "L\ta\t+\tb1\t+\t2M\nL\ta\t+\tb2\t+\t2M\nL\tb1\t+\tc\t+\t2M\n"
        "L\tb2\t+\tc\t+\t2M\nJ\ta\t+\tc\t+\t5\n"
        "P\tx\ta+,b1+,c+\t*\nP\ty\ta+,b2+,c+\t*\nP\tz\ta+;c+\t*\n"
    )
    paths = load_genome_paths(gfa)
    for other in ("y", "z"):
        result = compare_paths(paths, "x", other)
        assert result.collinear.bases <= result.content.shared_bases
        assert result.collinear.fraction_a <= 1
        swapped = compare_paths(paths, other, "x")
        assert swapped.collinear.bases == result.collinear.bases


def test_unknown_lengths_are_counted_as_known_bases_not_absence(tmp_path):
    gfa = tmp_path / "unknown.gfa"
    gfa.write_text(
        "S\ta\tAAAA\nS\tu\t*\nS\tc\tCCCC\nS\tg\tGGGG\n"
        "L\ta\t+\tu\t+\t0M\nL\tu\t+\tc\t+\t0M\nL\tc\t+\tg\t+\t0M\n"
        "P\tx\ta+,u+,c+,g+\t*\nP\ty\ta+,u+,c+,g+\t*\n"
    )
    result = compare_paths(load_genome_paths(gfa), "x", "y")
    (block,) = result.blocks
    assert (block.bases, block.complete) == (12, False)
    assert result.collinear.bases == 12 and result.collinear.fraction_a == 1
    assert result.collinear.complete is False


def test_overlaps_longer_than_segments_break_contiguity(tmp_path):
    gfa = tmp_path / "long.gfa"
    gfa.write_text("S\ta\tACGT\nS\tb\tACGT\nL\ta\t+\tb\t+\t10M\nP\tx\ta+,b+\t*\n")
    paths = load_genome_paths(gfa)
    layout = paths.layout("x")
    assert layout.contiguous == (False, False) and layout.starts == (0, None)
    result = compare_paths(paths, "x", "x")
    assert all(block.bases >= 0 for block in result.blocks)
    assert result.content.a.uncertain_overlap_steps == (1,)
    with pytest.raises(ValueError, match="max_rotations"):
        compare_paths(paths, "x", "x", circular=True, max_rotations=-1)


def chopped(sequence, prefix):
    names = [f"{prefix}{i}" for i in range(len(sequence))]
    segments = "".join(f"S\t{n}\t{base}\n" for n, base in zip(names, sequence))
    links = "".join(f"L\t{u}\t+\t{v}\t+\t0M\n" for u, v in zip(names, names[1:]))
    return names, segments, links


def summary(result):
    return (
        result.content.shared_bases,
        result.content.union_bases,
        result.collinear.bases,
        sorted(block.bases for block in result.blocks),
    )


def test_renaming_subdivision_and_strand_storage_are_invariant(tmp_path):
    paths_text = "P\tx\ta+,c+,g+\t*\nP\ty\ta+,g+,c+,t+\t*\n"
    coarse = compare_paths(load(tmp_path, paths_text), "x", "y")

    # Same sequences, every segment chopped into renamed one-base segments.
    pieces, segments, links = {}, "", ""
    for name, sequence in (("a", "AAAA"), ("c", "CCCC"), ("g", "GGGG"), ("t", "TTAT")):
        names, s, l = chopped(sequence, "seg_" + name)
        pieces[name], segments, links = names, segments + s, links + l
    for u, v in (("a", "c"), ("c", "g"), ("a", "g"), ("g", "c"), ("g", "t")):
        links += f"L\t{pieces[u][-1]}\t+\t{pieces[v][0]}\t+\t0M\n"

    def walk(*names):
        return ",".join(p + "+" for n in names for p in pieces[n])

    gfa = tmp_path / "chopped.gfa"
    gfa.write_text(
        segments + links + f"P\tx\t{walk('a', 'c', 'g')}\t*\n"
        f"P\ty\t{walk('a', 'g', 'c', 't')}\t*\n"
    )
    fine = compare_paths(load_genome_paths(gfa), "x", "y")
    assert summary(fine) == summary(coarse)
    # Step counts are encoding diagnostics and may differ.
    assert sorted(b.steps for b in fine.blocks) != sorted(
        b.steps for b in coarse.blocks
    )

    # c stored as its reverse complement and traversed on the minus strand.
    strand = SEGMENTS.replace("S\tc\tCCCC", "S\tc\tGGGG")
    flipped_links = (
        LINKS.replace("L\ta\t+\tc\t+", "L\ta\t+\tc\t-")
        .replace("L\tc\t+\tg\t+", "L\tc\t-\tg\t+")
        .replace("L\tg\t+\tc\t+", "L\tg\t+\tc\t-")
        .replace("L\tc\t+\ta\t+", "L\tc\t-\ta\t+")
    )
    gfa = tmp_path / "strand.gfa"
    gfa.write_text(strand + flipped_links + "P\tx\ta+,c-,g+\t*\nP\ty\ta+,g+,c-,t+\t*\n")
    stored = load_genome_paths(gfa)
    assert stored.spell("x").sequence == b"AAAACCCCGGGG"
    assert summary(compare_paths(stored, "x", "y")) == summary(coarse)


def test_circular_policy_allows_rotation_with_closing_evidence(tmp_path):
    paths = load(tmp_path, "P\tx\ta+,c+,g+,t+\t*\nP\ty\tg+,t+,a+,c+\t*\n")
    linear = compare_paths(paths, "x", "y")
    assert linear.collinear.bases == 8
    ring = compare_paths(paths, "x", "y", circular=True)
    assert ring.collinear.bases == 16 and ring.collinear.fraction_b == 1
    assert ring.collinear.exhaustive and ring.collinear.rotation is not None


def test_wrapping_blocks_require_closing_link_evidence(tmp_path):
    text = "P\tx\ta+,c+,g+,t+\t*\nP\ty\tc+,g+,t+,a+\t*\n"
    closed = compare_paths(load(tmp_path, text), "x", "y", circular=True)
    (block,) = closed.collinear.blocks
    assert block.bases == 16 and block.a.wraps and not block.b.wraps
    gfa = tmp_path / "graph.gfa"
    gfa.write_text(gfa.read_text().replace("L\tt\t+\ta\t+\t0M\n", ""))
    opened = load_genome_paths(gfa)
    assert opened.closing_transition("x").status == "missing"
    result = compare_paths(opened, "x", "y", circular=True)
    # Same bases, but the wrap is not contiguous sequence.
    assert result.collinear.bases == 16
    assert sorted(b.steps for b in result.collinear.blocks) == [1, 3]
    assert not any(b.a.wraps for b in result.collinear.blocks)


def test_walk_witnesses_use_source_coordinates(tmp_path):
    gfa = tmp_path / "walks.gfa"
    gfa.write_text(
        "H\tVN:Z:1.1\n" + SEGMENTS + LINKS + "W\ts1\t0\tchr\t100\t112\t>a>c>g\n"
        "W\ts2\t0\tchr\t0\t12\t>a>c>g\n"
    )
    result = compare_paths(load_genome_paths(gfa), "s1#0#chr", "s2#0#chr")
    (block,) = result.blocks
    assert (block.a.start, block.a.end, block.a.coordinates) == (100, 112, "source")
    assert (block.b.start, block.b.end) == (0, 12)


def test_matrix_matches_pairwise_comparisons(tmp_path):
    paths = load(
        tmp_path,
        "P\tx\ta+,c+,a+,g+\t*\nP\ty\ta+,c+,g+\t*\nP\tz\tg-,c-,a-\t*\nP\tw\tt+\t*\n",
    )
    for copies in ("unique", "count"):
        _, details = shared_content_matrix(paths, copies=copies)
        names = details["names"]
        for i, p in enumerate(names):
            for j, q in enumerate(names):
                pair = compare_paths(paths, p, q, copies=copies).content
                assert details["shared"][i, j] == pair.shared_bases
                assert details["union"][i, j] == pair.union_bases
    matrix, details = shared_content_matrix(paths, measure="containment")
    x, y = details["names"].index("x"), details["names"].index("y")
    assert details["containment"][y, x] == 1


def test_drb1_shared_content_and_reverse_stored_haplotype():
    paths = load_genome_paths(DRB1)
    _, details = shared_content_matrix(paths)
    jaccard = details["jaccard"]
    assert np.allclose(jaccard, jaccard.T) and np.allclose(np.diag(jaccard), 1)
    assert all(item.evidence == "complete" for item in details["paths"])
    # In the bundled fixture this haplotype is stored on the opposite strand.
    result = compare_paths(
        paths,
        "gi|568815529:3998044-4011446",
        "gi|345525392:5000-18402",
        orientation="both",
    )
    assert result.content.jaccard > 0.999
    assert result.collinear.bases == 0
    assert result.reverse_collinear.bases == 13401


def test_cli_compare_and_shared_content(tmp_path, capsys):
    load(tmp_path, "P\tx\ta+,c+,g+\t*\nP\ty\ta+,g+,c+\t*\n")
    gfa = str(tmp_path / "graph.gfa")
    main(["compare", gfa, "x", "y"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "gfa2network-path-comparison/1"
    assert payload["content"]["jaccard"] == 1 and payload["collinear"]["bases"] == 8
    assert payload["content"]["a"]["evidence"] == "complete"
    out = tmp_path / "shared.csv"
    main(["shared-content", gfa, "-o", str(out), "--measure", "shared"])
    assert np.loadtxt(out, delimiter=",").tolist() == [[12, 12], [12, 12]]
    metadata = json.loads((tmp_path / "shared.csv.metadata.json").read_text())
    assert metadata["measure"] == "shared" and metadata["input"]["sha256"]
    assert (tmp_path / "shared.csv.paths.tsv").read_text() == "0\tx\n1\ty\n"
    with pytest.raises(SystemExit, match="unknown path: q"):
        main(["compare", gfa, "x", "q"])
