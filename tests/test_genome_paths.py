"""Rich genome paths: declared steps, transition evidence, spelling and fidelity."""

from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import re
import sys

import pytest

from gfa2network import parse_gfa
from gfa2network.cli import main
from gfa2network.genome_paths import (
    GenomePathSet,
    load_genome_paths,
    overlap_length,
    read_fasta,
)
from gfa2network.model import reverse_complement
from gfa2network.parser import GFAParser, PathRecord

DRB1 = Path(__file__).parent / "data" / "DRB1-3123_unsorted.gfa"


def write(tmp_path, text, name="graph.gfa"):
    path = tmp_path / name
    path.write_text(text)
    return path


def codes(spelled):
    return [issue.code for issue in spelled.issues]


def test_p_records_keep_overlaps_and_link_jump_separators(tmp_path):
    gfa = write(tmp_path, "S\ta\tA\nS\tb\tC\nS\tc\tG\nP\tp\ta+,b-;c+\t0M,*\n")
    record = [r for r in GFAParser(str(gfa)) if isinstance(r, PathRecord)][0]
    assert record.segments == [(b"a", "+"), (b"b", "-"), (b"c", "+")]
    assert record.separators == [",", ";"]
    assert record.overlaps == [b"0M", b"*"]


def test_ordered_oriented_repeated_steps_round_trip(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tAC\nS\tb\tGG\nS\tc\tTT\n"
        "L\ta\t+\tb\t+\t0M\nL\tb\t+\ta\t+\t0M\nL\ta\t+\tc\t-\t0M\n"
        "P\tp\ta+,b+,a+,c-\t*\n",
    )
    paths = load_genome_paths(gfa)
    path = paths["p"]
    assert [(s.segment, s.orientation, s.ordinal) for s in path.steps] == [
        ("a", "+", 0),
        ("b", "+", 1),
        ("a", "+", 2),
        ("c", "-", 3),
    ]
    assert path.vertices() == ["a", "b", "a", "c"]
    assert path.vertices(oriented=True) == ["a:+", "b:+", "a:+", "c:-"]
    assert path.validated and path.transition_counts == {"supported": 3}
    assert paths.spell("p").sequence == b"ACGGACAA"
    for bidirected in (False, True):
        _, legacy = parse_gfa(
            gfa,
            build_graph=True,
            build_matrix=False,
            bidirected=bidirected,
            return_paths=True,
        )
        _, rich = parse_gfa(
            gfa,
            build_graph=True,
            build_matrix=False,
            bidirected=bidirected,
            return_paths=True,
            rich_paths=True,
        )
        assert isinstance(rich, GenomePathSet)
        assert rich.vertex_lists() == legacy
        # Strand is a property of the step, not of the chosen graph projection.
        assert [s.orientation for s in rich["p"].steps] == ["+", "+", "+", "-"]


def test_reverse_strand_relation_supports_forward_traversal(tmp_path):
    # L b- a- 0M is the same adjacency as a+ -> b+ read on the other strand.
    gfa = write(tmp_path, "S\ta\tAAAC\nS\tb\tGGT\nL\tb\t-\ta\t-\t0M\nP\tp\ta+,b+\t*\n")
    paths = load_genome_paths(gfa)
    assert paths["p"].transitions[0].evidence == ("L:3",)
    assert paths.spell("p").sequence == b"AAACGGT"


def test_reverse_complement_steps_are_spelled_on_their_strand(tmp_path):
    gfa = write(tmp_path, "S\ta\tAAAC\nS\tb\tGGT\nL\ta\t+\tb\t-\t0M\nP\tp\ta+,b-\t*\n")
    assert load_genome_paths(gfa).spell("p").sequence == b"AAAC" + b"ACC"


@pytest.mark.parametrize(
    "cigar,expected",
    [
        (None, ("unspecified", None)),
        (b"*", ("unspecified", None)),
        (b"0M", ("exact", 0)),
        (b"3M2=", ("exact", 5)),
        (b"2M1I2M", ("unsupported", None)),
        (b"4X", ("unsupported", None)),
        (b"1,2,3", ("unsupported", None)),
    ],
)
def test_overlap_classification(cigar, expected):
    assert overlap_length(cigar) == expected


def test_exact_overlaps_are_verified_before_merging(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tACGTA\nS\tb\tTACCC\nS\tc\tGGGGG\n"
        "L\ta\t+\tb\t+\t2M\nL\ta\t+\tc\t+\t2M\n"
        "P\tgood\ta+,b+\t*\nP\tbad\ta+,c+\t*\n",
    )
    paths = load_genome_paths(gfa)
    assert paths.spell("good").sequence == b"ACGTACCC"
    bad = paths.spell("bad")
    assert bad.status == "partial" and codes(bad) == ["overlap_mismatch"]
    assert [p.sequence for p in bad.pieces] == [b"ACGTA", b"GGGGG"]


def test_overlap_cannot_reach_past_the_previous_segment(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tACG\nS\tb\tT\nS\tc\tGTAA\n"
        "L\ta\t+\tb\t+\t0M\nL\tb\t+\tc\t+\t2M\nP\tp\ta+,b+,c+\t*\n",
    )
    spelled = load_genome_paths(gfa).spell("p")
    assert codes(spelled) == ["overlap_exceeds_sequence"]
    assert [p.sequence for p in spelled.pieces] == [b"ACGT", b"GTAA"]


def test_unknown_overlap_abstains_unless_policy_is_declared(tmp_path):
    gfa = write(tmp_path, "S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t+\t*\nP\tp\ta+,b+\t*\n")
    paths = load_genome_paths(gfa)
    strict = paths.spell("p")
    assert strict.status == "partial" and codes(strict) == ["unknown_overlap"]
    blunt = paths.spell("p", unknown_overlap="zero")
    assert blunt.sequence == b"ACGT" and blunt.assumed_overlaps == (0,)
    with pytest.raises(ValueError, match="strict or zero"):
        paths.spell("p", unknown_overlap="guess")


def test_path_overlap_field_resolves_or_conflicts_with_links(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tACGT\nS\tb\tGTAA\n"
        "L\ta\t+\tb\t+\t0M\nL\ta\t+\tb\t+\t2M\n"
        "P\tunresolved\ta+,b+\t*\n"
        "P\tresolved\ta+,b+\t2M\n"
        "P\tconflict\ta+,b+\t1M\n"
        "P\tunsupported\ta+,b+\t1M1I\n",
    )
    paths = load_genome_paths(gfa)
    assert paths["unresolved"].transitions[0].status == "ambiguous"
    assert paths["resolved"].transitions[0].overlap_length == 2
    assert paths.spell("resolved").sequence == b"ACGTAA"
    assert paths["conflict"].transitions[0].status == "conflict"
    assert codes(paths.spell("conflict")) == ["conflict_transition"]
    assert codes(paths.spell("unsupported")) == ["unsupported_overlap"]


def test_missing_transition_splits_the_spelling(tmp_path):
    gfa = write(
        tmp_path, "S\ta\tAA\nS\tb\tCC\nS\tc\tGG\nL\ta\t+\tb\t+\t0M\nP\tp\ta+,c+,b+\t*\n"
    )
    paths = load_genome_paths(gfa)
    assert [t.status for t in paths["p"].transitions] == ["missing", "missing"]
    assert not paths["p"].validated
    spelled = paths.spell("p")
    assert spelled.status == "partial" and len(spelled.pieces) == 3
    assert codes(spelled) == ["missing_transition", "missing_transition"]


def test_jumps_are_gaps_not_contiguous_sequence(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:1.2\nS\ta\tAA\nS\tb\tCC\nJ\ta\t+\tb\t+\t50\n"
        "P\tjump\ta+;b+\t*\nP\tlink\ta+,b+\t*\n",
    )
    paths = load_genome_paths(gfa)
    jump = paths["jump"].transitions[0]
    assert (jump.status, jump.relation, jump.distance) == ("supported", "jump", 50)
    spelled = paths.spell("jump")
    assert spelled.status == "partial" and codes(spelled) == ["jump"]
    assert spelled.failures == ()
    link = paths["link"].transitions[0]
    assert link.status == "missing" and "found jump" in link.detail


def test_missing_sequence_is_never_invented(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tAA\nS\tb\t*\tLN:i:4\nS\tc\tGG\n"
        "L\ta\t+\tb\t+\t0M\nL\tb\t+\tc\t+\t0M\nP\tp\ta+,b+,c+\t*\n",
    )
    spelled = load_genome_paths(gfa).spell("p")
    assert spelled.status == "partial"
    assert codes(spelled) == ["missing_sequence"]
    assert [p.sequence for p in spelled.pieces] == [b"AA", b"GG"]


def test_overlap_count_mismatch_is_reported_and_links_are_used(tmp_path):
    gfa = write(tmp_path, "S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t+\t0M\nP\tp\ta+,b+\t0M,0M\n")
    paths = load_genome_paths(gfa)
    spelled = paths.spell("p")
    assert spelled.sequence == b"ACGT"
    assert codes(spelled) == ["overlap_count_mismatch"]
    assert not paths.audit().ok


def test_walk_chunks_keep_identity_coordinates_and_gaps(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:1.1\nS\ta\tAAA\nS\tb\tCC\nS\tc\tGGGG\nS\td\tTT\n"
        "L\ta\t+\tb\t+\t0M\nL\tc\t+\td\t+\t0M\n"
        "W\tHG002\t1\tchr1\t10\t16\t>c>d\n"
        "W\tHG002\t1\tchr1\t0\t5\t>a>b\n",
    )
    paths = load_genome_paths(gfa)
    path = paths["HG002#1#chr1"]
    assert (path.record_type, path.sample, path.haplotype, path.sequence_id) == (
        "W",
        "HG002",
        1,
        "chr1",
    )
    assert [(c.start, c.end, c.first_step, c.last_step) for c in path.chunks] == [
        (0, 5, 0, 1),
        (10, 16, 2, 3),
    ]
    assert [t.status for t in path.transitions] == [
        "supported",
        "boundary",
        "supported",
    ]
    spelled = paths.spell("HG002#1#chr1")
    assert [p.sequence for p in spelled.pieces] == [b"AAACC", b"GGGGTT"]
    assert spelled.issues[0].code == "chunk_gap"
    assert spelled.issues[0].detail == "5 bases"
    assert spelled.failures == ()


def test_adjacent_walk_chunks_join_and_coordinates_are_checked(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:1.1\nS\ta\tAAA\nS\tb\tCC\n"
        "W\ts\t0\tseq\t0\t3\t>a\nW\ts\t0\tseq\t3\t5\t>b\n"
        "W\tt\t0\tseq\t0\t4\t>a\n",
    )
    paths = load_genome_paths(gfa)
    assert paths.spell("s#0#seq").sequence == b"AAACC"
    wrong = paths.spell("t#0#seq")
    assert codes(wrong) == ["coordinate_length_mismatch"]


def test_audit_compares_with_fasta_under_declared_equivalences(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tAAAC\nS\tb\tGGT\nL\ta\t+\tb\t+\t0M\n"
        "P\tsame\ta+,b+\t*\nP\trc\ta+,b+\t*\nP\tring\ta+,b+\t*\nP\tother\ta+,b+\t*\n",
    )
    fasta = tmp_path / "refs.fa"
    rotated = b"AAACGGT"[3:] + b"AAACGGT"[:3]
    fasta.write_bytes(
        b">same desc\nAAAC\nGGT\n>rc\n"
        + reverse_complement(b"AAACGGT")
        + b"\n>ring\n"
        + rotated
        + b"\n>other\nAAACGGA\n"
    )
    paths = load_genome_paths(gfa)
    strict = {p.name: p for p in paths.audit(fasta).paths}
    assert strict["same"].comparison == "identical"
    assert strict["rc"].comparison == "different"
    assert strict["ring"].comparison == "different"
    assert strict["other"].comparison == "different"
    assert strict["other"].first_difference == 6
    declared = paths.audit(
        str(fasta), circular=["ring"], reverse_complement=True
    ).to_dict()
    result = {p["name"]: p for p in declared["paths"]}
    assert result["rc"]["comparison"] == "reverse_complement"
    assert result["ring"]["comparison"] == "rotation"
    assert result["ring"]["offset"] == 4
    assert declared["policy"]["circular"] == ["ring"]
    assert declared["summary"]["passed"] == 3 and not declared["ok"]
    json.dumps(declared)


def test_audit_locates_pieces_and_walk_coordinates(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:1.2\nS\ta\tAAAC\nS\tb\tGGT\nJ\ta\t+\tb\t+\t3\n"
        "P\tgapped\ta+;b+\t*\n"
        "W\ts\t0\tchr\t2\t6\t>a\n",
    )
    references = {"gapped": "AAACNNNGGT", "s#0#chr": "TTAAACTT"}
    report = load_genome_paths(gfa).audit(references)
    result = {p.name: p for p in report.paths}
    assert result["gapped"].comparison == "pieces_located"
    assert result["s#0#chr"].comparison == "coordinates_match"
    assert result["s#0#chr"].chunks == [
        {"start": 2, "end": 6, "matches": True, "spelled_length": 4}
    ]
    assert report.ok
    shifted = load_genome_paths(gfa).audit({"s#0#chr": "AAACTTTT"}, names=["s#0#chr"])
    assert shifted.paths[0].comparison == "coordinates_differ"


def test_supplied_references_must_cover_audited_paths(tmp_path):
    gfa = write(tmp_path, "S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t+\t0M\nP\tp\ta+,b+\t*\n")
    paths = load_genome_paths(gfa)
    assert paths.audit().paths[0].comparison == "no_reference"
    report = paths.audit({"other": "ACGT"})
    assert report.paths[0].comparison == "reference_missing" and not report.ok
    mapped = paths.audit({"other": "ACGT"}, reference_names={"p": "other"})
    assert mapped.paths[0].comparison == "identical" and mapped.ok


def test_circular_pieces_cannot_overlap_after_wrapping(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:1.2\nS\ta\tAAACG\nS\tb\tCGGT\nJ\ta\t+\tb\t+\t3\nP\tp\ta+;b+\t*\n",
    )
    paths = load_genome_paths(gfa)
    # The pieces span 9 bases; on a 7-base circle they would share bases.
    report = paths.audit({"p": "AAACGGT"}, circular=["p"])
    assert report.paths[0].comparison == "pieces_unlocated"
    wrapped = paths.audit({"p": "GGTTAAACGTTC"}, circular=["p"])
    assert wrapped.paths[0].comparison == "pieces_located"


def test_circular_closing_overlap_is_trimmed_only_with_evidence(tmp_path):
    gfa = write(
        tmp_path,
        "S\ta\tACGT\nS\tb\tTTAC\nL\ta\t+\tb\t+\t1M\nL\tb\t+\ta\t+\t2M\n"
        "P\tring\ta+,b+\t*\n",
    )
    report = load_genome_paths(gfa).audit({"ring": "GTTAC"}, circular=True)
    path = report.paths[0]
    # ACGT + TAC = ACGTTAC; the evidenced 2 bp closing overlap removes the
    # trailing AC, leaving the circular sequence ACGTT.
    assert path.closing == {"status": "supported", "overlap_length": 2, "trimmed": 2}
    assert path.comparison == "rotation" and path.passed


def test_gfa2_dovetails_and_gaps(tmp_path):
    gfa = write(
        tmp_path,
        "H\tVN:Z:2.0\nS\ta\t5\tACGTA\nS\tb\t5\tTACCC\nS\tc\t3\tGGG\n"
        "E\te1\ta+\tb+\t3\t5$\t0\t2\t2M\n"
        "E\te2\tc-\tb-\t0\t0\t5\t5$\t0M\n"
        "G\tg1\ta+\tc+\t10\t*\n"
        "O\tlinked\ta+ b+ c+\nO\tgapped\ta+ c+\n",
    )
    paths = load_genome_paths(gfa)
    # e2 (c- then b-) is the reverse-strand form of b+ -> c+.
    assert [t.evidence for t in paths["linked"].transitions] == [("E:5",), ("E:6",)]
    assert paths.spell("linked").sequence == b"ACGTACCCGGG"
    gapped = paths["gapped"].transitions[0]
    assert (gapped.relation, gapped.distance, gapped.evidence) == (
        "jump",
        10,
        ("G:7",),
    )


@pytest.mark.parametrize(
    "segments,edge,group",
    [
        # Internal alignment of a whole short segment.
        ("S\ta\t6\tACGTAC\nS\tb\t2\tGT\n", "a+\tb+\t2\t4\t0\t2$\t2M", "a+ b+"),
        # a is contained at the start of b; its interval is a suffix and prefix.
        ("S\ta\t2\tAC\nS\tb\t4\tACGT\n", "a+\tb+\t0\t2$\t0\t2\t2M", "a+ b+"),
        # Coordinates beyond both segment ends.
        ("S\ta\t3\tACG\nS\tb\t3\tCGT\n", "a-\tb+\t0\t5\t0\t5\t5M", "a- b+"),
    ],
)
def test_containments_and_invalid_alignments_are_not_traversals(
    tmp_path, segments, edge, group
):
    gfa = write(
        tmp_path,
        "H\tVN:Z:2.0\n" + segments + "E\t*\t" + edge + "\nO\tp\t" + group + "\n",
    )
    path = load_genome_paths(gfa)["p"]
    assert path.transitions[0].status == "missing" and not path.validated


def test_split_paths_keep_parent_intervals_and_source_spelling(tmp_path, backend):
    gfa = write(
        tmp_path,
        "S\ta\tACGTAC\nS\tb\tTTGG\nS\tc\tCC\n"
        "C\ta\t+\tb\t+\t1\t4M\nL\ta\t+\tc\t+\t0M\n"
        "P\tp\ta+,c+\t*\nP\tq\tc-,a-\t*\n",
    )
    opts = dict(
        build_graph=True,
        build_matrix=False,
        split_on_alignment=True,
        return_paths=True,
        backend=backend,
    )
    _, legacy = parse_gfa(gfa, **opts)
    _, rich = parse_gfa(gfa, rich_paths=True, **opts)
    assert rich.subdivided and rich.vertex_lists() == legacy
    first = rich["p"].steps[0]
    assert first.segment == "a"
    assert [(i.start, i.end) for i in first.intervals] == [(0, 1), (1, 5), (5, 6)]
    reverse = rich["q"].steps[1]
    assert [(i.start, i.end) for i in reverse.intervals] == [(5, 6), (1, 5), (0, 1)]
    assert rich.spell("p").sequence == b"ACGTACCC"
    assert rich.spell("q").sequence == reverse_complement(b"ACGTACCC")


@pytest.fixture(params=["networkx", "igraph"])
def backend(request):
    if request.param == "igraph":
        pytest.importorskip("igraph")
    return request.param


def test_backends_and_byte_ids_share_rich_paths(tmp_path, backend):
    gfa = write(tmp_path, "S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t-\t0M\nP\tp\ta+,b-\t*\n")
    _, rich = parse_gfa(
        gfa,
        build_graph=True,
        build_matrix=False,
        backend=backend,
        bidirected=True,
        raw_bytes_id=True,
        return_paths=True,
        rich_paths=True,
    )
    assert list(rich) == [b"p"]
    assert rich.vertex_lists() == {b"p": [b"a:+", b"b:-"]}
    assert rich[b"p"].steps[1].segment == b"b"
    assert rich.spell(b"p").sequence == b"ACAC"


def test_stdin_single_pass(monkeypatch):
    text = b"S\ta\tAC\nS\tb\tGT\nL\ta\t+\tb\t+\t0M\nP\tp\ta+,b+\t*\n"
    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(text)))
    graph, rich = parse_gfa(
        "-", build_graph=True, build_matrix=False, return_paths=True, rich_paths=True
    )
    assert graph.number_of_edges() == 1 and rich.spell("p").sequence == b"ACGT"
    monkeypatch.setattr(sys, "stdin", TextIOWrapper(BytesIO(text)))
    assert load_genome_paths("-").spell("p").sequence == b"ACGT"


def test_declared_failures_match_the_list_api(tmp_path):
    undefined = write(tmp_path, "S\ta\tA\nP\tp\ta+,z+\t*\n", "undefined.gfa")
    with pytest.raises(ValueError, match="undefined segment"):
        load_genome_paths(undefined)
    unsupported = write(tmp_path, "S\ta\tA\nX\tsomething\nP\tp\ta+\t*\n", "x.gfa")
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="unsupported records: X"):
            load_genome_paths(unsupported)
    duplicate = write(tmp_path, "S\ta\tA\nS\ta\tC\n", "dup.gfa")
    with pytest.raises(ValueError, match=r"dup\.gfa:2: duplicate segment"):
        load_genome_paths(duplicate)
    with pytest.raises(ValueError, match="requires return_paths"):
        parse_gfa(duplicate, build_graph=True, build_matrix=False, rich_paths=True)
    structure = load_genome_paths(
        write(tmp_path, "S\ta\tA\nP\tp\ta+\t*\n", "s.gfa"), sequences=False
    )
    with pytest.raises(ValueError, match="sequences=True"):
        structure.spell("p")


def test_drb1_paths_are_validated_and_spelled():
    paths = load_genome_paths(DRB1)
    _, legacy = parse_gfa(DRB1, build_graph=True, build_matrix=False, return_paths=True)
    assert paths.vertex_lists() == legacy
    report = paths.audit()
    assert report.ok and report.summary["spelling"] == {"complete": 12}
    assert report.summary["transitions"] == {"supported": 21870}
    for name, path in paths.items():
        # Repeated visits survive; the list API keeps them and distances dedupe.
        assert len(path.steps) >= len({s.segment for s in path.steps})
        start, end = map(int, re.search(r":(\d+)-(\d+)$", name).groups())
        # Path names carry 1-based inclusive source intervals.
        assert len(paths.spell(name).sequence) == end - start + 1


def test_read_fasta_rejects_ambiguous_input(tmp_path):
    good = tmp_path / "good.fa"
    good.write_text(">x one\nAC\nGT\n\n>y\nTT\n")
    assert read_fasta(good) == {"x": b"ACGT", "y": b"TT"}
    duplicate = tmp_path / "dup.fa"
    duplicate.write_text(">x\nA\n>x\nC\n")
    with pytest.raises(ValueError, match="duplicate FASTA name"):
        read_fasta(duplicate)
    headless = tmp_path / "headless.fa"
    headless.write_text("ACGT\n")
    with pytest.raises(ValueError, match="before the first"):
        read_fasta(headless)


def test_paths_cli_writes_receipt_and_fasta(tmp_path, capsys):
    gfa = write(
        tmp_path,
        "S\ta\tAAAC\nS\tb\tGGT\nL\ta\t+\tb\t+\t0M\nP\tp\ta+,b+\t*\nP\tq\tb+,a+\t*\n",
    )
    refs = tmp_path / "refs.fa"
    refs.write_text(">p\nAAACGGT\n")
    receipt, spelled = tmp_path / "receipt.json", tmp_path / "spelled.fa"
    main(
        [
            "paths",
            str(gfa),
            "--reference",
            str(refs),
            "-o",
            str(receipt),
            "--fasta",
            str(spelled),
        ]
    )
    payload = json.loads(receipt.read_text())
    assert payload["format"] == "gfa2network-path-fidelity/1"
    assert payload["input"]["sha256"] and payload["reference"]["sha256"]
    assert payload["summary"]["comparison"] == {"identical": 1, "reference_missing": 1}
    assert not payload["ok"]
    assert read_fasta(spelled) == {"p": b"AAACGGT"}
    with pytest.raises(SystemExit, match="1 of 2 paths"):
        main(["paths", str(gfa), "--strict", "-o", str(receipt)])
    main(["paths", str(gfa), "--path", "p", "--strict"])
    assert json.loads(capsys.readouterr().out)["summary"]["paths"] == 1
    with pytest.raises(SystemExit, match="unknown path: z"):
        main(["paths", str(gfa), "--path", "z"])
