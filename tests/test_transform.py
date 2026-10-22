"""Sequence-preserving chop and rename transforms."""

from pathlib import Path

import numpy as np
import pytest

from gfa2network import parse_gfa
from gfa2network.genome_paths import load_genome_paths
from gfa2network.path_compare import compare_paths, shared_content_matrix
from gfa2network.transform import chop, rename

DRB1 = Path(__file__).parent / "data" / "DRB1-3123_unsorted.gfa"


def spelled(paths):
    return {name: paths.spell(name).sequence for name in paths}


def test_chop_and_rename_preserve_sequence_level_results(tmp_path):
    chopped, renamed = tmp_path / "chopped.gfa", tmp_path / "renamed.gfa"
    assert chop(DRB1, chopped, 3) > 0
    mapping = rename(chopped, renamed, prefix="n")
    original = load_genome_paths(DRB1)
    for path in (chopped, renamed):
        other = load_genome_paths(path)
        assert spelled(other) == spelled(original)
        _, mine = shared_content_matrix(original)
        _, theirs = shared_content_matrix(other)
        np.testing.assert_array_equal(mine["shared"], theirs["shared"])
    names = list(original)[:2]
    before = compare_paths(original, *names)
    after = compare_paths(load_genome_paths(renamed), *names)
    assert after.collinear.bases == before.collinear.bases
    assert sorted(b.bases for b in after.blocks) == sorted(
        b.bases for b in before.blocks
    )
    assert len(mapping) > 3214
    # Vertex-level statistics change with the representation.
    graph = parse_gfa(chopped, build_graph=True, build_matrix=False)
    assert graph.number_of_nodes() > 3214


def test_chop_handles_orientation_jumps_and_walks(tmp_path):
    source = tmp_path / "in.gfa"
    source.write_text(
        "H\tVN:Z:1.2\nS\ta\tACGTACG\nS\tb\tTTGCA\tLN:i:5\nS\tc\t*\tLN:i:9\n"
        "L\ta\t+\tb\t-\t0M\nJ\tb\t-\tc\t+\t40\n"
        "P\tp\ta+,b-;c+\t0M,*\nW\ts\t0\tchr\t0\t12\t<b<a\n"
    )
    out = tmp_path / "out.gfa"
    assert chop(source, out, 3) == 2
    text = out.read_text()
    assert "S\tb.1\tTTG\nS\tb.2\tCA\n" in text and "LN:i:5" not in text
    assert "S\tc\t*\tLN:i:9" in text  # no sequence: left whole
    paths = load_genome_paths(out)
    original = load_genome_paths(source)
    assert spelled(paths) == spelled(original)
    p = paths["p"]
    assert [t.relation for t in p.transitions][-1] == "jump"
    assert p.vertices(oriented=True)[:4] == ["a.1:+", "a.2:+", "a.3:+", "b.2:-"]


def test_transforms_refuse_what_they_cannot_preserve(tmp_path):
    overlap = tmp_path / "overlap.gfa"
    overlap.write_text("S\ta\tACGTA\nS\tb\tTACCC\nL\ta\t+\tb\t+\t2M\n")
    with pytest.raises(ValueError, match="blunt links"):
        chop(overlap, tmp_path / "x.gfa", 2)
    gfa2 = tmp_path / "gfa2.gfa"
    gfa2.write_text("H\tVN:Z:2.0\nS\ta\t4\tACGT\nE\te\ta+\ta+\t0\t0\t4\t4$\t0M\n")
    for transform in (chop, rename):
        with pytest.raises(ValueError, match="E records are not supported"):
            transform(gfa2, tmp_path / "y.gfa", *((2,) if transform is chop else ()))
    clash = tmp_path / "clash.gfa"
    clash.write_text("S\ta\tACGT\nS\ta.1\tA\n")
    with pytest.raises(ValueError, match="already exists"):
        chop(clash, tmp_path / "z.gfa", 2)
    with pytest.raises(ValueError, match="positive integer"):
        chop(clash, tmp_path / "z.gfa", 0)
