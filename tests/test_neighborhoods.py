"""Bridge blocks: contiguous sequence shared across labeled path groups."""

from gfa2network.genome_paths import load_genome_paths
from gfa2network.neighborhoods import bridge_blocks
from gfa2network.transform import chop

SEGMENTS = "".join(
    f"S\t{name}\t{seq}\n"
    for name, seq in (
        ("a", "AAAA"),
        ("b", "CCCC"),
        ("c", "GGGG"),
        ("d", "TTTT"),
        ("e", "ACAC"),
        ("f", "AGAG"),
        ("h", "ATAT"),
        ("x", "CACA"),
        ("y", "CTCT"),
    )
)


def graph(tmp_path, paths):
    links = set()
    for line in paths.splitlines():
        steps = line.split("\t")[2].split(",")
        for u, v in zip(steps, steps[1:]):
            links.add(f"L\t{u[:-1]}\t{u[-1]}\t{v[:-1]}\t{v[-1]}\t0M\n")
    gfa = tmp_path / "groups.gfa"
    gfa.write_text(SEGMENTS + "".join(sorted(links)) + paths)
    return load_genome_paths(gfa)


def test_blocks_contexts_and_reverse_traversals(tmp_path):
    paths = graph(
        tmp_path,
        "P\tp1\tx+,a+,b+,c+,y+\t*\n"
        "P\tp2\tx+,a+,b+,f+\t*\n"
        "P\tq1\td+,a+,b+,c+,e+\t*\n"
        "P\tq2\tc-,b-,a-,d-\t*\n"
        "P\tr1\ta+,h+,b+\t*\n"
        "P\tother\tx+,a+,b+\t*\n",
    )
    groups = {"p1": "G1", "p2": "G1", "q1": "G2", "q2": "G2"}
    blocks = {block.segments: block for block in bridge_blocks(paths, groups)}
    # x is shared only within G1; the unlabeled paths are ignored.
    assert set(blocks) == {("a+", "b+"), ("c+",)}
    ab = blocks[("a+", "b+")]
    assert (ab.bases, ab.sequence) == (8, b"AAAACCCC")
    assert ab.paths == ("p1", "p2", "q1", "q2") and ab.groups == ("G1", "G2")
    # r1 visits a and b without the a->b link, so they become separate blocks.
    split = {b.segments for b in bridge_blocks(paths, dict(groups, r1="G2"))}
    assert split == {("a+",), ("b+",), ("c+",)}
    reverse = next(t for t in ab.traversals if t.path == "q2")
    assert reverse.witness.reverse
    assert (reverse.witness.first_step, reverse.witness.last_step) == (1, 2)
    contexts = {(c["left"], c["right"]): c for c in ab.contexts}
    assert contexts[("d+", "c+")]["paths"] == ["q1", "q2"]
    assert contexts[("x+", "c+")]["groups"] == ["G1"]
    assert ab.group_specific_contexts
    witness = next(t.witness for t in ab.traversals if t.path == "p1")
    assert (witness.start, witness.end, witness.coordinates) == (4, 12, "path")
    data = ab.to_dict()
    assert data["sequence"] == "AAAACCCC" and data["group_specific_contexts"]


def test_a_context_shared_across_groups_is_reported(tmp_path):
    paths = graph(
        tmp_path,
        "P\tp\tx+,a+,b+,y+\t*\nP\tq\tx+,a+,b+,y+\t*\n",
    )
    (block,) = bridge_blocks(paths, {"p": "G1", "q": "G2"})
    assert block.segments == ("x+", "a+", "b+", "y+")
    assert not block.group_specific_contexts
    assert block.contexts[0]["groups"] == ["G1", "G2"]


def test_noncontiguous_supporters_are_partial_not_members(tmp_path):
    paths = graph(
        tmp_path,
        "P\tp\ta+,b+,c+\t*\nP\tq\ta+,b+,c+\t*\nP\tw\ta+,b+,h+,b+,c+\t*\n",
    )
    (block,) = bridge_blocks(paths, {"p": "G1", "q": "G2", "w": "G2"})
    assert block.segments == ("a+", "b+", "c+")
    assert block.paths == ("p", "q") and block.partial == ("w",)
    # Without q, w does not read a,b,c as one run (it revisits b). The shared
    # sequence is kept as smaller blocks rather than lost; how the run splits
    # depends on which path the file lists first.
    groups = {"p": "G1", "w": "G2"}
    found = {b.segments: b.paths for b in bridge_blocks(paths, groups)}
    assert found == {("a+",): ("p", "w"), ("b+", "c+"): ("p", "w")}
    (tmp_path / "first").mkdir()
    reordered = graph(
        tmp_path / "first", "P\tw\ta+,b+,h+,b+,c+\t*\nP\tp\ta+,b+,c+\t*\n"
    )
    found = {b.segments: b.paths for b in bridge_blocks(reordered, groups)}
    assert found == {("a+", "b+"): ("p", "w"), ("c+",): ("p", "w")}


def test_min_groups_and_min_bases(tmp_path):
    paths = graph(
        tmp_path,
        "P\tp\ta+,b+\t*\nP\tq\ta+,b+\t*\nP\tr\ta+\t*\n",
    )
    groups = {"p": "G1", "q": "G2", "r": "G3"}
    assert [b.segments for b in bridge_blocks(paths, groups, min_groups=3)] == [("a+",)]
    assert bridge_blocks(paths, groups, min_groups=3, min_bases=5) == []
    # r visits only a, so a and b are separate 4 bp blocks.
    assert bridge_blocks(paths, groups, min_bases=5) == []
    pair = {"p": "G1", "q": "G2"}
    assert [b.bases for b in bridge_blocks(paths, pair, min_bases=8)] == [8]


def test_walk_witnesses_use_source_coordinates(tmp_path):
    gfa = tmp_path / "walks.gfa"
    gfa.write_text(
        "H\tVN:Z:1.1\n" + SEGMENTS + "L\ta\t+\tb\t+\t0M\n"
        "W\ts1\t0\tchr\t100\t108\t>a>b\nW\ts2\t0\tchr\t0\t8\t>a>b\n"
    )
    paths = load_genome_paths(gfa)
    (block,) = bridge_blocks(paths, {"s1#0#chr": "G1", "s2#0#chr": "G2"})
    starts = {t.path: (t.witness.start, t.witness.end) for t in block.traversals}
    assert starts == {"s1#0#chr": (100, 108), "s2#0#chr": (0, 8)}


def test_blocks_do_not_depend_on_segmentation(tmp_path):
    paths = graph(
        tmp_path,
        "P\tp\tx+,a+,b+,c+,y+\t*\nP\tq\td+,a+,b+,c+,e+\t*\nP\tr\th+,b+,f+\t*\n",
    )
    groups = {"p": "G1", "q": "G2", "r": "G2"}
    chopped = tmp_path / "chopped.gfa"
    chop(tmp_path / "groups.gfa", chopped, 1)
    pieces = load_genome_paths(chopped)

    def summary(collection):
        return sorted(
            (b.sequence, b.paths, b.partial) for b in bridge_blocks(collection, groups)
        )

    # r shares only b, so b is its own block in both representations.
    assert summary(paths) == summary(pieces)
    assert [s for s, _, _ in summary(paths)] == [b"AAAA", b"CCCC", b"GGGG"]
