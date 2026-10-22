"""The topology report example runs end to end on a small labeled graph."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "topology_report.py"


def load_script():
    spec = importlib.util.spec_from_file_location("topology_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_case(tmp_path):
    segments = {
        "a": "ACGTAC",
        "b": "GGATCC",
        "c": "TTGACA",
        "d": "CAGTTG",
        "e": "AACCGG",
    }
    paths = {
        "g1#x1": ["a", "b", "c"],
        "g1#x2": ["a", "b", "d"],
        "g2#y1": ["e", "b", "c"],
        "g2#y2": ["e", "b", "d"],
    }
    links = {(u, v) for walk in paths.values() for u, v in zip(walk, walk[1:])}
    links |= {(walk[-1], walk[0]) for walk in paths.values()}
    gfa = tmp_path / "graph.gfa"
    gfa.write_text(
        "".join(f"S\t{k}\t{v}\n" for k, v in segments.items())
        + "".join(f"L\t{u}\t+\t{v}\t+\t0M\n" for u, v in sorted(links))
        + "".join(
            f"P\t{name}\t{','.join(s + '+' for s in walk)}\t*\n"
            for name, walk in paths.items()
        )
    )
    fasta = tmp_path / "genomes.fa"
    fasta.write_text(
        "".join(
            f">{name}\n{''.join(segments[s] for s in walk)}\n"
            for name, walk in paths.items()
        )
    )
    return gfa, fasta


@pytest.mark.parametrize("circular", [False, True])
def test_report_writes_every_output(tmp_path, circular):
    gfa, fasta = write_case(tmp_path)
    out = tmp_path / "report"
    argv = [
        "--gfa", str(gfa), "--reference", str(fasta), "--label-regex", r"^(g\d)#",
        "--subset-per-group", "2", "--min-bases", "6", "--kmer", "4",
        "--chop", "2", "--out", str(out),
    ] + (["--circular"] if circular else [])  # fmt: skip
    load_script().main(argv)
    for name in (
        "manifest.json",
        "fidelity.json",
        "content_groups.tsv",
        "subset.tsv",
        "subset_pairs.tsv",
        "bridges.json",
        "bridge_sizes.tsv",
        "sensitivity.tsv",
        "summary.md",
    ):
        assert (out / name).exists(), name
    fidelity = json.loads((out / "fidelity.json").read_text())
    assert fidelity["classes"] == {"identical": 4}
    bridges = json.loads((out / "bridges.json").read_text())
    shared = {tuple(b["segments"]) for b in bridges["blocks"]}
    assert ("b+",) in shared  # b is visited by both groups
    rows = [
        line.split("\t") for line in (out / "sensitivity.tsv").read_text().splitlines()
    ]
    for row in rows[1:]:
        assert row[2] == "0" and row[3] == "0" and row[5] == "True", row


def test_report_requires_labels(tmp_path):
    gfa, fasta = write_case(tmp_path)
    with pytest.raises(SystemExit):
        load_script().main(["--gfa", str(gfa), "--out", str(tmp_path / "r")])


def test_report_survives_a_subset_too_small_to_pair(tmp_path):
    gfa, fasta = write_case(tmp_path)
    out = tmp_path / "small"
    load_script().main(
        [
            "--gfa",
            str(gfa),
            "--label-regex",
            r"^(g1)#x1",
            "--out",
            str(out),
            "--chop",
            "0",
        ]
    )
    assert (out / "summary.md").exists()
    assert (out / "subset_pairs.tsv").read_text().count("\n") == 1
