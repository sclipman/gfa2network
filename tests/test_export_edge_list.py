from pathlib import Path
import subprocess
import sys

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t+\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "e.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_export_edge_list(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "edges.tsv"
    subprocess.run([
        sys.executable,
        "-m",
        "gfa2network",
        "export",
        str(gfa),
        "--format",
        "edge-list",
        "--output",
        str(out),
    ], check=True)
    assert out.read_text().strip() == "s1\ts2"
