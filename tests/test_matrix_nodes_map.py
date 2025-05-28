from pathlib import Path
import subprocess
import sys
import scipy.sparse as sp

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""

def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


def test_matrix_node_map(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "adj.npz"
    subprocess.run([
        sys.executable,
        "-m",
        "gfa2network",
        "convert",
        str(gfa),
        "--matrix",
        str(out),
    ], check=True)
    A = sp.load_npz(out)
    map_file = Path(str(out) + ".nodes.tsv")
    assert map_file.exists()
    lines = map_file.read_text().strip().splitlines()
    assert len(lines) == A.shape[0]
    for i, line in enumerate(lines):
        idx, node = line.split("\t")
        assert int(idx) == i
