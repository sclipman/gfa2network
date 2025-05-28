import sys
import subprocess
import re
import sys
from pathlib import Path
import pytest

SAMPLE_GFA = b"""S\ts1\t4\nS\ts2\t4\nL\ts1\t+\ts2\t-\t0M\n"""


def write_gfa(tmp_path: Path) -> Path:
    gfa = tmp_path / "sample.gfa"
    gfa.write_bytes(SAMPLE_GFA)
    return gfa


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="timing output differs on Windows"
)
def test_verbose_timing(tmp_path: Path):
    gfa = write_gfa(tmp_path)
    out = tmp_path / "adj.npz"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gfa2network",
            "convert",
            str(gfa),
            "--matrix",
            str(out),
            "--verbose",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pattern = r"Parsed in \d+\.\d{3} s"
    assert re.search(pattern, result.stdout)
    pattern = r"Built (graph|matrix|graph and matrix) in \d+\.\d{3} s"
    assert re.search(pattern, result.stdout)
    pattern = r"Exported in \d+\.\d{3} s"
    assert re.search(pattern, result.stdout)
