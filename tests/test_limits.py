from pathlib import Path
from pathlib import Path
import pytest
from gfa2network.cli import main


def write_gfa(path: Path, n: int) -> Path:
    lines = []
    for i in range(n):
        lines.append(f"S\t{i}\t*\n")
    for i in range(n - 1):
        lines.append(f"L\t{i}\t+\t{i+1}\t+\t0M\n")
    gfa = path / "big.gfa"
    gfa.write_text("".join(lines))
    return gfa


def test_dense_matrix_limit(tmp_path: Path):
    gfa = write_gfa(tmp_path, 400)
    out = tmp_path / "dense.npy"
    with pytest.raises(SystemExit):
        main([
            "convert",
            str(gfa),
            "--matrix",
            str(out),
            "--max-dense-gb",
            "0.001",
        ])


def test_dense_matrix_limit_respects_dtype(tmp_path: Path):
    gfa = write_gfa(tmp_path, 400)
    out = tmp_path / "dense.npy"
    main([
        "--max-dense-gb",
        "0.001",
        "convert",
        str(gfa),
        "--matrix",
        str(out),
        "--dtype",
        "float32",
    ])
    assert out.exists()

