from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False


try:
    import scipy.sparse as sp

    _HAS_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAS_SCIPY = False


def available_memory() -> int:
    """Return approximate available RAM in bytes (Linux only)."""
    try:
        with open("/proc/meminfo", "r") as fh:
            info = {line.split(":", 1)[0]: int(line.split()[1]) for line in fh}
        if "MemAvailable" in info:
            return info["MemAvailable"] * 1024
        if "MemTotal" in info:
            return info["MemTotal"] * 1024
    except Exception:
        pass
    return 0


def convert_format(A, fmt: str, *, verbose: bool = False):
    """Convert COO -> *fmt* with optional progress indication."""
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy required for matrix conversion")
    fmt = fmt.lower()
    if fmt not in {"csr", "csc", "coo", "dok"}:
        raise ValueError("matrix-format must be csr|csc|coo|dok")
    if fmt == "coo":
        return A
    if verbose:
        if _HAS_TQDM:
            bar = tqdm(total=1, bar_format="{desc} …{elapsed}", desc=f"[convert→{fmt}")
        else:
            start = time.perf_counter()
            print(f"[convert] -> {fmt} …", end="", file=sys.stderr, flush=True)
    out = A.asformat(fmt)
    if verbose:
        if _HAS_TQDM:
            bar.update(1)
            bar.close()
        else:
            dt = time.perf_counter() - start
            print(f" done in {dt:,.1f}s", file=sys.stderr)
    return out


def save_matrix(A, dest: Path, *, verbose: bool = False):
    """Write *A* to *dest* with progress bar and dense-size guard."""
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy required for matrix output")
    MAX_DENSE_BYTES = 5_000_000_000  # 5 GB
    if dest.suffix in {".csv", ".npy"}:
        nnz = A.nnz if sp.issparse(A) else A.size
        if nnz * 8 > MAX_DENSE_BYTES:
            raise MemoryError(
                f"dense export would allocate {nnz*8/1e9:.1f} GB; choose a sparse .npz or write an edge list instead"
            )
    if verbose:
        msg = f"[save] {dest.suffix[1:]} → {dest}"
        if _HAS_TQDM:
            bar = tqdm(total=1, bar_format="{desc} …{elapsed}", desc=msg)
        else:
            start = time.perf_counter()
            print(msg, "...", end="", file=sys.stderr, flush=True)
    if dest.suffix == ".npz":
        sp.save_npz(dest, A)
    elif dest.suffix == ".npy":
        import numpy as np

        np.save(dest, A.toarray() if sp.issparse(A) else A)
    elif dest.suffix == ".csv":
        import numpy as np

        np.savetxt(
            dest, A.toarray() if sp.issparse(A) else A, delimiter=",", fmt="%.6g"
        )
    else:
        raise ValueError("matrix path must end with .npz, .npy, or .csv")
    if verbose:
        if _HAS_TQDM:
            bar.update(1)
            bar.close()
        else:
            dt = time.perf_counter() - start
            print(f" done in {dt:,.1f}s", file=sys.stderr)


def save_node_map(nodes: Sequence[bytes | str], dest: Path) -> None:
    """Write node index mapping to *dest* as TSV."""
    with open(dest, "w") as fh:
        for i, node in enumerate(nodes):
            if isinstance(node, (bytes, bytearray)):
                node = node.decode()
            fh.write(f"{i}\t{node}\n")
