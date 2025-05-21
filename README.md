# GFA2Network

`GFA2NetworkX` converts large [GFA-1](https://github.com/GFA-spec/GFA-spec) or
[GFA-2](https://github.com/GFA-spec/GFA-spec/blob/master/GFA2.md) pangenome
variation graphs into handy Python objects.

The script **GFA2Network.py** can build

- a `networkx.Graph` / `DiGraph`, and/or
- a SciPy sparse adjacency matrix.

It reads the input file in a single pass and keeps memory usage roughly
proportional to the number of edges, so multi‑million node graphs can be
processed on ordinary hardware.

## Quick start

```bash
# build both outputs (directed graph + CSR matrix)
python GFA2Network.py input.gfa --graph --matrix adj.npz

# matrix only (lowest RAM) using COO format
python GFA2Network.py input.gfa --matrix adj.npz --matrix-format coo

# directed graph only with verbose progress
python GFA2Network.py input.gfa --graph --verbose
```

See `python GFA2Network.py -h` for all command line options.

## Dependencies

- Python 3.8+
- `networkx` and `numpy`
- `scipy` (optional, only required for matrix output)
- `tqdm` (optional, pretty progress bars)

## Implementation notes

Only segment (`S`) and link (`L`) records are used. Orientation symbols
`+`/`-` are stripped from node IDs. Additional GFA tags can be used as edge
weights with `--weight-tag TAG`.

## Output

If `--graph` is provided, a NetworkX graph is exposed as `G` when running the
script directly. With `--matrix PATH`, an adjacency matrix is written to the
specified path (`.npz`, `.npy` or `.csv`).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
