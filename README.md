# GFA2Network

`GFA2Network` converts large [GFA-1](https://github.com/GFA-spec/GFA-spec) or
[GFA-2](https://github.com/GFA-spec/GFA-spec/blob/master/GFA2.md) pangenome
variation graphs into handy Python objects.

The command **gfa2network** can build

- a `networkx.Graph` / `DiGraph`, and/or
- a SciPy sparse adjacency matrix.

It reads the input file in a single pass and keeps memory usage roughly
proportional to the number of edges, so multi‑million node graphs can be
processed on ordinary hardware.

## Quick start

```bash
# build both outputs (directed graph + CSR matrix)
gfa2network input.gfa --graph --matrix adj.npz

# matrix only (lowest RAM) using COO format
gfa2network input.gfa --matrix adj.npz --matrix-format coo

# directed graph only with verbose progress
gfa2network input.gfa --graph --verbose
```


See `gfa2network -h` for all command line options.

| Option             | Purpose |
| ------------------ | ------- |
| `--graph`          | Build a NetworkX object |
| `--matrix PATH`    | Write adjacency matrix to PATH |
| `--matrix-format`  | Sparse format for `.npz` (csr\|csc\|coo\|dok) |
| `--directed`       | Treat graph as directed (default) |
| `--undirected`     | Treat graph as undirected |
| `--weight-tag TAG` | Use numeric value of GFA tag `TAG` as edge weight |
| `--store-seq`      | Keep sequences from `S` records on nodes |
| `--verbose`        | Emit progress information |

`--store-seq` may drastically increase memory usage. The parser will warn when the
stored sequences exceed half of the available RAM. The flag is ignored when only
`--matrix` is requested.

## Using in Python

Import `parse_gfa` to build graphs in your own code:

```python
from gfa2network import parse_gfa

# build a NetworkX graph
G = parse_gfa("input.gfa", build_graph=True, build_matrix=False)
```

Pass `directed=False` for an undirected graph or specify a `weight_tag`
to use numeric edge weights.

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
# or
pip install .
```

This requires Python 3.8+ and the packages:
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
