# GFA2Network

`GFA2Network` converts [GFA-1](https://github.com/GFA-spec/GFA-spec) or
[GFA-2](https://github.com/GFA-spec/GFA-spec/blob/master/GFA2.md) pangenome
variation graphs into handy Python objects.  It can stream even very large
graphs and materialise them as a NetworkX graph and/or a SciPy sparse adjacency
matrix.

The command **gfa2network** can build

- a `networkx.Graph` / `DiGraph`, and/or
- a SciPy sparse adjacency matrix.

It reads the input file in a single pass and keeps memory usage roughly
proportional to the number of edges, so multi‑million node graphs can be
processed on ordinary hardware.

## Features

- Handles GFA‑1 and GFA‑2 link syntax
- Build directed or undirected graphs
- Stream parsing keeps memory proportional to edge count
- Optional edge weights from a tag (e.g. `RC`)
- Optional sequence storage on nodes
- Adjacency matrices in CSR/CSC/COO/DOK formats
- Helper utilities to convert or save matrices
- Bidirected graph representation via `--bidirected`
- Export edges to various formats with the `export` subcommand

The repository ships with a real world test graph,
[`DRB1-3123_unsorted.gfa`](tests/data/DRB1-3123_unsorted.gfa), containing about
9500 lines from the human *DRB1* region.  Use it to experiment with the CLI
or run the unit tests:

```bash
gfa2network tests/data/DRB1-3123_unsorted.gfa \
    --graph --matrix drb1.npz --verbose
```

Run the small test suite with:

```bash
pytest
```


## Quick start

```bash
# build both outputs (directed graph + CSR matrix)
gfa2network convert input.gfa --graph --matrix adj.npz

# matrix only (lowest RAM) using COO format
gfa2network convert input.gfa --matrix adj.npz --matrix-format coo

# directed graph only with verbose progress
gfa2network convert input.gfa --graph --verbose

# stream from stdin and strip orientations (legacy behaviour)
cat input.gfa | gfa2network convert - --graph --strip-orientation

# print basic statistics
gfa2network stats input.gfa

# export edge list
gfa2network export input.gfa --format edge-list > edges.txt
```


See `gfa2network -h` for all command line options.

| Option             | Purpose |
| ------------------ | ------- |
| `convert`          | Convert a GFA into graph and/or matrix |
| `stats`            | Print basic statistics |
| `export`           | Stream edges in various formats |
| `--graph`          | Build a NetworkX object |
| `--matrix PATH`    | Write adjacency matrix to PATH |
| `--matrix-format`  | Sparse format for `.npz` (csr\|csc\|coo\|dok) |
| `--directed`       | Treat graph as directed (default) |
| `--undirected`     | Treat graph as undirected |
| `--weight-tag TAG` | Use numeric value of GFA tag `TAG` as edge weight |
| `--store-seq`      | Keep sequences from `S` records on nodes |
| `--strip-orientation` | Remove `+/-` from IDs (legacy) |
| `--bidirected`     | Use bidirected node representation |
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
to use numeric edge weights.  The module also exposes a helper
`convert_format` to turn the returned COO matrix into CSR/CSC/DOK formats.

## Dependencies

Install dependencies with:

```bash
pip install -e .[tqdm]
```

This requires Python 3.8+ and the packages:
- `networkx` and `scipy`
- `tqdm` (optional, pretty progress bars)
- `pytest` (optional, to run the tests)

## Implementation notes

Segment (`S`), link (`L`), edge (`E`), containment (`C`) and path/walk (`P`/`O`)
records are parsed. Orientation symbols `+`/`-` are preserved on links and
paths. Use `--strip-orientation` to reproduce the legacy behaviour of removing
them. Additional GFA tags are parsed into a dictionary and can be used as edge
weights with `--weight-tag TAG`.

## Output

If `--graph` is provided, a NetworkX graph is exposed as `G` when running the
script directly. With `--matrix PATH`, an adjacency matrix is written to the
specified path (`.npz`, `.npy` or `.csv`).  Matrices are produced in COO format
and can be converted to other sparse formats via the `convert_format` helper.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
