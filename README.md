<p><img src="https://raw.githubusercontent.com/sclipman/gfa2network/main/assets/logo.png" width="125" alt="GFA2Network logo" /></p>

# GFA2Network

Convert GFA sequence-graph connectivity into NetworkX, igraph, or SciPy sparse
adjacency matrices. Input is read incrementally, including gzip files and stdin.
The resulting graph/matrix remains in memory. Storage includes nodes, edges,
path definitions when requested, and optional sequence/tag payloads.

## Installation

```bash
python -m pip install git+https://github.com/sclipman/gfa2network.git
# Optional igraph backend and progress display
python -m pip install "gfa2network[igraph,tqdm] @ git+https://github.com/sclipman/gfa2network.git"
```

Python 3.8+ is supported. NetworkX, NumPy, and SciPy are required. igraph, pandas
(labeled Python distance results), and tqdm are optional.

## Examples

```bash
# Directed graph and adjacency; save the graph as a pickle
# Loading pickle files requires trusting their producer.
gfa2network convert input.gfa --graph -o graph.pkl --matrix adjacency.npz

# Batched igraph construction with sequence and tag metadata
gfa2network convert input.gfa.gz --graph --backend igraph --store-seq --store-tags -o graph.pkl

# Binary adjacency; a row/column label sidecar is written by default
gfa2network convert input.gfa --matrix adjacency.npz --dtype bool

# Explicit undirected graph, or an explicit symmetric matrix projection
gfa2network convert input.gfa --graph --undirected
gfa2network convert input.gfa --matrix symmetric.npz --symmetric

# Subdivide segments at alignment boundaries
gfa2network convert input.gfa --graph --split-on-alignment --bidirected --store-seq -o split.pkl

# Standard input works for statistics and path analysis
cat input.gfa | gfa2network stats -
cat input.gfa | gfa2network distance - --path sample1 sample2 --undirected --details

# Export the normalized topology; add weights explicitly when wanted
gfa2network export input.gfa --format edge-list --with-weights > edges.tsv
gfa2network export input.gfa --format graphml > graph.graphml

# Pairwise path connectivity, with labels and reachable-coverage sidecars
gfa2network distance-matrix input.gfa -o distances.csv --method mean-pairwise

# Only sequence adjacency (no containments, alignments or gap estimates)
gfa2network convert input.gfa --graph -o traversal.pkl --projection traversal --manifest traversal.json

# Validate path transitions, spell sequences and audit them against source FASTA
gfa2network paths input.gfa --reference genomes.fa -o receipt.json --fasta spelled.fa

# Base-aware shared content between paths, and ordered shared blocks for a pair
gfa2network shared-content input.gfa -o jaccard.csv --measure jaccard
gfa2network compare input.gfa sample1 sample2 --orientation both -o blocks.json
```

Use `gfa2network <subcommand> --help` for options. Shared flags
`--raw-bytes-id`, `--max-dense-gb`, and `--max-tag-mb` work before or after a
subcommand. Verbose progress goes to stderr.

## Graph and matrix contract

Both backends use the same builder and edge policy:

| Setting | Meaning |
| --- | --- |
| Default `edge_mode="sum"` | One edge per endpoint pair. Repeated record weights are summed; absent weights contribute 1. `multiplicity` records the number of contributing records. |
| `edge_mode="parallel"` / `--edge-mode parallel` | Retain individual records as parallel graph edges. Sparse adjacency still sums their weights. Shortest paths use the cheapest parallel edge. |
| Undirected graph | Reciprocal input records contribute to the same summed edge. A self-loop contributes once per input record. |
| Directed graph (default) | Preserve directed connections. Matrices use the same direction by default. |
| `--symmetric` / `asymmetric=False` | Explicitly sum both directions into a symmetric projection; self-loops still count once. |
| `--bidirected` | Create `<id>:+` and `<id>:-` vertices and the forward/reverse-complement connection. The default graph is undirected. |
| `--keep-directed-bidir` | Keep oriented connections directed, including reverse-complement arcs. |
| `--strip-orientation` | Suppress orientation metadata; literal `+`/`-` characters in a base identifier are preserved. |
| `--weight-tag TAG` | Read finite numeric edge weights from that tag; a missing tag contributes 1. Negative weights can be exported, but distance analysis rejects them. |

For ordinary collapsed graphs, the first orientation attributes are retained and
additional distinct orientation pairs are recorded when needed. Undirected
orientation attributes follow the builder's node order. Use bidirected mode when
orientation belongs in the topology. Requested tags retain record provenance on
aggregated edges. GraphML/GEXF/JSON exports encode nested attributes as JSON strings.

`--matrix-format` accepts CSR, CSC, COO and DOK. NPZ serialization normalizes DOK
to a supported on-disk format. `--dtype` accepts bool, int8, int32, float32 and
float64; numeric overflow/truncation is rejected. Boolean matrices describe
nonzero aggregate weights, so a zero-cost edge has a false numeric entry even
though it exists structurally in sparse storage. A `<matrix>.nodes.tsv` sidecar
maps row/column indexes to identifiers; `--no-node-map` suppresses it.

Dense CSV/NPY exports check the complete shape times item size before allocating.
`--max-dense-gb` defaults to 5 decimal GB. This bounds the requested dense payload,
not total process RSS. `--max-tag-mb` warns on estimated stored tag payloads in
both backends; it is a warning threshold, not a hard memory budget.

## Comparing genome paths

Path comparisons count bases, not vertices, so renaming segments or
re-segmenting identical sequence leaves them unchanged. They use the rich
genome paths above.

```python
from gfa2network import compare_paths, load_genome_paths, shared_content_matrix

paths = load_genome_paths("input.gfa")
jaccard, details = shared_content_matrix(paths)          # details["shared"], ["union"], ...
result = compare_paths(paths, "sample1", "sample2", orientation="both")
result.content.jaccard, result.collinear.bases, result.blocks[0].a  # witness
```

| Quantity | Definition |
| --- | --- |
| Content | Segment bases a path traverses. The bases of an exact overlap are removed from one of the two segments, chosen by the link's strand-canonical form, so a reverse-complement traversal (or a rotation of a declared circular path, whose closing link then counts) loses the same bases. A complete path's content with repeats equals its spelled length unless overlaps on both sides of a segment meet. |
| `copies="unique"` / `"count"` | Each segment base once per path, or repeated visits counted (shared = smaller copy number). |
| Shared, union, Jaccard, containment | Shared bases; union = A + B − shared; Jaccard = shared / union; containment = shared / the row path. A zero denominator gives `None` / `NaN`, not 0. |
| Shared block | Maximal run of consecutive steps with identical segments and orientations in both paths (`forward`), or reversed with opposite orientations (`reverse`). Blocks never cross a jump, a W chunk gap or a transition without supported link evidence. |
| Collinear chain | Heaviest set of whole blocks in the same order in both paths. A block weighs the bases its matched step pairs share, so a chain never exceeds the copy-counted shared content and is symmetric in the two paths. Reordering lowers it; shared content is unchanged. |
| Witness | For each block and path: step range, start/end coordinates (`source` for W chunks with coordinates, else `path`), `reverse` and `wraps` flags. Unknown coordinates are `None`. |

Steps with an unknown segment length are excluded and listed; steps with an
unknown incoming overlap count their whole segment and are listed as uncertain;
either makes a path's evidence `partial`. `orientation="both"` finds a genome
stored on the opposite strand. `circular=True` declares both genomes circular:
the chain search tries a rotation at the start of each of the heaviest
`max_rotations` maximal blocks (`exhaustive` reports whether every start was
tried); with repeated segments the best rotation can start inside a block, so the
circular chain is then a lower bound. A block may wrap a path's end only across a
supported closing link, and blocks with unknown segment lengths count their known
bases and are marked incomplete. Shared content means the paths
traverse the same graph segments; it is not alignment identity (ANI) and implies
no homology beyond what the graph construction asserted.

Expected behavior on controls (tested): identical paths share everything in one
block; reordered paths keep Jaccard 1 while the collinear chain drops (and graph
distances stay 0); repeated visits change only `copies="count"` results;
reverse-complement traversals match only under `orientation="both"`; a jump splits
blocks but not content; distinct content shares nothing; paths without resolved
bases give no ratio. In the bundled DRB1 graph, `gi|345525392:5000-18402` is the
reverse-strand traversal of `gi|568815529:3998044-4011446` apart from one terminal
segment: content Jaccard > 0.999, forward collinear 0, reverse collinear 13,401 bp.

## Relation projections and manifests

GFA records declare different relationships. Every relation record is assigned
one class, and `projection=` / `--projection` selects the classes that become
edges:

| Class | Records |
| --- | --- |
| `adjacency` | `L` links, GFA2 dovetail `E` edges, coordinate-free (historical) `E` edges |
| `containment` | `C` records; GFA2 `E` edges whose interval spans a whole segment |
| `alignment` | Other GFA2 `E` alignments between segment interiors |
| `gap` | `J` jumps and GFA2 `G` gaps: estimated distances, not observed sequence |
| `unclassified` | GFA2 `E` edges with an unknown segment length or out-of-range coordinates |

| Projection | Classes |
| --- | --- |
| `all` (default) | every class; identical to 2.0 behavior |
| `traversal` | `adjacency` |
| `gap-inclusive` | `adjacency`, `gap` |
| `alignment` | `containment`, `alignment` |
| custom | any set, e.g. `projection={"gap"}` or `--projection adjacency,gap` |

Scientific analyses should choose a projection explicitly: under `all`, a
containment or an estimated gap is an edge like any other, so connectivity can
pass through sequence that was never observed as adjacent. GFA2 `E` records are
classified from forward-strand coordinates and segment lengths; records that
arrive before their segments are classified when the input ends. Subdivided
graphs keep each record's original class; links between abutting intervals of
one segment have the class `subdivision` and are kept by every projection,
because they restore the segment itself. Vertices come from `S` records and
included relations.

Each graph carries a manifest in its `projection` attribute (NetworkX
`G.graph["projection"]`, igraph `g["projection"]`); `return_manifest=True`
returns `(result, manifest)` for any result shape, including matrix-only
conversion. The manifest names the projection and included classes, counts
included and excluded records by class and record type, and states direction,
orientation handling, aggregation, weight meaning, edge-cost use, self-loop
counting, subdivision and vertex/edge/nonzero totals. `records` counts input
records once each, including duplicate and reciprocal records, which the graph
aggregates as described in the contract above; under subdivision,
`subdivided_relations` separately counts the generated interval relations. `convert --manifest` and `export --manifest` write it with an
`exports` section stating what each written format retains and drops (for
example, matrices and edge lists drop multiplicity, record provenance and gap
estimates). `distance-matrix` metadata embeds it with the cost policy.

## Supported records

The parser targets connectivity and segment-based paths, rather than complete
round-trip GFA editing. The authoritative syntax is in the
[GFA1 specification](https://gfa-spec.github.io/GFA-spec/GFA1.html) and
[GFA2 specification](https://gfa-spec.github.io/GFA-spec/GFA2.html).

| Format / records | Support |
| --- | --- |
| GFA1 `S`, `L`, `C`, `P` | Segment lengths/sequences/tags, oriented connections, containment positions/CIGARs, and comma-separated paths. |
| GFA1.1 `W` | Oriented walks; identity is escaped `sample#haplotype#sequence`. Multiple chunks are sorted by coordinates and must not overlap. Multiple chunks with unknown coordinates are rejected. Chunk gaps do not invent graph edges. |
| GFA1.2 `J`, semicolons in `P` | Estimated jumps and paths using jumps. |
| GFA2 `S`, `E`, `G` | Oriented references, coordinate intervals including `$` terminal markers, and gap estimates. Gap distances/variances are metadata; they do not automatically become shortest-path costs. |
| GFA2 `O` | Space-separated oriented **segment** references. Edge/group references are rejected during path analysis. Unnamed groups receive `@O:<ordinal>` labels. |
| `H`, `F`, `U` | Headers are read; fragment alignments and unordered collections are recognized metadata and do not add connections or ordered paths. |
| Unknown records | Conversion warns once per record type. Path analysis fails explicitly instead of silently dropping potentially relevant records. |

Missing sequences (`*`) are represented as absent, not as a literal sequence.
GFA1 lengths come from `LN` or the sequence. CRLF files are accepted, and malformed
record errors include the input location. Historical orientation-only and
interleaved-coordinate `E` layouts remain accepted for compatibility.

## Alignment subdivision

`--split-on-alignment` uses the same builder for both backends. It propagates cuts
through CIGAR match blocks, handles reverse orientation and indels, retains
sequence/tag metadata, and expands path traversals into interval order. Ordinary
links/jumps attach to the source exit and target entry according to orientation.
Interval nodes expose `parent_segment`, `start`, and `end` attributes.

Coordinates stay in the original segment frame. Each resulting alignment
connection represents an aligned interval; segmentation changes graph distances
and does not identify homologous bases as the same vertex. Sequence lengths must
agree with coordinates when sequences are present. CIGAR consumption is checked.
An absent alignment can be retained when its interval needs no internal
subdivision; otherwise an exact split cannot be inferred and the operation fails
explicitly. Dazzler trace strings are accepted for ordinary connectivity but need
an explicit CIGAR for subdivision. `--max-split-nodes` defaults to 1,000,000 and
bounds interval expansion before graph construction; the flag can be increased
for deliberate larger conversions. Expansion beyond ten times the original node
count also emits a warning.

## Genome paths

`parse_gfa(..., return_paths=True)` returns each path as a list of vertex
identifiers. For genome-level work, request rich paths instead:

```python
from gfa2network import load_genome_paths, parse_gfa

paths = load_genome_paths("input.gfa")          # one pass; stdin and gzip work
G, paths = parse_gfa("input.gfa", build_graph=True, build_matrix=False,
                     return_paths=True, rich_paths=True)
path = paths["sample1"]
path.steps          # ordered, oriented; repeated visits are separate steps
path.transitions    # relation evidence for each consecutive step
path.chunks         # W chunk coordinates (P/O records form one chunk)
paths.vertex_lists()                 # equals the list result for this graph
spelled = paths.spell("sample1")     # complete / partial / unavailable, with reasons
report = paths.audit("genomes.fa", circular=["sample1"])
```

A `GenomePathSet` keeps what each path declares: P overlaps and link (`,`) versus
jump (`;`) separators, W sample/haplotype/sequence identity and coordinates, record
lines and tags. Step strand is independent of whether the graph is bidirected.
Under `split_on_alignment`, steps remain input segments and list their interval
vertices with parent coordinates.

Each consecutive transition is checked against the relations the input declares,
in either strand: L links and J jumps, GFA2 dovetail `E` edges (overlap from
coordinates) and `G` gaps. Containments and internal alignments are not
traversals; a GFA2 interval spanning a whole segment is a containment.
A transition is `supported`, `missing`, `ambiguous` (relations
declare different overlaps, or both a link and a jump), `conflict` (the P overlap
disagrees with every relation) or a W chunk `boundary`.

Spelling never fills in sequence. Overlaps of only `M`/`=` operations are applied
after verifying the overlapping bases; unspecified (`*`) overlaps abstain unless
`unknown_overlap="zero"` is requested, which is recorded. Missing sequence,
indel-bearing overlaps, mismatching overlap bases, missing or ambiguous
transitions, jumps and W chunk gaps split the spelling into labeled pieces. Jumps
and chunk gaps are declared unknown sequence, not failures. Adjacent W chunks join;
a chunk whose spelled length disagrees with its coordinates is reported.

`audit()` compares spelled paths with source sequences by name (or an explicit
`reference_names` map); when references are supplied, an audited path without
one fails as `reference_missing`. Equivalences are declared, never inferred:
`reverse_complement=True` accepts strand reversal, and `circular` lists paths whose
comparison accepts rotation; an evidenced closing overlap is trimmed first. W
chunks with coordinates are compared with `reference[start:end]`; gapped paths
pass when their pieces occur in order, at most once around a declared circle. The receipt records policy, per-path
transition counts, issues, offsets and first differences; `gfa2network paths`
adds input and reference SHA-256 hashes. A match establishes fidelity to that
source, not biological validity of the source.

## Distances

These functions summarize graph connectivity, using edge costs. They do not
estimate nucleotide divergence, preserve genome path order, or establish sequence
identity. `min` is zero whenever two node sets share any node.

Edge cost is explicit. `cost="weight"` (default, `--cost weight`) uses edge
weights; in the default summing edge mode, duplicate records add to that weight,
so record support raises cost. `cost="hops"` charges 1 per edge regardless of
multiplicity, support or weight tags. Choose the relations with `projection`.

| Method | Definition, shared by scalar and matrix APIs |
| --- | --- |
| `min` | Minimum cost over all source/target node pairs. |
| `mean-pairwise` (alias `mean`) | Arithmetic mean over all source/target node pairs. |
| `mean-nearest` | Mean of source-to-nearest-target and nearest-source-to-target costs. |

Node sets are deduplicated; repeated path visits do not receive extra weight,
and path order is not used (rich paths keep both).
Graph-based functions respect the supplied graph direction. Distance matrices
are undirected by default; `--directed` produces directional matrices. Pairwise
mean diagonal entries are computed and can be nonzero, just as in the scalar API.

Means include unreachable contributions as infinity by default. Explicit
`unreachable="ignore"` / `--unreachable ignore` restricts means to reachable
contributions; `raise` reports an error. Minimum distance can be finite despite
partial disconnection; always inspect coverage when that matters. Pairwise
coverage counts reachable pairs; minimum/nearest coverage counts source and target
nodes with at least one reachable counterpart. Costs must be finite/nonnegative.
Read-support tags require a deliberate interpretation before use as distance costs.

Each distance-matrix export writes:

- `<output>.paths.tsv`: path order for both axes.
- `<output>.coverage.npy`: contribution coverage matrix.
- `<output>.metadata.json`: method, direction, unreachable policy and coverage definition.

Dense matrix workspace is checked before allocation. Searches retain O(V+E)
workspace rather than caching P full vertex-distance maps; path memberships and
O(P²) output/aggregation arrays are additional. Minimum/nearest search once per
path (twice if directed); pairwise means search once per distinct source node.
Actual time also depends on graph topology and path lengths.

```python
from gfa2network import parse_gfa
from gfa2network.analysis import genome_distance, genome_distance_matrix

G, paths = parse_gfa("input.gfa", build_graph=True, build_matrix=False,
                     directed=False, return_paths=True)
result = genome_distance(G, paths["sample1"], paths["sample2"],
                         method="mean-pairwise", return_details=True)
print(result.distance, result.coverage)

matrix, details = genome_distance_matrix("input.gfa", backend="igraph",
                                         method="mean-nearest", return_details=True)
```

`sequence_distance` finds exact stored sequence strings; build the graph with
`store_seq=True`. Reverse-oriented vertices store reverse-complement sequences.
`load_paths` reads P/O/W paths and can emit oriented vertex identifiers.

## Development and measured performance

```bash
python -m pip install -e ".[dev,igraph,tqdm]"
python -m pytest -q
bash scripts/lint_check.sh
python -m build
python scripts/benchmark.py --nodes 20000 --repeats 3 --output benchmark.json
```

The bundled [DRB1 graph](tests/data/DRB1-3123_unsorted.gfa) exercises unsorted real
input. CI tests Python 3.8, 3.12 and 3.14 with and without optional backends.
External large-graph tests use `LARGE_GRAPH=/path/to/file.gfa[.gz]` and skip when
unavailable. Reproducible benchmark scope, raw timings, peak RSS, input hashes and
output-equivalence checks are in [benchmarks](benchmarks/README.md). Measurements
on these inputs do not establish million-node capacity.

## Migrating from 1.0

Version 2.0 makes these deliberate corrections:

- Duplicate weights are summed in both backends and matrices. Choose `parallel`
  when individual edges are needed. Default graphs are simple graphs even in
  bidirected mode.
- Directed matrices retain graph direction by default; request `--symmetric` for
  the old symmetric projection explicitly.
- Both `mean` APIs use all-pairs means. Select `mean-nearest` for the previous
  distance-matrix definition. Unreachable contributions are no longer silently
  omitted; use `ignore` explicitly and inspect coverage.
- GFA2 `O` references are space-separated; migrate older comma-separated `O` files.
- Directed bidirected graphs include reverse-complement arcs. Subdivision maps
  links to the correct oriented ends and may produce more intervals.
- One version declaration drives both package metadata and `--version`.

## License

MIT; see [LICENSE](LICENSE). The software and documentation are the author's
responsibility and do not represent the views of Johns Hopkins University.
