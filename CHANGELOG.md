# Changelog

## Unreleased (2.1.0.dev0)

Additive; default graph topology, matrix, list-path and distance outputs are
unchanged. Graphs gain a `projection` manifest attribute.

- Add optional rich genome paths: `load_genome_paths()` and
  `parse_gfa(..., return_paths=True, rich_paths=True)` return a `GenomePathSet`
  keeping ordered oriented steps, repeated visits, P overlaps and link/jump
  separators, W chunk coordinates and sample/haplotype/sequence identity.
  Strand belongs to each step, independent of the graph projection.
- Check every consecutive path transition against L/J records and GFA2 dovetail
  `E`/`G` records in either strand; report `missing`, `ambiguous` or `conflict`
  evidence instead of assuming a traversal.
- Spell path sequence only where sequences, overlaps and transitions are
  unambiguous; overlapping bases are verified. Missing sequence, unspecified or
  indel-bearing overlaps, jumps and W chunk gaps split the result into labeled
  pieces with reasons. `unknown_overlap='zero'` is an explicit, recorded policy.
- Add a path-fidelity audit against source FASTA, with declared reverse-complement
  and circular-rotation equivalences, W coordinate checks and a JSON receipt;
  exposed as `gfa2network paths`.
- Add explicit relation projections (`all` default, `traversal`,
  `gap-inclusive`, `alignment`, or custom class sets) for graphs, matrices,
  statistics and distances. GFA2 `E` records are classified as dovetail
  adjacency, containment, internal alignment or unclassified, including records
  that precede their segments; subdivision keeps each record's class.
- Attach a projection manifest to every graph (`projection` attribute;
  `return_manifest=True`), with per-class included/excluded record counts and
  direction, orientation, aggregation, weight and cost policies. `convert` and
  `export` write it with `--manifest`, declaring what each format drops;
  distance-matrix metadata embeds it.
- Add `cost="hops"` / `--cost hops` so shortest-path cost can be separated from
  summed record weight.
- Add base-aware path comparison: `shared_content_matrix()` (shared bases,
  union, Jaccard, containment; unique or copy-counted) and `compare_paths()`
  (maximal forward/reverse shared blocks with step and coordinate witnesses, and
  a base-weighted collinear chain, optionally circular). Results are invariant
  to segment renaming and sequence-preserving re-segmentation; missing lengths
  and unknown overlaps are reported, never read as absence. CLI: `compare`,
  `shared-content`.
- Add `bridge_blocks()` for contiguous sequence shared across externally
  labeled path groups, with witnesses, flanking contexts and partial visitors;
  `transform.chop()`/`transform.rename()` for sequence-preserving
  representation checks; `genome_distance_matrix(names=...)`; and
  `examples/topology_report.py`.
- `audit(ambiguity="as-n")` / `paths --ambiguity as-n` declares IUPAC
  ambiguity codes equal to N; `audit(circular=False)` no longer raises.
- `GenomePathSet.layout()` exposes per-step lengths, overlaps, contiguity and
  start coordinates.
- Subdivided paths keep parent segments with interval coordinates. Interval links
  now carry an explicit `0M` overlap, and expanded P records keep consistent
  separators and overlaps.

## 2.0.0 — 2026-10-22

- Unify NetworkX, igraph, and sparse adjacency construction under one edge policy:
  duplicate weights sum by default, parallel records remain available explicitly,
  self-loops count once, and direction is preserved unless projection is requested.
- Replace the duplicated split builder with exact interval/CIGAR boundary
  propagation, correct reverse orientation and link endpoints, metadata retention,
  path expansion, and a configurable node-expansion guard.
- Support GFA1.1 W walks, ordered chunks, GFA1.2 jumps, GFA2 gaps, source-line
  errors, and single-pass statistics/path analysis including stdin.
- Use the same min, mean-pairwise, and mean-nearest definitions in scalar and
  matrix APIs. Implement both backends, explicit disconnected policies,
  nonnegative-cost validation, reachable coverage, and labeled exports.
- Replace O(PV) search caching with streamed CSR distance vectors; batch igraph
  construction and use typed sparse buffers. Include fresh-process benchmarks
  with output fingerprints and measured peak RSS rather than timing assertions.
- Repair standard GFA2 edge/group parsing, missing sequences, lengths, CRLF,
  typed tags, dense-allocation guards, NPZ formats, export serialization, CLI
  option placement, and optional fixture discovery.
- Consolidate version metadata, remove placeholder author email, add regression
  and independent-reference tests, and exercise Python 3.8/3.12/3.14 in CI.

See README migration notes for intentional changes from 1.0. Earlier tagged
release history is retained below; package metadata had advanced to 1.0 without
an accompanying changelog section before this maintenance pass.

## v0.12.0
- Added `--split-on-alignment` option to split segments at alignment
  boundaries. Emits a warning if the number of sub-nodes exceeds 10× the
  original segment count.

## v0.11.0
- Clarified documentation on graph interpretation and distance semantics.
- Distance helpers now emit a warning when called on a directed bidirected graph.
- Finishing-touches: performance warnings, configurable safety limits, CLI polish.

## v0.10.0
- Saving an adjacency matrix now writes `<outfile>.nodes.tsv` mapping row
  indices to node identifiers.
- Added `--no-node-map` option to suppress this sidecar file.

## v0.9.0
- Optimised `genome_distance_matrix` to reuse a cached multi-source Dijkstra per
  path. Runtime now scales linearly with the number of paths and output is
  unchanged.

## v0.8.0
- Added `--store-tags` option to keep tag dictionaries and segment lengths on
  nodes and edges. A `RuntimeWarning` is issued when stored tags exceed
  100&nbsp;MB.

## v0.7.0
- Node identifiers are now decoded to strings when building NetworkX graphs.
- Added `--raw-bytes-id` CLI flag for legacy byte behaviour.
- GraphML and GEXF exports use plain string IDs.

## v0.6.0
- Added optional pandas support in `analysis` for DataFrame results
- Introduced `genome_distance_matrix` helper
- Added `distance-matrix` CLI subcommand
- Updated README and tests

## v0.5.0
- Added sequence_distance and genome_distance helpers.
- Introduced distance CLI subcommand and load_paths utility for querying sequences or paths.
- Expanded README with Python API details and examples.
- Added unit tests covering the new functionality.


## v0.4.0

- Added igraph backend (--backend igraph) via IGraphBuilder—requires python-igraph.

## v0.3.0

- Expanded parser with edge (`E`), containment (`C`) and walk (`O`) records
- Tags are parsed into dictionaries with proper value types
- New `--bidirected` option and `export` subcommand
- Documentation and README updated
- Version bump

## v0.2.0

- Refactored into modular package layout
- Added `GFAParser` class and path support
- Orientation is preserved on links; use `--strip-orientation` for legacy mode
- New subcommands: `convert` and `stats`
- CLI exposes `--version`
- Added basic stats analysis
- Added CI workflow and tests
