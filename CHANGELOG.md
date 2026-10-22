# Changelog

## Unreleased
- Parse standard GFA2 edge columns, terminal `$` coordinates, and space-separated
  ordered groups; preserve identifier suffixes and handle CRLF input.
- Infer GFA1 segment lengths, recognize absent sequences, and retain containment
  positions/CIGARs and typed numeric arrays.
- Check the full dense allocation before export; support dense distance matrices
  and DOK input when saving sparse NPZ files.
- Repair igraph reverse-complement vertex naming, metadata on unsorted input,
  adjacency export, matrix dtype, and symmetry options. Reject unsupported igraph
  alignment splitting and byte IDs explicitly.
- Accept shared CLI options after subcommands, repair the `stat` alias, and write
  GraphML/GEXF to stdout when requested.
- Build distance matrices on undirected topology so path record order does not
  change their values. Validate method names and clearly restrict distance
  helpers to NetworkX until native igraph analysis is implemented.
- Add specification-based regression cases and a CI matrix; skip optional large
  fixtures when absent. Document remaining splitting and distance limitations.

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
