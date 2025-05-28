# Changelog

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
