# Changelog

## v0.5.0

- Added distance helpers `sequence_distance`, `genome_distance` and `load_paths`
- New CLI subcommand `distance` supporting sequence or path queries
- Expanded README with analysis examples and usage instructions
- Added unit tests covering the new functionality

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
