"""GFA2Network package."""

from .builders import parse_gfa
from .genome_paths import GenomePath, GenomePathSet, load_genome_paths
from .igraph_builder import parse_gfa_igraph, IGraphBuilder
from .path_compare import compare_paths, shared_content_matrix
from .utils import convert_format
from .version import __version__

__all__ = [
    "parse_gfa",
    "parse_gfa_igraph",
    "IGraphBuilder",
    "convert_format",
    "load_genome_paths",
    "GenomePath",
    "GenomePathSet",
    "compare_paths",
    "shared_content_matrix",
    "__version__",
]
