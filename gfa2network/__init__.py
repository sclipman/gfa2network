"""GFA2Network package."""

from .builders import parse_gfa
from .igraph_builder import parse_gfa_igraph
from .utils import convert_format
from .version import __version__

__all__ = ["parse_gfa", "parse_gfa_igraph", "convert_format", "__version__"]
