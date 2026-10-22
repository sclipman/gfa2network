"""igraph compatibility entry points using the shared batch accumulator."""

from __future__ import annotations

try:
    import igraph as ig

    _HAS_IGRAPH = True
except ImportError:  # pragma: no cover - optional dependency
    ig = None
    _HAS_IGRAPH = False


class IGraphBuilder:
    """Accumulate records and build igraph vertices/edges in one batch."""

    def __init__(self, **kwargs):
        if not _HAS_IGRAPH:
            raise RuntimeError("python-igraph is not available")
        from .model import GraphAccumulator

        self._builder = GraphAccumulator(
            backend="igraph", build_graph=True, build_matrix=True, **kwargs
        )
        self._node_index = self._builder.node_index
        self.directed = self._builder.directed

    @property
    def graph(self):
        return self._builder.finish_graph()

    def add_segment(self, record):
        self._builder.add_segment(record)

    def add_edge_record(self, record):
        self._builder.add_edge_record(record)

    def to_matrix(self):
        return self._builder.finish_matrix().tocsr()


def parse_gfa_igraph(path, **kwargs):
    """Equivalent to ``parse_gfa(path, backend='igraph', **kwargs)``."""
    from .builders import parse_gfa

    return parse_gfa(path, backend="igraph", **kwargs)
