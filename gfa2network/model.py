"""One graph/matrix contract shared by both backends and split conversion."""

from __future__ import annotations

from array import array
import math
import pickle
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .parser import GapRecord

_COMPLEMENT = bytes.maketrans(
    b"ACGTRYMKBDHVNacgtrymkbdhvn", b"TGCAYRKMVHDBNtgcayrkmvhdbn"
)


def reverse_complement(sequence):
    return sequence.translate(_COMPLEMENT)[::-1]


class GraphAccumulator:
    """Sum duplicate weights by default; optional parallel edges retain records.

    Undirected self-loops and self-reverse bidirected links count once per record.
    Sparse buffers use typed arrays, and igraph vertices/edges are created in bulk.
    """

    def __init__(
        self,
        *,
        build_graph=True,
        build_matrix=False,
        backend="networkx",
        directed=True,
        bidirected=False,
        keep_directed_bidir=False,
        weight_tag=None,
        store_seq=False,
        store_tags=False,
        raw_bytes_id=False,
        strip_orientation=False,
        edge_mode="sum",
        max_tag_mb=100.0,
    ):
        if edge_mode not in {"sum", "parallel"}:
            raise ValueError("edge_mode must be sum or parallel")
        if not math.isfinite(max_tag_mb) or max_tag_mb < 0:
            raise ValueError("max_tag_mb must be finite and nonnegative")
        self.build_graph, self.build_matrix, self.backend = (
            build_graph,
            build_matrix,
            backend,
        )
        self.directed = keep_directed_bidir if bidirected else directed
        self.bidirected, self.weight_tag = bidirected, weight_tag
        self.store_seq, self.store_tags = (
            store_seq and build_graph,
            store_tags and build_graph,
        )
        self.raw_bytes_id, self.strip_orientation = raw_bytes_id, strip_orientation
        self.edge_mode, self.max_tag_mb = edge_mode, max_tag_mb
        self.node_index = {}
        self.defined_segments = set()
        self.node_attributes = []
        self.edge_attributes = {} if edge_mode == "sum" else []
        self.rows, self.cols, self.data = array("q"), array("q"), array("d")
        self.tag_bytes = self.sequence_bytes = 0
        self._warned_tags = False
        self.graph = None
        if build_graph and backend == "networkx":
            cls = (
                (nx.MultiDiGraph if self.directed else nx.MultiGraph)
                if edge_mode == "parallel"
                else (nx.DiGraph if self.directed else nx.Graph)
            )
            self.graph = cls()

    def name(self, node):
        return node if self.raw_bytes_id else node.decode("ascii")

    def vertex(self, node, attributes=None):
        idx = self.node_index.get(node)
        if idx is None:
            idx = len(self.node_index)
            self.node_index[node] = idx
            if self.build_graph and self.backend == "igraph":
                self.node_attributes.append({})
        if self.build_graph:
            if self.backend == "networkx":
                self.graph.add_node(self.name(node), **(attributes or {}))
            elif attributes:
                self.node_attributes[idx].update(attributes)
        return idx

    def add_segment(self, record):
        if record.id in self.defined_segments:
            raise ValueError(f"duplicate segment {record.id!r}")
        self.defined_segments.add(record.id)
        for orientation in ("+", "-") if self.bidirected else (None,):
            node = (
                record.id
                if orientation is None
                else record.id + b":" + orientation.encode()
            )
            attrs = {}
            if self.store_seq and record.sequence is not None:
                attrs["sequence"] = (
                    reverse_complement(record.sequence)
                    if orientation == "-"
                    else record.sequence
                )
                self.sequence_bytes += len(record.sequence)
            if self.store_tags:
                if record.length is not None:
                    attrs["length"] = record.length
                if record.tags is not None:
                    attrs["tags"] = record.tags
                    self.count_tags(record.tags)
            if record.parent is not None:
                attrs.update(
                    parent_segment=self.name(record.parent),
                    start=record.start,
                    end=record.end,
                )
            self.vertex(node, attrs)

    def count_tags(self, tags):
        self.tag_bytes += len(pickle.dumps(tags))
        if not self._warned_tags and self.tag_bytes > self.max_tag_mb * 1_000_000:
            warnings.warn(
                f"stored tag dictionaries use more than {self.max_tag_mb:g} MB",
                RuntimeWarning,
            )
            self._warned_tags = True

    @staticmethod
    def merge(existing, attributes):
        weight = existing["weight"] + attributes["weight"]
        if not math.isfinite(weight):
            raise ValueError("summed edge weight is not finite")
        existing["weight"] = weight
        existing["multiplicity"] += 1
        if "orientation_from" in attributes:
            first = (existing["orientation_from"], existing["orientation_to"])
            second = (attributes["orientation_from"], attributes["orientation_to"])
            if first != second or "orientations" in existing:
                observed = existing.setdefault("orientations", [first])
                if second not in observed:
                    observed.append(second)
        if "tags" in attributes:
            existing.setdefault("tag_records", [existing.get("tags")]).append(
                attributes["tags"]
            )
        if (
            attributes["record_type"] != existing["record_type"]
            or "record_types" in existing
        ):
            existing.setdefault("record_types", [existing["record_type"]]).append(
                attributes["record_type"]
            )
        if "gap_distance" in attributes:
            existing.setdefault("gaps", []).append(
                (attributes.get("gap_distance"), attributes.get("gap_variance"))
            )

    def edge(self, u, v, attrs):
        a, b = self.vertex(u), self.vertex(v)
        if not self.directed and a > b and "orientation_from" in attrs:
            attrs = dict(attrs)
            attrs["orientation_from"], attrs["orientation_to"] = (
                attrs["orientation_to"],
                attrs["orientation_from"],
            )
        if self.build_matrix:
            self.rows.append(a)
            self.cols.append(b)
            self.data.append(attrs["weight"])
            if not self.directed and a != b:
                self.rows.append(b)
                self.cols.append(a)
                self.data.append(attrs["weight"])
        if not self.build_graph:
            return
        if self.backend == "networkx":
            u_name, v_name = self.name(u), self.name(v)
            if self.edge_mode == "sum" and self.graph.has_edge(u_name, v_name):
                self.merge(self.graph[u_name][v_name], attrs)
            else:
                self.graph.add_edge(u_name, v_name, **attrs)
        else:
            key = (a, b) if self.directed or a <= b else (b, a)
            if self.edge_mode == "parallel":
                self.edge_attributes.append((key, attrs))
            elif key in self.edge_attributes:
                self.merge(self.edge_attributes[key], attrs)
            else:
                self.edge_attributes[key] = attrs

    def add_edge_record(self, record):
        if record.orientation_from not in {"+", "-"} or record.orientation_to not in {
            "+",
            "-",
        }:
            raise ValueError("invalid edge orientation")
        weight = (
            (record.tags or {}).get(self.weight_tag, 1.0) if self.weight_tag else 1.0
        )
        if not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError(
                f"edge weight {self.weight_tag!r} must be finite and numeric"
            )
        u, v = record.from_segment, record.to_segment
        attrs = {"weight": float(weight), "multiplicity": 1}
        if not self.strip_orientation and not self.bidirected:
            attrs.update(
                orientation_from=record.orientation_from,
                orientation_to=record.orientation_to,
            )
        if self.store_tags:
            attrs["tags"] = record.tags
            if record.tags is not None:
                self.count_tags(record.tags)
        if isinstance(record, GapRecord):
            attrs.update(gap_distance=record.distance, gap_variance=record.variance)
        attrs["record_type"] = type(record).__name__
        if not self.bidirected:
            self.edge(u, v, attrs)
            return
        first = (
            u + b":" + record.orientation_from.encode(),
            v + b":" + record.orientation_to.encode(),
        )
        flip = {"+": b"-", "-": b"+"}
        second = (
            v + b":" + flip[record.orientation_to],
            u + b":" + flip[record.orientation_from],
        )
        self.edge(*first, attrs)
        equivalent = second == first or (not self.directed and second == first[::-1])
        if not equivalent:
            # Aggregation mutates attribute lists, so the two strand edges must
            # own independent list containers.
            copied = {
                key: list(value) if isinstance(value, list) else value
                for key, value in attrs.items()
            }
            self.edge(*second, copied)

    def finish_graph(self):
        if not self.build_graph:
            return None
        if self.backend == "networkx":
            return self.graph
        from .igraph_builder import ig

        entries = (
            list(self.edge_attributes.items())
            if self.edge_mode == "sum"
            else self.edge_attributes
        )
        graph = ig.Graph(
            n=len(self.node_index),
            edges=[key for key, _ in entries],
            directed=self.directed,
        )
        graph.vs["name"] = [self.name(node) for node in self.node_index]
        for key in {key for attrs in self.node_attributes for key in attrs}:
            graph.vs[key] = [attrs.get(key) for attrs in self.node_attributes]
        for key in {key for _, attrs in entries for key in attrs}:
            graph.es[key] = [attrs.get(key) for _, attrs in entries]
        return graph

    def finish_matrix(self, dtype="float64", asymmetric=None):
        if not self.build_matrix:
            return None
        n = len(self.node_index)
        data = np.frombuffer(self.data, dtype=np.float64)
        rows = np.frombuffer(self.rows, dtype=np.int64)
        cols = np.frombuffer(self.cols, dtype=np.int64)
        if asymmetric is False and self.directed:
            off_diagonal = rows != cols
            data = np.concatenate((data, data[off_diagonal]))
            rows, cols = (
                np.concatenate((rows, cols[off_diagonal])),
                np.concatenate((cols, rows[off_diagonal])),
            )
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        matrix.sum_duplicates()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("summed matrix weights must be finite")
        dt = np.dtype(dtype)
        if dt.kind == "f" and np.any(np.abs(matrix.data) > np.finfo(dt).max):
            raise ValueError(f"matrix weights cannot be represented by {dt}")
        if dt.kind in "iu" and matrix.nnz:
            limits = np.iinfo(dt)
            if (
                np.any(matrix.data < limits.min)
                or np.any(matrix.data > limits.max)
                or np.any(matrix.data != np.trunc(matrix.data))
            ):
                raise ValueError(f"matrix weights cannot be represented by {dt}")
        return matrix.astype(dt).tocsr()
