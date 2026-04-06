"""Condensate Graph Builder — delegates to Rust AccessGraph."""
import condensate_core


class GraphBuilder:
    def __init__(self, causal_window_ns=5_000_000, cluster_threshold=0.7):
        self._graph = condensate_core.AccessGraph(causal_window_ns, cluster_threshold)

    def build(self, events):
        """Build graph from (timestamp_ns, path, size_bytes) events."""
        self._graph.build(events)

    def node_count(self):
        return self._graph.node_count()

    def edge_count(self):
        return self._graph.edge_count()

    def cluster_count(self):
        return self._graph.cluster_count()

    def get_node_stats(self):
        return self._graph.get_node_stats()

    @property
    def inner(self):
        """Access the Rust AccessGraph directly."""
        return self._graph
