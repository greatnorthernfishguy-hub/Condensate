"""
Condensate Layer 1: The Graph Builder

Takes access logs from the Membrane (Layer 0) and builds a weighted
graph of memory access patterns. Discovers:

  - Temporal edges: A accessed near B → weighted edge
  - Causal chains: A always before B → directed edge with timing
  - Clusters: groups of regions always accessed together (proto-hyperedges)
  - Hot/cold classification: access frequency distribution

This is the substrate's raw material. Layer 2 (predictor) will use
this graph to predict future accesses.

Usage:
    from membrane import Membrane
    from graph_builder import GraphBuilder

    # ... run workload with Membrane wrapping ...
    log = Membrane.get_log()

    graph = GraphBuilder()
    graph.build(log)
    graph.print_analysis()
    graph.save("access_graph.json")
"""

import numpy as np
from collections import defaultdict
import json


class AccessNode:
    """A memory region tracked in the graph."""

    __slots__ = ['path', 'access_count', 'read_count', 'write_count',
                 'total_bytes', 'first_access_ns', 'last_access_ns',
                 'access_times_ns', '_temp_class']

    def __init__(self, path):
        self.path = path
        self.access_count = 0
        self.read_count = 0
        self.write_count = 0
        self.total_bytes = 0
        self.first_access_ns = float('inf')
        self.last_access_ns = 0
        self.access_times_ns = []
        self._temp_class = "WARM"  # default

    def record(self, ts_ns, event_type, size_bytes):
        self.access_count += 1
        if event_type == "READ":
            self.read_count += 1
        else:
            self.write_count += 1
        self.total_bytes += size_bytes
        self.first_access_ns = min(self.first_access_ns, ts_ns)
        self.last_access_ns = max(self.last_access_ns, ts_ns)
        self.access_times_ns.append(ts_ns)

    @property
    def temperature(self):
        """Normalized access frequency. Higher = hotter."""
        return self.access_count

    def to_dict(self):
        return {
            "path": self.path,
            "access_count": self.access_count,
            "reads": self.read_count,
            "writes": self.write_count,
            "total_bytes": self.total_bytes,
        }


class CausalEdge:
    """A directed edge: source is accessed BEFORE target."""

    __slots__ = ['source', 'target', 'count', 'timing_deltas_ns',
                 'mean_delta_ns', 'std_delta_ns', 'weight']

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.count = 0
        self.timing_deltas_ns = []
        self.mean_delta_ns = 0.0
        self.std_delta_ns = 0.0
        self.weight = 0.0  # computed after all edges built

    def add_observation(self, delta_ns):
        self.count += 1
        self.timing_deltas_ns.append(delta_ns)

    def finalize(self):
        """Compute statistics after all observations."""
        if self.timing_deltas_ns:
            arr = np.array(self.timing_deltas_ns, dtype=np.float64)
            self.mean_delta_ns = float(np.mean(arr))
            self.std_delta_ns = float(np.std(arr))
            # Weight: frequency × timing consistency
            # High count + low variance = strong causal edge
            consistency = 1.0 / (1.0 + self.std_delta_ns / max(self.mean_delta_ns, 1.0))
            self.weight = self.count * consistency

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "count": self.count,
            "mean_delta_ms": round(self.mean_delta_ns / 1_000_000, 3),
            "std_delta_ms": round(self.std_delta_ns / 1_000_000, 3),
            "weight": round(self.weight, 2),
        }


class Cluster:
    """A group of paths always accessed together — proto-hyperedge."""

    def __init__(self, cluster_id, members):
        self.cluster_id = cluster_id
        self.members = set(members)
        self.total_coaccesses = 0

    def to_dict(self):
        return {
            "id": self.cluster_id,
            "members": sorted(self.members),
            "size": len(self.members),
            "total_coaccesses": self.total_coaccesses,
        }


class GraphBuilder:
    """Builds a weighted access pattern graph from Membrane logs.

    The graph has:
      - Nodes: memory regions (paths) with access statistics
      - Causal edges: directed, weighted, with timing information
      - Clusters: groups of paths that always co-access (proto-hyperedges)
    """

    def __init__(self, causal_window_ns=5_000_000, cluster_threshold=0.7):
        """
        Args:
            causal_window_ns: Max time gap (ns) to consider causal.
                              Default 5ms — wide enough for Python overhead.
            cluster_threshold: Co-access ratio to form a cluster.
                               0.7 = paths must co-access 70%+ of the time.
        """
        self.causal_window_ns = causal_window_ns
        self.cluster_threshold = cluster_threshold

        self.nodes = {}          # path → AccessNode
        self.edges = {}          # (source, target) → CausalEdge
        self.clusters = []       # list of Cluster
        self._built = False

    def build(self, log_entries):
        """Build the graph from Membrane log entries.

        Args:
            log_entries: list of (timestamp_ns, event_type, path, size_bytes)
        """
        if not log_entries:
            print("  Warning: empty log, nothing to build")
            return

        # Phase 1: Build nodes
        for ts, event_type, path, size_bytes in log_entries:
            if path not in self.nodes:
                self.nodes[path] = AccessNode(path)
            self.nodes[path].record(ts, event_type, size_bytes)

        # Phase 2: Build causal edges
        # Sort by timestamp for sequential scanning
        sorted_log = sorted(log_entries, key=lambda e: e[0])

        for i, (ts_i, _, path_i, _) in enumerate(sorted_log):
            # Look forward within the causal window
            for j in range(i + 1, len(sorted_log)):
                ts_j, _, path_j, _ = sorted_log[j]
                delta = ts_j - ts_i

                if delta > self.causal_window_ns:
                    break  # past the window

                if path_i == path_j:
                    continue  # self-loop, skip

                # Directed edge: i happened before j
                key = (path_i, path_j)
                if key not in self.edges:
                    self.edges[key] = CausalEdge(path_i, path_j)
                self.edges[key].add_observation(delta)

        # Finalize edge statistics
        for edge in self.edges.values():
            edge.finalize()

        # Phase 3: Discover clusters (proto-hyperedges)
        self._discover_clusters()

        # Phase 4: Classify temperature
        self._classify_temperature()

        self._built = True

    def _discover_clusters(self):
        """Find groups of paths that are consistently co-accessed.

        Uses a simple greedy approach:
        1. For each pair of paths, compute co-access ratio
        2. Build adjacency from pairs above threshold
        3. Connected components = clusters
        """
        if len(self.nodes) < 2:
            return

        paths = list(self.nodes.keys())
        n = len(paths)

        # Build co-access matrix
        # co_access[i][j] = times i and j were accessed within window / min(count_i, count_j)
        path_to_idx = {p: i for i, p in enumerate(paths)}

        cocount = np.zeros((n, n), dtype=np.int32)

        for (src, tgt), edge in self.edges.items():
            i, j = path_to_idx.get(src), path_to_idx.get(tgt)
            if i is not None and j is not None:
                cocount[i][j] += edge.count
                cocount[j][i] += edge.count

        # Normalize to co-access ratio
        counts = np.array([self.nodes[p].access_count for p in paths], dtype=np.float64)
        min_counts = np.minimum.outer(counts, counts)
        min_counts = np.maximum(min_counts, 1.0)  # avoid div by zero
        coratio = cocount / min_counts

        # Build adjacency and find connected components
        adjacency = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if coratio[i][j] >= self.cluster_threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # BFS to find connected components
        visited = set()
        cluster_id = 0

        for start in range(n):
            if start in visited:
                continue
            if start not in adjacency:
                continue

            # BFS
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adjacency.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 2:
                members = [paths[i] for i in component]
                cluster = Cluster(cluster_id, members)

                # Sum co-access counts within cluster
                for i in component:
                    for j in component:
                        if i != j:
                            cluster.total_coaccesses += cocount[i][j]

                self.clusters.append(cluster)
                cluster_id += 1

    def _classify_temperature(self):
        """Tag nodes as hot/warm/cold based on access distribution."""
        if not self.nodes:
            return

        counts = [n.access_count for n in self.nodes.values()]
        if not counts:
            return

        # Use percentiles for classification
        p75 = np.percentile(counts, 75)
        p25 = np.percentile(counts, 25)

        for node in self.nodes.values():
            if node.access_count >= p75:
                node._temp_class = "HOT"
            elif node.access_count >= p25:
                node._temp_class = "WARM"
            else:
                node._temp_class = "COLD"

    def get_causal_chains(self, min_weight=2.0, max_depth=10):
        """Extract causal chains — sequences of A→B→C with strong edges.

        Returns list of chains, each chain is [(path, mean_delta_ms), ...]
        """
        if not self._built:
            return []

        # Build adjacency list of strong edges, sorted by weight
        successors = defaultdict(list)
        for (src, tgt), edge in self.edges.items():
            if edge.weight >= min_weight:
                successors[src].append((tgt, edge))

        # Sort successors by weight descending
        for src in successors:
            successors[src].sort(key=lambda x: -x[1].weight)

        # Find chains starting from each node
        chains = []
        visited_starts = set()

        # Start from nodes that have strong outgoing but weak incoming
        incoming_weight = defaultdict(float)
        outgoing_weight = defaultdict(float)
        for (src, tgt), edge in self.edges.items():
            if edge.weight >= min_weight:
                outgoing_weight[src] += edge.weight
                incoming_weight[tgt] += edge.weight

        # Good chain starts: strong outgoing, weaker incoming
        candidates = []
        for path in successors:
            out_w = outgoing_weight.get(path, 0)
            in_w = incoming_weight.get(path, 0)
            if out_w > 0:
                candidates.append((path, out_w - in_w))

        candidates.sort(key=lambda x: -x[1])

        for start, _ in candidates:
            if start in visited_starts:
                continue

            # Follow the strongest chain
            chain = [(start, 0.0)]
            current = start
            seen = {start}

            for _ in range(max_depth):
                if current not in successors:
                    break
                # Take the strongest unvisited successor
                found = False
                for next_path, edge in successors[current]:
                    if next_path not in seen:
                        chain.append((next_path, edge.mean_delta_ns / 1_000_000))
                        seen.add(next_path)
                        current = next_path
                        found = True
                        break
                if not found:
                    break

            if len(chain) >= 2:
                chains.append(chain)
                visited_starts.update(p for p, _ in chain)

        return chains

    def print_analysis(self):
        """Print a comprehensive analysis of the access graph."""
        if not self._built:
            print("  Graph not built yet. Call build() first.")
            return

        print(f"\n{'='*60}")
        print(f"  CONDENSATE — Layer 1 Graph Analysis")
        print(f"{'='*60}")

        # Node summary
        hot = [n for n in self.nodes.values() if getattr(n, '_temp_class', '') == 'HOT']
        warm = [n for n in self.nodes.values() if getattr(n, '_temp_class', '') == 'WARM']
        cold = [n for n in self.nodes.values() if getattr(n, '_temp_class', '') == 'COLD']

        print(f"\n  Nodes: {len(self.nodes)} total")
        print(f"    HOT:  {len(hot)} (top 25% access frequency)")
        print(f"    WARM: {len(warm)} (middle 50%)")
        print(f"    COLD: {len(cold)} (bottom 25%)")

        if hot:
            print(f"\n  Hottest nodes:")
            for node in sorted(hot, key=lambda n: -n.access_count)[:10]:
                print(f"    {node.path:<42} {node.access_count:>5} accesses")

        if cold:
            print(f"\n  Coldest nodes:")
            for node in sorted(cold, key=lambda n: n.access_count)[:5]:
                print(f"    {node.path:<42} {node.access_count:>5} accesses")

        # Edge summary
        strong_edges = [(k, e) for k, e in self.edges.items() if e.weight >= 2.0]
        print(f"\n  Edges: {len(self.edges)} total, {len(strong_edges)} strong (weight >= 2.0)")

        if strong_edges:
            print(f"\n  Strongest causal edges (A → B):")
            print(f"  {'Source':<25} {'→ Target':<25} {'Count':>5} {'Δt(ms)':>7} {'Wt':>6}")
            print(f"  {'-'*25} {'-'*25} {'-'*5} {'-'*7} {'-'*6}")

            sorted_edges = sorted(strong_edges, key=lambda x: -x[1].weight)
            for (src, tgt), edge in sorted_edges[:15]:
                src_short = src if len(src) <= 25 else "..." + src[-22:]
                tgt_short = tgt if len(tgt) <= 25 else "..." + tgt[-22:]
                print(f"  {src_short:<25} {tgt_short:<25} "
                      f"{edge.count:>5} {edge.mean_delta_ns/1e6:>7.3f} {edge.weight:>6.1f}")

        # Cluster summary
        if self.clusters:
            print(f"\n  Clusters (proto-hyperedges): {len(self.clusters)}")
            for cluster in sorted(self.clusters, key=lambda c: -len(c.members)):
                print(f"\n    Cluster {cluster.cluster_id} "
                      f"({len(cluster.members)} members, "
                      f"{cluster.total_coaccesses} co-accesses):")
                for member in sorted(cluster.members):
                    node = self.nodes.get(member)
                    temp = getattr(node, '_temp_class', '?') if node else '?'
                    count = node.access_count if node else 0
                    print(f"      [{temp:>4}] {member:<40} {count:>4}x")
        else:
            print(f"\n  Clusters: none found (threshold: {self.cluster_threshold})")

        # Causal chains
        chains = self.get_causal_chains()
        if chains:
            print(f"\n  Causal chains discovered: {len(chains)}")
            for i, chain in enumerate(chains[:5]):
                parts = []
                for path, delta_ms in chain:
                    short = path.split(".")[-1] if "." in path else path
                    if delta_ms > 0:
                        parts.append(f"--({delta_ms:.2f}ms)--> {short}")
                    else:
                        parts.append(short)
                print(f"    Chain {i}: {' '.join(parts)}")
            if len(chains) > 5:
                print(f"    ... and {len(chains) - 5} more chains")

        # Condensation potential
        if hot and cold:
            hot_accesses = sum(n.access_count for n in hot)
            total_accesses = sum(n.access_count for n in self.nodes.values())
            hot_pct = hot_accesses / total_accesses * 100
            print(f"\n  Condensation potential:")
            print(f"    {len(hot)} hot nodes handle {hot_pct:.0f}% of all accesses")
            print(f"    {len(cold)} cold nodes could be compressed/paged")
            if self.clusters:
                print(f"    {len(self.clusters)} clusters enable batch promote/demote")
            if chains:
                print(f"    {len(chains)} causal chains enable predictive prefetch")

        print(f"\n{'='*60}\n")

    def save(self, filepath):
        """Save the graph to JSON for later analysis."""
        data = {
            "nodes": {p: n.to_dict() for p, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges.values() if e.weight >= 1.0],
            "clusters": [c.to_dict() for c in self.clusters],
            "chains": self.get_causal_chains(),
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "strong_edges": sum(1 for e in self.edges.values() if e.weight >= 2.0),
                "clusters": len(self.clusters),
                "chains": len(self.get_causal_chains()),
            }
        }
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return super().default(obj)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        print(f"  Saved graph ({len(self.nodes)} nodes, "
              f"{len(self.edges)} edges) to {filepath}")
