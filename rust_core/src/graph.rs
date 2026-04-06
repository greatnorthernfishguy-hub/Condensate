//! Access Graph — learns memory access topology from observations.
//!
//! Nodes are memory regions. Edges are causal access correlations
//! with timing. Clusters are co-access groups (proto-hyperedges).
//!
//! This replaces the Python GraphBuilder with sub-microsecond performance.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::collections::HashMap;

/// A single access event recorded by the membrane.
#[derive(Clone, Debug)]
pub struct AccessEvent {
    pub timestamp_ns: u64,
    pub path: String,
    pub size_bytes: u64,
}

/// Causal edge: source accessed BEFORE target, with timing statistics.
#[derive(Clone, Debug)]
pub struct CausalEdge {
    pub source_id: u32,
    pub target_id: u32,
    pub count: u32,
    pub mean_delta_ns: f64,
    pub std_delta_ns: f64,
    pub weight: f64,
}

impl CausalEdge {
    fn new(source_id: u32, target_id: u32) -> Self {
        Self {
            source_id,
            target_id,
            count: 0,
            mean_delta_ns: 0.0,
            std_delta_ns: 0.0,
            weight: 0.0,
        }
    }

    /// Welford online update for mean and variance of timing deltas.
    fn add_observation(&mut self, delta_ns: f64) {
        self.count += 1;
        let n = self.count as f64;
        let old_mean = self.mean_delta_ns;
        self.mean_delta_ns += (delta_ns - old_mean) / n;
        // Welford variance accumulator
        self.std_delta_ns += (delta_ns - old_mean) * (delta_ns - self.mean_delta_ns);
    }

    /// Finalize statistics after all observations.
    fn finalize(&mut self) {
        if self.count > 1 {
            self.std_delta_ns = (self.std_delta_ns / (self.count as f64 - 1.0)).sqrt();
        } else {
            self.std_delta_ns = 0.0;
        }
        // Weight: frequency × timing consistency
        // High count + low variance = strong causal edge
        let consistency = 1.0 / (1.0 + self.std_delta_ns / self.mean_delta_ns.max(1.0));
        self.weight = self.count as f64 * consistency;
    }
}

/// A discovered cluster of co-accessed regions (proto-hyperedge).
#[derive(Clone, Debug)]
pub struct Cluster {
    pub id: u32,
    pub member_ids: Vec<u32>,
}

/// Node info: a tracked memory region.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: u32,
    pub path: String,
    pub access_count: u32,
    pub total_bytes: u64,
    pub first_access_ns: u64,
    pub last_access_ns: u64,
}

/// Holographic node boundary — lightweight representation for cold nodes.
/// Fixed size, no heap allocation. Enough for Lenia temperature management,
/// cluster membership checks, and promotion decisions.
/// Full NodeInfo is reconstructed from the path_to_id map only when needed.
#[derive(Clone, Copy, Debug)]
pub struct NodeBoundary {
    pub id: u32,
    pub access_count: u32,
    pub last_access_ns: u64,
    pub cluster_id: Option<u32>,
    pub edge_count: u16,
}

/// The access graph — learns memory access topology.
///
/// Exposed to Python via PyO3.
#[cfg_attr(feature = "python", pyclass)]
pub struct AccessGraph {
    /// Path → node ID mapping
    path_to_id: HashMap<String, u32>,
    /// Node ID → info
    nodes: Vec<NodeInfo>,
    /// (source_id, target_id) → edge
    edges: HashMap<(u32, u32), CausalEdge>,
    /// Discovered clusters
    pub clusters: Vec<Cluster>,
    /// Causal window in nanoseconds
    causal_window_ns: u64,
    /// Cluster co-access threshold
    cluster_threshold: f64,
    /// Whether build() has been called
    built: bool,
    /// Per-node successor list (sorted by weight, top-K)
    successors: Vec<Vec<(u32, f64, f64)>>, // (target_id, weight, mean_delta_ms)
    /// Node → cluster membership
    cluster_map: Vec<Option<u32>>,
}

impl AccessGraph {
    pub fn new(causal_window_ns: u64, cluster_threshold: f64) -> Self {
        Self {
            path_to_id: HashMap::new(),
            nodes: Vec::new(),
            edges: HashMap::new(),
            clusters: Vec::new(),
            causal_window_ns,
            cluster_threshold,
            built: false,
            successors: Vec::new(),
            cluster_map: Vec::new(),
        }
    }

    /// Build the graph from a list of (timestamp_ns, path, size_bytes) events.
    ///
    /// Called from Python with the membrane's access log.
    pub fn build(&mut self, events: Vec<(u64, String, u64)>) {
        if events.is_empty() {
            return;
        }

        // Phase 1: Register nodes
        for (ts, path, size) in &events {
            let id = self.get_or_create_node(path);
            let node = &mut self.nodes[id as usize];
            node.access_count += 1;
            node.total_bytes += size;
            if *ts < node.first_access_ns {
                node.first_access_ns = *ts;
            }
            if *ts > node.last_access_ns {
                node.last_access_ns = *ts;
            }
        }

        // Phase 2: Build causal edges (events are already sorted by timestamp)
        let n = events.len();
        for i in 0..n {
            let (ts_i, ref path_i, _) = events[i];
            let id_i = self.path_to_id[path_i];

            for j in (i + 1)..n {
                let (ts_j, ref path_j, _) = events[j];
                let delta = ts_j - ts_i;

                if delta > self.causal_window_ns {
                    break;
                }

                let id_j = self.path_to_id[path_j];
                if id_i == id_j {
                    continue;
                }

                let edge = self.edges
                    .entry((id_i, id_j))
                    .or_insert_with(|| CausalEdge::new(id_i, id_j));
                edge.add_observation(delta as f64);
            }
        }

        // Finalize edges
        for edge in self.edges.values_mut() {
            edge.finalize();
        }

        // Phase 3: Discover clusters
        self.discover_clusters();

        // Phase 4: Build successor lists for fast prediction
        self.build_successors();

        self.built = true;
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get cluster count.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get node access counts as (path, count) pairs.
    pub fn get_node_stats(&self) -> Vec<(String, u32)> {
        self.nodes.iter()
            .map(|n| (n.path.clone(), n.access_count))
            .collect()
    }

    /// Check if graph has been built.
    pub fn is_built(&self) -> bool {
        self.built
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AccessGraph {
    #[new]
    #[pyo3(signature = (causal_window_ns=5_000_000, cluster_threshold=0.7))]
    fn py_new(causal_window_ns: u64, cluster_threshold: f64) -> Self {
        Self::new(causal_window_ns, cluster_threshold)
    }

    #[pyo3(name = "build")]
    fn py_build(&mut self, events: Vec<(u64, String, u64)>) {
        self.build(events);
    }

    #[pyo3(name = "node_count")]
    fn py_node_count(&self) -> usize {
        self.node_count()
    }

    #[pyo3(name = "edge_count")]
    fn py_edge_count(&self) -> usize {
        self.edge_count()
    }

    #[pyo3(name = "cluster_count")]
    fn py_cluster_count(&self) -> usize {
        self.cluster_count()
    }

    #[pyo3(name = "get_node_stats")]
    fn py_get_node_stats(&self) -> Vec<(String, u32)> {
        self.get_node_stats()
    }
}

// Non-PyO3 internal methods
impl AccessGraph {
    /// Get top edges by weight as (source_path, target_path, count, mean_delta_ms, weight).
    pub fn get_top_edges(&self, limit: usize) -> Vec<(String, String, u32, f64, f64)> {
        let mut edges: Vec<_> = self.edges.values().collect();
        edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        edges.iter()
            .take(limit)
            .map(|e| {
                let src = &self.nodes[e.source_id as usize].path;
                let tgt = &self.nodes[e.target_id as usize].path;
                (src.clone(), tgt.clone(), e.count, e.mean_delta_ns / 1_000_000.0, e.weight)
            })
            .collect()
    }

    fn get_or_create_node(&mut self, path: &str) -> u32 {
        if let Some(&id) = self.path_to_id.get(path) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.path_to_id.insert(path.to_string(), id);
        self.nodes.push(NodeInfo {
            id,
            path: path.to_string(),
            access_count: 0,
            total_bytes: 0,
            first_access_ns: u64::MAX,
            last_access_ns: 0,
        });
        id
    }

    fn discover_clusters(&mut self) {
        let n = self.nodes.len();
        if n < 2 {
            return;
        }

        // Build adjacency directly from edges — O(E), not O(N²).
        // Only node pairs that actually have causal edges get compared.
        // The edges are the evidence; pairs without edges have no
        // co-access relationship and can't be in the same cluster.
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
        for ((src, tgt), edge) in &self.edges {
            let min_count = self.nodes[*src as usize].access_count
                .min(self.nodes[*tgt as usize].access_count)
                .max(1);
            let ratio = edge.count as f64 / min_count as f64;
            if ratio >= self.cluster_threshold {
                adjacency[*src as usize].push(*tgt);
                adjacency[*tgt as usize].push(*src);
            }
        }

        // BFS to find connected components
        let mut visited = vec![false; n];
        let mut cluster_id: u32 = 0;

        // Initialize cluster map
        self.cluster_map = vec![None; n];

        for start in 0..n {
            if visited[start] || adjacency[start].is_empty() {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = vec![start];

            while let Some(node) = queue.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                component.push(node as u32);

                for &neighbor in &adjacency[node] {
                    if !visited[neighbor as usize] {
                        queue.push(neighbor as usize);
                    }
                }
            }

            if component.len() >= 2 {
                for &member_id in &component {
                    self.cluster_map[member_id as usize] = Some(cluster_id);
                }
                self.clusters.push(Cluster {
                    id: cluster_id,
                    member_ids: component,
                });
                cluster_id += 1;
            }
        }
    }

    fn build_successors(&mut self) {
        let n = self.nodes.len();
        let max_weight = self.edges.values()
            .map(|e| e.weight)
            .fold(0.0f64, f64::max)
            .max(1.0);

        self.successors = vec![Vec::new(); n];

        for edge in self.edges.values() {
            if edge.weight < 1.0 {
                continue;
            }
            let norm_weight = edge.weight / max_weight;
            self.successors[edge.source_id as usize].push((
                edge.target_id,
                norm_weight,
                edge.mean_delta_ns / 1_000_000.0,
            ));
        }

        // Sort by weight descending, keep top 10
        for succs in &mut self.successors {
            succs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            succs.truncate(10);
        }
    }

    /// Get successors for a node by path. Used by the predictor.
    pub fn get_successors(&self, path: &str) -> &[(u32, f64, f64)] {
        if let Some(&id) = self.path_to_id.get(path) {
            &self.successors[id as usize]
        } else {
            &[]
        }
    }

    /// Get cluster members for a node by path.
    pub fn get_cluster_members(&self, path: &str) -> Option<&[u32]> {
        let &id = self.path_to_id.get(path)?;
        let cluster_id = self.cluster_map.get(id as usize)?.as_ref()?;
        self.clusters.get(*cluster_id as usize)
            .map(|c| c.member_ids.as_slice())
    }

    /// Get path for a node ID.
    pub fn get_path(&self, id: u32) -> Option<&str> {
        self.nodes.get(id as usize).map(|n| n.path.as_str())
    }

    /// Get holographic boundary for a node — lightweight, no heap allocation.
    /// Enough for temperature management and promotion decisions.
    pub fn get_boundary(&self, id: u32) -> Option<NodeBoundary> {
        let node = self.nodes.get(id as usize)?;
        let edge_count = self.edges.iter()
            .filter(|((s, t), _)| *s == id || *t == id)
            .count() as u16;
        Some(NodeBoundary {
            id: node.id,
            access_count: node.access_count,
            last_access_ns: node.last_access_ns,
            cluster_id: self.cluster_map.get(id as usize).and_then(|c| *c),
            edge_count,
        })
    }

    /// Get boundaries for all nodes — bulk operation for Lenia field seeding.
    /// O(N + E) — scans edges once to count per-node.
    pub fn get_all_boundaries(&self) -> Vec<NodeBoundary> {
        let n = self.nodes.len();
        let mut edge_counts = vec![0u16; n];
        for ((s, t), _) in &self.edges {
            if (*s as usize) < n { edge_counts[*s as usize] = edge_counts[*s as usize].saturating_add(1); }
            if (*t as usize) < n { edge_counts[*t as usize] = edge_counts[*t as usize].saturating_add(1); }
        }
        self.nodes.iter().enumerate().map(|(i, node)| {
            NodeBoundary {
                id: node.id,
                access_count: node.access_count,
                last_access_ns: node.last_access_ns,
                cluster_id: self.cluster_map.get(i).and_then(|c| *c),
                edge_count: edge_counts[i],
            }
        }).collect()
    }

    /// Get node ID for a path.
    pub fn get_id(&self, path: &str) -> Option<u32> {
        self.path_to_id.get(path).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_graph() {
        let mut graph = AccessGraph::new(5_000_000, 0.7);

        // Simulate: A→B→C repeated 10 times, 1ms apart
        let mut events = Vec::new();
        for i in 0..10 {
            let base = i * 3_000_000; // 3ms between iterations
            events.push((base as u64, "A".to_string(), 100));
            events.push((base as u64 + 1_000_000, "B".to_string(), 100));
            events.push((base as u64 + 2_000_000, "C".to_string(), 100));
        }

        graph.build(events);

        assert_eq!(graph.node_count(), 3);
        assert!(graph.edge_count() > 0);
        assert!(graph.is_built());

        // A→B should be a strong edge
        let top = graph.get_top_edges(5);
        assert!(!top.is_empty());
        println!("Top edges: {:?}", top);
    }

    #[test]
    fn test_cluster_discovery() {
        let mut graph = AccessGraph::new(3_000_000, 0.6);

        // Cluster 1: X,Y,Z always together (tight timing)
        // Cluster 2: P,Q always together
        // Gap between clusters
        let mut events = Vec::new();
        for i in 0..30 {
            let base = i * 20_000_000; // 20ms between iterations
            // Cluster 1
            events.push((base as u64, "X".to_string(), 100));
            events.push((base as u64 + 100_000, "Y".to_string(), 100));
            events.push((base as u64 + 200_000, "Z".to_string(), 100));
            // Gap
            // Cluster 2
            events.push((base as u64 + 10_000_000, "P".to_string(), 100));
            events.push((base as u64 + 10_100_000, "Q".to_string(), 100));
        }

        graph.build(events);

        assert!(graph.cluster_count() >= 2, "Should find at least 2 clusters, found {}", graph.cluster_count());
    }

    #[test]
    fn test_successor_lookup() {
        let mut graph = AccessGraph::new(5_000_000, 0.7);

        let mut events = Vec::new();
        for i in 0..50 {
            let base = i * 5_000_000;
            events.push((base as u64, "src".to_string(), 100));
            events.push((base as u64 + 1_000_000, "dst".to_string(), 100));
        }

        graph.build(events);

        let succs = graph.get_successors("src");
        assert!(!succs.is_empty(), "src should have successors");
        assert_eq!(graph.get_path(succs[0].0), Some("dst"));
    }
}
