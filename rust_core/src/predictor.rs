//! Predictor — predicts next memory access from learned topology.
//!
//! Proto-SNN: causal spike propagation through learned graph.
//! When a path is accessed, spikes propagate to predicted-next
//! paths via direct successors, causal chains, and cluster co-activation.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use crate::graph::AccessGraph;

/// A single prediction: what will be accessed, when, how confident.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct Prediction {
    #[cfg_attr(feature = "python", pyo3(get))]
    pub path: String,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub confidence: f64,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub expected_delta_ms: f64,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub source_path: String,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub chain_depth: u32,
}

#[cfg_attr(feature = "python", pymethods)]
impl Prediction {
    fn __repr__(&self) -> String {
        format!(
            "Prediction({}, conf={:.2}, dt={:.2}ms, depth={})",
            self.path, self.confidence, self.expected_delta_ms, self.chain_depth
        )
    }
}

/// Scoring results from prediction evaluation.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug)]
pub struct ScoreResult {
    #[cfg_attr(feature = "python", pyo3(get))]
    pub predictions_made: u32,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub hits: u32,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub misses: u32,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub accuracy: f64,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub direct_hits: u32,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub chain_hits: u32,
    #[cfg_attr(feature = "python", pyo3(get))]
    pub cluster_hits: u32,
}

/// The predictor — predicts next access from learned causal topology.
///
/// This is the proto-SNN. Production replaces this with real NeuroGraph
/// spike propagation.
#[cfg_attr(feature = "python", pyclass)]
pub struct RustPredictor {
    /// Reference to the graph we learned from
    /// (We store a copy of the data we need)
    learned: bool,

    /// Successors per node: (target_id, weight, delta_ms)
    successors: Vec<Vec<(u32, f64, f64)>>,

    /// Cluster membership: node_id → cluster_id
    cluster_map: Vec<Option<u32>>,

    /// Cluster members: cluster_id → [node_ids]
    cluster_members: Vec<Vec<u32>>,

    /// Path ↔ ID mapping
    path_to_id: std::collections::HashMap<String, u32>,
    id_to_path: Vec<String>,

    /// Scoring window in nanoseconds
    score_window_ns: u64,
}

#[cfg_attr(feature = "python", pymethods)]
impl RustPredictor {
    #[cfg_attr(feature = "python", new)]
    pub fn new() -> Self {
        Self {
            learned: false,
            successors: Vec::new(),
            cluster_map: Vec::new(),
            cluster_members: Vec::new(),
            path_to_id: std::collections::HashMap::new(),
            id_to_path: Vec::new(),
            score_window_ns: 10_000_000, // 10ms
        }
    }

    /// Learn from a built AccessGraph.
    pub fn learn(&mut self, graph: &AccessGraph) {
        // Copy the data we need from the graph
        let stats = graph.get_node_stats();
        let n = stats.len();

        self.id_to_path = Vec::with_capacity(n);
        self.path_to_id = std::collections::HashMap::with_capacity(n);

        for (i, (path, _count)) in stats.iter().enumerate() {
            self.id_to_path.push(path.to_string());
            self.path_to_id.insert(path.to_string(), i as u32);
        }

        // Copy successors
        self.successors = Vec::with_capacity(n);
        for (path, _) in &stats {
            let succs = graph.get_successors(path);
            self.successors.push(succs.to_vec());
        }

        // Copy cluster data
        self.cluster_map = Vec::with_capacity(n);
        for (path, _) in &stats {
            if let Some(members) = graph.get_cluster_members(path) {
                // Find which cluster this node belongs to
                let cluster_id = self.cluster_members.len();
                // Check if we already added this cluster
                let mut found = false;
                for (cid, existing) in self.cluster_members.iter().enumerate() {
                    if existing.contains(&self.path_to_id[path]) {
                        self.cluster_map.push(Some(cid as u32));
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.cluster_map.push(Some(cluster_id as u32));
                    self.cluster_members.push(members.to_vec());
                }
            } else {
                self.cluster_map.push(None);
            }
        }

        self.learned = true;
    }

    /// Predict what will be accessed next after `path`.
    ///
    /// Returns top-K predictions sorted by confidence.
    #[cfg_attr(feature = "python", pyo3(signature = (path, top_k=10)))]
    pub fn predict(&self, path: &str, top_k: usize) -> Vec<Prediction> {
        if !self.learned {
            return Vec::new();
        }

        let id = match self.path_to_id.get(path) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        // Collect predictions, keeping best confidence per target
        let mut best: std::collections::HashMap<u32, Prediction> =
            std::collections::HashMap::new();

        // Source 1: Direct successors
        for &(target_id, weight, delta_ms) in &self.successors[id as usize] {
            let target_path = &self.id_to_path[target_id as usize];
            let pred = Prediction {
                path: target_path.clone(),
                confidence: weight,
                expected_delta_ms: delta_ms,
                source_path: path.to_string(),
                chain_depth: 1,
            };
            let entry = best.entry(target_id).or_insert(pred.clone());
            if pred.confidence > entry.confidence {
                *entry = pred;
            }
        }

        // Source 2: Cluster co-activation
        if let Some(cluster_id) = self.cluster_map.get(id as usize).and_then(|c| *c) {
            if let Some(members) = self.cluster_members.get(cluster_id as usize) {
                for &member_id in members {
                    if member_id == id {
                        continue;
                    }
                    let member_path = &self.id_to_path[member_id as usize];
                    let pred = Prediction {
                        path: member_path.clone(),
                        confidence: 0.85,
                        expected_delta_ms: 0.1,
                        source_path: path.to_string(),
                        chain_depth: 1,
                    };
                    let entry = best.entry(member_id).or_insert(pred.clone());
                    if pred.confidence > entry.confidence {
                        *entry = pred;
                    }
                }
            }
        }

        // Sort by confidence, return top_k
        let mut result: Vec<Prediction> = best.into_values().collect();
        result.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        result.truncate(top_k);
        result
    }

    /// Score prediction accuracy against an access log.
    ///
    /// events: list of (timestamp_ns, path, size_bytes)
    pub fn score(&self, events: Vec<(u64, String, u64)>) -> ScoreResult {
        if !self.learned || events.is_empty() {
            return ScoreResult {
                predictions_made: 0, hits: 0, misses: 0, accuracy: 0.0,
                direct_hits: 0, chain_hits: 0, cluster_hits: 0,
            };
        }

        let mut hits: u32 = 0;
        let mut misses: u32 = 0;
        let mut predictions_made: u32 = 0;
        let mut direct_hits: u32 = 0;
        let mut chain_hits: u32 = 0;
        let mut cluster_hits: u32 = 0;

        let n = events.len();

        for i in 0..n.saturating_sub(1) {
            let (ts_i, ref path_i, _) = events[i];

            let preds = self.predict(path_i, 10);
            if preds.is_empty() {
                continue;
            }
            predictions_made += 1;

            // Build prediction set for fast lookup
            let pred_set: std::collections::HashMap<&str, &Prediction> = preds
                .iter()
                .map(|p| (p.path.as_str(), p))
                .collect();

            // Check what actually came next within scoring window
            let mut hit = false;
            for j in (i + 1)..n {
                let (ts_j, ref path_j, _) = events[j];
                let delta = ts_j - ts_i;

                if delta > self.score_window_ns {
                    break;
                }

                if let Some(pred) = pred_set.get(path_j.as_str()) {
                    hit = true;
                    if pred.chain_depth > 1 {
                        chain_hits += 1;
                    } else if self.path_to_id.get(path_j.as_str())
                        .and_then(|&id| self.cluster_map.get(id as usize))
                        .and_then(|c| *c)
                        .is_some()
                    {
                        cluster_hits += 1;
                    } else {
                        direct_hits += 1;
                    }
                    break;
                }
            }

            if hit {
                hits += 1;
            } else {
                misses += 1;
            }
        }

        let accuracy = if predictions_made > 0 {
            (hits as f64 / predictions_made as f64) * 100.0
        } else {
            0.0
        };

        ScoreResult {
            predictions_made,
            hits,
            misses,
            accuracy: (accuracy * 10.0).round() / 10.0,
            direct_hits,
            chain_hits,
            cluster_hits,
        }
    }

    /// Check if predictor has learned.
    pub fn is_learned(&self) -> bool {
        self.learned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AccessGraph;

    #[test]
    fn test_predict_sequential() {
        let mut graph = AccessGraph::new(5_000_000, 0.7);

        // A→B→C, repeated 20 times
        let mut events = Vec::new();
        for i in 0..20 {
            let base = i * 5_000_000u64;
            events.push((base, "A".to_string(), 100));
            events.push((base + 500_000, "B".to_string(), 100));
            events.push((base + 1_000_000, "C".to_string(), 100));
        }

        graph.build(events);

        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        let preds = predictor.predict("A", 5);
        assert!(!preds.is_empty(), "Should have predictions for A");

        let pred_paths: Vec<&str> = preds.iter().map(|p| p.path.as_str()).collect();
        assert!(pred_paths.contains(&"B"), "Should predict B after A, got {:?}", pred_paths);
    }

    #[test]
    fn test_score_accuracy() {
        let mut graph = AccessGraph::new(5_000_000, 0.7);

        let mut events = Vec::new();
        for i in 0..50 {
            let base = i * 5_000_000u64;
            events.push((base, "X".to_string(), 100));
            events.push((base + 1_000_000, "Y".to_string(), 100));
            events.push((base + 2_000_000, "Z".to_string(), 100));
        }

        graph.build(events.clone());

        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        let result = predictor.score(events);
        println!("Accuracy: {}%", result.accuracy);
        assert!(result.accuracy > 50.0, "Accuracy should be > 50%, got {}%", result.accuracy);
        assert!(result.predictions_made > 0);
    }
}
