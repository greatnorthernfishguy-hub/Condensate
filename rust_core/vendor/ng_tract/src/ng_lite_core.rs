//! NG-Lite Rust Core — Hebbian learning substrate for the E-T Systems ecosystem.
//!
//! This is the hot-path engine behind ng_lite.py. All data structures and
//! learning algorithms live here. Python calls in via PyO3.
//!
//! # ---- Changelog ----
//! # [2026-04-04] Claude Code (Opus 4.6) — Rebuild from scratch after sed destruction
//! #   What: Complete NG-Lite core with Hebbian learning, Welford variance,
//! #         receptor layer, constitutional nodes, binary persistence (msgpack).
//! #   Why:  Original file destroyed by bad sed command. Rebuild to spec.
//! #   How:  PyO3 pyclass with all hot-path methods matching Python ng_lite.py.
//! # -------------------

use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::types::PyList;
#[cfg(feature = "python")]
use pyo3::IntoPyObjectExt;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods};

use sha2::{Sha256, Digest};

// --- Internal structs ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Node {
    node_id: String,
    embedding_hash: String,
    activation_count: u32,
    last_activation: f64,
    metadata: HashMap<String, String>,
    embedding: Option<Vec<f32>>,
    constitutional: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Synapse {
    source_id: String,
    target_id: String,
    weight: f32,
    activation_count: u32,
    success_count: u32,
    failure_count: u32,
    last_updated: f64,
    metadata: HashMap<String, String>,
    welford_count: u32,
    welford_mean: f64,
    welford_m2: f64,
}

impl Synapse {
    fn variance(&self) -> f64 {
        if self.welford_count < 2 {
            0.0
        } else {
            self.welford_m2 / (self.welford_count as f64 - 1.0)
        }
    }

    fn is_contested(&self) -> bool {
        self.variance() > 0.002 && self.weight >= 0.15 && self.weight <= 0.85
    }
}

// --- State snapshot for binary persistence ---

#[derive(serde::Serialize, serde::Deserialize)]
struct StateSnapshot {
    module_id: String,
    nodes: HashMap<String, Node>,
    synapses: HashMap<String, Synapse>,
    embedding_cache: HashMap<String, Vec<f32>>,
    prototypes: Option<Vec<f32>>,
    prototype_counts: Option<Vec<u32>>,
    prototype_k: usize,
    receptor_input_count: u32,
    max_nodes: usize,
    max_synapses: usize,
    success_boost: f32,
    failure_penalty: f32,
    novelty_threshold: f32,
    pruning_threshold: f32,
    embedding_dim: usize,
    hash_dims: usize,
    receptor_enabled: bool,
    receptor_prototype_threshold: f32,
    receptor_ema_alpha: f32,
    receptor_warmup_count: u32,
    node_id_counter: u32,
    total_outcomes: u64,
    total_successes: u64,
}

// --- Helper functions ---

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    let inv = 1.0 / norm;
    v.iter().map(|x| x * inv).collect()
}

fn hash_embedding(emb: &[f32], hash_dims: usize) -> String {
    let dims = hash_dims.min(emb.len());
    let mut hasher = Sha256::new();
    for &val in &emb[..dims] {
        hasher.update(val.to_le_bytes());
    }
    let result = hasher.finalize();
    let hex = format!("{:x}", result);
    hex[..32.min(hex.len())].to_string()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// --- Tunable parameter bounds ---

struct ParamBounds {
    min: f64,
    max: f64,
}

fn tunable_params() -> HashMap<&'static str, ParamBounds> {
    let mut m = HashMap::new();
    m.insert("success_boost", ParamBounds { min: 0.01, max: 0.50 });
    m.insert("failure_penalty", ParamBounds { min: 0.01, max: 0.50 });
    m.insert("novelty_threshold", ParamBounds { min: 0.30, max: 0.95 });
    m.insert("pruning_threshold", ParamBounds { min: 0.001, max: 0.10 });
    m.insert("receptor_ema_alpha", ParamBounds { min: 0.0001, max: 0.01 });
    m.insert("receptor_prototype_threshold", ParamBounds { min: 0.50, max: 0.95 });
    m.insert("relevance_threshold", ParamBounds { min: 0.10, max: 0.70 });
    m
}

// --- PyO3 class ---

#[cfg(feature = "python")]
#[pyclass(name = "NGLiteCore")]
pub struct PyNGLiteCore {
    module_id: String,
    nodes: HashMap<String, Node>,
    synapses: HashMap<(String, String), Synapse>,
    embedding_cache: HashMap<String, Vec<f32>>,
    prototypes: Option<Vec<f32>>,
    prototype_counts: Option<Vec<u32>>,
    prototype_k: usize,
    receptor_input_count: u32,
    max_nodes: usize,
    max_synapses: usize,
    success_boost: f32,
    failure_penalty: f32,
    novelty_threshold: f32,
    pruning_threshold: f32,
    embedding_dim: usize,
    hash_dims: usize,
    receptor_enabled: bool,
    receptor_prototype_threshold: f32,
    receptor_ema_alpha: f32,
    receptor_warmup_count: u32,
    node_id_counter: u32,
    total_outcomes: u64,
    total_successes: u64,
    history_max: usize,
}

// --- Internal methods ---

#[cfg(feature = "python")]
impl PyNGLiteCore {
    fn find_similar(&self, emb: &[f32]) -> Option<(&str, f32)> {
        let threshold = 1.0 - self.novelty_threshold;
        let mut best: Option<(&str, f32)> = None;
        for (hash, cached_emb) in &self.embedding_cache {
            let sim = dot_product(emb, cached_emb);
            if sim >= threshold {
                if best.is_none() || sim > best.unwrap().1 {
                    best = Some((hash.as_str(), sim));
                }
            }
        }
        best
    }

    fn snap_to_prototype(&mut self, emb: &[f32]) -> Vec<f32> {
        if !self.receptor_enabled {
            return emb.to_vec();
        }
        let protos = match &self.prototypes {
            Some(p) => p,
            None => return emb.to_vec(),
        };
        let k = self.prototype_k;
        if k == 0 {
            return emb.to_vec();
        }
        let dim = self.embedding_dim;

        // Find best prototype
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for i in 0..k {
            let start = i * dim;
            let end = start + dim;
            if end > protos.len() {
                break;
            }
            let proto = &protos[start..end];
            let sim = dot_product(emb, proto);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        if best_sim < self.receptor_prototype_threshold {
            return emb.to_vec();
        }

        // EMA drift toward prototype
        let alpha = self.receptor_ema_alpha;
        let start = best_idx * dim;
        let protos_mut = match &mut self.prototypes {
            Some(p) => p,
            None => return emb.to_vec(),
        };
        let mut snapped = Vec::with_capacity(dim);
        for j in 0..dim {
            let proto_val = protos_mut[start + j];
            let new_val = proto_val * (1.0 - alpha) + emb[j] * alpha;
            protos_mut[start + j] = new_val;
            snapped.push(new_val);
        }

        // Update count
        if let Some(ref mut counts) = self.prototype_counts {
            if best_idx < counts.len() {
                counts[best_idx] += 1;
            }
        }

        self.receptor_input_count += 1;
        normalize(&snapped)
    }

    fn init_prototypes(&mut self) {
        let dim = self.embedding_dim;
        let k = self.prototype_k;
        if self.embedding_cache.len() < k || k == 0 {
            return;
        }

        // Collect embeddings
        let embeddings: Vec<&Vec<f32>> = self.embedding_cache.values().collect();
        let n = embeddings.len();

        // Initialize prototypes by picking k evenly-spaced embeddings
        let mut protos = vec![0.0f32; k * dim];
        for i in 0..k {
            let idx = (i * n) / k;
            let src = embeddings[idx];
            let start = i * dim;
            for j in 0..dim.min(src.len()) {
                protos[start + j] = src[j];
            }
        }

        // K-means: 20 iterations
        let mut assignments = vec![0usize; n];
        for _iter in 0..20 {
            // Assign each embedding to nearest prototype
            for (ei, emb) in embeddings.iter().enumerate() {
                let mut best_k = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for ki in 0..k {
                    let start = ki * dim;
                    let proto = &protos[start..start + dim];
                    let sim = dot_product(emb, proto);
                    if sim > best_sim {
                        best_sim = sim;
                        best_k = ki;
                    }
                }
                assignments[ei] = best_k;
            }

            // Recompute prototypes
            let mut new_protos = vec![0.0f32; k * dim];
            let mut counts = vec![0u32; k];
            for (ei, emb) in embeddings.iter().enumerate() {
                let ki = assignments[ei];
                counts[ki] += 1;
                let start = ki * dim;
                for j in 0..dim.min(emb.len()) {
                    new_protos[start + j] += emb[j];
                }
            }
            for ki in 0..k {
                if counts[ki] > 0 {
                    let start = ki * dim;
                    let inv = 1.0 / counts[ki] as f32;
                    for j in 0..dim {
                        new_protos[start + j] *= inv;
                    }
                    // Normalize prototype
                    let norm: f32 = (0..dim).map(|j| new_protos[start + j].powi(2)).sum::<f32>().sqrt();
                    if norm > 1e-12 {
                        let inv_n = 1.0 / norm;
                        for j in 0..dim {
                            new_protos[start + j] *= inv_n;
                        }
                    }
                }
            }
            protos = new_protos;
        }

        let counts = vec![0u32; k];
        self.prototypes = Some(protos);
        self.prototype_counts = Some(counts);
    }

    fn prune_least_used_node(&mut self) {
        let mut worst: Option<(String, u32)> = None;
        for (nid, node) in &self.nodes {
            if node.constitutional {
                continue;
            }
            match &worst {
                None => worst = Some((nid.clone(), node.activation_count)),
                Some((_, count)) => {
                    if node.activation_count < *count {
                        worst = Some((nid.clone(), node.activation_count));
                    }
                }
            }
        }
        if let Some((nid, _)) = worst {
            // Remove associated synapses
            self.synapses.retain(|k, _| k.0 != nid && k.1 != nid);
            // Remove from embedding cache
            if let Some(node) = self.nodes.get(&nid) {
                self.embedding_cache.remove(&node.embedding_hash);
            }
            self.nodes.remove(&nid);
        }
    }

    fn prune_weakest_synapse(&mut self) {
        let mut worst: Option<((String, String), f32)> = None;
        for (key, syn) in &self.synapses {
            match &worst {
                None => worst = Some((key.clone(), syn.weight)),
                Some((_, w)) => {
                    if syn.weight < *w {
                        worst = Some((key.clone(), syn.weight));
                    }
                }
            }
        }
        if let Some((key, _)) = worst {
            self.synapses.remove(&key);
        }
    }

    fn build_local_reasoning(syn: &Synapse) -> String {
        let total = syn.success_count + syn.failure_count;
        let avg_strength = if total > 0 {
            syn.weight as f64 / total as f64
        } else {
            0.0
        };
        format!(
            "Hebbian: {}/{} success (w={:.3}, {} activations, avg_strength={:.4})",
            syn.success_count, total, syn.weight, syn.activation_count, avg_strength
        )
    }

    fn to_snapshot(&self) -> StateSnapshot {
        // Convert (String,String) keyed synapses to "src|tgt" keyed
        let mut syn_map = HashMap::new();
        for ((src, tgt), syn) in &self.synapses {
            let key = format!("{}|{}", src, tgt);
            syn_map.insert(key, syn.clone());
        }
        StateSnapshot {
            module_id: self.module_id.clone(),
            nodes: self.nodes.clone(),
            synapses: syn_map,
            embedding_cache: self.embedding_cache.clone(),
            prototypes: self.prototypes.clone(),
            prototype_counts: self.prototype_counts.clone(),
            prototype_k: self.prototype_k,
            receptor_input_count: self.receptor_input_count,
            max_nodes: self.max_nodes,
            max_synapses: self.max_synapses,
            success_boost: self.success_boost,
            failure_penalty: self.failure_penalty,
            novelty_threshold: self.novelty_threshold,
            pruning_threshold: self.pruning_threshold,
            embedding_dim: self.embedding_dim,
            hash_dims: self.hash_dims,
            receptor_enabled: self.receptor_enabled,
            receptor_prototype_threshold: self.receptor_prototype_threshold,
            receptor_ema_alpha: self.receptor_ema_alpha,
            receptor_warmup_count: self.receptor_warmup_count,
            node_id_counter: self.node_id_counter,
            total_outcomes: self.total_outcomes,
            total_successes: self.total_successes,
        }
    }

    fn from_snapshot(&mut self, snap: StateSnapshot) {
        self.module_id = snap.module_id;
        self.nodes = snap.nodes;
        // Convert "src|tgt" keyed synapses back to (String,String)
        self.synapses.clear();
        for (key, syn) in snap.synapses {
            if let Some((src, tgt)) = key.split_once('|') {
                self.synapses.insert((src.to_string(), tgt.to_string()), syn);
            }
        }
        self.embedding_cache = snap.embedding_cache;
        self.prototypes = snap.prototypes;
        self.prototype_counts = snap.prototype_counts;
        self.prototype_k = snap.prototype_k;
        self.receptor_input_count = snap.receptor_input_count;
        self.max_nodes = snap.max_nodes;
        self.max_synapses = snap.max_synapses;
        self.success_boost = snap.success_boost;
        self.failure_penalty = snap.failure_penalty;
        self.novelty_threshold = snap.novelty_threshold;
        self.pruning_threshold = snap.pruning_threshold;
        self.embedding_dim = snap.embedding_dim;
        self.hash_dims = snap.hash_dims;
        self.receptor_enabled = snap.receptor_enabled;
        self.receptor_prototype_threshold = snap.receptor_prototype_threshold;
        self.receptor_ema_alpha = snap.receptor_ema_alpha;
        self.receptor_warmup_count = snap.receptor_warmup_count;
        self.node_id_counter = snap.node_id_counter;
        self.total_outcomes = snap.total_outcomes;
        self.total_successes = snap.total_successes;
    }
}

// --- PyO3 methods ---

#[cfg(feature = "python")]
#[pymethods]
impl PyNGLiteCore {
    #[new]
    #[pyo3(signature = (module_id, config))]
    fn new(module_id: &str, config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let get_usize = |key: &str, default: usize| -> usize {
            config.get_item(key).ok().flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            config.get_item(key).ok().flatten()
                .and_then(|v| v.extract::<f32>().ok())
                .unwrap_or(default)
        };
        let get_bool = |key: &str, default: bool| -> bool {
            config.get_item(key).ok().flatten()
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(default)
        };
        let get_u32 = |key: &str, default: u32| -> u32 {
            config.get_item(key).ok().flatten()
                .and_then(|v| v.extract::<u32>().ok())
                .unwrap_or(default)
        };

        let prototype_k = get_usize("receptor_layer_k", 256);

        Ok(Self {
            module_id: module_id.to_string(),
            nodes: HashMap::new(),
            synapses: HashMap::new(),
            embedding_cache: HashMap::new(),
            prototypes: None,
            prototype_counts: None,
            prototype_k,
            receptor_input_count: 0,
            max_nodes: get_usize("max_nodes", 10000),
            max_synapses: get_usize("max_synapses", 50000),
            success_boost: get_f32("success_boost", 0.15),
            failure_penalty: get_f32("failure_penalty", 0.20),
            novelty_threshold: get_f32("novelty_threshold", 0.70),
            pruning_threshold: get_f32("pruning_threshold", 0.01),
            embedding_dim: get_usize("embedding_dim", 768),
            hash_dims: get_usize("hash_dims", 128),
            receptor_enabled: get_bool("receptor_layer_enabled", false),
            receptor_prototype_threshold: get_f32("receptor_prototype_threshold", 0.75),
            receptor_ema_alpha: get_f32("receptor_ema_alpha", 0.001),
            receptor_warmup_count: get_u32("receptor_warmup_count", 256),
            node_id_counter: 0,
            total_outcomes: 0,
            total_successes: 0,
            history_max: get_usize("history_max", 10000),
        })
    }

    /// Find or create a node from an embedding vector.
    /// Returns a dict with node_id, embedding_hash, is_new, similarity.
    #[pyo3(signature = (embedding))]
    fn find_or_create_node<'py>(
        &mut self,
        py: Python<'py>,
        embedding: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<PyObject> {
        let readonly = embedding.readonly();
        let raw = readonly.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("embedding must be contiguous f32: {}", e)
            ))?;
        let emb = normalize(raw);
        let emb = self.snap_to_prototype(&emb);
        let emb_hash = hash_embedding(&emb, self.hash_dims);

        let dict = PyDict::new(py);

        // Check for existing similar node
        if let Some((existing_hash, sim)) = self.find_similar(&emb) {
            let existing_hash = existing_hash.to_string();
            // Find node_id by hash
            if let Some((nid, node)) = self.nodes.iter_mut().find(|(_, n)| n.embedding_hash == existing_hash) {
                node.activation_count += 1;
                node.last_activation = now_timestamp();
                dict.set_item("node_id", nid.clone())?;
                dict.set_item("embedding_hash", &existing_hash)?;
                dict.set_item("is_new", false)?;
                dict.set_item("similarity", sim)?;
                return Ok(dict.into_py_any(py)?);
            }
        }

        // Create new node
        if self.nodes.len() >= self.max_nodes {
            self.prune_least_used_node();
        }

        self.node_id_counter += 1;
        let node_id = format!("n_{}", self.node_id_counter);
        let now = now_timestamp();

        let node = Node {
            node_id: node_id.clone(),
            embedding_hash: emb_hash.clone(),
            activation_count: 1,
            last_activation: now,
            metadata: HashMap::new(),
            embedding: Some(emb.clone()),
            constitutional: false,
        };

        self.nodes.insert(node_id.clone(), node);
        self.embedding_cache.insert(emb_hash.clone(), emb);

        // Initialize prototypes when we have enough embeddings
        if self.receptor_enabled && self.prototypes.is_none()
            && self.embedding_cache.len() >= self.prototype_k
        {
            self.init_prototypes();
        }

        dict.set_item("node_id", &node_id)?;
        dict.set_item("embedding_hash", &emb_hash)?;
        dict.set_item("is_new", true)?;
        dict.set_item("similarity", 0.0f32)?;
        Ok(dict.into_py_any(py)?)
    }

    /// Record an outcome (success/failure) for a node-target pair.
    /// Hebbian update + Welford variance tracking.
    #[pyo3(signature = (embedding, target_id, success, strength, metadata=None))]
    fn record_outcome<'py>(
        &mut self,
        py: Python<'py>,
        embedding: &Bound<'py, PyArray1<f32>>,
        target_id: &str,
        success: bool,
        strength: f32,
        metadata: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyObject> {
        // Find or create source node
        let node_result = self.find_or_create_node(py, embedding)?;
        let node_dict = node_result.bind(py).downcast::<PyDict>()?;
        let source_id: String = node_dict.get_item("node_id")?.unwrap().extract()?;

        // Constitutional nodes have frozen synapses — the bucket comes up empty.
        if let Some(node) = self.nodes.get(&source_id) {
            if node.constitutional {
                let result = PyDict::new(py);
                result.set_item("node_id", &node.node_id)?;
                result.set_item("target_id", target_id)?;
                result.set_item("success", success)?;
                result.set_item("weight_after", 0.0f32)?;
                result.set_item("activation_count", 0u32)?;
                result.set_item("variance", 0.0f64)?;
                result.set_item("contested", false)?;
                result.set_item("constitutional", true)?;
                return Ok(result.into_py_any(py)?);
            }
        }

        // Get or create synapse
        let syn_key = (source_id.clone(), target_id.to_string());
        let now = now_timestamp();

        if self.synapses.len() >= self.max_synapses && !self.synapses.contains_key(&syn_key) {
            self.prune_weakest_synapse();
        }

        let syn = self.synapses.entry(syn_key).or_insert_with(|| {
            Synapse {
                source_id: source_id.clone(),
                target_id: target_id.to_string(),
                weight: 0.5,
                activation_count: 0,
                success_count: 0,
                failure_count: 0,
                last_updated: now,
                metadata: HashMap::new(),
                welford_count: 0,
                welford_mean: 0.0,
                welford_m2: 0.0,
            }
        });

        syn.activation_count += 1;
        syn.last_updated = now;

        // Hebbian learning — MUST match Python exactly
        let delta = if success {
            syn.success_count += 1;
            self.total_successes += 1;
            self.success_boost * (1.0 - syn.weight) * strength
        } else {
            syn.failure_count += 1;
            self.failure_penalty * syn.weight * strength
        };

        if success {
            syn.weight += delta;
        } else {
            syn.weight -= delta;
        }
        syn.weight = syn.weight.clamp(0.0, 1.0);

        // Welford variance
        syn.welford_count += 1;
        let w_delta = if success { delta } else { -delta };
        let old_mean = syn.welford_mean;
        syn.welford_mean += (w_delta as f64 - old_mean) / syn.welford_count as f64;
        syn.welford_m2 += (w_delta as f64 - old_mean) * (w_delta as f64 - syn.welford_mean);

        self.total_outcomes += 1;

        // Store metadata if provided
        if let Some(md) = metadata {
            for (k, v) in md.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                syn.metadata.insert(key, val);
            }
        }

        // Collect values before releasing the mutable borrow
        let weight = syn.weight;
        let contested = syn.is_contested();
        let variance = syn.variance();

        let result = PyDict::new(py);
        result.set_item("node_id", &source_id)?;
        result.set_item("target_id", target_id)?;
        result.set_item("success", success)?;
        result.set_item("weight_after", weight)?;
        result.set_item("activation_count", syn.activation_count)?;
        result.set_item("variance", variance)?;
        result.set_item("contested", contested)?;
        Ok(result.into_py_any(py)?)
    }

    /// Get recommendations for a given embedding.
    /// Returns list of (target_id, weight, reasoning) tuples sorted by weight.
    #[pyo3(signature = (embedding, top_k=10))]
    fn get_recommendations<'py>(
        &self,
        py: Python<'py>,
        embedding: &Bound<'py, PyArray1<f32>>,
        top_k: usize,
    ) -> PyResult<Vec<PyObject>> {
        let readonly = embedding.readonly();
        let raw = readonly.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("embedding must be contiguous f32: {}", e)
            ))?;
        let emb = normalize(raw);

        // Find matching node
        let source_id = match self.find_similar(&emb) {
            Some((hash, _)) => {
                let hash = hash.to_string();
                self.nodes.iter()
                    .find(|(_, n)| n.embedding_hash == hash)
                    .map(|(nid, _)| nid.clone())
            }
            None => None,
        };

        let source_id = match source_id {
            Some(id) => id,
            None => return Ok(vec![]),
        };

        // Collect synapses from this source
        let mut candidates: Vec<(&Synapse,)> = Vec::new();
        for ((src, _), syn) in &self.synapses {
            if *src == source_id {
                candidates.push((syn,));
            }
        }

        // Sort by weight descending
        candidates.sort_by(|a, b| b.0.weight.partial_cmp(&a.0.weight).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        let mut results = Vec::new();
        for (syn,) in &candidates {
            let reasoning = Self::build_local_reasoning(syn);
            let tuple = (syn.target_id.as_str(), syn.weight, reasoning);
            results.push(tuple.into_py_any(py)?);
        }
        Ok(results)
    }

    /// Detect novelty of an embedding.
    /// Returns 1.0 - max_similarity (high = novel, low = familiar).
    #[pyo3(signature = (embedding))]
    fn detect_novelty(
        &self,
        embedding: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<f32> {
        let readonly = embedding.readonly();
        let raw = readonly.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("embedding must be contiguous f32: {}", e)
            ))?;
        let emb = normalize(raw);

        let mut max_sim = 0.0f32;
        for cached in self.embedding_cache.values() {
            let sim = dot_product(&emb, cached);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        Ok(1.0 - max_sim)
    }

    /// Export state as Python dict (for debugging/migration only).
    fn export_state<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("module_id", &self.module_id)?;
        dict.set_item("node_count", self.nodes.len())?;
        dict.set_item("synapse_count", self.synapses.len())?;
        dict.set_item("total_outcomes", self.total_outcomes)?;
        dict.set_item("total_successes", self.total_successes)?;

        // Nodes
        let nodes_dict = PyDict::new(py);
        for (nid, node) in &self.nodes {
            let nd = PyDict::new(py);
            nd.set_item("node_id", &node.node_id)?;
            nd.set_item("embedding_hash", &node.embedding_hash)?;
            nd.set_item("activation_count", node.activation_count)?;
            nd.set_item("last_activation", node.last_activation)?;
            nd.set_item("constitutional", node.constitutional)?;
            let meta = PyDict::new(py);
            for (k, v) in &node.metadata {
                meta.set_item(k, v)?;
            }
            nd.set_item("metadata", meta)?;
            nodes_dict.set_item(nid, nd)?;
        }
        dict.set_item("nodes", nodes_dict)?;

        // Synapses
        let synapses_dict = PyDict::new(py);
        for ((src, tgt), syn) in &self.synapses {
            let key = format!("{}|{}", src, tgt);
            let sd = PyDict::new(py);
            sd.set_item("source_id", &syn.source_id)?;
            sd.set_item("target_id", &syn.target_id)?;
            sd.set_item("weight", syn.weight)?;
            sd.set_item("activation_count", syn.activation_count)?;
            sd.set_item("success_count", syn.success_count)?;
            sd.set_item("failure_count", syn.failure_count)?;
            sd.set_item("last_updated", syn.last_updated)?;
            sd.set_item("welford_count", syn.welford_count)?;
            sd.set_item("welford_mean", syn.welford_mean)?;
            sd.set_item("welford_m2", syn.welford_m2)?;
            sd.set_item("contested", syn.is_contested())?;
            sd.set_item("variance", syn.variance())?;
            synapses_dict.set_item(key, sd)?;
        }
        dict.set_item("synapses", synapses_dict)?;

        // Config
        let config = PyDict::new(py);
        config.set_item("max_nodes", self.max_nodes)?;
        config.set_item("max_synapses", self.max_synapses)?;
        config.set_item("success_boost", self.success_boost)?;
        config.set_item("failure_penalty", self.failure_penalty)?;
        config.set_item("novelty_threshold", self.novelty_threshold)?;
        config.set_item("pruning_threshold", self.pruning_threshold)?;
        config.set_item("embedding_dim", self.embedding_dim)?;
        config.set_item("hash_dims", self.hash_dims)?;
        config.set_item("receptor_enabled", self.receptor_enabled)?;
        config.set_item("receptor_prototype_threshold", self.receptor_prototype_threshold)?;
        config.set_item("receptor_ema_alpha", self.receptor_ema_alpha)?;
        config.set_item("receptor_warmup_count", self.receptor_warmup_count)?;
        dict.set_item("config", config)?;

        Ok(dict.into_py_any(py)?)
    }

    /// Import state from Python dict (for JSON migration only).
    fn import_state(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(mid) = state.get_item("module_id")? {
            self.module_id = mid.extract()?;
        }
        if let Some(to) = state.get_item("total_outcomes")? {
            self.total_outcomes = to.extract()?;
        }
        if let Some(ts) = state.get_item("total_successes")? {
            self.total_successes = ts.extract()?;
        }

        // Import nodes
        if let Some(nodes_obj) = state.get_item("nodes")? {
            let nodes_dict = nodes_obj.downcast::<PyDict>()?;
            self.nodes.clear();
            self.embedding_cache.clear();
            for (key, val) in nodes_dict.iter() {
                let nid: String = key.extract()?;
                let nd = val.downcast::<PyDict>()?;
                let emb_hash: String = nd.get_item("embedding_hash")?
                    .map_or(Ok(String::new()), |v| v.extract())?;
                let node = Node {
                    node_id: nid.clone(),
                    embedding_hash: emb_hash,
                    activation_count: nd.get_item("activation_count")?
                        .map_or(Ok(0), |v| v.extract())?,
                    last_activation: nd.get_item("last_activation")?
                        .map_or(Ok(0.0), |v| v.extract())?,
                    metadata: nd.get_item("metadata")?
                        .and_then(|v| v.extract().ok())
                        .unwrap_or_default(),
                    embedding: nd.get_item("embedding")?
                        .and_then(|v| v.extract().ok()),
                    constitutional: nd.get_item("constitutional")?
                        .map_or(Ok(false), |v| v.extract())?,
                };
                // Rebuild embedding cache if embedding present
                if let Some(ref emb) = node.embedding {
                    self.embedding_cache.insert(node.embedding_hash.clone(), emb.clone());
                }
                self.nodes.insert(nid, node);
            }
        }

        // Import synapses
        if let Some(syn_obj) = state.get_item("synapses")? {
            let syn_dict = syn_obj.downcast::<PyDict>()?;
            self.synapses.clear();
            for (key, val) in syn_dict.iter() {
                let key_str: String = key.extract()?;
                let sd = val.downcast::<PyDict>()?;
                let source_id: String = sd.get_item("source_id")?
                    .map_or(Ok(String::new()), |v| v.extract())?;
                let target_id: String = sd.get_item("target_id")?
                    .map_or(Ok(String::new()), |v| v.extract())?;
                let syn = Synapse {
                    source_id: source_id.clone(),
                    target_id: target_id.clone(),
                    weight: sd.get_item("weight")?
                        .map_or(Ok(0.5), |v| v.extract())?,
                    activation_count: sd.get_item("activation_count")?
                        .map_or(Ok(0), |v| v.extract())?,
                    success_count: sd.get_item("success_count")?
                        .map_or(Ok(0), |v| v.extract())?,
                    failure_count: sd.get_item("failure_count")?
                        .map_or(Ok(0), |v| v.extract())?,
                    last_updated: sd.get_item("last_updated")?
                        .map_or(Ok(0.0), |v| v.extract())?,
                    metadata: sd.get_item("metadata")?
                        .and_then(|v| v.extract().ok())
                        .unwrap_or_default(),
                    welford_count: sd.get_item("welford_count")?
                        .map_or(Ok(0), |v| v.extract())?,
                    welford_mean: sd.get_item("welford_mean")?
                        .map_or(Ok(0.0), |v| v.extract())?,
                    welford_m2: sd.get_item("welford_m2")?
                        .map_or(Ok(0.0), |v| v.extract())?,
                };
                // Use key_str or source|target
                let _ = key_str; // key_str format is "src|tgt"
                self.synapses.insert((source_id, target_id), syn);
            }
        }

        // Import config
        if let Some(config_obj) = state.get_item("config")? {
            let cd = config_obj.downcast::<PyDict>()?;
            if let Some(v) = cd.get_item("max_nodes")? { self.max_nodes = v.extract()?; }
            if let Some(v) = cd.get_item("max_synapses")? { self.max_synapses = v.extract()?; }
            if let Some(v) = cd.get_item("success_boost")? { self.success_boost = v.extract()?; }
            if let Some(v) = cd.get_item("failure_penalty")? { self.failure_penalty = v.extract()?; }
            if let Some(v) = cd.get_item("novelty_threshold")? { self.novelty_threshold = v.extract()?; }
            if let Some(v) = cd.get_item("pruning_threshold")? { self.pruning_threshold = v.extract()?; }
            if let Some(v) = cd.get_item("embedding_dim")? { self.embedding_dim = v.extract()?; }
            if let Some(v) = cd.get_item("hash_dims")? { self.hash_dims = v.extract()?; }
            if let Some(v) = cd.get_item("receptor_enabled")? { self.receptor_enabled = v.extract()?; }
            if let Some(v) = cd.get_item("receptor_prototype_threshold")? { self.receptor_prototype_threshold = v.extract()?; }
            if let Some(v) = cd.get_item("receptor_ema_alpha")? { self.receptor_ema_alpha = v.extract()?; }
            if let Some(v) = cd.get_item("receptor_warmup_count")? { self.receptor_warmup_count = v.extract()?; }
        }

        Ok(())
    }

    /// Update a tunable parameter. Returns dict with old and new values.
    #[pyo3(signature = (key, value))]
    fn update_config<'py>(
        &mut self,
        py: Python<'py>,
        key: &str,
        value: f64,
    ) -> PyResult<PyObject> {
        let params = tunable_params();
        let bounds = params.get(key).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(
                format!("unknown tunable param: '{}'. Valid: {:?}", key, params.keys().collect::<Vec<_>>())
            )
        })?;

        let clamped = value.clamp(bounds.min, bounds.max);
        let old_value: f64;

        match key {
            "success_boost" => { old_value = self.success_boost as f64; self.success_boost = clamped as f32; }
            "failure_penalty" => { old_value = self.failure_penalty as f64; self.failure_penalty = clamped as f32; }
            "novelty_threshold" => { old_value = self.novelty_threshold as f64; self.novelty_threshold = clamped as f32; }
            "pruning_threshold" => { old_value = self.pruning_threshold as f64; self.pruning_threshold = clamped as f32; }
            "receptor_ema_alpha" => { old_value = self.receptor_ema_alpha as f64; self.receptor_ema_alpha = clamped as f32; }
            "receptor_prototype_threshold" => { old_value = self.receptor_prototype_threshold as f64; self.receptor_prototype_threshold = clamped as f32; }
            "relevance_threshold" => {
                // relevance_threshold is derived from novelty_threshold in Python
                // Store as novelty_threshold inverse
                old_value = self.novelty_threshold as f64;
                self.novelty_threshold = clamped as f32;
            }
            _ => {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("unhandled tunable param: '{}'", key)
                ));
            }
        }

        let dict = PyDict::new(py);
        dict.set_item("key", key)?;
        dict.set_item("old_value", old_value)?;
        dict.set_item("new_value", clamped)?;
        dict.set_item("clamped", clamped != value)?;
        Ok(dict.into_py_any(py)?)
    }

    /// Get substrate statistics.
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("module_id", &self.module_id)?;
        dict.set_item("node_count", self.nodes.len())?;
        dict.set_item("synapse_count", self.synapses.len())?;
        dict.set_item("embedding_cache_size", self.embedding_cache.len())?;
        dict.set_item("total_outcomes", self.total_outcomes)?;
        dict.set_item("total_successes", self.total_successes)?;
        dict.set_item("node_id_counter", self.node_id_counter)?;
        dict.set_item("max_nodes", self.max_nodes)?;
        dict.set_item("max_synapses", self.max_synapses)?;
        dict.set_item("receptor_enabled", self.receptor_enabled)?;
        dict.set_item("receptor_input_count", self.receptor_input_count)?;
        dict.set_item("has_prototypes", self.prototypes.is_some())?;
        dict.set_item("history_max", self.history_max)?;

        // Constitutional node count
        let constitutional_count = self.nodes.values().filter(|n| n.constitutional).count();
        dict.set_item("constitutional_nodes", constitutional_count)?;

        // Contested synapse count
        let contested_count = self.synapses.values().filter(|s| s.is_contested()).count();
        dict.set_item("contested_synapses", contested_count)?;

        // Success rate
        let success_rate = if self.total_outcomes > 0 {
            self.total_successes as f64 / self.total_outcomes as f64
        } else {
            0.0
        };
        dict.set_item("success_rate", success_rate)?;

        Ok(dict.into_py_any(py)?)
    }

    /// Seed constitutional nodes from a list of embedding dicts.
    /// Each dict should have "embedding" (list of f32) and optionally "label".
    #[pyo3(signature = (embeddings))]
    fn seed_constitutional(
        &mut self,
        embeddings: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        for item in embeddings.iter() {
            let dict = item.downcast::<PyDict>()?;
            let emb_raw: Vec<f32> = dict.get_item("embedding")?.unwrap().extract()?;
            let label: String = dict.get_item("label")?
                .map_or(Ok(String::new()), |v| v.extract())?;

            let emb = normalize(&emb_raw);
            let emb_hash = hash_embedding(&emb, self.hash_dims);

            self.node_id_counter += 1;
            let node_id = format!("n_{}", self.node_id_counter);
            let now = now_timestamp();

            let mut metadata = HashMap::new();
            if !label.is_empty() {
                metadata.insert("label".to_string(), label);
            }
            metadata.insert("constitutional".to_string(), "true".to_string());

            let node = Node {
                node_id: node_id.clone(),
                embedding_hash: emb_hash.clone(),
                activation_count: 0,
                last_activation: now,
                metadata,
                embedding: Some(emb.clone()),
                constitutional: true,
            };

            self.nodes.insert(node_id, node);
            self.embedding_cache.insert(emb_hash, emb);
        }
        Ok(())
    }

    /// Save state to binary file using msgpack. Atomic write (tmp + rename).
    /// Releases GIL for I/O.
    #[pyo3(signature = (path))]
    fn save_binary(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let snapshot = self.to_snapshot();
        let data = rmp_serde::to_vec(&snapshot)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("msgpack serialize failed: {}", e)
            ))?;
        let path = path.to_string();
        py.allow_threads(move || {
            let tmp_path = format!("{}.tmp.{}", path, std::process::id());
            std::fs::write(&tmp_path, &data)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                    format!("write failed: {}", e)
                ))?;
            std::fs::rename(&tmp_path, &path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                    format!("rename failed: {}", e)
                ))
        })
    }

    /// Load state from binary msgpack file. Releases GIL for I/O.
    #[pyo3(signature = (path))]
    fn load_binary(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path = path.to_string();
        let data = py.allow_threads(move || {
            std::fs::read(&path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                    format!("read failed: {}", e)
                ))
        })?;
        let snapshot: StateSnapshot = rmp_serde::from_slice(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("msgpack deserialize failed: {}", e)
            ))?;
        self.from_snapshot(snapshot);
        Ok(())
    }
}
