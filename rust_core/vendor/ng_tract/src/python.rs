//! PyO3 bindings for the Binary Tract Format.
//!
//! Python API:
//!   - ng_tract.write_outcome(...) -> bytes
//!   - ng_tract.write_topology(...) -> bytes
//!   - ng_tract.TractReader(data) -> iterator of typed entries
//!   - ng_tract.is_btf(data) -> bool
//!   - ng_tract.extract_topology_features(entry, k) -> (he_np, global_np)
//!
//! Embeddings are exposed as zero-copy numpy arrays where possible.
//!
//! ---- Changelog ----
//! [2026-04-11] Claude Code (Sonnet 4.6) — Punchlist #119 Step 5
//!   What: extract_topology_features(entry, k) — BTF fast path for encoder bucket
//!   Why:  graph_encoder.py and encoder_v2.py call ng_tract.extract_topology_features
//!         to extract (K,8) hyperedge features + (8,) global features directly from
//!         a PyTopologyEntry without Python dict traversal.  This closes the
//!         missing Rust function that was blocking the BTF encoder path.
//!   How:  Pure Rust feature extraction matching _read_from_delta and _read_global
//!         in graph_encoder.py.  Builds fired-node sets, iterates hyperedges,
//!         scores by member_activity, returns top-K via ndarray -> numpy.
//! [2026-04-04] Claude Code (Opus 4.6) — PyTopologyEntry + LeniaEngine bindings
//! -------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::IntoPyObjectExt;

#[cfg(feature = "python")]
use pyo3::types::{PyBytes, PyDict, PyList};

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};

#[cfg(feature = "python")]
use crate::format::*;

// --- Module ---

#[cfg(feature = "python")]
#[pymodule]
fn ng_tract(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add("ENTRY_OUTCOME", ENTRY_OUTCOME)?;
    m.add("ENTRY_TOPOLOGY", ENTRY_TOPOLOGY)?;
    m.add("ENTRY_EXPERIENCE", ENTRY_EXPERIENCE)?;
    m.add_function(wrap_pyfunction!(is_btf, m)?)?;
    m.add_function(wrap_pyfunction!(write_outcome, m)?)?;
    m.add_function(wrap_pyfunction!(write_topology, m)?)?;
    m.add_function(wrap_pyfunction!(extract_topology_features, m)?)?;
    m.add_function(wrap_pyfunction!(deposit_outcome, m)?)?;
    m.add_function(wrap_pyfunction!(deposit_experience, m)?)?;
    m.add_function(wrap_pyfunction!(deposit_topology, m)?)?;
    m.add_class::<PyTractReader>()?;
    m.add_class::<PyOutcomeEntry>()?;
    m.add_class::<PyExperienceEntry>()?;
    m.add_class::<PyTopologyEntry>()?;
    m.add_class::<PyFiredNode>()?;
    m.add_class::<PySalienceSignal>()?;
    m.add_class::<PyFiredHyperedge>()?;
    m.add_class::<PyOutgoingSynapse>()?;
    m.add_class::<PyLeniaEngine>()?;
    Ok(())
}

// --- Free functions ---

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "is_btf")]
fn is_btf(data: &[u8]) -> bool {
    Envelope::is_btf(data)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (timestamp, module_id, target_id, success, embedding, metadata=None))]
#[pyo3(name = "write_outcome")]
fn write_outcome(
    py: Python<'_>,
    timestamp: f64,
    module_id: &str,
    target_id: &str,
    success: bool,
    embedding: Vec<f32>,
    metadata: Option<&Bound<'_, PyBytes>>,
) -> PyResult<Py<PyBytes>> {
    let meta_bytes = match metadata {
        Some(b) => b.as_bytes().to_vec(),
        None => vec![],
    };
    let entry = OutcomeEntry {
        timestamp,
        module_id: module_id.to_string(),
        target_id: target_id.to_string(),
        success,
        embedding_dim: embedding.len() as u16,
        embedding,
        metadata: meta_bytes,
    };
    let bytes = crate::write::write_outcome(&entry);
    Ok(PyBytes::new(py, &bytes).into())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    timestamp, timestep,
    fired_nodes,
    predictions_confirmed=0, predictions_surprised=0,
    synapses_pruned=0, synapses_sprouted=0,
    fired_hyperedges=None, salience=None
))]
#[pyo3(name = "write_topology")]
fn write_topology(
    py: Python<'_>,
    timestamp: f64,
    timestep: u32,
    fired_nodes: &Bound<'_, PyList>,
    predictions_confirmed: u16,
    predictions_surprised: u16,
    synapses_pruned: u16,
    synapses_sprouted: u16,
    fired_hyperedges: Option<&Bound<'_, PyList>>,
    salience: Option<&Bound<'_, PyList>>,
) -> PyResult<Py<PyBytes>> {
    // Parse fired_nodes from list of dicts
    let mut nodes = Vec::new();
    for item in fired_nodes.iter() {
        let d = item.downcast::<PyDict>()?;
        let node_id_str: String = d.get_item("node_id")?.unwrap().extract()?;
        let node_id = uuid::Uuid::parse_str(&node_id_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let label: String = d.get_item("label")?.map_or(Ok(String::new()), |v| v.extract())?;
        let embedding: Vec<f32> = d.get_item("embedding")?
            .map_or(Ok(vec![]), |v| v.extract())?;
        let outgoing_list = d.get_item("outgoing_synapses")?;
        let mut outgoing = Vec::new();
        if let Some(syn_list) = outgoing_list {
            for syn_item in syn_list.downcast::<PyList>()?.iter() {
                let sd = syn_item.downcast::<PyDict>()?;
                let post_str: String = sd.get_item("post_node_id")?.unwrap().extract()?;
                let post_id = uuid::Uuid::parse_str(&post_str)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let weight: f32 = sd.get_item("weight")?.unwrap().extract()?;
                let trace: f32 = sd.get_item("eligibility_trace")?
                    .map_or(Ok(0.0), |v| v.extract())?;
                outgoing.push(OutgoingSynapse {
                    post_node_id: post_id, weight, eligibility_trace: trace,
                });
            }
        }
        nodes.push(FiredNode {
            node_id, label,
            embedding_dim: embedding.len() as u16,
            embedding, outgoing,
        });
    }

    // Parse fired_hyperedges
    let mut hes = Vec::new();
    if let Some(he_list) = fired_hyperedges {
        for item in he_list.iter() {
            let d = item.downcast::<PyDict>()?;
            let he_id_str: String = d.get_item("hyperedge_id")?.unwrap().extract()?;
            let he_id = uuid::Uuid::parse_str(&he_id_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let label: String = d.get_item("label")?.map_or(Ok(String::new()), |v| v.extract())?;
            let act_count: u32 = d.get_item("activation_count")?.map_or(Ok(0), |v| v.extract())?;
            let members: Vec<String> = d.get_item("member_node_ids")?
                .map_or(Ok(vec![]), |v| v.extract())?;
            let outputs: Vec<String> = d.get_item("output_target_ids")?
                .map_or(Ok(vec![]), |v| v.extract())?;
            let member_ids: Result<Vec<uuid::Uuid>, _> = members.iter()
                .map(|s| uuid::Uuid::parse_str(s)).collect();
            let output_ids: Result<Vec<uuid::Uuid>, _> = outputs.iter()
                .map(|s| uuid::Uuid::parse_str(s)).collect();
            hes.push(FiredHyperedge {
                hyperedge_id: he_id, label, activation_count: act_count,
                member_node_ids: member_ids.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
                output_target_ids: output_ids.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            });
        }
    }

    // Parse salience
    let mut sal = Vec::new();
    if let Some(sal_list) = salience {
        for item in sal_list.iter() {
            let d = item.downcast::<PyDict>()?;
            let sig_type: u8 = d.get_item("signal_type")?.unwrap().extract()?;
            let nid_str: String = d.get_item("node_id")?
                .map_or(Ok("00000000-0000-0000-0000-000000000000".to_string()), |v| v.extract())?;
            let node_id = uuid::Uuid::parse_str(&nid_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let value: f32 = d.get_item("value")?.unwrap().extract()?;
            let reference: f32 = d.get_item("reference")?.map_or(Ok(0.0), |v| v.extract())?;
            sal.push(SalienceSignal { signal_type: sig_type, node_id, value, reference });
        }
    }

    let entry = TopologyEntry {
        timestamp, timestep,
        predictions_confirmed, predictions_surprised,
        synapses_pruned, synapses_sprouted,
        fired_nodes: nodes, fired_hyperedges: hes, salience: sal,
    };
    let bytes = crate::write::write_topology(&entry);
    Ok(PyBytes::new(py, &bytes).into())
}

// --- Reader ---

#[cfg(feature = "python")]
#[pyclass(name = "TractReader")]
struct PyTractReader {
    data: Vec<u8>,
    pos: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTractReader {
    #[new]
    fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let mut reader = crate::read::TractReader::new(&self.data[self.pos..]);
        match reader.next_entry() {
            None => Ok(None),
            Some(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
            Some(Ok(result)) => {
                self.pos += reader.position();

                match result {
                    crate::read::ReadResult::Entry(TractEntry::Outcome(entry)) => {
                        Ok(Some(PyOutcomeEntry::from_entry(py, entry)?.into_py_any(py)?))
                    }
                    crate::read::ReadResult::Entry(TractEntry::Topology(entry)) => {
                        Ok(Some(PyTopologyEntry::from_entry(py, entry)?.into_py_any(py)?))
                    }
                    crate::read::ReadResult::Entry(TractEntry::Experience(entry)) => {
                        Ok(Some(PyExperienceEntry::from_entry(py, entry)?.into_py_any(py)?))
                    }
                    crate::read::ReadResult::JsonlLine(line) => {
                        let bytes = PyBytes::new(py, line);
                        Ok(Some(bytes.into_py_any(py)?))
                    }
                }
            }
        }
    }

    /// Bytes consumed so far — the cursor-drain API the vendored
    /// ng_tract_bridge.py relies on (new_offset = start_offset + reader.position()).
    /// PyTractReader already tracks self.pos in __next__; this exposes it.
    /// 2026-06-08 CC — close the UniAI/ng-tract-rs fork: UniAI had the correct
    /// [u8;2] magic + format but lacked the exposed position() that ng-tract-rs
    /// (divergent u16 magic) had. Make the UniAI pattern complete.
    fn position(&self) -> usize {
        self.pos
    }
}

// --- Python entry wrappers ---

#[cfg(feature = "python")]
#[pyclass]
struct PyOutcomeEntry {
    #[pyo3(get)]
    entry_type: u8,
    #[pyo3(get)]
    timestamp: f64,
    #[pyo3(get)]
    module_id: String,
    #[pyo3(get)]
    target_id: String,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    embedding_dim: u16,
    embedding: Vec<f32>,
    metadata_bytes: Vec<u8>,
}

#[cfg(feature = "python")]
impl PyOutcomeEntry {
    fn from_entry(_py: Python<'_>, e: OutcomeEntry) -> PyResult<Self> {
        Ok(Self {
            entry_type: ENTRY_OUTCOME,
            timestamp: e.timestamp,
            module_id: e.module_id,
            target_id: e.target_id,
            success: e.success,
            embedding_dim: e.embedding_dim,
            embedding: e.embedding,
            metadata_bytes: e.metadata,
        })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOutcomeEntry {
    /// Zero-copy numpy array of the embedding.
    fn embedding_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.embedding.clone().into_pyarray(py)
    }

    /// Metadata decoded from MessagePack to Python dict.
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        if self.metadata_bytes.is_empty() {
            return Ok(PyDict::new(py).into_py_any(py)?);
        }
        // Decode MessagePack to rmpv::Value, then convert to Python
        match rmp_serde::from_slice::<std::collections::HashMap<String, String>>(&self.metadata_bytes) {
            Ok(map) => {
                let dict = PyDict::new(py);
                for (k, v) in &map {
                    dict.set_item(k, v)?;
                }
                Ok(dict.into_py_any(py)?)
            }
            Err(_) => Ok(PyDict::new(py).into_py_any(py)?),
        }
    }
}

// --- Experience entry ---

#[cfg(feature = "python")]
#[pyclass]
struct PyExperienceEntry {
    #[pyo3(get)]
    entry_type: u8,
    #[pyo3(get)]
    timestamp: f64,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    content_type: String,
    #[pyo3(get)]
    content: String,
}

#[cfg(feature = "python")]
impl PyExperienceEntry {
    fn from_entry(_py: Python<'_>, e: ExperienceEntry) -> PyResult<Self> {
        Ok(Self {
            entry_type: ENTRY_EXPERIENCE,
            timestamp: e.timestamp,
            source: e.source,
            content_type: e.content_type,
            content: String::from_utf8_lossy(&e.content).to_string(),
        })
    }
}

// --- Topology entry ---

#[cfg(feature = "python")]
#[pyclass]
struct PyTopologyEntry {
    #[pyo3(get)]
    entry_type: u8,
    #[pyo3(get)]
    timestamp: f64,
    #[pyo3(get)]
    timestep: u32,
    #[pyo3(get)]
    predictions_confirmed: u16,
    #[pyo3(get)]
    predictions_surprised: u16,
    #[pyo3(get)]
    synapses_pruned: u16,
    #[pyo3(get)]
    synapses_sprouted: u16,
    fired_nodes_data: Vec<FiredNode>,
    fired_hyperedges_data: Vec<FiredHyperedge>,
    salience_data: Vec<SalienceSignal>,
}

#[cfg(feature = "python")]
impl PyTopologyEntry {
    fn from_entry(_py: Python<'_>, e: TopologyEntry) -> PyResult<Self> {
        Ok(Self {
            entry_type: ENTRY_TOPOLOGY,
            timestamp: e.timestamp,
            timestep: e.timestep,
            predictions_confirmed: e.predictions_confirmed,
            predictions_surprised: e.predictions_surprised,
            synapses_pruned: e.synapses_pruned,
            synapses_sprouted: e.synapses_sprouted,
            fired_nodes_data: e.fired_nodes,
            fired_hyperedges_data: e.fired_hyperedges,
            salience_data: e.salience,
        })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTopologyEntry {
    /// Fired nodes as a list of PyFiredNode objects.
    fn fired_nodes(&self) -> PyResult<Vec<PyFiredNode>> {
        self.fired_nodes_data.iter().map(|n| {
            Ok(PyFiredNode {
                node_id: n.node_id.to_string(),
                label: n.label.clone(),
                embedding_dim: n.embedding_dim,
                embedding: n.embedding.clone(),
                outgoing_data: n.outgoing.to_vec(),
            })
        }).collect()
    }

    fn fired_hyperedges(&self) -> PyResult<Vec<PyFiredHyperedge>> {
        self.fired_hyperedges_data.iter().map(|h| {
            Ok(PyFiredHyperedge {
                hyperedge_id: h.hyperedge_id.to_string(),
                label: h.label.clone(),
                activation_count: h.activation_count,
                member_node_ids: h.member_node_ids.iter().map(|u| u.to_string()).collect(),
                output_target_ids: h.output_target_ids.iter().map(|u| u.to_string()).collect(),
            })
        }).collect()
    }

    fn salience(&self) -> PyResult<Vec<PySalienceSignal>> {
        self.salience_data.iter().map(|s| {
            Ok(PySalienceSignal {
                signal_type: s.signal_type,
                node_id: s.node_id.to_string(),
                value: s.value,
                reference: s.reference,
            })
        }).collect()
    }
}

// --- Fired node ---

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
struct PyFiredNode {
    #[pyo3(get)]
    node_id: String,
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    embedding_dim: u16,
    embedding: Vec<f32>,
    outgoing_data: Vec<OutgoingSynapse>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFiredNode {
    fn embedding_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.embedding.clone().into_pyarray(py)
    }

    fn outgoing_synapses(&self) -> Vec<PyOutgoingSynapse> {
        self.outgoing_data.iter().map(|s| PyOutgoingSynapse {
            post_node_id: s.post_node_id.to_string(),
            weight: s.weight,
            eligibility_trace: s.eligibility_trace,
        }).collect()
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
struct PyOutgoingSynapse {
    #[pyo3(get)]
    post_node_id: String,
    #[pyo3(get)]
    weight: f32,
    #[pyo3(get)]
    eligibility_trace: f32,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
struct PyFiredHyperedge {
    #[pyo3(get)]
    hyperedge_id: String,
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    activation_count: u32,
    #[pyo3(get)]
    member_node_ids: Vec<String>,
    #[pyo3(get)]
    output_target_ids: Vec<String>,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
struct PySalienceSignal {
    #[pyo3(get)]
    signal_type: u8,
    #[pyo3(get)]
    node_id: String,
    #[pyo3(get)]
    value: f32,
    #[pyo3(get)]
    reference: f32,
}

// =====================================================================
// extract_topology_features — BTF fast path for the encoder bucket
//
// Takes a PyTopologyEntry (already in Rust memory from the River read)
// and extracts the two feature arrays the encoder needs.  No Python
// dict traversal, no serialization boundary — in-memory processing of
// what the bucket already contains.
//
// Returns:
//   he_features_np   — (K, 8) f32 array, top-K hyperedge features
//   global_features_np — (8,) f32 array, global step summary
//
// Feature layout matches _read_from_delta in graph_encoder.py:
//   [fired, member_activity, size, output_fan_out,
//    activation_log, connectivity, mean_weight, hot_eligibility]
//
// Global layout matches _read_global:
//   [fired_nodes_norm, fired_hes_norm, confirmed_norm, surprised_norm,
//    pruned_norm, sprouted_norm, surprise_ratio, timestep_norm]
// =====================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn extract_topology_features<'py>(
    py: Python<'py>,
    entry: pyo3::PyRef<'_, PyTopologyEntry>,
    k: usize,
) -> PyResult<(
    pyo3::Bound<'py, PyArray2<f32>>,
    pyo3::Bound<'py, PyArray1<f32>>,
)> {
    use numpy::ndarray::Array2;
    use std::collections::{HashMap, HashSet};
    use crate::format::{FiredNode, SALIENCE_HOT_ELIGIBILITY};

    // --- Global features (8) — always computable from entry scalars ---
    let confirmed = entry.predictions_confirmed as f32;
    let surprised = entry.predictions_surprised as f32;
    let surprise_ratio = (surprised / (confirmed + surprised).max(1.0)).min(1.0);

    let global_vec: Vec<f32> = vec![
        (entry.fired_nodes_data.len() as f32 / 100.0).min(1.0),
        (entry.fired_hyperedges_data.len() as f32 / 20.0).min(1.0),
        (confirmed / 10.0).min(1.0),
        (surprised / 5.0).min(1.0),
        (entry.synapses_pruned as f32 / 10.0).min(1.0),
        (entry.synapses_sprouted as f32 / 10.0).min(1.0),
        surprise_ratio,
        (entry.timestep as f32 / 10000.0).min(1.0),
    ];
    let global_arr = global_vec.into_pyarray(py);

    // --- Hyperedge features — empty fast path ---
    if entry.fired_hyperedges_data.is_empty() {
        let empty = Array2::<f32>::zeros((0, 8));
        return Ok((empty.into_pyarray(py), global_arr));
    }

    // Build fired node UUID set for member activity calculation
    let fired_node_ids: HashSet<uuid::Uuid> =
        entry.fired_nodes_data.iter().map(|n| n.node_id).collect();

    // Node map: UUID → FiredNode reference for synapse lookup
    let node_map: HashMap<uuid::Uuid, &FiredNode> =
        entry.fired_nodes_data.iter().map(|n| (n.node_id, n)).collect();

    // Score and compute features for each fired hyperedge
    let mut scored: Vec<(f32, [f32; 8])> = Vec::with_capacity(entry.fired_hyperedges_data.len());

    for he in &entry.fired_hyperedges_data {
        let n_members = he.member_node_ids.len();

        // Member activity: fraction of member nodes that fired this step
        let members_fired = he.member_node_ids.iter()
            .filter(|id| fired_node_ids.contains(id))
            .count();
        let member_activity = if n_members > 0 {
            members_fired as f32 / n_members as f32
        } else {
            0.0f32
        };

        // Connectivity, mean weight, and hot eligibility from outgoing synapses
        // of fired member nodes — mirrors _synapse_stats_from_delta
        let mut syn_count = 0u32;
        let mut weight_sum = 0.0f32;
        let mut hot_eligibility = 0.0f32;

        for &member_id in &he.member_node_ids {
            if fired_node_ids.contains(&member_id) {
                if let Some(node) = node_map.get(&member_id) {
                    for syn in &node.outgoing {
                        syn_count += 1;
                        weight_sum += syn.weight;
                        if syn.eligibility_trace > hot_eligibility {
                            hot_eligibility = syn.eligibility_trace;
                        }
                    }
                }
            }
        }

        // Hot eligibility from salience signals — mirrors _hot_eligibility
        // Salience carries SALIENCE_HOT_ELIGIBILITY signals for member nodes.
        for sig in &entry.salience_data {
            if sig.signal_type == SALIENCE_HOT_ELIGIBILITY
                && he.member_node_ids.contains(&sig.node_id)
            {
                if sig.value > hot_eligibility {
                    hot_eligibility = sig.value;
                }
            }
        }

        let mean_weight = if syn_count > 0 {
            weight_sum / syn_count as f32
        } else {
            0.0f32
        };

        // log1p(activation_count) / 10.0 — activation history
        let act_log = ((he.activation_count as f32 + 1.0).ln() / 10.0).min(1.0);

        let features: [f32; 8] = [
            1.0f32,                                                   // fired flag
            member_activity,                                           // member activity
            (n_members as f32 / 100.0).min(1.0),                     // size
            (he.output_target_ids.len() as f32 / 50.0).min(1.0),    // output fan-out
            act_log,                                                   // activation history
            (syn_count as f32 / 50.0).min(1.0),                      // connectivity
            mean_weight,                                               // mean synapse weight
            hot_eligibility.min(1.0),                                  // hot eligibility
        ];

        // Score: all entries are fired; rank by member activity weight
        let score = 1.0f32 + member_activity * 0.5;
        scored.push((score, features));
    }

    // Sort descending by score, take top k
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(scored.len());

    // Build row-major flat vec → ndarray → numpy
    let mut flat: Vec<f32> = Vec::with_capacity(take * 8);
    for (_, feats) in &scored[..take] {
        flat.extend_from_slice(feats);
    }

    let he_arr = Array2::from_shape_vec((take, 8), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);

    Ok((he_arr, global_arr))
}

// =====================================================================
// Lenia Engine — Zero-Copy PyO3 bindings
//
// The engine never copies weight data. It receives numpy arrays (which
// ARE PyTorch tensor buffers), gets mutable f32 slices into them, and
// operates in place. Data stays binary the entire time.
// =====================================================================

#[cfg(feature = "python")]
use numpy::PyReadwriteArray1;

/// Registered matrix: metadata + a reference to the Python numpy array.
/// The numpy array IS the PyTorch parameter's data buffer.
#[cfg(feature = "python")]
struct RegisteredMatrix {
    meta: crate::lenia::WeightMeta,
    array: PyObject, // holds the numpy array alive (prevents GC)
}

/// Registered splat matrix: per-splat dynamics state + raw numpy array references.
///
/// Python owns mu/sigma/alpha/amp as numpy arrays. Rust holds raw pointers
/// into those arrays and modifies them in-place during step_splats(). Internal
/// state (phases, energies, etc.) is owned by Rust and not exposed to Python.
#[cfg(feature = "python")]
struct RegisteredSplatMatrix {
    name: String,
    meta: crate::lenia::SplatMeta,
    // Python array references (prevent GC). Flat f32 arrays:
    //   mu_array:    2N floats [row_0, col_0, row_1, col_1, ...]
    //   sigma_array: N floats
    //   alpha_array: N floats
    //   amp_array:   N floats
    mu_array: PyObject,
    sigma_array: PyObject,
    alpha_array: PyObject,
    amp_array: PyObject,
}

#[cfg(feature = "python")]
#[pyclass(name = "LeniaEngine")]
struct PyLeniaEngine {
    kernel: crate::lenia::RingKernel,
    config: crate::lenia::LeniaConfig,
    matrices: Vec<RegisteredMatrix>,
    topology: Option<crate::lenia::NeighborTopology>,
    splat_matrices: Vec<RegisteredSplatMatrix>,
    step_count: u64,
    splat_step_count: u64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLeniaEngine {
    #[new]
    #[pyo3(signature = (
        kernel_radius=3, kernel_sigma=1.0,
        growth_mu=0.15, growth_sigma=0.015, growth_scale=0.001,
        max_weight_delta=0.01, activation_coupling=1.0,
        energy_consumption_rate=0.1, energy_recovery_rate=0.05,
        inhibitor_production_rate=0.15, inhibitor_diffusion_rate=0.3,
        inhibitor_decay_rate=0.1,
        phase_dt=0.2, freq_adaptation_rate=0.01,
        amplitude_decay=0.02, amplitude_coupling=0.3, interference_rate=0.1
    ))]
    fn new(
        kernel_radius: usize,
        kernel_sigma: f32,
        growth_mu: f32,
        growth_sigma: f32,
        growth_scale: f32,
        max_weight_delta: f32,
        activation_coupling: f32,
        energy_consumption_rate: f32,
        energy_recovery_rate: f32,
        inhibitor_production_rate: f32,
        inhibitor_diffusion_rate: f32,
        inhibitor_decay_rate: f32,
        phase_dt: f32,
        freq_adaptation_rate: f32,
        amplitude_decay: f32,
        amplitude_coupling: f32,
        interference_rate: f32,
    ) -> Self {
        let config = crate::lenia::LeniaConfig {
            kernel_radius,
            kernel_sigma,
            growth_mu,
            growth_sigma,
            growth_scale,
            max_weight_delta,
            activation_coupling,
            energy_consumption_rate,
            energy_recovery_rate,
            inhibitor_production_rate,
            inhibitor_diffusion_rate,
            inhibitor_decay_rate,
            phase_dt,
            freq_adaptation_rate,
            amplitude_decay,
            amplitude_coupling,
            interference_rate,
            ..Default::default()
        };
        let kernel = crate::lenia::RingKernel::new(kernel_radius, kernel_sigma);
        Self {
            kernel,
            config,
            matrices: Vec::new(),
            topology: None,
            splat_matrices: Vec::new(),
            step_count: 0,
            splat_step_count: 0,
        }
    }

    /// Register a weight matrix for zero-copy Lenia processing.
    ///
    /// Pass a FLAT numpy f32 array (e.g. `param.data.numpy().ravel()`).
    /// The engine holds a reference to this array and operates on it
    /// directly — no copy. The array must stay alive while registered.
    ///
    /// Args:
    ///     name: parameter name (e.g. "layers.0.q_proj.weight")
    ///     array: numpy f32 array (flat, contiguous)
    ///     rows: number of rows
    ///     cols: number of cols
    fn register_matrix(
        &mut self,
        py: Python<'_>,
        name: String,
        array: &Bound<'_, PyArray1<f32>>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let len = array.len();
        if len != rows * cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("array length {} != rows*cols {}*{}", len, rows, cols)
            ));
        }
        // Verify contiguous
        if !array.is_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "array must be contiguous (C-order)"
            ));
        }
        self.matrices.push(RegisteredMatrix {
            meta: crate::lenia::WeightMeta {
                name,
                rows,
                cols,
                activation_mag: None,
                initial_l1: None,
                energy: 1.0,
                inhibitor: 0.0,
                phase: 0.0,
                amplitude: 0.5,   // start at mid-amplitude
                natural_freq: 0.1,
                group_id: 0,
                contribution: 0.0,
            },
            array: array.as_any().clone().unbind(), // prevent GC
        });
        Ok(())
    }

    /// Build neighbor topology from registered matrix names.
    /// Call once after all matrices are registered.
    /// Assigns phase angles, natural frequencies, and group IDs.
    fn build_topology(&mut self) {
        use std::f32::consts::TAU;
        let metas: Vec<crate::lenia::WeightMeta> = self.matrices.iter()
            .map(|r| crate::lenia::WeightMeta {
                name: r.meta.name.clone(),
                rows: r.meta.rows, cols: r.meta.cols,
                activation_mag: r.meta.activation_mag,
                initial_l1: r.meta.initial_l1,
                energy: r.meta.energy, inhibitor: r.meta.inhibitor,
                phase: r.meta.phase, amplitude: r.meta.amplitude,
                natural_freq: r.meta.natural_freq,
                group_id: r.meta.group_id, contribution: r.meta.contribution,
            }).collect();
        let topo = crate::lenia::NeighborTopology::build(&metas, &self.config);

        // Assign natural frequencies spread across the range
        let n = self.matrices.len();
        for (i, reg) in self.matrices.iter_mut().enumerate() {
            let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
            reg.meta.natural_freq = self.config.freq_min
                + t * (self.config.freq_max - self.config.freq_min);
            // Spread initial phases evenly
            reg.meta.phase = TAU * (i as f32 / n as f32);
            // Assign group from topology
            for (gi, group) in topo.groups.iter().enumerate() {
                if group.members.contains(&i) {
                    reg.meta.group_id = gi as u16;
                    break;
                }
            }
        }

        self.topology = Some(topo);
    }

    /// Update activation magnitudes.
    fn update_activations(&mut self, activations: &Bound<'_, PyDict>) -> PyResult<()> {
        for reg in &mut self.matrices {
            let name = &reg.meta.name;
            if let Some(val) = activations.get_item(name)? {
                reg.meta.activation_mag = Some(val.extract()?);
            } else {
                let parent = name.rsplit_once('.').map(|(p, _)| p).unwrap_or(name);
                if let Some(val) = activations.get_item(parent)? {
                    reg.meta.activation_mag = Some(val.extract()?);
                }
            }
        }
        Ok(())
    }

    /// Run one Lenia dynamics step on all registered matrices (parallel, zero-copy).
    ///
    /// Gets raw f32 pointers into numpy arrays, then releases the GIL and
    /// processes all matrices in parallel via rayon. Each matrix is independent
    /// memory — no overlap, no data race. PyTorch sees changes immediately.
    fn step(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();

        // Build topology on first step if not yet built
        if self.topology.is_none() && !self.matrices.is_empty() {
            self.build_topology();
        }

        // Inter-matrix coupling: Kuramoto phase + R-D inhibitor diffusion.
        // Must happen BEFORE the parallel per-matrix step (reads previous state).
        // Inter-matrix coupling: collect metas, couple, write back
        {
            let mut metas: Vec<crate::lenia::WeightMeta> = self.matrices.iter()
                .map(|r| crate::lenia::WeightMeta {
                    name: r.meta.name.clone(),
                    rows: r.meta.rows,
                    cols: r.meta.cols,
                    activation_mag: r.meta.activation_mag,
                    initial_l1: r.meta.initial_l1,
                    energy: r.meta.energy,
                    inhibitor: r.meta.inhibitor,
                    phase: r.meta.phase,
                    amplitude: r.meta.amplitude,
                    natural_freq: r.meta.natural_freq,
                    group_id: r.meta.group_id,
                    contribution: r.meta.contribution,
                })
                .collect();

            if let Some(ref topo) = self.topology {
                crate::lenia::inter_matrix_coupling(&mut metas, topo, &self.config);
            }

            // Write back coupled state (interference changes amplitude)
            for (reg, meta) in self.matrices.iter_mut().zip(metas.into_iter()) {
                reg.meta.inhibitor = meta.inhibitor;
                reg.meta.phase = meta.phase;
                reg.meta.amplitude = meta.amplitude;
                reg.meta.natural_freq = meta.natural_freq;
            }
        }

        // --- Activation normalization: finite resource ---
        // Activation magnitude becomes relative share (mean = 1.0).
        // Overexcited matrices compete for the same budget as quiet ones.
        {
            let acts: Vec<f32> = self.matrices.iter()
                .filter_map(|r| r.meta.activation_mag)
                .collect();
            if !acts.is_empty() {
                let mean_act: f32 = acts.iter().sum::<f32>() / acts.len() as f32;
                if mean_act > 0.0 {
                    for reg in self.matrices.iter_mut() {
                        if let Some(act) = reg.meta.activation_mag {
                            reg.meta.activation_mag = Some(act / mean_act);
                        }
                    }
                }
            }
        }

        // Phase 1: collect raw pointers while holding the GIL.
        // Each matrix's numpy buffer is independent memory.
        struct RawMatrix {
            ptr: *mut f32,
            len: usize,
        }
        // Safety: we send raw pointers to rayon threads. This is safe because:
        // - Each matrix is a separate contiguous allocation (no overlap)
        // - Each thread gets exactly one matrix (no shared writes)
        // - The numpy arrays are kept alive by self.matrices[].array
        // - We hold py (GIL) while collecting pointers, release for compute
        unsafe impl Send for RawMatrix {}

        let mut raw_matrices: Vec<RawMatrix> = Vec::with_capacity(self.matrices.len());

        for reg in &self.matrices {
            let array_bound = reg.array.bind(py).downcast::<PyArray1<f32>>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
                    format!("registered array is not numpy f32: {}", e)
                ))?
                .clone();
            // Get raw pointer — no borrow tracking, we manage safety ourselves
            let ptr = unsafe { array_bound.as_raw_array_mut().as_mut_ptr() };
            let len = array_bound.len();
            raw_matrices.push(RawMatrix { ptr, len });
        }

        // Phase 2: build work items, release GIL, process in parallel
        let kernel = &self.kernel;
        let config = &self.config;

        // Work item: raw pointer pair that's safe to send across threads.
        // Each item is independent memory — no overlap between items.
        struct Work {
            data_ptr: *mut f32,
            data_len: usize,
            meta_ptr: *mut crate::lenia::WeightMeta,
        }
        // Safety: each Work item points to independent, non-overlapping memory.
        unsafe impl Send for Work {}
        unsafe impl Sync for Work {}

        let work: Vec<Work> = raw_matrices.into_iter()
            .zip(self.matrices.iter_mut())
            .map(|(raw, reg)| Work {
                data_ptr: raw.ptr,
                data_len: raw.len,
                meta_ptr: &mut reg.meta as *mut _,
            })
            .collect();

        // Release GIL and process all matrices in parallel
        let results: Vec<crate::lenia::MatrixMetrics> = py.allow_threads(|| {
            use rayon::prelude::*;
            work.par_iter().map(|w| {
                let slice = unsafe { std::slice::from_raw_parts_mut(w.data_ptr, w.data_len) };
                let meta = unsafe { &mut *w.meta_ptr };
                crate::lenia::step_slice(slice, meta, kernel, config)
            }).collect()
        });

        // --- Amplitude pool conservation: finite resource ---
        // Total amplitude across all matrices = N * 0.5 (initial mean).
        // Interference and Hebbian growth redistribute it; they don't create it.
        // Matrices that dominate force others down — the dynamics decide who wins.
        {
            let total_amp: f32 = self.matrices.iter().map(|r| r.meta.amplitude).sum();
            let target = self.matrices.len() as f32 * 0.5;
            if total_amp > 0.0 {
                let scale = target / total_amp;
                for reg in self.matrices.iter_mut() {
                    reg.meta.amplitude = (reg.meta.amplitude * scale).clamp(0.0, 1.0);
                }
            }
        }

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.step_count += 1;

        let total_delta: f64 = results.iter().map(|r| r.delta_norm).sum();
        let total_energy: f64 = results.iter().map(|r| r.activation_energy).sum();
        let processed = results.iter().filter(|r| !r.skipped).count();
        let skipped = results.iter().filter(|r| r.skipped).count();

        // Energy scarcity stats
        let active_results: Vec<&crate::lenia::MatrixMetrics> =
            results.iter().filter(|r| !r.skipped).collect();
        let avg_energy = if !active_results.is_empty() {
            active_results.iter().map(|r| r.energy_level).sum::<f64>()
                / active_results.len() as f64
        } else { 1.0 };
        let min_energy = active_results.iter()
            .map(|r| r.energy_level).fold(1.0f64, f64::min);
        let depleted = active_results.iter()
            .filter(|r| r.energy_level < 0.05).count();

        let dict = PyDict::new(py);
        dict.set_item("step", self.step_count)?;
        dict.set_item("total_delta_norm", total_delta)?;
        dict.set_item("total_activation_energy", total_energy)?;
        dict.set_item("time_ms", elapsed_ms)?;
        dict.set_item("matrices_processed", processed)?;
        dict.set_item("matrices_skipped", skipped)?;
        dict.set_item("avg_energy", avg_energy)?;
        dict.set_item("min_energy", min_energy)?;
        dict.set_item("depleted_matrices", depleted)?;

        // Amplitude + inhibitor stats
        let avg_amplitude = if !active_results.is_empty() {
            active_results.iter().map(|r| r.amplitude).sum::<f64>()
                / active_results.len() as f64
        } else { 0.5 };
        let avg_born_gate = if !active_results.is_empty() {
            active_results.iter().map(|r| r.born_gate).sum::<f64>()
                / active_results.len() as f64
        } else { 0.25 };
        let avg_inhibitor = if !active_results.is_empty() {
            active_results.iter().map(|r| r.inhibitor_level).sum::<f64>()
                / active_results.len() as f64
        } else { 0.0 };
        let high_amp = active_results.iter().filter(|r| r.amplitude > 0.7).count();
        let low_amp = active_results.iter().filter(|r| r.amplitude < 0.3).count();

        dict.set_item("avg_amplitude", avg_amplitude)?;
        dict.set_item("avg_born_gate", avg_born_gate)?;
        dict.set_item("avg_inhibitor", avg_inhibitor)?;
        dict.set_item("high_amplitude", high_amp)?;
        dict.set_item("low_amplitude", low_amp)?;

        Ok(dict.into_py_any(py)?)
    }

    /// Get energy levels for all matrices as a dict {name: energy}.
    fn get_energy_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for reg in &self.matrices {
            dict.set_item(&reg.meta.name, reg.meta.energy)?;
        }
        Ok(dict)
    }

    /// Get amplitude levels for all matrices as a dict {name: amplitude}.
    /// Used by Python hooks for Born rule forward pass scaling.
    fn get_amplitude_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for reg in &self.matrices {
            dict.set_item(&reg.meta.name, reg.meta.amplitude)?;
        }
        Ok(dict)
    }

    /// Get inhibitor levels for all matrices as a dict {name: inhibitor}.
    fn get_inhibitor_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for reg in &self.matrices {
            dict.set_item(&reg.meta.name, reg.meta.inhibitor)?;
        }
        Ok(dict)
    }

    #[getter]
    fn num_matrices(&self) -> usize {
        self.matrices.len()
    }

    #[getter]
    fn steps(&self) -> u64 {
        self.step_count
    }

    // =========================================================================
    // Splat Matrix Methods
    // =========================================================================

    /// Register a splat matrix for Lenia processing.
    ///
    /// Args:
    ///     name:         identifier (e.g. "layers.0.myelin_splats")
    ///     mu_flat:      flat f32 numpy array, 2N elements [r0, c0, r1, c1, ...]
    ///                   positions normalized to [0, 1]
    ///     sigma_flat:   flat f32 numpy array, N elements — isotropic spread
    ///     alpha_flat:   flat f32 numpy array, N elements — weight values (signed)
    ///     amp_flat:     flat f32 numpy array, N elements — Born amplitudes [0, 1]
    ///     is_myelin:    list of bools, length N — True = learnable, False = base
    ///     rows, cols:   original weight matrix shape
    #[pyo3(signature = (name, mu_flat, sigma_flat, alpha_flat, amp_flat, is_myelin, rows, cols))]
    fn register_splat_matrix(
        &mut self,
        py: Python<'_>,
        name: String,
        mu_flat: &Bound<'_, PyArray1<f32>>,
        sigma_flat: &Bound<'_, PyArray1<f32>>,
        alpha_flat: &Bound<'_, PyArray1<f32>>,
        amp_flat: &Bound<'_, PyArray1<f32>>,
        is_myelin: Vec<bool>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let n = is_myelin.len();
        if mu_flat.len() != 2 * n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("mu_flat length {} != 2*n ({})", mu_flat.len(), 2 * n)
            ));
        }
        if sigma_flat.len() != n || alpha_flat.len() != n || amp_flat.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("sigma/alpha/amp must all have length n={}", n)
            ));
        }
        for arr in &[mu_flat, sigma_flat, alpha_flat, amp_flat] {
            if !arr.is_contiguous() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "all splat arrays must be contiguous (C-order)"
                ));
            }
        }
        let meta = crate::lenia::SplatMeta::new(n, rows, cols, is_myelin);
        self.splat_matrices.push(RegisteredSplatMatrix {
            name,
            meta,
            mu_array: mu_flat.as_any().clone().unbind(),
            sigma_array: sigma_flat.as_any().clone().unbind(),
            alpha_array: alpha_flat.as_any().clone().unbind(),
            amp_array: amp_flat.as_any().clone().unbind(),
        });
        let _ = py; // py used implicitly via bind(py) calls during step
        Ok(())
    }

    /// Update activation magnitudes for splat matrices.
    /// Same API as update_activations but for the splat population.
    fn update_splat_activations(&mut self, activations: &Bound<'_, PyDict>) -> PyResult<()> {
        for reg in &mut self.splat_matrices {
            if let Some(val) = activations.get_item(&reg.name)? {
                reg.meta.activation_mag = Some(val.extract()?);
            } else {
                // Try parent name (strip trailing ".myelin_splats" etc.)
                let parent = reg.name.rsplit_once('.').map(|(p, _)| p).unwrap_or(&reg.name);
                if let Some(val) = activations.get_item(parent)? {
                    reg.meta.activation_mag = Some(val.extract()?);
                }
            }
        }
        Ok(())
    }

    /// Run one Lenia step on all registered splat matrices (parallel, zero-copy).
    ///
    /// The four splat arrays (mu, sigma, alpha, amp) are modified in-place.
    /// Python sees changes immediately — no copy required.
    fn step_splats(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();

        // Collect raw pointers while holding the GIL.
        struct RawSplat {
            mu_ptr:    *mut f32,
            sigma_ptr: *mut f32,
            alpha_ptr: *mut f32,
            amp_ptr:   *mut f32,
        }
        // Safety: each RegisteredSplatMatrix holds independent non-overlapping arrays.
        unsafe impl Send for RawSplat {}

        let mut raw: Vec<RawSplat> = Vec::with_capacity(self.splat_matrices.len());
        for reg in &self.splat_matrices {
            let mu = reg.mu_array.bind(py).downcast::<PyArray1<f32>>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
                    format!("[splat] mu array is not f32: {}", e)))?
                .clone();
            let sigma = reg.sigma_array.bind(py).downcast::<PyArray1<f32>>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
                    format!("[splat] sigma array is not f32: {}", e)))?
                .clone();
            let alpha = reg.alpha_array.bind(py).downcast::<PyArray1<f32>>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
                    format!("[splat] alpha array is not f32: {}", e)))?
                .clone();
            let amp = reg.amp_array.bind(py).downcast::<PyArray1<f32>>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
                    format!("[splat] amp array is not f32: {}", e)))?
                .clone();
            raw.push(RawSplat {
                mu_ptr:    unsafe { mu.as_raw_array_mut().as_mut_ptr() },
                sigma_ptr: unsafe { sigma.as_raw_array_mut().as_mut_ptr() },
                alpha_ptr: unsafe { alpha.as_raw_array_mut().as_mut_ptr() },
                amp_ptr:   unsafe { amp.as_raw_array_mut().as_mut_ptr() },
            });
        }

        // Release GIL and process all splat matrices in parallel.
        let config = &self.config;
        struct Work {
            raw: RawSplat,
            meta_ptr: *mut crate::lenia::SplatMeta,
            name_ptr: *const String,
        }
        unsafe impl Send for Work {}
        unsafe impl Sync for Work {}

        let work: Vec<Work> = raw.into_iter()
            .zip(self.splat_matrices.iter_mut())
            .map(|(r, reg)| Work {
                raw: r,
                meta_ptr: &mut reg.meta as *mut _,
                name_ptr: &reg.name as *const _,
            })
            .collect();

        let results: Vec<crate::lenia::SplatMetrics> = py.allow_threads(|| {
            use rayon::prelude::*;
            work.par_iter().map(|w| {
                let n = unsafe { (*w.meta_ptr).n };
                let mu    = unsafe { std::slice::from_raw_parts_mut(w.raw.mu_ptr, 2 * n) };
                let sigma = unsafe { std::slice::from_raw_parts_mut(w.raw.sigma_ptr, n) };
                let alpha = unsafe { std::slice::from_raw_parts_mut(w.raw.alpha_ptr, n) };
                let amp   = unsafe { std::slice::from_raw_parts_mut(w.raw.amp_ptr, n) };
                let meta  = unsafe { &mut *w.meta_ptr };
                let name  = unsafe { &*w.name_ptr };
                crate::lenia::step_splats(mu, sigma, alpha, amp, meta, config, name)
            }).collect()
        });

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.splat_step_count += 1;

        // Build return dict
        let total_myelin_active: usize = results.iter().map(|r| r.n_myelin_active).sum();
        let total_drift: f32 = results.iter().map(|r| r.total_drift).sum();
        let avg_amp = if !results.is_empty() {
            results.iter().map(|r| r.avg_amp as f64).sum::<f64>() / results.len() as f64
        } else { 0.5 };
        let total_high_amp: usize = results.iter().map(|r| r.high_amp).sum();
        let total_low_amp: usize = results.iter().map(|r| r.low_amp).sum();
        let total_splats: usize = results.iter().map(|r| r.n_splats).sum();

        let dict = PyDict::new(py);
        dict.set_item("step", self.splat_step_count)?;
        dict.set_item("time_ms", elapsed_ms)?;
        dict.set_item("splat_matrices", results.len())?;
        dict.set_item("total_splats", total_splats)?;
        dict.set_item("myelin_active", total_myelin_active)?;
        dict.set_item("total_drift", total_drift)?;
        dict.set_item("avg_amp", avg_amp)?;
        dict.set_item("high_amp_splats", total_high_amp)?;
        dict.set_item("low_amp_splats", total_low_amp)?;

        Ok(dict.into_py_any(py)?)
    }

    /// Get amplitude levels for all splat matrices as a dict {name: avg_amp}.
    fn get_splat_amp_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for reg in &self.splat_matrices {
            // amp is in the numpy array; we can't read it here without binding.
            // Return meta stats instead: energy mean as proxy.
            let mean_energy: f32 = reg.meta.energies.iter().sum::<f32>()
                / reg.meta.n.max(1) as f32;
            dict.set_item(&reg.name, mean_energy)?;
        }
        Ok(dict)
    }

    #[getter]
    fn num_splat_matrices(&self) -> usize {
        self.splat_matrices.len()
    }

    #[getter]
    fn splat_steps(&self) -> u64 {
        self.splat_step_count
    }
}

// =====================================================================
// deposit_outcome / deposit_topology
//
// Zero-copy ingestion path: numpy array -> BTF binary -> tract files.
// Rust owns the bytes end-to-end. Python never touches them.
// =====================================================================

#[cfg(feature = "python")]
use numpy::PyReadonlyArray1;

/// deposit_outcome(timestamp, module_id, target_id, success, embedding, tract_paths, metadata=None)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "deposit_outcome")]
#[pyo3(signature = (timestamp, module_id, target_id, success, embedding, tract_paths, metadata=None))]
fn deposit_outcome(
    py: Python<'_>,
    timestamp: f64,
    module_id: &str,
    target_id: &str,
    success: bool,
    embedding: &Bound<'_, PyArray1<f32>>,
    tract_paths: Vec<String>,
    metadata: Option<&Bound<'_, PyBytes>>,
) -> PyResult<()> {
    let ro: PyReadonlyArray1<f32> = embedding.readonly();
    let emb_slice = ro.as_slice()?;
    let meta_bytes = match metadata {
        Some(b) => b.as_bytes().to_vec(),
        None => vec![],
    };
    let entry = OutcomeEntry {
        timestamp,
        module_id: module_id.to_string(),
        target_id: target_id.to_string(),
        success,
        embedding_dim: emb_slice.len() as u16,
        embedding: emb_slice.to_vec(),
        metadata: meta_bytes,
    };
    let bytes = crate::write::write_outcome(&entry);
    py.allow_threads(|| {
        for path in &tract_paths {
            if let Err(e) = crate::write::deposit_to_file(path, &bytes) {
                eprintln!("[ng_tract] deposit_outcome failed ({}): {}", path, e);
            }
        }
    });
    Ok(())
}

/// deposit_experience(content, source, tract_path, content_type="text")
///
/// Write raw experience to the feeder tract. No embedding, no classification,
/// no transformation. The content enters as raw bytes and stays that way
/// until extraction by the topology owner. Law 7.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "deposit_experience")]
#[pyo3(signature = (content, source, tract_path, content_type="text"))]
fn deposit_experience(
    py: Python<'_>,
    content: &[u8],
    source: &str,
    tract_path: &str,
    content_type: &str,
) -> PyResult<()> {
    // content is RAW BYTES (LAW 7 — the substrate receives raw experience; no UTF-8
    // gate at the deposit boundary). ExperienceEntry.content is Vec<u8>; matching it
    // here. 2026-06-09 CC — was &str, which forced UTF-8 validity on raw experience
    // and broke Python callers (TID/NG) that correctly pass bytes.
    let entry = ExperienceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        source: source.to_string(),
        content_type: content_type.to_string(),
        content: content.to_vec(),
    };
    let bytes = crate::write::write_experience(&entry);
    py.allow_threads(|| {
        if let Err(e) = crate::write::deposit_to_file(tract_path, &bytes) {
            eprintln!("[ng_tract] deposit_experience failed ({}): {}", tract_path, e);
        }
    });
    Ok(())
}

/// deposit_topology(step_result, graph, vector_db, tract_paths)
///
/// Accepts neuro_foundation StepResult + Graph objects directly.
/// Verified field names: StepResult.fired_node_ids/fired_hyperedge_ids/synapses_pruned/
/// synapses_sprouted/predictions_confirmed/predictions_surprised; Graph.timestep/nodes/hyperedges;
/// Node.metadata['label']; Hyperedge.member_nodes(Set[str])/output_targets(List[str])/
/// activation_count/metadata['label']. Embeddings omitted (live in vector DB, not on Node).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "deposit_topology")]
fn deposit_topology(
    py: Python<'_>,
    step_result: &Bound<'_, pyo3::PyAny>,
    graph: &Bound<'_, pyo3::PyAny>,
    _vector_db: &Bound<'_, pyo3::PyAny>,
    tract_paths: Vec<String>,
) -> PyResult<()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let fired_node_ids: Vec<String> = step_result.getattr("fired_node_ids")?.extract()?;
    let fired_hyperedge_ids: Vec<String> = step_result.getattr("fired_hyperedge_ids")?.extract()?;
    let synapses_pruned: u16 = step_result.getattr("synapses_pruned")?.extract::<i64>()? as u16;
    let synapses_sprouted: u16 = step_result.getattr("synapses_sprouted")?.extract::<i64>()? as u16;
    let predictions_confirmed: u16 = step_result.getattr("predictions_confirmed")?.extract::<i64>()? as u16;
    let predictions_surprised: u16 = step_result.getattr("predictions_surprised")?.extract::<i64>()? as u16;
    let timestep: u32 = graph.getattr("timestep")?.extract::<i64>()? as u32;
    let nodes_dict = graph.getattr("nodes")?;
    let hes_dict = graph.getattr("hyperedges")?;
    let mut fired_nodes = Vec::new();
    for node_id_str in &fired_node_ids {
        let node_id = match uuid::Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => continue,
        };
        let node_obj = match nodes_dict.get_item(node_id_str) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let label: String = if let Ok(meta) = node_obj.getattr("metadata") {
            if let Ok(lbl) = meta.get_item("label") {
                lbl.extract::<String>().unwrap_or_default()
            } else { String::new() }
        } else { String::new() };
        fired_nodes.push(FiredNode {
            node_id, label, embedding_dim: 0, embedding: vec![], outgoing: vec![],
        });
    }
    let mut fired_hyperedges = Vec::new();
    for he_id_str in &fired_hyperedge_ids {
        let he_id = match uuid::Uuid::parse_str(he_id_str) {
            Ok(u) => u,
            Err(_) => continue,
        };
        let he_obj = match hes_dict.get_item(he_id_str) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let label: String = if let Ok(meta) = he_obj.getattr("metadata") {
            if let Ok(lbl) = meta.get_item("label") {
                lbl.extract::<String>().unwrap_or_default()
            } else { String::new() }
        } else { String::new() };
        let activation_count: u32 = he_obj.getattr("activation_count")
            .ok().and_then(|v: Bound<'_, pyo3::PyAny>| v.extract::<i64>().ok())
            .unwrap_or(0) as u32;
        let member_ids: Vec<uuid::Uuid> = he_obj.getattr("member_nodes")
            .ok().and_then(|v: Bound<'_, pyo3::PyAny>| v.extract::<Vec<String>>().ok())
            .unwrap_or_default().iter()
            .filter_map(|s| uuid::Uuid::parse_str(s).ok()).collect();
        let output_ids: Vec<uuid::Uuid> = he_obj.getattr("output_targets")
            .ok().and_then(|v: Bound<'_, pyo3::PyAny>| v.extract::<Vec<String>>().ok())
            .unwrap_or_default().iter()
            .filter_map(|s| uuid::Uuid::parse_str(s).ok()).collect();
        fired_hyperedges.push(FiredHyperedge {
            hyperedge_id: he_id, label, activation_count,
            member_node_ids: member_ids, output_target_ids: output_ids,
        });
    }
    let entry = TopologyEntry {
        timestamp, timestep, predictions_confirmed, predictions_surprised,
        synapses_pruned, synapses_sprouted,
        fired_nodes, fired_hyperedges, salience: vec![],
    };
    let bytes = crate::write::write_topology(&entry);
    py.allow_threads(|| {
        for path in &tract_paths {
            if let Err(e) = crate::write::deposit_to_file(path, &bytes) {
                eprintln!("[ng_tract] deposit_topology failed ({}): {}", path, e);
            }
        }
    });
    Ok(())
}
