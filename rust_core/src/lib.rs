//! Condensate Core — Rust implementation
//!
//! Living memory manager: learns access patterns through causal topology,
//! predicts future accesses, manages memory tiers via continuous thermal
//! field dynamics.
//!
//! # Modules
//!
//! ## Core pipeline (original)
//! - `graph` — AccessGraph: learns memory access topology
//! - `predictor` — RustPredictor: causal spike propagation predictions
//! - `membrane` — LD_PRELOAD malloc/free interception
//! - `condenser` — HOT/WARM/COLD tier management with real memory ops
//! - `pipeline` — Living loop connecting all components
//! - `lenia` — Continuous thermal field dynamics
//!
//! ## Condensing strategies (Phase 1 blocks F-L)
//! - `keyframe` — Keyframe/delta encoding (video codec model)
//! - `sparse` — Partial decompression (serve exactly what's needed)
//! - `locality` — Manufactured spatial locality + software prefetch
//! - `sleep` — Biological sleep consolidation cycle
//! - `gate` — Prediction gate (KISS overhead reduction)
//! - `splat` — Gaussian splat field geometry
//! - `erasure` — Erasure coding + holographic boundaries
//!
//! # Build targets
//!
//! - `cargo build --features python` → Python module (.so)
//! - `cargo build --no-default-features --features preload` → LD_PRELOAD .so

pub mod graph;
pub mod predictor;
pub mod membrane;
pub mod condenser;
pub mod pipeline;
pub mod lenia;
pub mod keyframe;
pub mod sparse;
pub mod gate;
pub mod locality;
pub mod sleep;
pub mod splat;
pub mod erasure;
mod bench;
mod pybind;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module: condensate_core
///
/// Exposes the core pipeline types and condensing strategies to Python.
/// Python is orchestration only — the data path is Rust.
#[cfg(feature = "python")]
#[pymodule]
fn condensate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_class::<graph::AccessGraph>()?;
    m.add_class::<predictor::RustPredictor>()?;
    m.add_class::<predictor::Prediction>()?;
    // Demo measurement + tiering
    pybind::py::register(m)?;
    Ok(())
}
