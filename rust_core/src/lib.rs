//! Condensate Core — Rust implementation
//!
//! Living memory manager: learns access patterns through causal topology,
//! predicts future accesses, manages memory tiers.
//!
//! This crate provides:
//! - AccessGraph: learns memory access topology from observations
//! - Predictor: predicts next access from causal spike propagation
//! - Membrane: system-level memory allocation interceptor (LD_PRELOAD)
//! - Python bindings via PyO3 (optional, feature-gated)

pub mod graph;
pub mod predictor;
pub mod membrane;
pub mod condenser;
pub mod pipeline;
pub mod lenia;
mod bench;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module: condensate_core
#[cfg(feature = "python")]
#[pymodule]
fn condensate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<graph::AccessGraph>()?;
    m.add_class::<predictor::RustPredictor>()?;
    m.add_class::<predictor::Prediction>()?;
    Ok(())
}
