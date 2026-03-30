//! Condensate Core — Rust implementation
//!
//! Living memory manager: learns access patterns through causal topology,
//! predicts future accesses, manages memory tiers.
//!
//! This crate provides:
//! - AccessGraph: learns memory access topology from observations
//! - Predictor: predicts next access from causal spike propagation
//! - Python bindings via PyO3

mod graph;
mod predictor;
mod bench;

use pyo3::prelude::*;

/// Python module: condensate_core
#[pymodule]
fn condensate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<graph::AccessGraph>()?;
    m.add_class::<predictor::RustPredictor>()?;
    m.add_class::<predictor::Prediction>()?;
    Ok(())
}
