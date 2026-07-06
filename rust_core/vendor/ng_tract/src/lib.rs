//! ng_tract — Binary Tract Format for the NeuroGraph substrate data path.
//!
//! Implements the BTF v0.1 spec: cache-line aligned, zero-copy readable,
//! appendable binary entries that replace JSONL on River tracts.
//!
//! Two entry types:
//! - Outcome: what a module learned (record_outcome)
//! - Topology: what the Graph produced (graph.step output)

pub mod format;
pub mod read;
pub mod write;

// Heavy substrate compute — gated behind the "engine" feature so BTF-only
// consumers (default-features=false) pull no rayon/BLAS. The "python" feature
// enables "engine" (python.rs wraps these). See Cargo.toml [features].
#[cfg(feature = "engine")]
pub mod lenia;
#[cfg(feature = "engine")]
pub mod features;
#[cfg(feature = "engine")]
pub mod ng_lite_core;

#[cfg(feature = "python")]
pub mod python;

pub use format::*;
