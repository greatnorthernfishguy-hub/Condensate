//! PyO3 bindings for the demo: real lz4 measurement + tiering.
//! Pure `*_impl` fns are always compiled and unit-tested; thin #[pyfunction]
//! wrappers (python feature) expose them to Python. lz4 calls mirror
//! condenser.rs::ManagedRegion (compress_prepend_size / decompress_size_prepended).
//
// ---- Changelog ----
// [2026-06-24] Claude Code — Task 1: Expose lz4 measure + tier via PyO3 abi3
// What: New pybind.rs with pure *_impl fns + #[cfg(feature="python")] py wrappers
// Why: Demo Space rebuild needs real Rust lz4 measurement exposed to Python
// How: Always-compiled impl fns keep cargo test working without pyo3; thin
//      #[pyfunction] wrappers behind feature gate register into condensate_core module
// -------------------

/// lz4-compressed length of `data` (same call as the engine's compressor).
pub fn lz4_compress_len_impl(data: &[u8]) -> usize {
    lz4_flex::compress_prepend_size(data).len()
}

/// Lossless round-trip: compress then decompress, must equal the original.
/// Backs the demo's "✓ lossless verified" claim.
pub fn lz4_verify_roundtrip_impl(data: &[u8]) -> bool {
    let c = lz4_flex::compress_prepend_size(data);
    matches!(lz4_flex::decompress_size_prepended(&c), Ok(d) if d == data)
}

/// HOT/WARM/COLD by access fraction vs the busiest region.
pub fn classify_tier_impl(access_count: u64, max_access: u64) -> &'static str {
    let f = if max_access == 0 { 0.0 } else { access_count as f64 / max_access as f64 };
    if f >= 0.50 { "HOT" } else if f >= 0.10 { "WARM" } else { "COLD" }
}

#[cfg(feature = "python")]
pub mod py {
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn lz4_compress_len(data: &[u8]) -> usize { super::lz4_compress_len_impl(data) }

    #[pyfunction]
    pub fn lz4_verify_roundtrip(data: &[u8]) -> bool { super::lz4_verify_roundtrip_impl(data) }

    #[pyfunction]
    pub fn classify_tier(access_count: u64, max_access: u64) -> String {
        super::classify_tier_impl(access_count, max_access).to_string()
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(lz4_compress_len, m)?)?;
        m.add_function(wrap_pyfunction!(lz4_verify_roundtrip, m)?)?;
        m.add_function(wrap_pyfunction!(classify_tier, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressible_data_shrinks() {
        let data = vec![7u8; 100_000];
        assert!(lz4_compress_len_impl(&data) < data.len());
    }

    #[test]
    fn roundtrip_is_lossless() {
        let data: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        assert!(lz4_verify_roundtrip_impl(&data));
    }

    #[test]
    fn tiers_classify_by_fraction() {
        assert_eq!(classify_tier_impl(100, 100), "HOT");
        assert_eq!(classify_tier_impl(20, 100), "WARM");
        assert_eq!(classify_tier_impl(1, 100), "COLD");
        assert_eq!(classify_tier_impl(5, 0), "COLD"); // no accesses anywhere
    }
}
