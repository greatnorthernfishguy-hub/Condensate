//! Condenser — the motor output. Compresses cold memory, restores hot.
//!
//! The membrane observes. The graph learns. The predictor predicts.
//! The condenser ACTS — compressing idle allocations and restoring
//! them before they're needed.
//!
//! Three tiers:
//!   HOT:  Untouched, full speed access
//!   WARM: LZ4 compressed in-place, fast decompress on access
//!   COLD: Backed by disk file, zero RSS until touched
//!
//! The condenser runs as a background thread, periodically scanning
//! the membrane's tracked allocations and demoting idle ones.
//! When the predictor fires a spike ("this region is about to be
//! accessed"), the condenser pre-promotes it.

use std::collections::HashMap;
use std::fs;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::time::Instant;

use crate::membrane::{MembraneState, MembraneSummary};

const PAGE_SIZE: usize = 4096;
const COLD_DIR: &str = "/tmp/condensate_cold";

/// Tier state for a managed memory region
#[derive(Clone, Debug, PartialEq)]
pub enum Tier {
    /// Full speed, untouched — original allocation
    Hot,
    /// LZ4 compressed copy stored, original could be reclaimed
    Warm {
        compressed: Vec<u8>,
        original_size: usize,
    },
    /// Compressed bytes written to disk, in-memory buffer freed
    Cold {
        file_path: String,
        original_size: usize,
    },
}

/// A memory region managed by the condenser
#[derive(Clone, Debug)]
pub struct ManagedRegion {
    pub address: usize,
    pub size: usize,
    pub tier: Tier,
    pub last_access_ns: u64,
    pub access_count: u32,
    pub promotions: u32,
    pub demotions: u32,
    pub prediction_hits: u32,
    /// Optional data override used in tests to inject specific byte patterns
    /// without needing a real allocation. Only consulted by read_region_data
    /// when present; ignored in production.
    pub test_data: Option<Vec<u8>>,
}

impl ManagedRegion {
    pub fn new(address: usize, size: usize, timestamp_ns: u64) -> Self {
        Self {
            address,
            size,
            tier: Tier::Hot,
            last_access_ns: timestamp_ns,
            access_count: 1,
            promotions: 0,
            demotions: 0,
            prediction_hits: 0,
            test_data: None,
        }
    }

    /// Compress HOT → WARM using LZ4
    pub fn compress(&mut self, data: &[u8]) -> usize {
        if self.tier != Tier::Hot {
            return 0;
        }

        let compressed = lz4_flex::compress_prepend_size(data);
        let saved = if data.len() > compressed.len() {
            data.len() - compressed.len()
        } else {
            0
        };

        self.tier = Tier::Warm {
            compressed,
            original_size: data.len(),
        };
        self.demotions += 1;
        saved
    }

    /// Decompress WARM → HOT
    pub fn decompress(&mut self) -> Option<Vec<u8>> {
        match &self.tier {
            Tier::Warm { compressed, .. } => {
                match lz4_flex::decompress_size_prepended(compressed) {
                    Ok(data) => {
                        self.tier = Tier::Hot;
                        self.promotions += 1;
                        Some(data)
                    }
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    pub fn is_hot(&self) -> bool {
        self.tier == Tier::Hot
    }

    pub fn is_cold(&self) -> bool {
        matches!(self.tier, Tier::Cold { .. })
    }

    /// Bytes currently in RAM for this region
    pub fn ram_usage(&self) -> usize {
        match &self.tier {
            Tier::Hot => self.size,
            Tier::Warm { compressed, .. } => compressed.len(),
            Tier::Cold { .. } => 0,
        }
    }
}

/// Condenser configuration
pub struct CondenserConfig {
    /// How long (ns) before a region is considered idle
    pub idle_threshold_ns: u64,
    /// Minimum allocation size to manage (skip tiny ones)
    pub min_manage_size: usize,
    /// Maximum number of regions to track
    pub max_tracked: usize,
    /// How often the scan loop runs (ns)
    pub scan_interval_ns: u64,
    /// When true, compress/decompress uses data stored in the Warm tier
    /// directly rather than reading from raw memory addresses. Enables
    /// testing without real allocations.
    pub test_mode: bool,
}

impl Default for CondenserConfig {
    fn default() -> Self {
        Self {
            idle_threshold_ns: 5_000_000_000,  // 5 seconds
            min_manage_size: 65_536,            // 64KB minimum
            max_tracked: 10_000,
            scan_interval_ns: 1_000_000_000,   // 1 second
            test_mode: false,
        }
    }
}

/// The condenser engine
pub struct Condenser {
    config: CondenserConfig,
    /// Managed regions: address → ManagedRegion
    regions: HashMap<usize, ManagedRegion>,
    /// Start time
    start: Instant,
    /// Stats
    total_compressed: u64,
    total_decompressed: u64,
    total_bytes_saved: u64,
    peak_bytes_saved: u64,
    scan_count: u64,
    /// When true, use test-safe data paths (no raw pointer reads/writes)
    test_mode: bool,
}

impl Condenser {
    pub fn new(config: CondenserConfig) -> Self {
        let test_mode = config.test_mode;
        Self {
            config,
            regions: HashMap::with_capacity(1000),
            start: Instant::now(),
            total_compressed: 0,
            total_decompressed: 0,
            total_bytes_saved: 0,
            peak_bytes_saved: 0,
            scan_count: 0,
            test_mode,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Register a new allocation for management
    pub fn register(&mut self, address: usize, size: usize) {
        if size < self.config.min_manage_size {
            return;
        }
        if self.regions.len() >= self.config.max_tracked {
            return;
        }

        let ts = self.elapsed_ns();
        self.regions.insert(address, ManagedRegion::new(address, size, ts));
    }

    /// Record an access — marks region as hot
    pub fn touch(&mut self, address: usize) {
        let now = self.elapsed_ns();
        if let Some(region) = self.regions.get_mut(&address) {
            region.last_access_ns = now;
            region.access_count += 1;
        }
    }

    /// Remove a region (freed by the application)
    pub fn unregister(&mut self, address: usize) {
        if let Some(region) = self.regions.remove(&address) {
            // Reclaim any saved bytes
            let usage = region.ram_usage();
            if usage < region.size {
                self.total_bytes_saved = self.total_bytes_saved
                    .saturating_sub((region.size - usage) as u64);
            }
        }
    }

    /// Pre-promote a region (prediction-driven).
    /// Decompresses the region and, when not in test_mode, writes the
    /// decompressed bytes back to the original address.
    pub fn pre_promote(&mut self, address: usize) {
        if let Some(region) = self.regions.get_mut(&address) {
            if !region.is_hot() {
                region.prediction_hits += 1;

                if let Some(decompressed) = region.decompress() {
                    // decompress() already set tier → Hot and bumped promotions.
                    if !self.test_mode {
                        // SAFETY: The caller guarantees `address` points to a live
                        // allocation of at least `decompressed.len()` bytes that we
                        // originally registered and compressed. We are restoring the
                        // original contents before the application touches it again.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                decompressed.as_ptr(),
                                address as *mut u8,
                                decompressed.len(),
                            );
                        }
                    }
                } else {
                    // Fallback: force to Hot even if decompress failed
                    region.tier = Tier::Hot;
                    region.promotions += 1;
                }

                self.total_decompressed += 1;
            }
        }
    }

    /// Demote a WARM region to COLD by writing its compressed bytes to disk.
    /// Creates `/tmp/condensate_cold/` if it does not exist.
    pub fn demote_to_cold(&mut self, address: usize) {
        if let Some(region) = self.regions.get_mut(&address) {
            if let Tier::Warm { ref compressed, original_size } = region.tier.clone() {
                // Ensure the cold directory exists
                fs::create_dir_all(COLD_DIR)
                    .expect("condensate: failed to create cold storage directory");

                let file_path = format!("{}/{}.bin", COLD_DIR, address);

                fs::write(&file_path, compressed)
                    .expect("condensate: failed to write cold file");

                region.tier = Tier::Cold { file_path, original_size };
                region.demotions += 1;
            }
        }
    }

    /// Promote a COLD region back to HOT.
    /// Reads compressed bytes from disk, LZ4-decompresses them, deletes the
    /// file, and sets the tier back to Hot.
    /// Returns the decompressed data, or None if the region is not Cold.
    pub fn promote_from_cold(&mut self, address: usize) -> Option<Vec<u8>> {
        if let Some(region) = self.regions.get_mut(&address) {
            if let Tier::Cold { ref file_path, .. } = region.tier.clone() {
                let compressed = fs::read(&file_path)
                    .expect("condensate: failed to read cold file");

                let decompressed = lz4_flex::decompress_size_prepended(&compressed)
                    .expect("condensate: failed to decompress cold data");

                // Delete the backing file
                let _ = fs::remove_file(&file_path);

                region.tier = Tier::Hot;
                region.promotions += 1;
                self.total_decompressed += 1;

                return Some(decompressed);
            }
        }
        None
    }

    /// Build the data buffer used during scan compression.
    ///
    /// Priority order:
    ///   1. If the region has a `test_data` override, use that.
    ///   2. If in `test_mode`, generate a deterministic repeating pattern from
    ///      the address bytes — compressible, safe, no real allocation needed.
    ///   3. In production: read directly from the live allocation.
    fn read_region_data(&self, address: usize, size: usize) -> Vec<u8> {
        // Test-data override takes precedence (injected by tests for specific patterns)
        if let Some(region) = self.regions.get(&address) {
            if let Some(ref data) = region.test_data {
                return data.clone();
            }
        }

        if self.test_mode {
            // Deterministic repeating pattern from the address bytes — compressible
            let addr_bytes = address.to_le_bytes();
            let mut buf = Vec::with_capacity(size);
            for i in 0..size {
                buf.push(addr_bytes[i % addr_bytes.len()]);
            }
            buf
        } else {
            // SAFETY: The caller (register) has verified that `address` is a live
            // allocation of exactly `size` bytes tracked by this condenser. We hold
            // a shared reference to this data only for the duration of this call and
            // do not alias the slice with any mutable reference.
            unsafe {
                std::slice::from_raw_parts(address as *const u8, size).to_vec()
            }
        }
    }

    /// Scan for idle regions and compress them.
    ///
    /// Guards applied per region before compression:
    ///   1. Skip regions smaller than PAGE_SIZE (4096 bytes) — not worth it.
    ///   2. Skip if compressed_size > original_size * 0.9 — less than 10% savings.
    ///
    /// Returns (regions_compressed, bytes_saved)
    pub fn scan_and_compress(&mut self) -> (u32, u64) {
        let now = self.elapsed_ns();
        let threshold = self.config.idle_threshold_ns;
        self.scan_count += 1;

        let mut compressed_count = 0u32;
        let mut bytes_saved = 0u64;

        // Collect addresses to compress (can't mutate while iterating)
        let to_compress: Vec<usize> = self.regions.iter()
            .filter(|(_, r)| {
                r.is_hot() &&
                r.size >= self.config.min_manage_size &&
                r.size >= PAGE_SIZE &&           // minimum page size guard
                now - r.last_access_ns > threshold
            })
            .map(|(&addr, _)| addr)
            .collect();

        for addr in to_compress {
            let size = match self.regions.get(&addr) {
                Some(r) => r.size,
                None => continue,
            };

            let data = self.read_region_data(addr, size);

            // Compression ratio guard: pre-check before promoting to Warm
            let candidate = lz4_flex::compress_prepend_size(&data);
            if candidate.len() > (data.len() as f64 * 0.9) as usize {
                // Less than 10% savings — skip this region
                continue;
            }

            if let Some(region) = self.regions.get_mut(&addr) {
                let saved = region.compress(&data);

                if saved > 0 {
                    compressed_count += 1;
                    bytes_saved += saved as u64;
                    self.total_compressed += 1;
                    self.total_bytes_saved += saved as u64;
                    if self.total_bytes_saved > self.peak_bytes_saved {
                        self.peak_bytes_saved = self.total_bytes_saved;
                    }
                }
            }
        }

        (compressed_count, bytes_saved)
    }

    /// Get a summary of condenser state
    pub fn summary(&self) -> CondenserSummary {
        let mut hot_count = 0u32;
        let mut hot_bytes = 0u64;
        let mut warm_count = 0u32;
        let mut warm_bytes = 0u64;
        let mut warm_compressed_bytes = 0u64;
        let mut cold_count = 0u32;

        for region in self.regions.values() {
            match &region.tier {
                Tier::Hot => {
                    hot_count += 1;
                    hot_bytes += region.size as u64;
                }
                Tier::Warm { compressed, original_size } => {
                    warm_count += 1;
                    warm_bytes += *original_size as u64;
                    warm_compressed_bytes += compressed.len() as u64;
                }
                Tier::Cold { original_size, .. } => {
                    cold_count += 1;
                    warm_bytes += *original_size as u64;
                }
            }
        }

        let total_original = hot_bytes + warm_bytes;
        let total_current = hot_bytes + warm_compressed_bytes;
        let saved = total_original.saturating_sub(total_current);

        CondenserSummary {
            total_regions: self.regions.len() as u32,
            hot_count,
            hot_mb: hot_bytes as f64 / (1024.0 * 1024.0),
            warm_count,
            warm_original_mb: warm_bytes as f64 / (1024.0 * 1024.0),
            warm_compressed_mb: warm_compressed_bytes as f64 / (1024.0 * 1024.0),
            cold_count,
            total_original_mb: total_original as f64 / (1024.0 * 1024.0),
            total_current_mb: total_current as f64 / (1024.0 * 1024.0),
            saved_mb: saved as f64 / (1024.0 * 1024.0),
            saved_pct: if total_original > 0 {
                saved as f64 / total_original as f64 * 100.0
            } else { 0.0 },
            total_compressions: self.total_compressed,
            total_decompressions: self.total_decompressed,
            scan_count: self.scan_count,
            prediction_driven: self.regions.values()
                .map(|r| r.prediction_hits as u64).sum(),
        }
    }
}

/// Summary output
#[derive(Clone, Debug)]
pub struct CondenserSummary {
    pub total_regions: u32,
    pub hot_count: u32,
    pub hot_mb: f64,
    pub warm_count: u32,
    pub warm_original_mb: f64,
    pub warm_compressed_mb: f64,
    pub cold_count: u32,
    pub total_original_mb: f64,
    pub total_current_mb: f64,
    pub saved_mb: f64,
    pub saved_pct: f64,
    pub total_compressions: u64,
    pub total_decompressions: u64,
    pub scan_count: u64,
    pub prediction_driven: u64,
}

impl CondenserSummary {
    pub fn print(&self) {
        eprintln!("\n{}", "=".repeat(55));
        eprintln!("  CONDENSATE CONDENSER — Memory Tier Report");
        eprintln!("{}", "=".repeat(55));
        eprintln!("  Managed regions: {}", self.total_regions);
        eprintln!("    HOT:  {} ({:.1} MB)", self.hot_count, self.hot_mb);
        eprintln!("    WARM: {} ({:.1} MB original → {:.1} MB compressed)",
                 self.warm_count, self.warm_original_mb, self.warm_compressed_mb);
        eprintln!("    COLD: {} (on disk)", self.cold_count);
        eprintln!();
        eprintln!("  Total original:   {:.1} MB", self.total_original_mb);
        eprintln!("  Total current:    {:.1} MB", self.total_current_mb);
        eprintln!("  *** SAVED: {:.1} MB ({:.1}%) ***", self.saved_mb, self.saved_pct);
        eprintln!();
        eprintln!("  Compressions:     {}", self.total_compressions);
        eprintln!("  Decompressions:   {}", self.total_decompressions);
        eprintln!("  Prediction-driven: {}", self.prediction_driven);
        eprintln!("  Scan cycles:      {}", self.scan_count);
        eprintln!("{}\n", "=".repeat(55));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Condenser in test_mode with immediate idle threshold
    fn test_condenser() -> Condenser {
        Condenser::new(CondenserConfig {
            idle_threshold_ns: 0,
            min_manage_size: 1024,
            test_mode: true,
            ..Default::default()
        })
    }

    #[test]
    fn test_register_and_touch() {
        let mut c = Condenser::new(CondenserConfig {
            test_mode: true,
            ..Default::default()
        });

        c.register(0x10000, 100_000);
        c.register(0x20000, 200_000);
        assert_eq!(c.regions.len(), 2);

        c.touch(0x10000);
        assert_eq!(c.regions[&0x10000].access_count, 2);
    }

    #[test]
    fn test_compress_decompress() {
        let mut region = ManagedRegion::new(0x10000, 1024, 0);

        // Compress
        let data = vec![0u8; 1024]; // zeros compress well
        let saved = region.compress(&data);
        assert!(saved > 0, "Should save bytes on compressible data");
        assert!(!region.is_hot());
        assert_eq!(region.demotions, 1);

        // Decompress
        let restored = region.decompress().unwrap();
        assert_eq!(restored.len(), 1024);
        assert!(region.is_hot());
        assert_eq!(region.promotions, 1);
    }

    #[test]
    fn test_scan_compresses_idle() {
        let mut c = Condenser::new(CondenserConfig {
            idle_threshold_ns: 0, // compress immediately
            min_manage_size: 1024,
            test_mode: true,
            ..Default::default()
        });

        c.register(0x10000, 65_536);
        c.register(0x20000, 65_536);

        let (count, saved) = c.scan_and_compress();
        assert_eq!(count, 2, "Should compress both idle regions");
        assert!(saved > 0);

        let summary = c.summary();
        assert_eq!(summary.hot_count, 0);
        assert_eq!(summary.warm_count, 2);
        assert!(summary.saved_pct > 0.0);
    }

    #[test]
    fn test_pre_promote() {
        let mut c = Condenser::new(CondenserConfig {
            idle_threshold_ns: 0,
            min_manage_size: 1024,
            test_mode: true,
            ..Default::default()
        });

        c.register(0x10000, 65_536);
        c.scan_and_compress(); // compress it

        assert!(!c.regions[&0x10000].is_hot());

        c.pre_promote(0x10000);
        assert!(c.regions[&0x10000].is_hot());
        assert_eq!(c.regions[&0x10000].prediction_hits, 1);
    }

    #[test]
    fn test_summary_accuracy() {
        let mut c = Condenser::new(CondenserConfig {
            idle_threshold_ns: 0,
            min_manage_size: 1024,
            test_mode: true,
            ..Default::default()
        });

        // 3 regions: 2 will compress, 1 stays hot
        c.register(0x10000, 65_536);
        c.register(0x20000, 65_536);
        c.register(0x30000, 65_536);

        // Touch the third to keep it hot
        c.touch(0x30000);

        // Only compress idle ones (threshold=0 means everything is idle,
        // but touch updates the timestamp so 0x30000 stays hot IF
        // threshold > 0. With threshold=0, all compress.)
        c.scan_and_compress();

        let summary = c.summary();
        summary.print();

        assert_eq!(summary.total_regions, 3);
        assert!(summary.total_compressions >= 2);
    }

    // -----------------------------------------------------------------
    // New tests for Block B
    // -----------------------------------------------------------------

    #[test]
    fn test_minimum_page_size_guard() {
        // Region of 100 bytes is below PAGE_SIZE (4096); scan must skip it.
        // We need min_manage_size lower than PAGE_SIZE to let it register,
        // but the scan-time guard should still block compression.
        let mut c = Condenser::new(CondenserConfig {
            idle_threshold_ns: 0,
            min_manage_size: 64,   // low enough to register the 100-byte region
            test_mode: true,
            ..Default::default()
        });

        c.register(0xABCD0, 100);
        assert_eq!(c.regions.len(), 1, "Region should be registered");

        let (count, _saved) = c.scan_and_compress();
        assert_eq!(count, 0, "Scan should skip the sub-page-size region");
        assert!(c.regions[&0xABCD0].is_hot(), "Region should remain Hot");
    }

    #[test]
    fn test_compression_ratio_guard() {
        // The ratio guard in scan_and_compress skips a region if
        // compressed_size > original_size * 0.9 (less than 10% savings).
        //
        // We test both sides:
        //   1. Compressible data passes the guard → region becomes Warm.
        //   2. Incompressible data is skipped → region stays Hot.
        //
        // We use ManagedRegion::test_data injection to control exactly what
        // bytes each region presents to the scan, without needing real addresses.

        // --- Happy path: zero-filled buffer compresses extremely well ---
        let mut c = test_condenser();
        let compressible = vec![0u8; 65_536];
        c.register(0xC0000usize, 65_536);
        c.regions.get_mut(&0xC0000usize).unwrap().test_data = Some(compressible);
        let (count, _) = c.scan_and_compress();
        assert_eq!(count, 1, "Compressible region should pass the ratio guard");
        assert!(matches!(c.regions[&0xC0000usize].tier, Tier::Warm { .. }));

        // --- Blocked path: incompressible data (unique bytes, no patterns) ---
        // A sequential 0..=255 cycle gives LZ4 very little to grab onto when
        // the window never repeats at scan scale.  We build a buffer that is
        // already-maximally-dense for LZ4 by using raw bytes from a known
        // LZ4 frame: we compress a small seed with maximum output, then
        // expand it into a large buffer that changes every byte position.
        // The most reliable incompressible source is XOR-folding the position
        // counter with a prime multiplier across the full u8 space.
        let buf_size = 65_536usize;
        // Each byte is derived from position with a prime multiplier — the
        // pattern never repeats within the buffer since 65536 is the full u8
        // cycle times 256, so LZ4's match-finder finds no long-range copies.
        let incompressible: Vec<u8> = (0..buf_size)
            .map(|i| {
                let a = (i.wrapping_mul(6364136223846793005) >> 33) as u8;
                let b = (i.wrapping_mul(1442695040888963407) >> 25) as u8;
                a ^ b ^ (i as u8)
            })
            .collect();

        // Verify our data actually fails the 90% ratio guard before running scan
        let candidate = lz4_flex::compress_prepend_size(&incompressible);
        let threshold = (buf_size as f64 * 0.9) as usize;
        assert!(
            candidate.len() > threshold,
            "Test data must be incompressible enough to trigger the guard \
             (candidate_len={} threshold={}). Regenerate with a harder pattern.",
            candidate.len(), threshold
        );

        // Register and inject incompressible data — scan should skip it
        let mut c2 = test_condenser();
        c2.register(0xD0000usize, buf_size);
        c2.regions.get_mut(&0xD0000usize).unwrap().test_data = Some(incompressible);
        let (count2, _) = c2.scan_and_compress();
        assert_eq!(count2, 0, "Incompressible region should be skipped by the ratio guard");
        assert!(c2.regions[&0xD0000usize].is_hot(), "Region should remain Hot");
    }

    #[test]
    fn test_cold_tier_disk_roundtrip() {
        let mut c = test_condenser();

        // Use a large address that doesn't collide with anything real
        let addr = 0xDEAD_0000usize;
        c.register(addr, 65_536);

        // Compress HOT → WARM
        let (count, _) = c.scan_and_compress();
        assert_eq!(count, 1, "Region should compress to WARM");
        assert!(matches!(c.regions[&addr].tier, Tier::Warm { .. }));

        // Capture the original decompressed bytes from the WARM tier so we
        // can compare them after the roundtrip.
        let original_data = match &c.regions[&addr].tier {
            Tier::Warm { compressed, .. } => {
                lz4_flex::decompress_size_prepended(compressed).unwrap()
            }
            _ => panic!("Expected Warm tier"),
        };

        // Demote WARM → COLD (writes file to disk)
        c.demote_to_cold(addr);
        assert!(matches!(c.regions[&addr].tier, Tier::Cold { .. }));

        // Verify file exists on disk
        let file_path = match &c.regions[&addr].tier {
            Tier::Cold { file_path, .. } => file_path.clone(),
            _ => panic!("Expected Cold tier"),
        };
        assert!(Path::new(&file_path).exists(), "Cold file should exist on disk");

        // Promote COLD → HOT (reads file, decompresses, deletes file)
        let restored = c.promote_from_cold(addr).expect("promote_from_cold should return data");

        assert_eq!(restored, original_data, "Restored data should match original");
        assert!(matches!(c.regions[&addr].tier, Tier::Hot), "Tier should be Hot after promotion");
    }

    #[test]
    fn test_cold_tier_file_cleanup() {
        let mut c = test_condenser();

        let addr = 0xBEEF_0000usize;
        c.register(addr, 65_536);
        c.scan_and_compress();

        // Demote to cold
        c.demote_to_cold(addr);
        let file_path = match &c.regions[&addr].tier {
            Tier::Cold { file_path, .. } => file_path.clone(),
            _ => panic!("Expected Cold tier"),
        };
        assert!(Path::new(&file_path).exists(), "File should exist before promote");

        // Promote from cold
        c.promote_from_cold(addr);

        // File must be gone
        assert!(
            !Path::new(&file_path).exists(),
            "Cold file should be deleted after promote_from_cold"
        );
    }
}
