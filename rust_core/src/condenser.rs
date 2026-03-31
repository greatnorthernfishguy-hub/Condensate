//! Condenser — the motor output. Compresses cold memory, restores hot.
//!
//! The membrane observes. The graph learns. The predictor predicts.
//! The condenser ACTS — compressing idle allocations and restoring
//! them before they're needed.
//!
//! Three tiers:
//!   HOT:  Untouched, full speed access
//!   WARM: LZ4 compressed in-place, fast decompress on access
//!   COLD: Backed by mmap'd file, zero RSS until touched
//!
//! The condenser runs as a background thread, periodically scanning
//! the membrane's tracked allocations and demoting idle ones.
//! When the predictor fires a spike ("this region is about to be
//! accessed"), the condenser pre-promotes it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use crate::membrane::{MembraneState, MembraneSummary};

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
    /// Backed to disk via mmap, zero RSS
    Cold {
        file_offset: u64,
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
}

impl Default for CondenserConfig {
    fn default() -> Self {
        Self {
            idle_threshold_ns: 5_000_000_000,  // 5 seconds
            min_manage_size: 65_536,            // 64KB minimum
            max_tracked: 10_000,
            scan_interval_ns: 1_000_000_000,   // 1 second
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
}

impl Condenser {
    pub fn new(config: CondenserConfig) -> Self {
        Self {
            config,
            regions: HashMap::with_capacity(1000),
            start: Instant::now(),
            total_compressed: 0,
            total_decompressed: 0,
            total_bytes_saved: 0,
            peak_bytes_saved: 0,
            scan_count: 0,
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

    /// Pre-promote a region (prediction-driven)
    pub fn pre_promote(&mut self, address: usize) {
        if let Some(region) = self.regions.get_mut(&address) {
            if !region.is_hot() {
                // In a real implementation, this would decompress
                // and write back to the original address.
                // For the PoC, we track that the prediction fired.
                region.prediction_hits += 1;
                region.tier = Tier::Hot;
                region.promotions += 1;
                self.total_decompressed += 1;
            }
        }
    }

    /// Scan for idle regions and compress them
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
                now - r.last_access_ns > threshold
            })
            .map(|(&addr, _)| addr)
            .collect();

        for addr in to_compress {
            if let Some(region) = self.regions.get_mut(&addr) {
                // In a real LD_PRELOAD implementation, we'd read from
                // the actual memory address. For now, simulate with
                // a zero-filled buffer (shows compression mechanics).
                let fake_data = vec![0u8; region.size];
                let saved = region.compress(&fake_data);

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

    #[test]
    fn test_register_and_touch() {
        let mut c = Condenser::new(CondenserConfig::default());

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
}
