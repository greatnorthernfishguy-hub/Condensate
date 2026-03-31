//! System-Level Membrane — LD_PRELOAD memory allocation interceptor
//!
//! Hooks malloc/free/mmap/munmap to track memory allocation patterns
//! at the C level. Works for ANY process, not just Python.
//!
//! Usage:
//!   LD_PRELOAD=libcondensate_membrane.so ./any_program
//!
//! The membrane records:
//!   - Allocation events: address, size, timestamp
//!   - Free events: address, timestamp
//!   - Access frequency: which allocations are touched and when
//!   - Size distribution: what sizes dominate
//!
//! This data feeds the AccessGraph for pattern discovery.

use libc::{c_void, size_t};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::Instant;

use crate::pipeline::{Pipeline, PipelineConfig};

/// Global state for the membrane
static INITIALIZED: AtomicBool = AtomicBool::new(false);

// Thread-local re-entrancy guard since our hooks call malloc internally
thread_local! {
    static REENTRANT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A tracked memory allocation
#[derive(Clone, Debug)]
pub struct Allocation {
    pub address: usize,
    pub size: usize,
    pub alloc_time_ns: u64,
    pub last_access_ns: u64,
    pub access_count: u32,
}

/// Size bucket for allocation pattern analysis
#[derive(Clone, Debug, Default)]
pub struct SizeBucket {
    pub label: &'static str,
    pub min_bytes: usize,
    pub max_bytes: usize,
    pub count: u64,
    pub total_bytes: u64,
    pub freed_count: u64,
}

/// The membrane's recorded state
pub struct MembraneState {
    /// Start time for relative timestamps
    start: Instant,
    /// Active allocations: address → Allocation
    active: HashMap<usize, Allocation>,
    /// Size distribution buckets
    buckets: Vec<SizeBucket>,
    /// Total allocated bytes (current)
    total_allocated: u64,
    /// Peak allocated bytes
    peak_allocated: u64,
    /// Total allocation events
    total_alloc_events: u64,
    /// Total free events
    total_free_events: u64,
    /// Sampling rate: record 1 in N allocations (reduces overhead)
    sample_rate: u32,
    /// Sample counter
    sample_counter: u32,
    /// Minimum allocation size to track (skip tiny allocs)
    min_track_size: usize,
}

impl MembraneState {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            active: HashMap::with_capacity(10_000),
            buckets: vec![
                SizeBucket { label: "tiny",   min_bytes: 0,          max_bytes: 64,         ..Default::default() },
                SizeBucket { label: "small",  min_bytes: 64,         max_bytes: 1_024,      ..Default::default() },
                SizeBucket { label: "medium", min_bytes: 1_024,      max_bytes: 64_000,     ..Default::default() },
                SizeBucket { label: "large",  min_bytes: 64_000,     max_bytes: 1_000_000,  ..Default::default() },
                SizeBucket { label: "huge",   min_bytes: 1_000_000,  max_bytes: 64_000_000, ..Default::default() },
                SizeBucket { label: "massive",min_bytes: 64_000_000, max_bytes: usize::MAX, ..Default::default() },
            ],
            total_allocated: 0,
            peak_allocated: 0,
            total_alloc_events: 0,
            total_free_events: 0,
            sample_rate: 100,  // Track 1 in 100 allocs by default
            sample_counter: 0,
            min_track_size: 4096, // Skip allocs under 4KB
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    pub fn record_alloc(&mut self, address: usize, size: usize) {
        self.total_alloc_events += 1;

        // Bucket the size
        for bucket in &mut self.buckets {
            if size >= bucket.min_bytes && size < bucket.max_bytes {
                bucket.count += 1;
                bucket.total_bytes += size as u64;
                break;
            }
        }

        // Skip tiny allocations for detailed tracking
        if size < self.min_track_size {
            return;
        }

        // Sampling: only track 1 in N large allocations
        self.sample_counter += 1;
        if self.sample_counter % self.sample_rate != 0 {
            // Still track total bytes even if not recording the allocation
            self.total_allocated += size as u64;
            if self.total_allocated > self.peak_allocated {
                self.peak_allocated = self.total_allocated;
            }
            return;
        }

        let ts = self.elapsed_ns();

        self.active.insert(address, Allocation {
            address,
            size,
            alloc_time_ns: ts,
            last_access_ns: ts,
            access_count: 1,
        });

        self.total_allocated += size as u64;
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }
    }

    pub fn record_free(&mut self, address: usize) {
        self.total_free_events += 1;

        if let Some(alloc) = self.active.remove(&address) {
            self.total_allocated = self.total_allocated.saturating_sub(alloc.size as u64);

            // Record in bucket freed count
            for bucket in &mut self.buckets {
                if alloc.size >= bucket.min_bytes && alloc.size < bucket.max_bytes {
                    bucket.freed_count += 1;
                    break;
                }
            }
        }
    }

    /// Get a summary of current state
    pub fn summary(&self) -> MembraneSummary {
        let mut hot_count = 0u64;
        let mut hot_bytes = 0u64;
        let mut cold_count = 0u64;
        let mut cold_bytes = 0u64;

        let now = self.elapsed_ns();
        let cold_threshold_ns = 5_000_000_000; // 5 seconds idle = cold

        for alloc in self.active.values() {
            let idle = now - alloc.last_access_ns;
            if idle > cold_threshold_ns {
                cold_count += 1;
                cold_bytes += alloc.size as u64;
            } else {
                hot_count += 1;
                hot_bytes += alloc.size as u64;
            }
        }

        MembraneSummary {
            tracked_allocations: self.active.len() as u64,
            total_alloc_events: self.total_alloc_events,
            total_free_events: self.total_free_events,
            current_allocated_mb: self.total_allocated as f64 / (1024.0 * 1024.0),
            peak_allocated_mb: self.peak_allocated as f64 / (1024.0 * 1024.0),
            hot_count,
            hot_mb: hot_bytes as f64 / (1024.0 * 1024.0),
            cold_count,
            cold_mb: cold_bytes as f64 / (1024.0 * 1024.0),
            buckets: self.buckets.clone(),
        }
    }
}

/// Summary output for display/logging
#[derive(Clone, Debug)]
pub struct MembraneSummary {
    pub tracked_allocations: u64,
    pub total_alloc_events: u64,
    pub total_free_events: u64,
    pub current_allocated_mb: f64,
    pub peak_allocated_mb: f64,
    pub hot_count: u64,
    pub hot_mb: f64,
    pub cold_count: u64,
    pub cold_mb: f64,
    pub buckets: Vec<SizeBucket>,
}

impl MembraneSummary {
    pub fn print(&self) {
        eprintln!("\n{}", "=".repeat(55));
        eprintln!("  CONDENSATE MEMBRANE — System Memory Profile");
        eprintln!("{}", "=".repeat(55));
        eprintln!("  Total alloc events:  {}", self.total_alloc_events);
        eprintln!("  Total free events:   {}", self.total_free_events);
        eprintln!("  Tracked allocations: {}", self.tracked_allocations);
        eprintln!("  Current allocated:   {:.1} MB", self.current_allocated_mb);
        eprintln!("  Peak allocated:      {:.1} MB", self.peak_allocated_mb);
        eprintln!();
        eprintln!("  HOT (accessed <5s ago): {} allocs, {:.1} MB", self.hot_count, self.hot_mb);
        eprintln!("  COLD (idle >5s):        {} allocs, {:.1} MB", self.cold_count, self.cold_mb);

        if self.cold_mb > 0.0 {
            let total = self.hot_mb + self.cold_mb;
            let pct = self.cold_mb / total * 100.0;
            eprintln!();
            eprintln!("  *** CONDENSATION POTENTIAL: {:.1}% ({:.1} MB cold) ***", pct, self.cold_mb);
        }

        eprintln!();
        eprintln!("  Size distribution:");
        eprintln!("  {:>10}  {:>10}  {:>12}  {:>8}", "Bucket", "Count", "Total MB", "Freed");
        eprintln!("  {:>10}  {:>10}  {:>12}  {:>8}", "------", "-----", "--------", "-----");
        for b in &self.buckets {
            if b.count > 0 {
                eprintln!("  {:>10}  {:>10}  {:>12.1}  {:>8}",
                         b.label, b.count, b.total_bytes as f64 / (1024.0 * 1024.0), b.freed_count);
            }
        }
        eprintln!("{}\n", "=".repeat(55));
    }
}

/// Global membrane state behind a mutex
static MEMBRANE: std::sync::LazyLock<Mutex<MembraneState>> =
    std::sync::LazyLock::new(|| Mutex::new(MembraneState::new()));

/// Global pipeline — the living loop
static PIPELINE: std::sync::LazyLock<Mutex<Pipeline>> =
    std::sync::LazyLock::new(|| Mutex::new(Pipeline::new(PipelineConfig::default())));

/// Scan counter — run condenser scan every N allocs
static SCAN_COUNTER: AtomicU64 = AtomicU64::new(0);
const SCAN_INTERVAL: u64 = 1_000; // scan every 1,000 allocs

// --- LD_PRELOAD hook functions ---

/// Get the original malloc function
unsafe fn real_malloc(size: size_t) -> *mut c_void {
    type MallocFn = unsafe extern "C" fn(size_t) -> *mut c_void;
    unsafe {
        let sym = libc::dlsym(libc::RTLD_NEXT, c"malloc".as_ptr());
        let func: MallocFn = std::mem::transmute(sym);
        func(size)
    }
}

/// Get the original free function
unsafe fn real_free(ptr: *mut c_void) {
    type FreeFn = unsafe extern "C" fn(*mut c_void);
    unsafe {
        let sym = libc::dlsym(libc::RTLD_NEXT, c"free".as_ptr());
        let func: FreeFn = std::mem::transmute(sym);
        func(ptr)
    }
}

/// Hooked malloc — records the allocation, feeds the pipeline, calls real malloc
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    let ptr = unsafe { real_malloc(size) };

    REENTRANT.with(|r| {
        if r.get() {
            return;
        }
        r.set(true);

        // Feed membrane state (observation/stats)
        if let Ok(mut state) = MEMBRANE.try_lock() {
            state.record_alloc(ptr as usize, size);
        }

        // Feed the living pipeline (learn → predict → act)
        if let Ok(mut pipeline) = PIPELINE.try_lock() {
            pipeline.process_alloc(ptr as usize, size);
        }

        // Periodic compression scan
        let count = SCAN_COUNTER.fetch_add(1, Ordering::Relaxed);
        if count > 0 && count % SCAN_INTERVAL == 0 {
            if let Ok(mut pipeline) = PIPELINE.try_lock() {
                pipeline.scan();
            }
        }

        r.set(false);
    });

    ptr
}

/// Hooked free — records the deallocation, updates pipeline, calls real free
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    REENTRANT.with(|r| {
        if r.get() {
            return;
        }
        r.set(true);

        if let Ok(mut state) = MEMBRANE.try_lock() {
            state.record_free(ptr as usize);
        }

        if let Ok(mut pipeline) = PIPELINE.try_lock() {
            pipeline.process_free(ptr as usize);
        }

        r.set(false);
    });

    unsafe { real_free(ptr) }
}

/// Print full pipeline summary on process exit
#[unsafe(no_mangle)]
pub extern "C" fn condensate_summary() {
    // Membrane stats
    if let Ok(state) = MEMBRANE.lock() {
        state.summary().print();
    }
    // Pipeline stats (the living loop)
    if let Ok(pipeline) = PIPELINE.lock() {
        pipeline.summary().print();
    }
}

/// Called when the shared library is loaded (constructor)
#[used]
#[unsafe(link_section = ".init_array")]
static INIT: extern "C" fn() = {
    extern "C" fn init() {
        INITIALIZED.store(true, Ordering::SeqCst);
        eprintln!("[condensate] Living pipeline active — membrane → graph → predictor → condenser");

        unsafe { libc::atexit(condensate_summary) };
    }
    init
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membrane_state() {
        let mut state = MembraneState::new();
        state.sample_rate = 1;      // track every alloc for testing
        state.min_track_size = 0;   // track all sizes

        state.record_alloc(0x1000, 8192);
        state.record_alloc(0x2000, 65536);
        state.record_alloc(0x3000, 1_000_000);

        assert_eq!(state.total_alloc_events, 3);

        let summary = state.summary();
        assert!(summary.current_allocated_mb > 0.0);
        assert_eq!(summary.tracked_allocations, 3);
    }

    #[test]
    fn test_free_tracking() {
        let mut state = MembraneState::new();
        state.sample_rate = 1;
        state.min_track_size = 0;

        state.record_alloc(0x1000, 100_000);
        state.record_alloc(0x2000, 200_000);
        assert_eq!(state.active.len(), 2);

        state.record_free(0x1000);
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.total_free_events, 1);
    }

    #[test]
    fn test_size_buckets() {
        let mut state = MembraneState::new();

        state.record_alloc(0x1000, 32);        // tiny
        state.record_alloc(0x2000, 512);       // small
        state.record_alloc(0x3000, 8192);      // medium
        state.record_alloc(0x4000, 100_000);   // large
        state.record_alloc(0x5000, 2_000_000); // huge

        let summary = state.summary();
        // Check that buckets have counts
        let total_bucket_count: u64 = summary.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total_bucket_count, 5);
    }
}
