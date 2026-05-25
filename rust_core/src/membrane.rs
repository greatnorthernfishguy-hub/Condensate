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
//!
//! ---- Changelog ----
//! [2026-05-25] CC — Fix: PIPELINE now uses test_mode=true in LD_PRELOAD context
//!   What: condenser.scan_and_compress() was reading from and writing to live
//!         Python/uvicorn heap addresses, causing use-after-free heap corruption
//!         and SIGSEGV in glibc's freelist traversal (~35-50s after TID start).
//!   Why:  The condenser tracks addresses it does not own. Freed allocations can
//!         be re-used by Python; condenser's copy_nonoverlapping overwrote glibc
//!         chunk metadata, crashing in libc+0x17934d (svcfd_create region).
//!   How:  PipelineConfig { test_mode: true } on global PIPELINE — condenser
//!         learns allocation patterns but never dereferences observed addresses.
//! -------------------

use libc::{c_void, size_t};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::Instant;
use std::fs;
use std::io::Write;

use crate::pipeline::{Pipeline, PipelineConfig};

/// Operating mode for the membrane
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MembraneMode {
    /// Record observations but don't feed the condenser
    ObserveOnly,
    /// Full condensation — observation + active pipeline feeding
    Active,
}

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

    // --- Observe-only mode ---
    /// Current operating mode (starts ObserveOnly)
    pub mode: MembraneMode,

    // --- Process identification ---
    /// Name of this process (from /proc/self/exe)
    pub process_name: String,
    /// PID of this process
    pub process_id: u32,

    // --- Confidence gating ---
    /// Number of observation cycles recorded
    pub observation_cycles: u64,
    /// Minimum cycles before mode can become Active
    pub min_observation_cycles: u64,

    // --- Self-interference detection ---
    /// Timestamp (ns) when we transitioned from ObserveOnly → Active
    pub engagement_timestamp_ns: Option<u64>,

    // --- Canary system ---
    /// Path to the active canary file (if armed)
    pub canary_file: Option<String>,
    /// How long (seconds) before a canary is considered expired
    pub canary_timeout_s: u64,

    // --- Quiet mode ---
    /// Suppress all stdout/stderr output when true
    pub quiet: bool,
}

impl MembraneState {
    pub fn new() -> Self {
        // Resolve process name from /proc/self/exe; fallback to "unknown"
        let process_name = std::fs::read_link("/proc/self/exe")
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "unknown".to_string());

        let process_id = std::process::id();

        // Quiet mode: suppress output when CONDENSATE_QUIET is set
        let quiet = std::env::var("CONDENSATE_QUIET").is_ok();

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
            mode: MembraneMode::ObserveOnly,
            process_name,
            process_id,
            observation_cycles: 0,
            min_observation_cycles: 1000,
            engagement_timestamp_ns: None,
            canary_file: None,
            canary_timeout_s: 60,
            quiet,
        }
    }

    // --- Observe-only mode ---

    /// Return the current operating mode
    pub fn mode(&self) -> MembraneMode {
        self.mode
    }

    /// Set the operating mode directly
    pub fn set_mode(&mut self, mode: MembraneMode) {
        self.mode = mode;
    }

    // --- Confidence gating ---

    /// Increment the observation cycle counter
    pub fn record_cycle(&mut self) {
        self.observation_cycles += 1;
    }

    /// True once enough cycles have been observed to trust the data
    pub fn is_confident(&self) -> bool {
        self.observation_cycles >= self.min_observation_cycles
    }

    // --- Self-interference detection ---

    /// Report this process as potentially dangerous; append to the blacklist file
    pub fn report_crash(&self) {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/condensate_blacklist")
        {
            let _ = writeln!(f, "{}", self.process_name);
        }
    }

    /// True if this process's name appears in the blacklist file
    pub fn is_blacklisted(&self) -> bool {
        fs::read_to_string("/tmp/condensate_blacklist")
            .map(|contents| {
                contents.lines().any(|line| line == self.process_name)
            })
            .unwrap_or(false)
    }

    // --- Canary system ---

    /// Arm the canary: write a file with the engagement timestamp and timeout.
    /// Also records engagement_timestamp_ns on the state and transitions to Active.
    pub fn arm_canary(&mut self) {
        let now_ns = self.elapsed_ns();
        self.engagement_timestamp_ns = Some(now_ns);
        self.mode = MembraneMode::Active;

        let path = format!("/tmp/condensate_canary_{}", self.process_id);
        if let Ok(mut f) = fs::File::create(&path) {
            let _ = writeln!(f, "engagement_ns={}", now_ns);
            let _ = writeln!(f, "timeout_s={}", self.canary_timeout_s);
        }
        self.canary_file = Some(path);
    }

    /// Confirm health: delete the canary file
    pub fn confirm_canary(&mut self) {
        if let Some(ref path) = self.canary_file {
            let _ = fs::remove_file(path);
        }
        self.canary_file = None;
    }

    /// True if the canary was armed and has now exceeded its timeout
    pub fn check_canary_expired(&self, now_ns: u64) -> bool {
        match self.engagement_timestamp_ns {
            Some(ts) => {
                let elapsed_s = now_ns.saturating_sub(ts) / 1_000_000_000;
                elapsed_s >= self.canary_timeout_s
            }
            None => false,
        }
    }

    /// Rollback: revert to ObserveOnly and clean up the canary file
    pub fn rollback(&mut self) {
        self.mode = MembraneMode::ObserveOnly;
        self.confirm_canary(); // deletes the canary file if present
    }

    pub fn elapsed_ns(&self) -> u64 {
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

// --- LD_PRELOAD hook functions ---
// Only compiled when building the standalone preload .so.
// NOT active during tests or when used as a Python module.
#[cfg(feature = "preload")]
mod preload_hooks {
use super::*;
use std::sync::atomic::AtomicUsize;

/// Event type tag for the ring buffer
const EVENT_ALLOC: u8 = 1;
const EVENT_FREE: u8 = 2;
const EVENT_EMPTY: u8 = 0;

/// Lock-free ring buffer capacity — must be power of 2.
/// 8K slots × 8 bytes = 64KB. Lives in .bss, zero heap allocation.
const RING_SIZE: usize = 8192;

/// Compact ring event — 8 bytes, packed into a single AtomicU64.
/// Layout: [tag:8][size_kb:16][address_low:32][_pad:8]
/// No heap allocation, no struct, no AtomicU8 issues.
/// The entire ring is a static array of AtomicU64 — lives in .bss.
static RING: [AtomicU64; RING_SIZE] = {
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; RING_SIZE]
};

/// Pack an event into a u64: tag in low byte, size_kb in bytes 1-2, address_low in bytes 3-6
#[inline(always)]
fn pack_event(tag: u8, address: usize, size: usize) -> u64 {
    let size_kb = (size / 1024).min(0xFFFF) as u64;
    let addr_low = (address as u32) as u64;
    (tag as u64) | (size_kb << 8) | (addr_low << 24)
}

/// Unpack: returns (tag, address_low, size_kb)
#[inline(always)]
fn unpack_event(packed: u64) -> (u8, usize, usize) {
    let tag = (packed & 0xFF) as u8;
    let size_kb = ((packed >> 8) & 0xFFFF) as usize;
    let addr_low = ((packed >> 24) & 0xFFFFFFFF) as usize;
    (tag, addr_low, size_kb * 1024)
}

/// Write cursor — atomically incremented by malloc/free hooks
static WRITE_POS: AtomicUsize = AtomicUsize::new(0);

/// Global membrane state — only accessed by drain thread
static MEMBRANE: std::sync::LazyLock<Mutex<MembraneState>> =
    std::sync::LazyLock::new(|| Mutex::new(MembraneState::new()));

/// Global pipeline — only accessed by drain thread
/// test_mode: true — never read from or write to live process memory.
/// The condenser tracks alloc patterns (graph, predictor) but never
/// dereferences the observed addresses. This is mandatory in LD_PRELOAD
/// context where we do not own the memory we observe.
static PIPELINE: std::sync::LazyLock<Mutex<Pipeline>> =
    std::sync::LazyLock::new(|| Mutex::new(Pipeline::new(PipelineConfig {
        test_mode: true,
        ..PipelineConfig::default()
    })));

/// Drain thread handle
static DRAIN_STARTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Membrane engagement state — three phases:
/// 0 = DORMANT: pure passthrough, don't even record. Process just started.
/// 1 = OBSERVING: push events to ring buffer, drain thread processes them.
/// 2 = ACTIVE: full condensation (future — currently same as OBSERVING).
const PHASE_DORMANT: u8 = 0;
const PHASE_OBSERVING: u8 = 1;
#[allow(dead_code)]
const PHASE_ACTIVE: u8 = 2;

static ENGAGEMENT_PHASE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(PHASE_DORMANT);

/// Grace period before engaging — let the process finish initializing.
/// Default 10 seconds. Solves V8/Node.js and xrdp SEGV during init.
const GRACE_PERIOD_NS: u64 = 10_000_000_000;

/// Timestamp when the membrane loaded (set in init)
static LOAD_TIME_NS: AtomicU64 = AtomicU64::new(0);

/// Cached real malloc/free function pointers — resolved ONCE at init,
/// not via dlsym on every call. Avoids dlsym during early process init.
static REAL_MALLOC: AtomicU64 = AtomicU64::new(0);
static REAL_FREE: AtomicU64 = AtomicU64::new(0);

// Early-alloc buffer — serves malloc calls during the bootstrap window before
// dlsym has resolved REAL_MALLOC. dlsym calls malloc internally; without this,
// real_malloc(ptr==0) calls dlsym, which calls malloc, which calls real_malloc(0),
// which calls dlsym... → infinite recursion → stack overflow → SIGSEGV.
// 1MB in .bss: zero-initialized, writable, no heap. Far more than dlsym needs.
static mut EARLY_BUF: [u8; 1024 * 1024] = [0u8; 1024 * 1024];
static EARLY_POS: AtomicUsize = AtomicUsize::new(0);
const EARLY_BUF_LEN: usize = 1024 * 1024;

unsafe fn early_alloc(size: usize) -> *mut c_void {
    if size == 0 { return 1usize as *mut c_void; }
    let aligned = (size + 15) & !15;
    let pos = EARLY_POS.fetch_add(aligned, Ordering::Relaxed);
    if pos + aligned <= EARLY_BUF_LEN {
        unsafe { EARLY_BUF.as_mut_ptr().add(pos) as *mut c_void }
    } else {
        std::ptr::null_mut()
    }
}

fn is_early_ptr(ptr: *mut c_void) -> bool {
    if ptr.is_null() || ptr as usize == 1 { return false; }
    unsafe {
        let start = EARLY_BUF.as_ptr() as usize;
        let p = ptr as usize;
        p >= start && p < start + EARLY_BUF_LEN
    }
}


/// Scan counter — run condenser scan every N allocs
static SCAN_COUNTER: AtomicU64 = AtomicU64::new(0);
const SCAN_INTERVAL: u64 = 1_000;

/// Start the background drain thread (called once when transitioning to OBSERVING)
fn start_drain_thread() {
    if DRAIN_STARTED.swap(true, Ordering::SeqCst) {
        return; // already started
    }
    // Now that we're actually observing, register the exit summary
    unsafe { libc::atexit(condensate_summary) };

    std::thread::Builder::new()
        .name("condensate-drain".to_string())
        .spawn(|| {
            let mut read_pos: usize = 0;
            loop {
                let mut drained = 0;
                // Drain up to 1024 events per batch
                for _ in 0..1024 {
                    let slot = &RING[read_pos & (RING_SIZE - 1)];
                    let packed = slot.load(Ordering::Acquire);
                    if packed == 0 {
                        break; // empty slot
                    }

                    let (tag, address, size) = unpack_event(packed);

                    match tag {
                        EVENT_ALLOC => {
                            if let Ok(mut state) = MEMBRANE.try_lock() {
                                state.record_alloc(address, size);
                            }
                            if let Ok(mut pipeline) = PIPELINE.try_lock() {
                                pipeline.process_alloc(address, size);
                            }
                            let count = SCAN_COUNTER.fetch_add(1, Ordering::Relaxed);
                            if count > 0 && count % SCAN_INTERVAL == 0 {
                                if let Ok(mut pipeline) = PIPELINE.try_lock() {
                                    pipeline.scan();
                                }
                            }
                        }
                        EVENT_FREE => {
                            if let Ok(mut state) = MEMBRANE.try_lock() {
                                state.record_free(address);
                            }
                            if let Ok(mut pipeline) = PIPELINE.try_lock() {
                                pipeline.process_free(address);
                            }
                        }
                        _ => {}
                    }

                    // Mark slot as consumed (zero = empty)
                    slot.store(0, Ordering::Release);
                    read_pos += 1;
                    drained += 1;
                }

                if drained == 0 {
                    // Nothing to drain — sleep briefly to avoid busy-spin
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        })
        .expect("Failed to spawn condensate drain thread");
}

/// Push an event to the ring buffer — lock-free, ~10ns, zero heap allocation
#[inline(always)]
fn push_event(tag: u8, address: usize, size: usize) {
    let pos = WRITE_POS.fetch_add(1, Ordering::Relaxed);
    let slot = &RING[pos & (RING_SIZE - 1)];

    // If slot isn't empty (non-zero), drain thread is behind — drop this event.
    // Better to lose an event than to stall malloc.
    if slot.load(Ordering::Relaxed) != 0 {
        return;
    }

    // Single atomic store — the packed value IS the fence
    slot.store(pack_event(tag, address, size), Ordering::Release);
}

/// Resolve and cache the real malloc/free function pointers.
/// Called once during init — after this, no more dlsym calls.
unsafe fn cache_real_functions() {
    unsafe {
        let m = libc::dlsym(libc::RTLD_NEXT, c"malloc".as_ptr());
        let f = libc::dlsym(libc::RTLD_NEXT, c"free".as_ptr());
        REAL_MALLOC.store(m as u64, Ordering::Release);
        REAL_FREE.store(f as u64, Ordering::Release);
    }
}

/// Call the real malloc — uses cached pointer, no dlsym
#[inline(always)]
unsafe fn real_malloc(size: size_t) -> *mut c_void {
    type MallocFn = unsafe extern "C" fn(size_t) -> *mut c_void;
    let ptr = REAL_MALLOC.load(Ordering::Relaxed);
    if ptr == 0 {
        // Bootstrap: dlsym hasn't returned REAL_MALLOC yet.
        // Calling dlsym here would recurse infinitely — use static buffer instead.
        return unsafe { early_alloc(size) };
    }
    unsafe {
        let func: MallocFn = std::mem::transmute(ptr);
        func(size)
    }
}

/// Call the real free — uses cached pointer, no dlsym
#[inline(always)]
unsafe fn real_free(ptr: *mut c_void) {
    type FreeFn = unsafe extern "C" fn(*mut c_void);
    let fptr = REAL_FREE.load(Ordering::Relaxed);
    // Early-buffer pointers live in .bss — never pass them to the real allocator.
    if is_early_ptr(ptr) { return; }
    if fptr == 0 { return; }
    unsafe {
        let func: FreeFn = std::mem::transmute(fptr);
        func(ptr)
    }
}

/// Get monotonic time in nanoseconds (no allocation, no syscall overhead)
#[inline(always)]
fn now_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Hooked malloc — born dormant, wakes after grace period.
///
/// DORMANT: pure passthrough. Single atomic load overhead (~1ns).
/// OBSERVING: push to ring buffer. Atomic increment (~10ns).
/// The process doesn't know the difference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    let ptr = unsafe { real_malloc(size) };

    // Fast path: dormant = pure passthrough, ~1ns overhead
    let phase = ENGAGEMENT_PHASE.load(Ordering::Relaxed);
    if phase == PHASE_DORMANT {
        // Check if grace period has elapsed — transition to observing
        let load_time = LOAD_TIME_NS.load(Ordering::Relaxed);
        if load_time > 0 && now_ns() - load_time > GRACE_PERIOD_NS {
            ENGAGEMENT_PHASE.store(PHASE_OBSERVING, Ordering::Release);
            // Start drain thread on first transition
            start_drain_thread();
        }
        return ptr;
    }

    // Observing/active: record the event
    REENTRANT.with(|r| {
        if r.get() {
            return;
        }
        r.set(true);
        push_event(EVENT_ALLOC, ptr as usize, size);
        r.set(false);
    });

    ptr
}

/// Hooked free — same dormant/observing phases as malloc.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let phase = ENGAGEMENT_PHASE.load(Ordering::Relaxed);
    if phase >= PHASE_OBSERVING {
        REENTRANT.with(|r| {
            if r.get() {
                return;
            }
            r.set(true);
            push_event(EVENT_FREE, ptr as usize, 0);
            r.set(false);
        });
    }

    unsafe { real_free(ptr) }
}

/// Print full pipeline summary on process exit — only if process ran long enough
#[unsafe(no_mangle)]
pub extern "C" fn condensate_summary() {
    // Only print for long-lived processes (>5 seconds)
    // Short-lived commands (ls, grep, cat) shouldn't flood stderr
    let (elapsed, quiet) = MEMBRANE.try_lock()
        .map(|s| (s.elapsed_ns(), s.quiet))
        .unwrap_or((0, false));

    if elapsed < 5_000_000_000 {
        return; // process ran < 5 seconds, skip summary
    }

    // Honour quiet mode — suppress all output
    if quiet {
        return;
    }

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

        // Cache real malloc/free pointers — one dlsym each, never again.
        unsafe { cache_real_functions() };

        // Record load time — grace period starts now.
        // The membrane stays DORMANT (pure passthrough) for GRACE_PERIOD_NS.
        // After that, the first malloc transitions to OBSERVING.
        // This lets processes like Node.js/V8 and xrdp finish their
        // initialization before we touch anything.
        LOAD_TIME_NS.store(now_ns(), Ordering::Release);

        // Don't start drain thread yet — it starts when DORMANT → OBSERVING.
        // Don't register atexit yet — only register when we actually observe.
    }
    init
};

} // mod preload_hooks

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

    #[test]
    fn test_observe_only_mode() {
        let state = MembraneState::new();
        assert_eq!(state.mode(), MembraneMode::ObserveOnly);
    }

    #[test]
    fn test_confidence_gating() {
        let mut state = MembraneState::new();
        state.min_observation_cycles = 5;

        // Before enough cycles: not confident
        assert!(!state.is_confident());

        for _ in 0..4 {
            state.record_cycle();
        }
        assert!(!state.is_confident());

        // After reaching min_observation_cycles: confident
        state.record_cycle();
        assert!(state.is_confident());
    }

    #[test]
    fn test_mode_transition() {
        let mut state = MembraneState::new();
        state.min_observation_cycles = 3;

        assert_eq!(state.mode(), MembraneMode::ObserveOnly);

        for _ in 0..3 {
            state.record_cycle();
        }
        assert!(state.is_confident());

        state.set_mode(MembraneMode::Active);
        assert_eq!(state.mode(), MembraneMode::Active);
    }

    #[test]
    fn test_quiet_mode() {
        // Without the env var set, quiet should be false
        std::env::remove_var("CONDENSATE_QUIET");
        let state = MembraneState::new();
        assert!(!state.quiet);

        // With the env var set, quiet should be true
        std::env::set_var("CONDENSATE_QUIET", "1");
        let state_quiet = MembraneState::new();
        assert!(state_quiet.quiet);

        // Clean up
        std::env::remove_var("CONDENSATE_QUIET");
    }

    #[test]
    fn test_canary_arm_and_confirm() {
        let mut state = MembraneState::new();

        // Before arming: no canary file
        assert!(state.canary_file.is_none());

        state.arm_canary();

        // After arming: file should exist on disk
        let path = state.canary_file.clone().expect("canary_file should be set after arm_canary");
        assert!(std::path::Path::new(&path).exists(), "canary file should exist after arm_canary");
        // Mode transitions to Active
        assert_eq!(state.mode(), MembraneMode::Active);
        // engagement timestamp is recorded
        assert!(state.engagement_timestamp_ns.is_some());

        state.confirm_canary();

        // After confirming: file should be gone and canary_file cleared
        assert!(state.canary_file.is_none());
        assert!(!std::path::Path::new(&path).exists(), "canary file should be removed after confirm_canary");
    }

    #[test]
    fn test_canary_expiry() {
        let mut state = MembraneState::new();
        state.canary_timeout_s = 2; // 2-second timeout

        state.arm_canary();

        let armed_ns = state.engagement_timestamp_ns.unwrap();

        // A timestamp just before expiry should not be expired
        let before_expiry_ns = armed_ns + 1_000_000_000; // 1 second later
        assert!(!state.check_canary_expired(before_expiry_ns));

        // A timestamp past the timeout should report expired
        let after_expiry_ns = armed_ns + 3_000_000_000; // 3 seconds later
        assert!(state.check_canary_expired(after_expiry_ns));

        // Clean up the canary file
        state.confirm_canary();
    }
}
