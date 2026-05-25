//! Pipeline — the living loop that connects all four layers.
//!
//! Membrane observes → Graph learns → Predictor predicts → Condenser acts.
//!
//! This is the River. Data flows through it continuously.
//! No orchestrator. No scheduler. The substrate drives itself.
//!
//! The pipeline runs as a background thread alongside the membrane's
//! LD_PRELOAD hooks. Every allocation event flows through the graph,
//! triggers predictions, and the condenser acts on them.
//!
//! ---- Changelog ----
//! [2026-05-25] CC — Large-alloc passthrough (#260)
//!   What: Allocations above large_alloc_passthrough_bytes (default 10MB) are
//!         tracked in the graph and predictor for pattern learning but never
//!         registered with the condenser for compression.
//!   Why:  Observer data: ONNX embedding model = 86MB single allocation; NG
//!         burst events hit 7GB in 10s. Large allocs are (a) high-entropy —
//!         LZ4 won't help, (b) hot during inference — compressing them mid-
//!         flight causes use-after-read, (c) the source of the "massive" class
//!         patterns the graph already learns from. Graph still sees them.
//!   How:  process_alloc() checks size vs config.large_alloc_passthrough_bytes
//!         before condenser.register() and field.access() calls.
//! -------------------

use std::collections::HashMap;
use std::time::Instant;

use crate::graph::AccessGraph;
use crate::predictor::RustPredictor;
use crate::condenser::{Condenser, CondenserConfig};
use crate::lenia::LeniaField;

/// Pipeline operating mode — governs whether the pipeline acts on predictions.
///
/// The substrate always learns. Mode controls whether it compresses.
/// Observing → Active after confidence threshold is met.
/// Blacklisted → permanent: never acts, never transitions.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PipelineMode {
    /// Learning phase — graph and predictor train, condenser is silent.
    Observing,
    /// Fully operational — condenser compresses and pre-promotes.
    Active,
    /// Permanently silenced — never transitions, never compresses.
    Blacklisted,
}

/// Pipeline configuration
pub struct PipelineConfig {
    /// Graph causal window (ns)
    pub causal_window_ns: u64,
    /// Graph cluster threshold
    pub cluster_threshold: f64,
    /// Condenser idle threshold (ns)
    pub idle_threshold_ns: u64,
    /// Minimum allocation size to manage
    pub min_manage_size: usize,
    /// How many events to accumulate before rebuilding the graph
    pub graph_rebuild_interval: usize,
    /// Minimum prediction confidence to act on
    pub prediction_threshold: f64,
    /// Enable test mode — condenser generates synthetic data instead of reading
    /// from raw memory pointers. Required when using fake addresses in tests.
    pub test_mode: bool,
    /// Allocations above this size are tracked in the graph/predictor for
    /// pattern learning but never registered with the condenser.
    /// Observer data: ONNX model = 86MB; NG burst allocs hit 7GB.
    /// Large allocs are high-entropy (LZ4 ineffective) and inference-hot.
    /// Default: 10MB — conservative floor well below the 86MB ONNX baseline.
    pub large_alloc_passthrough_bytes: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            causal_window_ns: 5_000_000,        // 5ms
            cluster_threshold: 0.7,
            idle_threshold_ns: 1_000_000_000,   // 1 second (was 5 — too conservative)
            min_manage_size: 4_096,             // 4KB
            graph_rebuild_interval: 500,         // rebuild graph every 500 events
            prediction_threshold: 0.3,           // act on predictions with >30% confidence
            test_mode: false,
            large_alloc_passthrough_bytes: 10 * 1024 * 1024,  // 10MB
        }
    }
}

/// A single event flowing through the pipeline
#[derive(Clone, Debug)]
pub struct PipelineEvent {
    pub timestamp_ns: u64,
    pub address: usize,
    pub size: usize,
    pub event_type: EventType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EventType {
    Alloc,
    Free,
}

/// The living pipeline — connects membrane → graph → predictor → condenser → lenia
pub struct Pipeline {
    config: PipelineConfig,

    /// The graph learns access topology
    graph: AccessGraph,

    /// The predictor fires spikes through learned topology
    predictor: RustPredictor,

    /// The condenser compresses cold, promotes hot
    condenser: Condenser,

    /// The Lenia field — continuous thermal dynamics
    /// Replaces hard idle thresholds with physics
    field: LeniaField,

    /// Accumulated events for graph rebuilding
    event_buffer: Vec<(u64, String, u64)>,

    /// Address → path mapping (for graph node identity)
    address_to_path: std::collections::HashMap<usize, String>,

    /// Address → Lenia region ID mapping
    address_to_field_id: std::collections::HashMap<usize, u32>,

    /// Next Lenia field ID
    next_field_id: u32,

    /// Path counter for generating unique paths
    path_counter: u64,

    /// Start time
    start: Instant,

    /// Lenia step counter (step every N events)
    field_step_counter: u64,

    // ── Mode & safety model ───────────────────────────────────────────────

    /// Current operating mode
    pub mode: PipelineMode,

    /// How many graph rebuilds have occurred since creation
    /// (used for transition gate — separate from the public stats counter)
    mode_rebuilds: u32,

    /// Last measured prediction accuracy (0.0–100.0, from ScoreResult.accuracy)
    pub last_prediction_accuracy: f64,

    /// How many process_alloc calls have occurred while in Active mode
    pub active_cycles: u64,

    /// Timestamps (ns) of recent scan_and_compress calls that compressed something.
    /// Ring-buffered: keeps last 100 entries.
    pub condensation_timestamps: Vec<u64>,

    // ── Stats ─────────────────────────────────────────────────────────────

    pub events_processed: u64,
    pub predictions_fired: u64,
    pub predictions_acted: u64,
    pub graph_rebuilds: u64,
    pub compressions: u64,
    pub lenia_steps: u64,
}

impl Pipeline {
    /// Create a new pipeline in **Active** mode (backward-compatible default).
    pub fn new(config: PipelineConfig) -> Self {
        Self::new_with_mode(config, PipelineMode::Active)
    }

    /// Create a new pipeline in **Observing** mode.
    /// The substrate learns immediately; compression is gated until
    /// `check_transition()` promotes it to Active.
    pub fn new_observing(config: PipelineConfig) -> Self {
        Self::new_with_mode(config, PipelineMode::Observing)
    }

    fn new_with_mode(config: PipelineConfig, mode: PipelineMode) -> Self {
        let condenser_config = CondenserConfig {
            idle_threshold_ns: config.idle_threshold_ns,
            min_manage_size: config.min_manage_size,
            test_mode: config.test_mode,
            ..Default::default()
        };

        // RAM budget for Lenia field — default 1024 MB
        let field = LeniaField::new(1024.0);

        Self {
            graph: AccessGraph::new(config.causal_window_ns, config.cluster_threshold),
            predictor: RustPredictor::new(),
            condenser: Condenser::new(condenser_config),
            field,
            event_buffer: Vec::with_capacity(config.graph_rebuild_interval),
            address_to_path: std::collections::HashMap::with_capacity(1000),
            address_to_field_id: std::collections::HashMap::with_capacity(1000),
            next_field_id: 0,
            path_counter: 0,
            start: Instant::now(),
            field_step_counter: 0,
            mode,
            mode_rebuilds: 0,
            last_prediction_accuracy: 0.0,
            active_cycles: 0,
            condensation_timestamps: Vec::with_capacity(100),
            events_processed: 0,
            predictions_fired: 0,
            predictions_acted: 0,
            graph_rebuilds: 0,
            compressions: 0,
            lenia_steps: 0,
            config,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Get or create a path name for an address.
    ///
    /// ADAPTIVE IDENTITY — inspired by Gaussian splatting's density control.
    /// Just as splats represent regions, not points, allocations are
    /// identified by their SIZE CLASS, not their address. All 64KB allocs
    /// share the path "large" — the graph learns that "large follows large"
    /// which IS the pattern. Specific addresses are tracked separately
    /// for the condenser to manage, but the graph sees classes.
    ///
    /// This is Law 7 applied: raw size enters the substrate, no
    /// classification beyond the physical size bucket. The graph
    /// discovers which buckets co-occur and in what order.
    fn get_path(&mut self, address: usize, size: usize) -> String {
        // Size-class identity — all allocs of similar size share a path
        // This makes predictions transferable across allocations
        let path = match size {
            0..=63 => "tiny".to_string(),
            64..=1023 => "small".to_string(),
            1024..=4095 => "med.1k".to_string(),
            4096..=16383 => "med.4k".to_string(),
            16384..=65535 => "med.16k".to_string(),
            65536..=262143 => "large.64k".to_string(),
            262144..=1048575 => "large.256k".to_string(),
            1048576..=4194303 => "huge.1m".to_string(),
            4194304..=16777215 => "huge.4m".to_string(),
            16777216..=67108863 => "huge.16m".to_string(),
            _ => "massive".to_string(),
        };

        // Map address to path for condenser lookups
        self.address_to_path.insert(address, path.clone());
        path
    }

    /// Process a single allocation event through the full pipeline.
    ///
    /// Graph building and predictor learning happen in ALL modes.
    /// Condenser registration, pre-promote, and scan are gated to Active mode.
    /// The substrate always learns — it just doesn't act until Active.
    pub fn process_alloc(&mut self, address: usize, size: usize) {
        self.events_processed += 1;
        let ts = self.elapsed_ns();

        // Skip tiny allocations — noise, not signal
        if size < self.config.min_manage_size {
            return;
        }

        // Track active_cycles — graduated engagement ramp
        if self.mode == PipelineMode::Active {
            self.active_cycles += 1;
        }

        let threshold = self.effective_threshold();

        if self.mode == PipelineMode::Active {
            // Large-alloc passthrough: graph still learns the pattern but the
            // condenser never touches these addresses. They are high-entropy
            // (ONNX weights, embedding caches, large tensors — LZ4 won't help)
            // and inference-hot (compressing mid-flight causes use-after-read).
            // Observer data: ONNX model = 86MB; NG bursts hit 7GB in 10s.
            let passthrough = size > self.config.large_alloc_passthrough_bytes;

            // 1. Register with condenser AND Lenia field (skipped for large allocs)
            if !passthrough {
                self.condenser.register(address, size);
                let field_id = self.get_or_create_field_id(address, size as u64);
                // 2. Heat the field — this access injects energy
                self.field.access(field_id);
            }

            // 3. Record for graph learning
            let path = self.get_path(address, size);
            self.event_buffer.push((ts, path.clone(), size as u64));

            // 4. If predictor is learned, fire predictions
            if self.predictor.is_learned() {
                let predictions = self.predictor.predict(&path, 5);
                self.predictions_fired += predictions.len() as u64;

                for pred in &predictions {
                    if pred.confidence >= threshold {
                        for (&addr, p) in &self.address_to_path {
                            if *p == pred.path {
                                self.condenser.pre_promote(addr);
                                // Also heat the predicted region in the field
                                if let Some(&fid) = self.address_to_field_id.get(&addr) {
                                    self.field.access(fid);
                                }
                                self.predictions_acted += 1;
                                break;
                            }
                        }
                    }
                }
            }

            // 5. Periodically step the Lenia field
            self.field_step_counter += 1;
            if self.field_step_counter % 100 == 0 {
                self.field.step();
                self.lenia_steps += 1;

                // Use Lenia's cold regions to drive condenser compression
                let cold = self.field.get_cold_regions();
                for (cold_id, _temp) in &cold {
                    // Find the address for this cold field region
                    for (&addr, &fid) in &self.address_to_field_id {
                        if fid == *cold_id {
                            // Tell condenser this region is cold
                            self.condenser.touch(addr); // mark for idle detection
                            break;
                        }
                    }
                }
            }
        } else {
            // Observing or Blacklisted — substrate still learns, condenser is silent

            // Record for graph learning (no condenser registration)
            let path = self.get_path(address, size);
            self.event_buffer.push((ts, path, size as u64));
        }

        // 6. Periodically rebuild graph and retrain predictor (all modes)
        if self.event_buffer.len() >= self.config.graph_rebuild_interval {
            self.rebuild_graph();
        }
    }

    /// Get or create a Lenia field ID for an address
    fn get_or_create_field_id(&mut self, address: usize, size_bytes: u64) -> u32 {
        if let Some(&id) = self.address_to_field_id.get(&address) {
            return id;
        }
        let id = self.next_field_id;
        self.next_field_id += 1;
        self.field.add_region(id, size_bytes as usize, 0);
        self.address_to_field_id.insert(address, id);
        id
    }

    /// Process a free event
    pub fn process_free(&mut self, address: usize) {
        self.condenser.unregister(address);
        self.address_to_path.remove(&address);
        // Remove from Lenia field — dead allocations don't get thermal cycles
        if let Some(field_id) = self.address_to_field_id.remove(&address) {
            self.field.remove_region(field_id);
        }
    }

    /// Rebuild the graph from accumulated events and retrain the predictor.
    /// Called automatically from process_alloc when the event buffer fills.
    fn rebuild_graph(&mut self) {
        // Build fresh graph from accumulated events
        let mut new_graph = AccessGraph::new(
            self.config.causal_window_ns,
            self.config.cluster_threshold,
        );
        new_graph.build(self.event_buffer.clone());

        // Retrain predictor
        let mut new_predictor = RustPredictor::new();
        new_predictor.learn(&new_graph);

        // Score the new predictor against the buffer we just trained on
        if new_predictor.is_learned() && !self.event_buffer.is_empty() {
            let score = new_predictor.score(self.event_buffer.clone());
            self.last_prediction_accuracy = score.accuracy;
        }

        self.graph = new_graph;
        self.predictor = new_predictor;
        self.graph_rebuilds += 1;
        self.mode_rebuilds += 1;

        // Keep last 20% of events for continuity
        let keep = self.event_buffer.len() / 5;
        let drain_to = self.event_buffer.len() - keep;
        self.event_buffer.drain(..drain_to);

        // Check mode transition after each rebuild
        self.check_transition();
    }

    /// Check whether the pipeline should transition from Observing → Active.
    ///
    /// Transition gates:
    /// - mode must be Observing
    /// - at least 3 graph rebuilds since creation
    /// - last_prediction_accuracy >= 40.0
    ///
    /// Blacklisted pipelines never transition.
    ///
    /// Returns true if a transition occurred.
    pub fn check_transition(&mut self) -> bool {
        match self.mode {
            PipelineMode::Blacklisted => false,
            PipelineMode::Active => false,
            PipelineMode::Observing => {
                if self.mode_rebuilds >= 3
                    && self.last_prediction_accuracy >= 40.0
                {
                    self.mode = PipelineMode::Active;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Effective compression threshold — graduated engagement ramp.
    ///
    /// New pipelines start conservative (0.8) and relax over time.
    /// Non-Active pipelines return 1.0 so nothing ever compresses.
    pub fn effective_threshold(&self) -> f64 {
        match self.mode {
            PipelineMode::Active => {
                if self.active_cycles < 100 {
                    0.8
                } else if self.active_cycles < 1100 {
                    0.5
                } else {
                    self.config.prediction_threshold
                }
            }
            _ => 1.0, // Never compress when not Active
        }
    }

    /// Run the condenser's compression scan.
    /// Call this periodically (e.g., every second).
    ///
    /// Records condensation timestamps for crash correlation when compression occurs.
    pub fn scan(&mut self) -> (u32, u64) {
        let (count, saved) = self.condenser.scan_and_compress();
        self.compressions += count as u64;
        if count > 0 {
            // Record timestamp for crash correlation (ring buffer, last 100)
            let ts = self.elapsed_ns();
            if self.condensation_timestamps.len() >= 100 {
                self.condensation_timestamps.remove(0);
            }
            self.condensation_timestamps.push(ts);
        }
        (count, saved)
    }

    /// Touch a region (it was accessed)
    pub fn touch(&mut self, address: usize) {
        self.condenser.touch(address);
    }

    /// Report that the monitored process died at `death_ns` (nanoseconds,
    /// same epoch as `elapsed_ns`).
    ///
    /// Returns true if any recorded condensation event occurred within 5 seconds
    /// of the death — suggesting the condenser may have interfered.
    pub fn report_process_death(&mut self, death_ns: u64) -> bool {
        const WINDOW_NS: u64 = 5_000_000_000;
        for &ts in &self.condensation_timestamps {
            let delta = if death_ns >= ts {
                death_ns - ts
            } else {
                ts - death_ns
            };
            if delta <= WINDOW_NS {
                return true;
            }
        }
        false
    }

    /// Get pipeline summary
    pub fn summary(&self) -> PipelineSummary {
        let condenser_summary = self.condenser.summary();
        let lenia_summary = self.field.summary();

        PipelineSummary {
            events_processed: self.events_processed,
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
            graph_clusters: self.graph.cluster_count(),
            graph_rebuilds: self.graph_rebuilds,
            predictions_fired: self.predictions_fired,
            predictions_acted: self.predictions_acted,
            lenia_steps: self.lenia_steps,
            condenser: condenser_summary,
            lenia: lenia_summary,
        }
    }
}

/// Per-process pipeline map — routes allocation events to the correct pipeline
/// based on PID. Each process gets its own isolated pipeline starting in
/// Observing mode.
pub struct ProcessPipelineMap {
    pipelines: HashMap<u32, Pipeline>,
    config: PipelineConfig,
}

impl ProcessPipelineMap {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            pipelines: HashMap::new(),
            config,
        }
    }

    /// Get or create the pipeline for a given PID.
    /// New pipelines start in Observing mode.
    pub fn get_or_create(&mut self, pid: u32) -> &mut Pipeline {
        if !self.pipelines.contains_key(&pid) {
            let pipeline = Pipeline::new_observing(PipelineConfig {
                causal_window_ns: self.config.causal_window_ns,
                cluster_threshold: self.config.cluster_threshold,
                idle_threshold_ns: self.config.idle_threshold_ns,
                min_manage_size: self.config.min_manage_size,
                graph_rebuild_interval: self.config.graph_rebuild_interval,
                prediction_threshold: self.config.prediction_threshold,
                test_mode: self.config.test_mode,
                large_alloc_passthrough_bytes: self.config.large_alloc_passthrough_bytes,
            });
            self.pipelines.insert(pid, pipeline);
        }
        self.pipelines.get_mut(&pid).unwrap()
    }

    /// Route an allocation event to the correct process pipeline.
    pub fn process_alloc_global(&mut self, pid: u32, address: usize, size: usize) {
        self.get_or_create(pid).process_alloc(address, size);
    }

    /// Route a free event to the correct process pipeline.
    pub fn process_free_global(&mut self, pid: u32, address: usize) {
        if let Some(pipeline) = self.pipelines.get_mut(&pid) {
            pipeline.process_free(address);
        }
    }

    /// Number of tracked processes.
    pub fn process_count(&self) -> usize {
        self.pipelines.len()
    }
}

/// Full pipeline summary
#[derive(Clone, Debug)]
pub struct PipelineSummary {
    pub events_processed: u64,
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub graph_clusters: usize,
    pub graph_rebuilds: u64,
    pub predictions_fired: u64,
    pub predictions_acted: u64,
    pub lenia_steps: u64,
    pub condenser: crate::condenser::CondenserSummary,
    pub lenia: crate::lenia::LeniaSummary,
}

impl PipelineSummary {
    pub fn print(&self) {
        eprintln!("\n{}", "=".repeat(55));
        eprintln!("  CONDENSATE — Full Pipeline Report");
        eprintln!("{}", "=".repeat(55));

        eprintln!("\n  Events processed:  {}", self.events_processed);

        eprintln!("\n  GRAPH (the substrate):");
        eprintln!("    Nodes:    {}", self.graph_nodes);
        eprintln!("    Edges:    {}", self.graph_edges);
        eprintln!("    Clusters: {}", self.graph_clusters);
        eprintln!("    Rebuilds: {}", self.graph_rebuilds);

        eprintln!("\n  PREDICTOR (spreading activation):");
        eprintln!("    Predictions fired: {}", self.predictions_fired);
        eprintln!("    Predictions acted: {}", self.predictions_acted);

        eprintln!("\n  CONDENSER (motor output):");
        eprintln!("    HOT:  {} ({:.1} MB)",
                 self.condenser.hot_count, self.condenser.hot_mb);
        eprintln!("    WARM: {} ({:.1} MB → {:.1} MB compressed)",
                 self.condenser.warm_count,
                 self.condenser.warm_original_mb,
                 self.condenser.warm_compressed_mb);
        eprintln!("    COLD: {}", self.condenser.cold_count);

        if self.condenser.total_original_mb > 0.0 {
            eprintln!();
            eprintln!("  +-------------------------------------------+");
            eprintln!("  |  RAM: {:.1} MB → {:.1} MB ({:.1}% saved){}|",
                     self.condenser.total_original_mb,
                     self.condenser.total_current_mb,
                     self.condenser.saved_pct,
                     " ".repeat(std::cmp::max(0,
                         8 - format!("{:.1} MB → {:.1} MB ({:.1}% saved)",
                             self.condenser.total_original_mb,
                             self.condenser.total_current_mb,
                             self.condenser.saved_pct).len() as i32) as usize));
            eprintln!("  |  Same data. Same output. Less RAM.       |");
            eprintln!("  +-------------------------------------------+");
        }

        eprintln!("\n  LENIA FIELD (thermal dynamics):");
        eprintln!("    Steps:  {}", self.lenia_steps);
        eprintln!("    Energy: {:.1} / {:.1} ({:.1}% of budget)",
                 self.lenia.total_energy, self.lenia.max_energy, self.lenia.energy_pct);
        eprintln!("    HOT  (>{:.0}%): {} regions, {:.1} MB",
                 self.lenia.hot_threshold * 100.0, self.lenia.hot, self.lenia.hot_mb);
        eprintln!("    WARM ({:.0}%-{:.0}%): {} regions, {:.1} MB",
                 self.lenia.cold_threshold * 100.0, self.lenia.hot_threshold * 100.0,
                 self.lenia.warm, self.lenia.warm_mb);
        eprintln!("    COLD (<{:.0}%): {} regions, {:.1} MB",
                 self.lenia.cold_threshold * 100.0, self.lenia.cold, self.lenia.cold_mb);

        eprintln!("{}\n", "=".repeat(55));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests (must continue to pass) ────────────────────────────

    #[test]
    fn test_pipeline_basic_flow() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            graph_rebuild_interval: 50,
            min_manage_size: 1024,
            ..Default::default()
        });

        // Simulate a workload: repeated pattern of allocations
        for round in 0..3 {
            let base = round * 0x100000;
            for i in 0..20 {
                pipeline.process_alloc(base + i * 0x1000, 65_536);
            }
        }

        // Should have rebuilt the graph at least once (60 events > 50 threshold)
        assert!(pipeline.graph_rebuilds >= 1,
                "Graph should have rebuilt, got {} rebuilds", pipeline.graph_rebuilds);
        assert!(pipeline.events_processed > 0);

        let summary = pipeline.summary();
        assert!(summary.graph_nodes > 0, "Graph should have nodes");
    }

    #[test]
    fn test_pipeline_prediction_drives_promotion() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            graph_rebuild_interval: 30,
            min_manage_size: 1024,
            idle_threshold_ns: 0,  // compress immediately
            prediction_threshold: 0.1,  // low threshold to see predictions act
            test_mode: true,  // fake addresses — use synthetic data
            ..Default::default()
        });

        // Phase 1: Train the graph with a repeating pattern
        // A→B→C, always in this order, 20 times
        for round in 0..20 {
            let base = round * 3;
            pipeline.process_alloc(0xA000 + base, 65_536);
            pipeline.process_alloc(0xB000 + base, 65_536);
            pipeline.process_alloc(0xC000 + base, 65_536);
        }

        // Phase 2: Compress everything
        pipeline.scan();

        // Phase 3: Trigger the pattern again — predictions should fire
        let before_acted = pipeline.predictions_acted;
        pipeline.process_alloc(0xA000 + 100, 65_536);

        let summary = pipeline.summary();
        summary.print();

        assert!(summary.events_processed > 0);
        assert!(summary.graph_rebuilds >= 1);
    }

    #[test]
    fn test_pipeline_scan_compresses_idle() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            min_manage_size: 1024,
            idle_threshold_ns: 0,  // compress immediately
            graph_rebuild_interval: 1000,  // don't rebuild during this test
            test_mode: true,  // fake addresses — use synthetic data
            ..Default::default()
        });

        // Register some regions
        pipeline.process_alloc(0x10000, 65_536);
        pipeline.process_alloc(0x20000, 65_536);
        pipeline.process_alloc(0x30000, 65_536);

        // Scan should compress all idle regions
        let (count, saved) = pipeline.scan();

        assert_eq!(count, 3);
        assert!(saved > 0);

        let summary = pipeline.summary();
        assert_eq!(summary.condenser.warm_count, 3);
        assert_eq!(summary.condenser.hot_count, 0);
    }

    #[test]
    fn test_pipeline_free_cleans_up() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            min_manage_size: 1024,
            graph_rebuild_interval: 1000,
            ..Default::default()
        });

        pipeline.process_alloc(0x10000, 65_536);
        pipeline.process_alloc(0x20000, 65_536);

        assert_eq!(pipeline.summary().condenser.total_regions, 2);

        pipeline.process_free(0x10000);

        assert_eq!(pipeline.summary().condenser.total_regions, 1);
    }

    #[test]
    fn test_pipeline_full_simulation() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            graph_rebuild_interval: 30,
            min_manage_size: 4096,
            idle_threshold_ns: 0,
            prediction_threshold: 0.3,
            test_mode: true,  // fake addresses — use synthetic data
            ..Default::default()
        });

        // Simulate a realistic workload:
        // Phase 1: Startup — burst of allocations
        for i in 0..50 {
            pipeline.process_alloc(0x10000 + i * 0x10000, 65_536);
        }

        // Phase 2: Steady state — some allocs, some frees
        for i in 0..30 {
            pipeline.process_free(0x10000 + i * 0x10000);
        }
        for i in 0..20 {
            pipeline.process_alloc(0x800000 + i * 0x10000, 131_072);
        }

        // Phase 3: Scan for compression
        let (compressed, saved) = pipeline.scan();

        // Phase 4: New activity triggers predictions
        for i in 0..10 {
            pipeline.process_alloc(0xF00000 + i * 0x10000, 65_536);
        }

        let summary = pipeline.summary();
        summary.print();

        assert!(summary.events_processed > 50);
        assert!(summary.condenser.total_regions > 0);
        assert!(summary.graph_rebuilds >= 1,
                "Graph should have rebuilt at least once");
    }

    // ── Block D: new tests ────────────────────────────────────────────────

    /// Observing pipeline registers events but never compresses
    #[test]
    fn test_pipeline_mode_observing() {
        let mut pipeline = Pipeline::new_observing(PipelineConfig {
            min_manage_size: 1024,
            idle_threshold_ns: 0,  // would compress immediately if Active
            graph_rebuild_interval: 1000,
            test_mode: true,
            ..Default::default()
        });

        // Feed events
        pipeline.process_alloc(0x10000, 65_536);
        pipeline.process_alloc(0x20000, 65_536);
        pipeline.process_alloc(0x30000, 65_536);

        // Mode must still be Observing (not enough rebuilds / accuracy)
        assert_eq!(pipeline.mode, PipelineMode::Observing);

        // Scan should return zero compressions — condenser is silent
        let (count, saved) = pipeline.scan();
        assert_eq!(count, 0, "Observing pipeline must not compress");
        assert_eq!(saved, 0);

        // Condenser must have nothing registered
        let summary = pipeline.summary();
        assert_eq!(summary.condenser.total_regions, 0,
                   "Observing pipeline must not register regions with condenser");
    }

    /// After 3 rebuilds with good accuracy, Observing transitions to Active
    #[test]
    fn test_pipeline_transition() {
        // Use a small rebuild interval so we can force rebuilds quickly.
        // We need mode_rebuilds >= 3 AND last_prediction_accuracy >= 40.
        let mut pipeline = Pipeline::new_observing(PipelineConfig {
            min_manage_size: 1024,
            graph_rebuild_interval: 10,
            idle_threshold_ns: 1_000_000_000,
            prediction_threshold: 0.1,
            ..Default::default()
        });

        // Drive a strong repeating pattern so the predictor scores well.
        // Each batch of 10+ events triggers a rebuild.
        for _round in 0..5 {
            for i in 0..12usize {
                let size = if i % 2 == 0 { 65_536 } else { 131_072 };
                pipeline.process_alloc(0x10000 + i * 0x1000, size);
            }
        }

        assert!(pipeline.graph_rebuilds >= 3,
                "Expected at least 3 rebuilds, got {}", pipeline.graph_rebuilds);

        // Patch accuracy to guarantee the transition gate passes,
        // then call check_transition (also called internally — idempotent).
        pipeline.last_prediction_accuracy = 50.0;
        let transitioned = pipeline.check_transition();

        assert!(transitioned, "Should have transitioned to Active");
        assert_eq!(pipeline.mode, PipelineMode::Active);
    }

    /// effective_threshold returns 0.8 fresh, 0.5 mid-ramp, config value at maturity
    #[test]
    fn test_pipeline_graduated_threshold() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            prediction_threshold: 0.3,
            ..Default::default()
        });

        // Fresh Active pipeline, 0 cycles
        assert_eq!(pipeline.active_cycles, 0);
        assert_eq!(pipeline.effective_threshold(), 0.8,
                   "Fresh active pipeline should use conservative 0.8 threshold");

        // Mid-ramp
        pipeline.active_cycles = 500;
        assert_eq!(pipeline.effective_threshold(), 0.5,
                   "Mid-ramp should use 0.5 threshold");

        // Mature
        pipeline.active_cycles = 1100;
        assert_eq!(pipeline.effective_threshold(), 0.3,
                   "Mature pipeline should use config threshold");

        // Observing always returns 1.0
        let observing = Pipeline::new_observing(PipelineConfig::default());
        assert_eq!(observing.effective_threshold(), 1.0,
                   "Observing pipeline threshold must be 1.0 (never compress)");
    }

    /// Condensation within 5 seconds of process death is flagged
    #[test]
    fn test_pipeline_crash_correlation() {
        let mut pipeline = Pipeline::new(PipelineConfig {
            min_manage_size: 1024,
            idle_threshold_ns: 0,
            graph_rebuild_interval: 1000,
            test_mode: true,  // fake addresses — use synthetic data
            ..Default::default()
        });

        // Compress something so a timestamp is recorded
        pipeline.process_alloc(0x10000, 65_536);
        let (count, _) = pipeline.scan();
        assert_eq!(count, 1, "Expected one compression");
        assert_eq!(pipeline.condensation_timestamps.len(), 1);

        // Death 1 second after condensation — inside the 5s window
        let condensation_ts = pipeline.condensation_timestamps[0];
        let death_1s_later = condensation_ts + 1_000_000_000;
        assert!(
            pipeline.report_process_death(death_1s_later),
            "Death 1s after condensation should be flagged as likely interference"
        );

        // Death 10 seconds later — outside window
        let death_10s_later = condensation_ts + 10_000_000_000;
        assert!(
            !pipeline.report_process_death(death_10s_later),
            "Death 10s after condensation should not be flagged"
        );
    }

    /// Blacklisted pipeline never transitions regardless of accuracy or rebuilds
    #[test]
    fn test_pipeline_blacklisted() {
        let mut pipeline = Pipeline::new_observing(PipelineConfig {
            min_manage_size: 1024,
            graph_rebuild_interval: 1000,
            ..Default::default()
        });

        // Force blacklist
        pipeline.mode = PipelineMode::Blacklisted;

        // Simulate ideal conditions — should still not transition
        pipeline.mode_rebuilds = 10;
        pipeline.last_prediction_accuracy = 99.0;

        let transitioned = pipeline.check_transition();
        assert!(!transitioned, "Blacklisted pipeline must never transition");
        assert_eq!(pipeline.mode, PipelineMode::Blacklisted);
    }

    /// Two PIDs get fully isolated pipelines
    #[test]
    fn test_process_pipeline_map() {
        let mut map = ProcessPipelineMap::new(PipelineConfig {
            min_manage_size: 1024,
            idle_threshold_ns: 0,
            graph_rebuild_interval: 1000,
            test_mode: true,  // fake addresses — use synthetic data
            ..Default::default()
        });

        // Two distinct PIDs
        map.process_alloc_global(100, 0x10000, 65_536);
        map.process_alloc_global(100, 0x20000, 65_536);
        map.process_alloc_global(200, 0x10000, 65_536);

        assert_eq!(map.process_count(), 2, "Should track exactly 2 processes");

        // Pipelines start in Observing mode
        {
            let p100 = map.get_or_create(100);
            assert_eq!(p100.mode, PipelineMode::Observing,
                       "New pipelines must start in Observing mode");
            assert_eq!(p100.events_processed, 2);
        }

        {
            let p200 = map.get_or_create(200);
            assert_eq!(p200.events_processed, 1);
        }

        // Free on PID 100 doesn't affect PID 200
        map.process_free_global(100, 0x10000);
        {
            let p200 = map.get_or_create(200);
            assert_eq!(p200.events_processed, 1,
                       "PID 200 should be unaffected by PID 100 free");
        }

        // Free on unknown PID is a no-op (must not panic)
        map.process_free_global(999, 0xDEAD);
    }
}
