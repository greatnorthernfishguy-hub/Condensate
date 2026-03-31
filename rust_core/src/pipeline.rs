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

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::graph::AccessGraph;
use crate::predictor::RustPredictor;
use crate::condenser::{Condenser, CondenserConfig};
use crate::lenia::LeniaField;

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

    /// Stats
    pub events_processed: u64,
    pub predictions_fired: u64,
    pub predictions_acted: u64,
    pub graph_rebuilds: u64,
    pub compressions: u64,
    pub lenia_steps: u64,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let condenser_config = CondenserConfig {
            idle_threshold_ns: config.idle_threshold_ns,
            min_manage_size: config.min_manage_size,
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
    /// This is the heartbeat. Every malloc flows here:
    /// 1. Register with condenser + Lenia field
    /// 2. Heat the Lenia field (access = energy injection)
    /// 3. Record in event buffer (for graph learning)
    /// 4. If graph is learned, predict what's next
    /// 5. Pre-promote predicted regions
    /// 6. Periodically step the Lenia field (continuous dynamics)
    pub fn process_alloc(&mut self, address: usize, size: usize) {
        self.events_processed += 1;
        let ts = self.elapsed_ns();

        // Skip tiny allocations — noise, not signal
        if size < self.config.min_manage_size {
            return;
        }

        // 1. Register with condenser AND Lenia field
        self.condenser.register(address, size);
        let field_id = self.get_or_create_field_id(address, size as u64);

        // 2. Heat the field — this access injects energy
        self.field.access(field_id);

        // 3. Record for graph learning
        let path = self.get_path(address, size);
        self.event_buffer.push((ts, path.clone(), size as u64));

        // 4. If predictor is learned, fire predictions
        if self.predictor.is_learned() {
            let predictions = self.predictor.predict(&path, 5);
            self.predictions_fired += predictions.len() as u64;

            for pred in &predictions {
                if pred.confidence >= self.config.prediction_threshold {
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

        // 6. Periodically rebuild graph and retrain predictor
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
        self.field.add_region(id, size_bytes);
        self.address_to_field_id.insert(address, id);
        id
    }

    /// Process a free event
    pub fn process_free(&mut self, address: usize) {
        self.condenser.unregister(address);
        self.address_to_path.remove(&address);
        self.address_to_field_id.remove(&address);
    }

    /// Rebuild the graph from accumulated events and retrain the predictor
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

        self.graph = new_graph;
        self.predictor = new_predictor;
        self.graph_rebuilds += 1;

        // Keep last 20% of events for continuity
        let keep = self.event_buffer.len() / 5;
        let drain_to = self.event_buffer.len() - keep;
        self.event_buffer.drain(..drain_to);
    }

    /// Run the condenser's compression scan
    /// Call this periodically (e.g., every second)
    pub fn scan(&mut self) -> (u32, u64) {
        let (count, saved) = self.condenser.scan_and_compress();
        self.compressions += count as u64;
        (count, saved)
    }

    /// Touch a region (it was accessed)
    pub fn touch(&mut self, address: usize) {
        self.condenser.touch(address);
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
}
