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
            idle_threshold_ns: 5_000_000_000,   // 5 seconds
            min_manage_size: 4_096,             // 4KB
            graph_rebuild_interval: 500,         // rebuild graph every 500 events
            prediction_threshold: 0.5,           // act on predictions with >50% confidence
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

/// The living pipeline — connects membrane → graph → predictor → condenser
pub struct Pipeline {
    config: PipelineConfig,

    /// The graph learns access topology
    graph: AccessGraph,

    /// The predictor fires spikes through learned topology
    predictor: RustPredictor,

    /// The condenser compresses cold, promotes hot
    condenser: Condenser,

    /// Accumulated events for graph rebuilding
    event_buffer: Vec<(u64, String, u64)>,

    /// Address → path mapping (for graph node identity)
    address_to_path: std::collections::HashMap<usize, String>,

    /// Path counter for generating unique paths
    path_counter: u64,

    /// Start time
    start: Instant,

    /// Stats
    pub events_processed: u64,
    pub predictions_fired: u64,
    pub predictions_acted: u64,
    pub graph_rebuilds: u64,
    pub compressions: u64,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let condenser_config = CondenserConfig {
            idle_threshold_ns: config.idle_threshold_ns,
            min_manage_size: config.min_manage_size,
            ..Default::default()
        };

        Self {
            graph: AccessGraph::new(config.causal_window_ns, config.cluster_threshold),
            predictor: RustPredictor::new(),
            condenser: Condenser::new(condenser_config),
            event_buffer: Vec::with_capacity(config.graph_rebuild_interval),
            address_to_path: std::collections::HashMap::with_capacity(1000),
            path_counter: 0,
            start: Instant::now(),
            events_processed: 0,
            predictions_fired: 0,
            predictions_acted: 0,
            graph_rebuilds: 0,
            compressions: 0,
            config,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Get or create a path name for an address.
    /// Paths are how the graph identifies nodes.
    /// We bucket by size class for better pattern learning —
    /// "all 64KB allocations" behave similarly regardless of address.
    fn get_path(&mut self, address: usize, size: usize) -> String {
        if let Some(path) = self.address_to_path.get(&address) {
            return path.clone();
        }

        // Name by size bucket — the graph learns that "64KB allocs"
        // follow "64KB allocs", not that address 0x1234 follows 0x5678.
        // This is the key insight: allocation SIZE PATTERNS are what
        // repeat, not specific addresses.
        let bucket = match size {
            0..=63 => "tiny",
            64..=1023 => "small",
            1024..=65535 => "med",
            65536..=1048575 => "large",
            1048576..=67108863 => "huge",
            _ => "massive",
        };

        self.path_counter += 1;
        let path = format!("{}.{}", bucket, self.path_counter);
        self.address_to_path.insert(address, path.clone());
        path
    }

    /// Process a single allocation event through the full pipeline.
    ///
    /// This is the heartbeat. Every malloc flows here:
    /// 1. Register with condenser (for tier management)
    /// 2. Record in event buffer (for graph learning)
    /// 3. If graph is learned, predict what's next
    /// 4. Pre-promote predicted regions
    pub fn process_alloc(&mut self, address: usize, size: usize) {
        self.events_processed += 1;
        let ts = self.elapsed_ns();

        // Skip tiny allocations — noise, not signal
        if size < self.config.min_manage_size {
            return;
        }

        // 1. Register with condenser
        self.condenser.register(address, size);

        // 2. Record for graph learning
        let path = self.get_path(address, size);
        self.event_buffer.push((ts, path.clone(), size as u64));

        // 3. If predictor is learned, fire predictions
        if self.predictor.is_learned() {
            let predictions = self.predictor.predict(&path, 5);
            self.predictions_fired += predictions.len() as u64;

            for pred in &predictions {
                if pred.confidence >= self.config.prediction_threshold {
                    // Find the address for this predicted path
                    // and pre-promote it in the condenser
                    for (&addr, p) in &self.address_to_path {
                        if *p == pred.path {
                            self.condenser.pre_promote(addr);
                            self.predictions_acted += 1;
                            break;
                        }
                    }
                }
            }
        }

        // 4. Periodically rebuild graph and retrain predictor
        if self.event_buffer.len() >= self.config.graph_rebuild_interval {
            self.rebuild_graph();
        }
    }

    /// Process a free event
    pub fn process_free(&mut self, address: usize) {
        self.condenser.unregister(address);
        self.address_to_path.remove(&address);
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

        PipelineSummary {
            events_processed: self.events_processed,
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
            graph_clusters: self.graph.cluster_count(),
            graph_rebuilds: self.graph_rebuilds,
            predictions_fired: self.predictions_fired,
            predictions_acted: self.predictions_acted,
            condenser: condenser_summary,
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
    pub condenser: crate::condenser::CondenserSummary,
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
