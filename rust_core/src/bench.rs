//! Benchmark: Rust graph build + predict vs equivalent Python workload.
//!
//! Run with: cargo test --release bench_ -- --nocapture

#[cfg(test)]
mod bench {
    use crate::graph::AccessGraph;
    use crate::predictor::RustPredictor;
    use std::time::Instant;

    /// Generate a realistic workload: N layers, some hot, some cold,
    /// with causal chains. Same pattern as test_predictor.py.
    fn generate_inference_workload(
        num_layers: u32,
        num_hot: u32,
        iterations: u32,
    ) -> Vec<(u64, String, u64)> {
        let mut events = Vec::new();
        let mut ts: u64 = 0;

        for _ in 0..iterations {
            // Hot layers accessed every iteration
            for i in 0..num_hot {
                events.push((ts, format!("layer_{}", i), 65536));
                ts += 100_000; // 0.1ms between accesses
            }

            // Cold layers: 3% chance
            for i in num_hot..num_layers {
                if (ts / 1000 + i as u64) % 33 == 0 {
                    events.push((ts, format!("layer_{}", i), 65536));
                    ts += 100_000;
                }
            }

            ts += 2_000_000; // 2ms between iterations
        }

        events
    }

    #[test]
    fn bench_graph_build_small() {
        // 16 layers, 4 hot, 100 iterations — ~similar to Python test
        let events = generate_inference_workload(16, 4, 100);
        println!("\nSmall workload: {} events", events.len());

        let start = Instant::now();
        let mut graph = AccessGraph::new(5_000_000, 0.7);
        graph.build(events.clone());
        let elapsed = start.elapsed();

        println!("  Graph build: {:?}", elapsed);
        println!("  Nodes: {}, Edges: {}, Clusters: {}",
                 graph.node_count(), graph.edge_count(), graph.cluster_count());

        // Predict
        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        let start = Instant::now();
        let result = predictor.score(events);
        let elapsed = start.elapsed();

        println!("  Score: {:?}", elapsed);
        println!("  Accuracy: {}%, Predictions: {}, Hits: {}",
                 result.accuracy, result.predictions_made, result.hits);
    }

    #[test]
    fn bench_graph_build_medium() {
        // 64 layers, 8 hot, 100 iterations
        let events = generate_inference_workload(64, 8, 100);
        println!("\nMedium workload: {} events", events.len());

        let start = Instant::now();
        let mut graph = AccessGraph::new(5_000_000, 0.7);
        graph.build(events.clone());
        let elapsed = start.elapsed();

        println!("  Graph build: {:?}", elapsed);
        println!("  Nodes: {}, Edges: {}, Clusters: {}",
                 graph.node_count(), graph.edge_count(), graph.cluster_count());

        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        let start = Instant::now();
        let result = predictor.score(events);
        let elapsed = start.elapsed();

        println!("  Score: {:?}", elapsed);
        println!("  Accuracy: {}%, Predictions: {}, Hits: {}",
                 result.accuracy, result.predictions_made, result.hits);
    }

    #[test]
    fn bench_graph_build_large() {
        // 256 layers, 16 hot, 50 iterations — stress test
        let events = generate_inference_workload(256, 16, 50);
        println!("\nLarge workload: {} events", events.len());

        let start = Instant::now();
        let mut graph = AccessGraph::new(5_000_000, 0.7);
        graph.build(events.clone());
        let elapsed = start.elapsed();

        println!("  Graph build: {:?}", elapsed);
        println!("  Nodes: {}, Edges: {}, Clusters: {}",
                 graph.node_count(), graph.edge_count(), graph.cluster_count());

        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        let start = Instant::now();
        let result = predictor.score(events);
        let elapsed = start.elapsed();

        println!("  Score: {:?}", elapsed);
        println!("  Accuracy: {}%, Predictions: {}, Hits: {}",
                 result.accuracy, result.predictions_made, result.hits);
    }

    #[test]
    fn bench_predict_latency() {
        // Measure single-prediction latency — this is the hot path
        let events = generate_inference_workload(64, 8, 100);

        let mut graph = AccessGraph::new(5_000_000, 0.7);
        graph.build(events);

        let mut predictor = RustPredictor::new();
        predictor.learn(&graph);

        // Warm up
        for _ in 0..100 {
            let _ = predictor.predict("layer_0", 10);
        }

        // Measure
        let iterations = 100_000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = predictor.predict("layer_0", 10);
        }
        let elapsed = start.elapsed();

        let per_predict_ns = elapsed.as_nanos() / iterations as u128;
        println!("\nSingle predict() latency:");
        println!("  {} iterations in {:?}", iterations, elapsed);
        println!("  *** {per_predict_ns} ns per prediction ***");
        println!("  ({:.1} million predictions/sec)",
                 1_000_000_000.0 / per_predict_ns as f64);
    }
}
