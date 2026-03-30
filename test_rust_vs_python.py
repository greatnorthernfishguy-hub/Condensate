"""
Condensate: Rust vs Python Benchmark

Runs the same workloads through both backends and compares:
- Build time
- Prediction latency
- Accuracy
- Scoring time

Run: python3 test_rust_vs_python.py
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from membrane import Membrane
from graph_builder import GraphBuilder
from predictor import Predictor
from rust_predictor import is_rust_available, RustPredictorWrapper


def generate_workload(num_layers, num_hot, iterations):
    """Generate access log from a simulated workload."""
    Membrane.clear()

    state = {f"layer_{i}": np.random.randn(64, 64).astype(np.float32)
             for i in range(num_layers)}
    wrapped = Membrane.wrap(state, "model")

    hot_set = set(range(num_hot))
    for _ in range(iterations):
        for i in range(num_layers):
            if i in hot_set:
                _ = wrapped[f"layer_{i}"]
            elif np.random.random() < 0.03:
                _ = wrapped[f"layer_{i}"]
        time.sleep(0.001)

    return Membrane.get_log()


def benchmark_python(log):
    """Benchmark the Python predictor."""
    # Build graph
    start = time.monotonic()
    graph = GraphBuilder(causal_window_ns=5_000_000)
    graph.build(log)
    build_time = time.monotonic() - start

    # Learn predictor
    predictor = Predictor()
    start = time.monotonic()
    predictor.learn(graph)
    learn_time = time.monotonic() - start

    # Score
    start = time.monotonic()
    result = predictor.score(log)
    score_time = time.monotonic() - start

    # Single prediction latency
    start = time.monotonic()
    for _ in range(1000):
        predictor.predict("model.layer_0")
    predict_time = (time.monotonic() - start) / 1000

    return {
        "build_ms": build_time * 1000,
        "learn_ms": learn_time * 1000,
        "score_ms": score_time * 1000,
        "predict_us": predict_time * 1_000_000,
        "accuracy": result["accuracy"],
        "predictions": result["predictions_made"],
        "hits": result["hits"],
    }


def benchmark_rust(log):
    """Benchmark the Rust predictor."""
    if not is_rust_available():
        return None

    wrapper = RustPredictorWrapper()

    # Build graph + learn (combined in Rust path)
    start = time.monotonic()
    wrapper.learn_from_log(log)
    build_learn_time = time.monotonic() - start

    # Score
    start = time.monotonic()
    result = wrapper.score(log)
    score_time = time.monotonic() - start

    # Single prediction latency
    start = time.monotonic()
    for _ in range(1000):
        wrapper.predict("model.layer_0")
    predict_time = (time.monotonic() - start) / 1000

    return {
        "build_ms": build_learn_time * 1000,
        "learn_ms": 0,  # combined with build
        "score_ms": score_time * 1000,
        "predict_us": predict_time * 1_000_000,
        "accuracy": result["accuracy"],
        "predictions": result["predictions_made"],
        "hits": result["hits"],
    }


def run_comparison(name, num_layers, num_hot, iterations):
    """Run both backends on the same workload and compare."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  {num_layers} layers, {num_hot} hot, {iterations} iterations")
    print(f"{'='*65}")

    log = generate_workload(num_layers, num_hot, iterations)
    print(f"  Events generated: {len(log)}")

    # Python
    py = benchmark_python(log)

    # Rust
    rs = benchmark_rust(log)

    # Print comparison
    print(f"\n  {'Metric':<25} {'Python':>12} {'Rust':>12} {'Speedup':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    if rs:
        py_build = py['build_ms'] + py['learn_ms']
        rs_build = rs['build_ms']
        build_speedup = py_build / rs_build if rs_build > 0 else float('inf')

        score_speedup = py['score_ms'] / rs['score_ms'] if rs['score_ms'] > 0 else float('inf')
        predict_speedup = py['predict_us'] / rs['predict_us'] if rs['predict_us'] > 0 else float('inf')

        print(f"  {'Build + Learn':<25} {py_build:>10.1f}ms {rs_build:>10.1f}ms {build_speedup:>9.1f}x")
        print(f"  {'Score':<25} {py['score_ms']:>10.1f}ms {rs['score_ms']:>10.1f}ms {score_speedup:>9.1f}x")
        print(f"  {'Single predict()':<25} {py['predict_us']:>10.1f}μs {rs['predict_us']:>10.1f}μs {predict_speedup:>9.1f}x")
        print(f"  {'Accuracy':<25} {py['accuracy']:>11.1f}% {rs['accuracy']:>11.1f}%")
        print(f"  {'Predictions':<25} {py['predictions']:>12} {rs['predictions']:>12}")
        print(f"  {'Hits':<25} {py['hits']:>12} {rs['hits']:>12}")
    else:
        print(f"  Rust backend not available — showing Python only")
        print(f"  {'Build + Learn':<25} {py['build_ms'] + py['learn_ms']:>10.1f}ms")
        print(f"  {'Score':<25} {py['score_ms']:>10.1f}ms")
        print(f"  {'Single predict()':<25} {py['predict_us']:>10.1f}μs")
        print(f"  {'Accuracy':<25} {py['accuracy']:>11.1f}%")

    return py, rs


if __name__ == "__main__":
    print("=" * 65)
    print("  CONDENSATE — Rust vs Python Benchmark")
    print("=" * 65)

    if is_rust_available():
        print("  Rust backend: AVAILABLE")
    else:
        print("  Rust backend: NOT AVAILABLE")
        print("  Build it: cd rust_core && maturin develop --release")
        print("  Showing Python-only results for now.")

    run_comparison("Small (inference-like)", 16, 4, 50)
    run_comparison("Medium (mixed workload)", 32, 8, 50)
    run_comparison("Large (stress test)", 64, 8, 30)

    print(f"\n{'='*65}")
    if is_rust_available():
        print("  Rust core is live. Production-ready prediction engine.")
    else:
        print("  Build the Rust core to see speedup numbers:")
        print("  cd rust_core && maturin develop --release")
    print(f"{'='*65}")
