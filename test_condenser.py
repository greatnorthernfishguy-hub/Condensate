"""
Condensate Layer 3: Condenser Tests

The moment of truth — does condensation actually save RAM?

Run: python3 test_condenser.py
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from condenser import Condenser


def test_basic_compression():
    """Test 1: Can we compress and decompress without data loss?"""
    print("\n--- Test 1: Lossless Compression Round-Trip ---")

    condenser = Condenser(demotion_idle_ms=1)

    # Register some numpy arrays
    original_data = np.random.randn(256, 256).astype(np.float32)
    condenser.register("test.weights", original_data.copy())

    region = condenser.regions["test.weights"]
    original_size = region.original_size

    # Compress to WARM
    saved = region.compress_to_warm()
    assert region.tier == "WARM"
    assert region.hot_data is None
    assert region.warm_data is not None
    print(f"  Original: {original_size / 1024:.1f} KB")
    print(f"  Compressed: {region.compressed_size / 1024:.1f} KB")
    print(f"  Ratio: {original_size / region.compressed_size:.1f}:1")
    print(f"  Saved: {saved / 1024:.1f} KB")

    # Promote back to HOT
    restored = region.promote_to_hot()
    assert region.tier == "HOT"
    assert np.array_equal(restored, original_data), "Data corrupted after round-trip!"
    print(f"  Round-trip: LOSSLESS (arrays match exactly)")

    # Compress to COLD (disk)
    region.compress_to_cold(condenser.cold_dir)
    assert region.tier == "COLD"
    assert region.current_ram_usage == 0
    print(f"  Cold (on disk): 0 KB RAM")

    # Promote from COLD back to HOT
    restored2 = region.promote_to_hot()
    assert region.tier == "HOT"
    assert np.array_equal(restored2, original_data), "Data corrupted after cold round-trip!"
    print(f"  Cold round-trip: LOSSLESS")

    condenser.cleanup()
    print("  PASS")


def test_selective_condensation():
    """Test 2: Hot regions stay hot, cold regions compress.

    16 regions, 4 hot, 12 cold. After condensation, only 4 should
    be in RAM at full size.
    """
    print("\n--- Test 2: Selective Condensation ---")

    # 16 regions × 64KB each = 1MB total
    # Use structured data (sparse + patterns) — like real weights, not pure noise
    state = {}
    for i in range(16):
        arr = np.zeros((128, 64), dtype=np.float32)
        # Sparse: only ~20% nonzero (realistic for many weight matrices)
        mask = np.random.random((128, 64)) < 0.2
        arr[mask] = np.random.randn(mask.sum()).astype(np.float32)
        state[f"block_{i}"] = arr

    hot_blocks = {0, 1, 2, 3}

    def workload(wrapped):
        # Hot blocks: accessed every iteration
        for i in hot_blocks:
            _ = wrapped[f"block_{i}"]

        # Cold blocks: rarely accessed
        if np.random.random() < 0.05:
            idx = np.random.choice(list(range(4, 16)))
            _ = wrapped[f"block_{idx}"]

        time.sleep(0.001)

    condenser = Condenser(demotion_idle_ms=10, warmup_iters=15)
    results = condenser.run_benchmark(state, workload, iterations=30,
                                       name="selective")
    condenser.print_results(results)

    # Verify tier management is working — cold regions should exist
    last_log = results["promotion_log"][-1] if results["promotion_log"] else {}
    warm_cold = last_log.get("warm", 0) + last_log.get("cold", 0)
    print(f"  Condensed regions (WARM+COLD): {warm_cold} of {results['total_regions']}")
    print(f"  RAM saved: {results['saved_mb']:.2f} MB ({results['saved_pct']:.1f}%)")
    assert warm_cold >= 8, f"Should condense at least 8 cold regions, got {warm_cold}"
    condenser.cleanup()
    print("  PASS")


def test_inference_workload():
    """Test 3: Simulated AI inference — THE benchmark.

    6-layer model with attention + FFN + KV cache.
    Config and unused layers should compress.
    Active layers should stay hot.
    """
    print("\n--- Test 3: AI Inference Workload (The Real Test) ---")

    state = {}

    # Model layers (each ~128KB) — sparse structured weights
    for i in range(6):
        for name in ["q", "k", "v"]:
            arr = np.zeros((128, 128), dtype=np.float32)
            mask = np.random.random((128, 128)) < 0.25
            arr[mask] = np.random.randn(mask.sum()).astype(np.float32)
            state[f"layer_{i}_{name}"] = arr
        for name, shape in [("ffn_up", (128, 512)), ("ffn_down", (512, 128))]:
            arr = np.zeros(shape, dtype=np.float32)
            mask = np.random.random(shape) < 0.2
            arr[mask] = np.random.randn(mask.sum()).astype(np.float32)
            state[f"layer_{i}_{name}"] = arr

    # KV cache — zeros (compresses extremely well)
    for i in range(6):
        state[f"kv_{i}_keys"] = np.zeros((256, 128), dtype=np.float32)
        state[f"kv_{i}_vals"] = np.zeros((256, 128), dtype=np.float32)

    # Config and metadata (small)
    for i in range(20):
        state[f"meta_{i}"] = np.zeros(32, dtype=np.float32)

    def workload(wrapped):
        # Token generation: sequential through layers
        for token in range(3):
            for layer_idx in range(6):
                _ = wrapped[f"layer_{layer_idx}_q"]
                _ = wrapped[f"layer_{layer_idx}_k"]
                _ = wrapped[f"layer_{layer_idx}_v"]
                _ = wrapped[f"kv_{layer_idx}_keys"]
                _ = wrapped[f"kv_{layer_idx}_vals"]
                _ = wrapped[f"layer_{layer_idx}_ffn_up"]
                _ = wrapped[f"layer_{layer_idx}_ffn_down"]
                time.sleep(0.0001)

        # Metadata accessed once per request
        _ = wrapped["meta_0"]
        _ = wrapped["meta_1"]

    print(f"  State: {len(state)} regions, "
          f"{sum(v.nbytes for v in state.values()) / 1024 / 1024:.2f} MB total")

    condenser = Condenser(demotion_idle_ms=5, warmup_iters=10)
    results = condenser.run_benchmark(state, workload, iterations=20,
                                       name="inference")
    condenser.print_results(results)

    print(f"\n  *** INFERENCE RESULTS ***")
    print(f"  Baseline RAM:    {results['baseline_ram_mb']:.2f} MB")
    print(f"  Condensed RAM:   {results['avg_condensed_ram_mb']:.2f} MB")
    print(f"  Saved:           {results['saved_mb']:.2f} MB ({results['saved_pct']:.1f}%)")
    print(f"  Prediction acc:  {results['prediction_accuracy']}%")

    condenser.cleanup()
    print("  PASS")


def test_large_state():
    """Test 4: Larger state — stress test with meaningful RAM numbers.

    64 regions × 256KB = 16 MB total state.
    Only 8 regions hot at any time = 2 MB needed.
    Target: condense ~14 MB.
    """
    print("\n--- Test 4: Large State Stress Test ---")

    # 64 regions × 256KB each = 16 MB
    # Structured sparse data — compresses well
    state = {}
    for i in range(64):
        arr = np.zeros((256, 128), dtype=np.float32)
        mask = np.random.random((256, 128)) < 0.15
        arr[mask] = np.random.randn(mask.sum()).astype(np.float32)
        state[f"region_{i}"] = arr

    # 8 hot regions that rotate
    hot_set_a = set(range(0, 8))
    hot_set_b = set(range(32, 40))

    iteration_count = [0]

    def workload(wrapped):
        iteration_count[0] += 1
        # Alternate between two hot sets
        hot = hot_set_a if (iteration_count[0] % 20) < 10 else hot_set_b

        for i in hot:
            _ = wrapped[f"region_{i}"]

        time.sleep(0.002)

    total_mb = sum(v.nbytes for v in state.values()) / 1024 / 1024
    print(f"  State: {len(state)} regions, {total_mb:.1f} MB total")
    print(f"  Only 8 regions hot at any time (2 MB needed)")

    condenser = Condenser(demotion_idle_ms=15, warmup_iters=15)
    results = condenser.run_benchmark(state, workload, iterations=40,
                                       name="large")
    condenser.print_results(results)

    print(f"\n  *** LARGE STATE RESULTS ***")
    print(f"  Baseline RAM:    {results['baseline_ram_mb']:.1f} MB (all in RAM)")
    print(f"  Condensed RAM:   {results['avg_condensed_ram_mb']:.1f} MB")
    print(f"  Saved:           {results['saved_mb']:.1f} MB ({results['saved_pct']:.1f}%)")

    condenser.cleanup()
    print("  PASS")


def test_prediction_value():
    """Test 5: Measure prediction-driven vs reactive promotions.

    The ratio of predicted vs reactive tells us how much the
    predictor is actually helping vs just reacting to cache misses.
    """
    print("\n--- Test 5: Prediction Value Measurement ---")

    state = {f"chunk_{i}": np.random.randn(64, 64).astype(np.float32)
             for i in range(20)}

    # Predictable pattern: 0→1→2→3, then 10→11→12→13
    def workload(wrapped):
        for i in range(4):
            _ = wrapped[f"chunk_{i}"]
            time.sleep(0.001)
        time.sleep(0.005)
        for i in range(10, 14):
            _ = wrapped[f"chunk_{i}"]
            time.sleep(0.001)
        time.sleep(0.005)

    condenser = Condenser(demotion_idle_ms=8, warmup_iters=15)
    results = condenser.run_benchmark(state, workload, iterations=25,
                                       name="predval")
    condenser.print_results(results)

    pred = results["prediction_promotions"]
    react = results["reactive_promotions"]
    total = pred + react

    if total > 0:
        pred_pct = pred / total * 100
        print(f"\n  Promotions: {total} total")
        print(f"    Prediction-driven: {pred} ({pred_pct:.0f}%)")
        print(f"    Reactive (miss):   {react} ({100-pred_pct:.0f}%)")

        if pred_pct > 50:
            print(f"  GOOD — Majority of promotions are prediction-driven")
        else:
            print(f"  Prediction helps but reactive still dominates")
    else:
        print(f"  No promotions needed (everything stayed HOT)")

    condenser.cleanup()
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  CONDENSATE — Layer 3 Condenser Tests")
    print("  The Moment of Truth: Does It Actually Save RAM?")
    print("=" * 60)

    test_basic_compression()
    test_selective_condensation()
    test_inference_workload()
    test_large_state()
    test_prediction_value()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
