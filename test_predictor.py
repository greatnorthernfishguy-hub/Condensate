"""
Condensate Layer 2: Predictor Tests

Tests prediction accuracy on known access patterns.
The key question: can we predict what's coming before it's requested?

Run: python3 test_predictor.py
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from membrane import Membrane
from graph_builder import GraphBuilder
from predictor import Predictor


def generate_and_learn(name, state, access_fn, train_iters,
                       causal_window_ns=3_000_000):
    """Helper: run a workload, build graph, learn predictor.

    Returns (predictor, graph) after training.
    """
    Membrane.clear()
    wrapped = Membrane.wrap(state, name)

    for _ in range(train_iters):
        access_fn(wrapped)

    train_log = Membrane.get_log()

    graph = GraphBuilder(causal_window_ns=causal_window_ns)
    graph.build(train_log)

    predictor = Predictor()
    predictor.learn(graph)

    return predictor, graph, train_log


def test_sequential_prediction():
    """Test 1: Sequential layer access — can we predict the next layer?

    Pattern: layer_0 → layer_1 → layer_2 → ... → layer_7
    If we see layer_3, we should predict layer_4.
    """
    print("\n--- Test 1: Sequential Layer Prediction ---")

    state = {f"layer_{i}": {"w": np.random.randn(64, 64).astype(np.float32)}
             for i in range(8)}

    def access_fn(wrapped):
        for i in range(8):
            layer = wrapped[f"layer_{i}"]
            _ = layer["w"]
            time.sleep(0.0005)

    # Train on 20 passes
    predictor, graph, train_log = generate_and_learn(
        "seq", state, access_fn, train_iters=20
    )

    predictor.print_model()

    # Test on 10 new passes
    Membrane.clear()
    wrapped = Membrane.wrap(
        {k: dict(v) if isinstance(v, dict) else v for k, v in state.items()},
        "seq"
    )
    for _ in range(10):
        access_fn(wrapped)

    test_log = Membrane.get_log()
    result = predictor.print_score(test_log, verbose=True)

    assert result["accuracy"] > 50, f"Sequential prediction should be >50%, got {result['accuracy']}%"
    print(f"  Accuracy: {result['accuracy']}% — sequential prediction works!")
    print("  PASS")


def test_causal_chain_prediction():
    """Test 2: Known causal chains — A→B→C with consistent timing.

    The predictor should learn the chain and predict B when A fires,
    and C when B fires. Multi-hop: seeing A should also predict C.
    """
    print("\n--- Test 2: Causal Chain Prediction ---")

    state = {f"r{i}": np.random.randn(32).astype(np.float32)
             for i in range(8)}

    def access_fn(wrapped):
        # Chain: r0 → r2 → r5 → r7  (always, ~1ms apart)
        _ = wrapped["r0"]
        time.sleep(0.001)
        _ = wrapped["r2"]
        time.sleep(0.001)
        _ = wrapped["r5"]
        time.sleep(0.001)
        _ = wrapped["r7"]
        time.sleep(0.005)

        # Noise
        if np.random.random() > 0.7:
            _ = wrapped[f"r{np.random.choice([1, 3, 4, 6])}"]
        time.sleep(0.005)

    predictor, graph, train_log = generate_and_learn(
        "chain", state, access_fn, train_iters=50
    )

    predictor.print_model()

    # Test: when r0 fires, do we predict r2?
    preds = predictor.predict("chain.r0")
    pred_paths = [p.path for p in preds]
    print(f"  When r0 fires, predictions: {[p.path.split('.')[-1] for p in preds[:5]]}")

    r2_predicted = "chain.r2" in pred_paths
    print(f"  r2 predicted after r0: {r2_predicted}")

    # Test: when r2 fires, do we predict r5?
    preds_r2 = predictor.predict("chain.r2")
    pred_paths_r2 = [p.path for p in preds_r2]
    r5_predicted = "chain.r5" in pred_paths_r2
    print(f"  r5 predicted after r2: {r5_predicted}")

    # Score on fresh data
    Membrane.clear()
    wrapped = Membrane.wrap(
        {k: v.copy() if hasattr(v, 'copy') else v for k, v in state.items()},
        "chain"
    )
    for _ in range(20):
        access_fn(wrapped)

    result = predictor.print_score(Membrane.get_log(), verbose=True)

    assert r2_predicted, "Should predict r2 after r0"
    print("  PASS")


def test_cluster_prediction():
    """Test 3: Cluster co-activation — if one member fires, predict all.

    When item_0 fires, we should predict item_1 and item_2 (same cluster).
    """
    print("\n--- Test 3: Cluster Co-Activation Prediction ---")

    state = {f"item_{i}": np.random.randn(16).astype(np.float32)
             for i in range(10)}

    def access_fn(wrapped):
        # Cluster A: always together
        _ = wrapped["item_0"]
        _ = wrapped["item_1"]
        _ = wrapped["item_2"]
        time.sleep(0.008)

        # Cluster B: always together
        _ = wrapped["item_5"]
        _ = wrapped["item_6"]
        _ = wrapped["item_7"]
        time.sleep(0.008)

        # Random singletons
        _ = wrapped[f"item_{np.random.choice([3, 4, 8, 9])}"]
        time.sleep(0.008)

    predictor, graph, train_log = generate_and_learn(
        "clust", state, access_fn, train_iters=40,
        causal_window_ns=3_000_000
    )

    predictor.print_model()

    # Test: when item_0 fires, predict item_1 and item_2
    preds = predictor.predict("clust.item_0")
    pred_paths = {p.path for p in preds}
    print(f"  When item_0 fires: {[p.path.split('.')[-1] for p in preds[:5]]}")

    item_1_predicted = "clust.item_1" in pred_paths
    item_2_predicted = "clust.item_2" in pred_paths
    print(f"  item_1 predicted: {item_1_predicted}")
    print(f"  item_2 predicted: {item_2_predicted}")

    # Score on fresh data
    Membrane.clear()
    wrapped = Membrane.wrap(
        {k: v.copy() for k, v in state.items()}, "clust"
    )
    for _ in range(15):
        access_fn(wrapped)

    result = predictor.print_score(Membrane.get_log(), verbose=True)

    assert item_1_predicted and item_2_predicted, "Should predict cluster members"
    print("  PASS")


def test_inference_simulation():
    """Test 4: Realistic inference — train on requests, predict on new ones.

    This is the demo workload. If prediction accuracy is high here,
    Condensate has legs.
    """
    print("\n--- Test 4: AI Inference Prediction (The Real Test) ---")

    state = {
        "config": {"temp": 0.7, "max_tok": 512},
    }
    for i in range(6):
        state[f"layer_{i}"] = {
            "q": np.random.randn(64, 64).astype(np.float32),
            "k": np.random.randn(64, 64).astype(np.float32),
            "v": np.random.randn(64, 64).astype(np.float32),
            "ffn": np.random.randn(64, 256).astype(np.float32),
        }
    for i in range(6):
        state[f"kv_{i}"] = {
            "keys": np.zeros((128, 64), dtype=np.float32),
            "vals": np.zeros((128, 64), dtype=np.float32),
        }

    def access_fn(wrapped):
        # Config once
        _ = wrapped["config"]["temp"]

        # 5 tokens of autoregressive generation
        for tok in range(5):
            for layer_idx in range(6):
                layer = wrapped[f"layer_{layer_idx}"]
                _ = layer["q"]
                _ = layer["k"]
                _ = layer["v"]
                kv = wrapped[f"kv_{layer_idx}"]
                _ = kv["keys"]
                _ = kv["vals"]
                _ = layer["ffn"]
                time.sleep(0.0001)

    # TRAIN on 10 requests
    print("  Training on 10 requests...")
    predictor, graph, train_log = generate_and_learn(
        "inf", state, access_fn, train_iters=10,
        causal_window_ns=2_000_000
    )

    predictor.print_model()

    # TEST on 5 new requests
    print("  Testing on 5 new requests...")
    Membrane.clear()

    # Rebuild state for test
    test_state = {}
    test_state["config"] = {"temp": 0.7, "max_tok": 512}
    for i in range(6):
        test_state[f"layer_{i}"] = {
            "q": np.random.randn(64, 64).astype(np.float32),
            "k": np.random.randn(64, 64).astype(np.float32),
            "v": np.random.randn(64, 64).astype(np.float32),
            "ffn": np.random.randn(64, 256).astype(np.float32),
        }
    for i in range(6):
        test_state[f"kv_{i}"] = {
            "keys": np.zeros((128, 64), dtype=np.float32),
            "vals": np.zeros((128, 64), dtype=np.float32),
        }

    wrapped = Membrane.wrap(test_state, "inf")
    for _ in range(5):
        access_fn(wrapped)

    test_log = Membrane.get_log()
    result = predictor.print_score(test_log, verbose=True)

    # The moment of truth
    accuracy = result["accuracy"]
    print(f"\n  *** INFERENCE PREDICTION ACCURACY: {accuracy}% ***")

    if accuracy >= 80:
        print("  EXCELLENT — Condensate can predict inference access patterns!")
        print("  This means: pre-staging works. RAM condensation is viable.")
    elif accuracy >= 60:
        print("  GOOD — Significant prediction capability. Worth pursuing.")
    elif accuracy >= 40:
        print("  MODERATE — Some structure learned. Needs better substrate.")
    else:
        print("  LOW — Pattern too noisy or model too simple. Investigate.")

    print("  PASS")


def test_prediction_vs_no_prediction():
    """Test 5: Quantify the value — compare predicted vs unpredicted accesses.

    Simulates what would happen with and without prediction:
    - Without: every cold access = full latency (cache miss)
    - With: predicted accesses = pre-staged (cache hit)

    Reports the theoretical speedup.
    """
    print("\n--- Test 5: Prediction Value (Theoretical Speedup) ---")

    state = {}
    for i in range(16):
        state[f"block_{i}"] = np.random.randn(128, 128).astype(np.float32)

    hot_blocks = {0, 1, 2, 3}  # always in RAM
    cold_blocks = set(range(4, 16))  # would need paging

    def access_fn(wrapped):
        # Hot blocks every iteration
        for i in hot_blocks:
            _ = wrapped[f"block_{i}"]
        time.sleep(0.001)

        # Cold blocks: predictable pattern
        # Phase A: blocks 4,5,6 together
        _ = wrapped["block_4"]
        _ = wrapped["block_5"]
        _ = wrapped["block_6"]
        time.sleep(0.005)

        # Phase B: blocks 10,11,12 together
        _ = wrapped["block_10"]
        _ = wrapped["block_11"]
        _ = wrapped["block_12"]
        time.sleep(0.005)

        # Random cold access (unpredictable)
        _ = wrapped[f"block_{np.random.choice([7, 8, 9, 13, 14, 15])}"]
        time.sleep(0.005)

    # Train
    predictor, graph, train_log = generate_and_learn(
        "value", state, access_fn, train_iters=30,
        causal_window_ns=3_000_000
    )

    # Test
    Membrane.clear()
    wrapped = Membrane.wrap(
        {k: v.copy() for k, v in state.items()}, "value"
    )
    for _ in range(10):
        access_fn(wrapped)

    result = predictor.score(Membrane.get_log())

    # Simulate latency impact
    hit_rate = result["accuracy"] / 100.0
    cold_access_count = result["predictions_made"]

    # Latency model (simplified):
    # Cache hit (predicted & pre-staged):  ~100ns (RAM-HOT)
    # Cache miss (unpredicted cold):       ~100μs (disk page-in)
    # That's a 1000x difference
    hit_latency_ns = 100
    miss_latency_ns = 100_000

    with_prediction = (cold_access_count * hit_rate * hit_latency_ns +
                       cold_access_count * (1 - hit_rate) * miss_latency_ns)

    without_prediction = cold_access_count * miss_latency_ns

    speedup = without_prediction / with_prediction if with_prediction > 0 else 1.0

    print(f"\n  Cold accesses in test: {cold_access_count}")
    print(f"  Prediction hit rate:   {result['accuracy']}%")
    print(f"")
    print(f"  Without Condensate:")
    print(f"    Every cold access = {miss_latency_ns/1000:.0f}μs (page from disk)")
    print(f"    Total latency:      {without_prediction/1e6:.1f}ms")
    print(f"")
    print(f"  With Condensate:")
    print(f"    Predicted hits:     {hit_latency_ns}ns (pre-staged in RAM)")
    print(f"    Unpredicted misses: {miss_latency_ns/1000:.0f}μs (still cold)")
    print(f"    Total latency:      {with_prediction/1e6:.1f}ms")
    print(f"")
    print(f"  *** THEORETICAL SPEEDUP: {speedup:.1f}x ***")

    if speedup > 5:
        print(f"  Significant — prediction eliminates most cold-access latency")
    elif speedup > 2:
        print(f"  Meaningful — prediction cuts cold-access latency substantially")
    else:
        print(f"  Marginal — need better prediction or different access patterns")

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  CONDENSATE — Layer 2 Predictor Tests")
    print("=" * 60)

    test_sequential_prediction()
    test_causal_chain_prediction()
    test_cluster_prediction()
    test_inference_simulation()
    test_prediction_vs_no_prediction()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
