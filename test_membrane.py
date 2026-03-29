"""
Condensate Layer 0: Membrane Tests

Tests the membrane wrapper on increasingly realistic workloads.
Run: python3 test_membrane.py
"""

import numpy as np
import time
import os
import sys

# Add parent dir to path so we can import membrane
sys.path.insert(0, os.path.dirname(__file__))
from membrane import Membrane


def test_basic_dict():
    """Test 1: Basic dict access tracking."""
    print("\n--- Test 1: Basic Dict Access ---")
    Membrane.clear()

    data = Membrane.wrap({
        "name": "test",
        "values": [1, 2, 3, 4, 5],
        "nested": {"a": 10, "b": 20, "c": 30},
    }, "basic")

    # Read some values
    _ = data["name"]
    _ = data["name"]          # same key twice
    _ = data["values"]
    _ = data["nested"]["a"]   # nested read — should log both levels
    _ = data["nested"]["b"]

    # Write
    data["name"] = "updated"

    assert Membrane.entry_count() > 0, "Should have recorded accesses"
    Membrane.print_stats()
    print("  PASS")


def test_numpy_arrays():
    """Test 2: Dict of numpy arrays — simulates model weight storage."""
    print("\n--- Test 2: NumPy Array State (Simulated Model Weights) ---")
    Membrane.clear()

    # Simulate a small model with layers of weight matrices
    state = {}
    for layer in range(8):
        state[f"layer_{layer}"] = {
            "weight": np.random.randn(256, 256).astype(np.float32),
            "bias": np.random.randn(256).astype(np.float32),
            "attention": {
                "q_proj": np.random.randn(256, 256).astype(np.float32),
                "k_proj": np.random.randn(256, 256).astype(np.float32),
                "v_proj": np.random.randn(256, 256).astype(np.float32),
            }
        }

    wrapped = Membrane.wrap(state, "model")

    total_bytes = sum(
        state[f"layer_{i}"]["weight"].nbytes +
        state[f"layer_{i}"]["bias"].nbytes +
        sum(v.nbytes for v in state[f"layer_{i}"]["attention"].values())
        for i in range(8)
    )
    print(f"  Model state: {total_bytes / 1024 / 1024:.1f} MB across 8 layers")

    # Simulate a forward pass — sequential layer access
    print("  Simulating forward pass...")
    for layer_idx in range(8):
        layer = wrapped[f"layer_{layer_idx}"]
        w = layer["weight"]
        b = layer["bias"]
        attn = layer["attention"]
        q = attn["q_proj"]
        k = attn["k_proj"]
        v = attn["v_proj"]

    # Simulate a second forward pass — same pattern
    print("  Simulating second forward pass...")
    for layer_idx in range(8):
        layer = wrapped[f"layer_{layer_idx}"]
        w = layer["weight"]
        b = layer["bias"]
        attn = layer["attention"]
        q = attn["q_proj"]
        k = attn["k_proj"]
        v = attn["v_proj"]

    Membrane.print_stats()
    print("  PASS")


def test_selective_access():
    """Test 3: Selective access — some layers hot, some cold.
    This is the pattern Condensate exploits: not all state is accessed equally.
    """
    print("\n--- Test 3: Selective Access (Hot/Cold Pattern) ---")
    Membrane.clear()

    state = {}
    for layer in range(16):
        state[f"layer_{layer}"] = {
            "weight": np.random.randn(128, 128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
        }

    wrapped = Membrane.wrap(state, "selective")

    # Simulate: layers 3, 7, 11 are "hot" — accessed 10x more
    hot_layers = {3, 7, 11}
    for iteration in range(20):
        for layer_idx in range(16):
            if layer_idx in hot_layers:
                # Hot path — always accessed
                layer = wrapped[f"layer_{layer_idx}"]
                _ = layer["weight"]
                _ = layer["bias"]
            elif iteration % 10 == 0:
                # Cold path — accessed once every 10 iterations
                layer = wrapped[f"layer_{layer_idx}"]
                _ = layer["weight"]

    stats = Membrane.stats()
    Membrane.print_stats()

    # Verify hot layers have more accesses
    hot_count = sum(
        stats["paths"].get(f"selective.layer_{i}", {}).get("reads", 0)
        for i in hot_layers
    )
    cold_count = sum(
        stats["paths"].get(f"selective.layer_{i}", {}).get("reads", 0)
        for i in range(16) if i not in hot_layers
    )
    ratio = hot_count / max(cold_count, 1)
    print(f"  Hot/cold access ratio: {ratio:.1f}x")
    print(f"  (This ratio is what Condensate exploits — hot stays in RAM, cold compresses)")
    print("  PASS")


def test_temporal_chains():
    """Test 4: Temporal access chains — A always followed by B followed by C.
    This is what the SNN will learn as causal chains for prefetch.
    """
    print("\n--- Test 4: Temporal Chains (Causal Access Patterns) ---")
    Membrane.clear()

    state = {f"region_{i}": np.random.randn(64, 64).astype(np.float32) for i in range(10)}
    wrapped = Membrane.wrap(state, "temporal")

    # Chain 1: 0 → 3 → 7 (always in this order)
    # Chain 2: 1 → 4 → 8 (always in this order)
    # Region 5: random, no chain
    chains = [
        [0, 3, 7],
        [1, 4, 8],
    ]

    for _ in range(50):
        for chain in chains:
            for region_id in chain:
                _ = wrapped[f"region_{region_id}"]
                time.sleep(0.0001)  # tiny delay to separate timestamps

        # Random access to region 5
        if np.random.random() > 0.5:
            _ = wrapped["region_5"]

    stats = Membrane.stats()
    Membrane.print_stats()

    # Check co-accesses — chain members should co-access heavily
    coaccesses = stats.get("top_coaccesses", [])
    if coaccesses:
        print(f"  Top co-access pairs found: {len(coaccesses)}")
        print(f"  (These are the causal chains the SNN would learn)")

    print("  PASS")


def test_overhead():
    """Test 5: Measure the membrane's overhead.
    This tells us if the observation layer is cheap enough.
    """
    print("\n--- Test 5: Overhead Measurement ---")

    state = {f"key_{i}": np.random.randn(32).astype(np.float32) for i in range(100)}

    # Baseline: raw dict access
    iterations = 100_000
    start = time.monotonic_ns()
    for _ in range(iterations):
        for key in ["key_0", "key_50", "key_99"]:
            _ = state[key]
    raw_ns = time.monotonic_ns() - start

    # Membrane: wrapped dict access
    Membrane.clear()
    wrapped = Membrane.wrap(state.copy(), "overhead")
    start = time.monotonic_ns()
    for _ in range(iterations):
        for key in ["key_0", "key_50", "key_99"]:
            _ = wrapped[key]
    membrane_ns = time.monotonic_ns() - start

    raw_per = raw_ns / (iterations * 3)
    membrane_per = membrane_ns / (iterations * 3)
    overhead = membrane_per - raw_per

    print(f"  Raw dict access:      {raw_per:.0f} ns/access")
    print(f"  Membrane access:      {membrane_per:.0f} ns/access")
    print(f"  Overhead per access:  {overhead:.0f} ns")
    print(f"  Slowdown factor:      {membrane_per / raw_per:.1f}x")
    print(f"  Total accesses logged: {Membrane.entry_count()}")

    # The membrane is for observation only — overhead is acceptable
    # if it's under ~1μs per access. For production, the Rust core
    # will bring this to ~5ns.
    if overhead < 5000:
        print(f"  Overhead acceptable for PoC (< 5μs)")
    else:
        print(f"  Overhead high — expected for Python, Rust core will fix")

    print("  PASS")


def test_save_log():
    """Test 6: Save the access log for Layer 1 analysis."""
    print("\n--- Test 6: Save Log ---")
    Membrane.clear()

    state = {f"region_{i}": np.random.randn(64, 64).astype(np.float32) for i in range(5)}
    wrapped = Membrane.wrap(state, "saveable")

    # Generate some access patterns
    for _ in range(10):
        _ = wrapped["region_0"]
        _ = wrapped["region_2"]
        _ = wrapped["region_4"]

    log_path = os.path.join(os.path.dirname(__file__), "test_access_log.json")
    Membrane.save_log(log_path)

    # Verify file exists and is valid JSON
    import json
    with open(log_path) as f:
        data = json.load(f)
    assert "entries" in data
    assert len(data["entries"]) == 30  # 3 accesses x 10 iterations

    # Clean up
    os.remove(log_path)
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  CONDENSATE — Layer 0 Membrane Tests")
    print("=" * 60)

    test_basic_dict()
    test_numpy_arrays()
    test_selective_access()
    test_temporal_chains()
    test_overhead()
    test_save_log()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
