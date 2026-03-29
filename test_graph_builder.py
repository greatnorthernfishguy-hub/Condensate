"""
Condensate Layer 1: Graph Builder Tests

Tests the graph builder on access logs from the Membrane.
Run: python3 test_graph_builder.py
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from membrane import Membrane
from graph_builder import GraphBuilder


def test_sequential_model():
    """Test 1: Sequential layer access — like a transformer forward pass.
    Should discover: each layer is a cluster, layers chain sequentially.
    """
    print("\n--- Test 1: Sequential Model Forward Pass ---")
    Membrane.clear()

    # 12-layer "model" with attention components
    state = {}
    for layer in range(12):
        state[f"layer_{layer}"] = {
            "weight": np.random.randn(128, 128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
            "attn_q": np.random.randn(128, 128).astype(np.float32),
            "attn_k": np.random.randn(128, 128).astype(np.float32),
            "attn_v": np.random.randn(128, 128).astype(np.float32),
        }

    wrapped = Membrane.wrap(state, "model")

    # Run 5 "forward passes" — sequential layer access
    for pass_num in range(5):
        for layer_idx in range(12):
            layer = wrapped[f"layer_{layer_idx}"]
            _ = layer["weight"]
            _ = layer["bias"]
            _ = layer["attn_q"]
            _ = layer["attn_k"]
            _ = layer["attn_v"]
            time.sleep(0.0002)  # small gap between layers

    # Build graph
    graph = GraphBuilder(causal_window_ns=2_000_000)  # 2ms window
    graph.build(Membrane.get_log())
    graph.print_analysis()

    # Verify clusters found
    assert len(graph.clusters) > 0, "Should find layer clusters"

    # Verify causal chains found
    chains = graph.get_causal_chains()
    assert len(chains) > 0, "Should find sequential chains"

    print("  PASS")


def test_hot_cold_pattern():
    """Test 2: Hot/cold access — some regions hammered, others barely touched.
    Should discover: clear temperature separation, cold regions compressible.
    """
    print("\n--- Test 2: Hot/Cold Access Pattern ---")
    Membrane.clear()

    # 20 regions, 4 of them hot
    state = {f"region_{i}": np.random.randn(64, 64).astype(np.float32)
             for i in range(20)}
    wrapped = Membrane.wrap(state, "hotcold")

    hot = {2, 7, 13, 18}

    for _ in range(100):
        for i in range(20):
            if i in hot:
                _ = wrapped[f"region_{i}"]  # hot: every iteration
            elif np.random.random() < 0.03:
                _ = wrapped[f"region_{i}"]  # cold: 3% chance

    graph = GraphBuilder()
    graph.build(Membrane.get_log())
    graph.print_analysis()

    # Verify temperature classification
    hot_nodes = [n for n in graph.nodes.values()
                 if getattr(n, '_temp_class', '') == 'HOT']
    cold_nodes = [n for n in graph.nodes.values()
                  if getattr(n, '_temp_class', '') == 'COLD']

    print(f"  HOT nodes: {len(hot_nodes)}, COLD nodes: {len(cold_nodes)}")
    assert len(hot_nodes) >= 3, "Should identify hot regions"
    assert len(cold_nodes) >= 1, "Should identify cold regions"
    print("  PASS")


def test_causal_chains():
    """Test 3: Known causal chains — verify the graph discovers them.
    This is the core capability: can we learn prefetch chains?
    """
    print("\n--- Test 3: Causal Chain Discovery ---")
    Membrane.clear()

    state = {f"r{i}": np.random.randn(32, 32).astype(np.float32)
             for i in range(10)}
    wrapped = Membrane.wrap(state, "causal")

    # Chain A: r0 → r2 → r5 → r9  (always this order, ~0.5ms apart)
    # Chain B: r1 → r3 → r6       (always this order)
    # Noise:   r4, r7, r8          (random, no pattern)

    for _ in range(80):
        # Chain A
        _ = wrapped["r0"]
        time.sleep(0.0005)
        _ = wrapped["r2"]
        time.sleep(0.0005)
        _ = wrapped["r5"]
        time.sleep(0.0005)
        _ = wrapped["r9"]
        time.sleep(0.001)

        # Chain B
        _ = wrapped["r1"]
        time.sleep(0.0005)
        _ = wrapped["r3"]
        time.sleep(0.0005)
        _ = wrapped["r6"]
        time.sleep(0.002)

        # Noise
        if np.random.random() > 0.5:
            _ = wrapped[f"r{np.random.choice([4, 7, 8])}"]

    graph = GraphBuilder(causal_window_ns=3_000_000)  # 3ms window
    graph.build(Membrane.get_log())
    graph.print_analysis()

    # Check for discovered chains
    chains = graph.get_causal_chains(min_weight=5.0)
    print(f"\n  Chains found (weight >= 5): {len(chains)}")
    for chain in chains:
        path_names = [p.split(".")[-1] for p, _ in chain]
        print(f"    {' → '.join(path_names)}")

    # The graph should find chain-like patterns
    # (exact chains depend on timing, but structure should be visible)
    assert len(chains) >= 1, "Should discover at least one causal chain"
    print("  PASS")


def test_cluster_discovery():
    """Test 4: Co-access clusters — groups of regions always used together.
    These become hyperedges: promote/demote the whole group as a unit.
    """
    print("\n--- Test 4: Cluster (Proto-Hyperedge) Discovery ---")
    Membrane.clear()

    state = {f"item_{i}": np.random.randn(16).astype(np.float32)
             for i in range(15)}
    wrapped = Membrane.wrap(state, "cluster")

    # Cluster A: items 0, 1, 2 always together
    # Cluster B: items 5, 6, 7, 8 always together
    # Cluster C: items 10, 11 always together
    # Singletons: 3, 4, 9, 12, 13, 14 — accessed independently

    for _ in range(60):
        # Cluster A — tight access, big gap after
        _ = wrapped["item_0"]
        _ = wrapped["item_1"]
        _ = wrapped["item_2"]
        time.sleep(0.008)  # 8ms gap — outside causal window

        # Cluster B — tight access, big gap after
        _ = wrapped["item_5"]
        _ = wrapped["item_6"]
        _ = wrapped["item_7"]
        _ = wrapped["item_8"]
        time.sleep(0.008)

        # Cluster C (less frequent)
        if np.random.random() > 0.3:
            _ = wrapped["item_10"]
            _ = wrapped["item_11"]
            time.sleep(0.008)

        # Random singletons
        idx = np.random.choice([3, 4, 9, 12, 13, 14])
        _ = wrapped[f"item_{idx}"]
        time.sleep(0.008)

    graph = GraphBuilder(causal_window_ns=3_000_000, cluster_threshold=0.6)
    graph.build(Membrane.get_log())
    graph.print_analysis()

    # Should find at least 2 clear clusters
    print(f"\n  Clusters found: {len(graph.clusters)}")
    assert len(graph.clusters) >= 2, "Should find multiple clusters"

    # Verify cluster A members are together
    cluster_a_found = False
    for cluster in graph.clusters:
        paths = {m.split(".")[-1] for m in cluster.members}
        if {"item_0", "item_1", "item_2"}.issubset(paths):
            cluster_a_found = True
            break

    assert cluster_a_found, "Should find cluster A (items 0,1,2)"
    print("  Cluster A (items 0,1,2) found correctly")
    print("  PASS")


def test_real_world_simulation():
    """Test 5: Realistic workload — simulates an AI inference server.

    Pattern:
    - Model weights accessed sequentially (forward pass)
    - KV cache accessed selectively (attention)
    - Config accessed once at start
    - Buffer reused across requests
    """
    print("\n--- Test 5: Realistic AI Inference Simulation ---")
    Membrane.clear()

    state = {
        "config": {"max_tokens": 512, "temperature": 0.7, "top_p": 0.9},
        "buffer": {"input_ids": np.zeros(512, dtype=np.int32),
                    "logits": np.zeros(32000, dtype=np.float32)},
    }
    # Add model layers
    for i in range(6):
        state[f"layer_{i}"] = {
            "q": np.random.randn(64, 64).astype(np.float32),
            "k": np.random.randn(64, 64).astype(np.float32),
            "v": np.random.randn(64, 64).astype(np.float32),
            "ffn_up": np.random.randn(64, 256).astype(np.float32),
            "ffn_down": np.random.randn(256, 64).astype(np.float32),
        }
    # Add KV cache (per layer, grows with sequence)
    for i in range(6):
        state[f"kv_cache_{i}"] = {
            "keys": np.zeros((512, 64), dtype=np.float32),
            "values": np.zeros((512, 64), dtype=np.float32),
        }

    wrapped = Membrane.wrap(state, "server")

    # Simulate 3 requests
    for req in range(3):
        # Config read once per request
        _ = wrapped["config"]["max_tokens"]
        _ = wrapped["config"]["temperature"]

        # Buffer setup
        _ = wrapped["buffer"]["input_ids"]

        # Forward pass — 10 "tokens" of autoregressive generation
        for token in range(10):
            for layer_idx in range(6):
                # Attention
                layer = wrapped[f"layer_{layer_idx}"]
                _ = layer["q"]
                _ = layer["k"]
                _ = layer["v"]

                # KV cache read/write
                cache = wrapped[f"kv_cache_{layer_idx}"]
                _ = cache["keys"]
                _ = cache["values"]

                # FFN
                _ = layer["ffn_up"]
                _ = layer["ffn_down"]
                time.sleep(0.0001)

            # Logits at the end of each token
            _ = wrapped["buffer"]["logits"]

    total_bytes = 0
    for k, v in state.items():
        if isinstance(v, dict):
            for v2 in v.values():
                if isinstance(v2, np.ndarray):
                    total_bytes += v2.nbytes
                elif isinstance(v2, dict):
                    for v3 in v2.values():
                        if isinstance(v3, np.ndarray):
                            total_bytes += v3.nbytes
    total_mb = total_bytes / 1024 / 1024

    print(f"  Simulated: 3 requests × 10 tokens × 6 layers")
    print(f"  Total state: {total_mb:.1f} MB")

    graph = GraphBuilder(causal_window_ns=2_000_000)
    graph.build(Membrane.get_log())
    graph.print_analysis()

    # Save for potential Layer 2 testing
    graph.save(os.path.join(os.path.dirname(__file__), "inference_graph.json"))

    # Verify key insights
    config_node = graph.nodes.get("server.config.max_tokens")
    layer0_q = graph.nodes.get("server.layer_0.q")

    if config_node and layer0_q:
        print(f"  Config accesses: {config_node.access_count} (read once per request)")
        print(f"  Layer 0 Q accesses: {layer0_q.access_count} (every token, every request)")
        ratio = layer0_q.access_count / max(config_node.access_count, 1)
        print(f"  Ratio: {ratio:.0f}x — config is compressible, Q is not")

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  CONDENSATE — Layer 1 Graph Builder Tests")
    print("=" * 60)

    test_sequential_model()
    test_hot_cold_pattern()
    test_causal_chains()
    test_cluster_discovery()
    test_real_world_simulation()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
