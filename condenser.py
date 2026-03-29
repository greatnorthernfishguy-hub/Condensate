"""
Condensate Layer 3: The Condenser

The actual RAM reduction engine. Takes predictions from Layer 2
and manages memory tiers:

  HOT:  Full Python objects in RAM (actively accessed)
  WARM: LZ4-compressed binary in RAM (predicted-soon or recently cold)
  COLD: Serialized to disk (not predicted, not recent)

When the predictor says "region B is coming," the condenser
pre-promotes B from WARM→HOT before the access arrives.
When a region goes quiet, the condenser demotes it HOT→WARM→COLD.

This is the layer that proves RAM savings are real and measurable.

Usage:
    from condenser import Condenser

    condenser = Condenser(ram_budget_mb=50)
    condenser.learn_and_manage(state_dict, workload_fn)
    condenser.print_results()
"""

import numpy as np
import pickle
import lz4.frame
import time
import sys
import os
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from membrane import Membrane
from graph_builder import GraphBuilder
from predictor import Predictor


class MemoryRegion:
    """A managed memory region with tier tracking."""

    __slots__ = ['path', 'tier', 'hot_data', 'warm_data', 'cold_path',
                 'original_size', 'compressed_size', 'access_count',
                 'last_access_ns', 'promotions', 'demotions',
                 'prediction_hits']

    def __init__(self, path, data):
        self.path = path
        self.tier = "HOT"
        self.hot_data = data
        self.warm_data = None       # LZ4 compressed bytes
        self.cold_path = None       # disk file path
        self.original_size = self._measure(data)
        self.compressed_size = 0
        self.access_count = 0
        self.last_access_ns = time.monotonic_ns()
        self.promotions = 0
        self.demotions = 0
        self.prediction_hits = 0

    def _measure(self, data):
        """Measure actual memory footprint."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (bytes, bytearray)):
            return len(data)
        else:
            try:
                return sys.getsizeof(data)
            except TypeError:
                return 64  # fallback estimate

    def compress_to_warm(self):
        """HOT → WARM: compress data, free the original."""
        if self.tier != "HOT" or self.hot_data is None:
            return 0

        serialized = pickle.dumps(self.hot_data, protocol=pickle.HIGHEST_PROTOCOL)
        self.warm_data = lz4.frame.compress(serialized)
        self.compressed_size = len(self.warm_data)

        saved = self.original_size - self.compressed_size
        self.hot_data = None
        self.tier = "WARM"
        self.demotions += 1
        return max(saved, 0)

    def compress_to_cold(self, cold_dir):
        """WARM → COLD: write to disk, free RAM entirely."""
        if self.tier == "COLD":
            return 0

        # If still HOT, compress first
        if self.tier == "HOT":
            self.compress_to_warm()

        if self.warm_data is None:
            return 0

        # Write compressed data to disk
        safe_name = self.path.replace(".", "_").replace("/", "_")
        self.cold_path = os.path.join(cold_dir, f"{safe_name}.cold")
        with open(self.cold_path, 'wb') as f:
            f.write(self.warm_data)

        saved = self.compressed_size
        self.warm_data = None
        self.compressed_size = 0
        self.tier = "COLD"
        self.demotions += 1
        return saved

    def promote_to_hot(self):
        """WARM/COLD → HOT: decompress and restore."""
        if self.tier == "HOT":
            return self.hot_data

        if self.tier == "COLD" and self.cold_path:
            # Load from disk first
            with open(self.cold_path, 'rb') as f:
                self.warm_data = f.read()
            self.compressed_size = len(self.warm_data)
            self.tier = "WARM"

        if self.tier == "WARM" and self.warm_data:
            decompressed = lz4.frame.decompress(self.warm_data)
            self.hot_data = pickle.loads(decompressed)
            self.warm_data = None
            self.compressed_size = 0
            self.tier = "HOT"
            self.promotions += 1

        return self.hot_data

    @property
    def current_ram_usage(self):
        """How much RAM this region currently uses."""
        if self.tier == "HOT":
            return self.original_size
        elif self.tier == "WARM":
            return self.compressed_size
        else:
            return 0  # on disk

    def touch(self):
        """Record an access."""
        self.access_count += 1
        self.last_access_ns = time.monotonic_ns()


class Condenser:
    """The RAM condensation engine.

    Manages memory regions across HOT/WARM/COLD tiers using
    predictions from the Layer 2 predictor to pre-stage data.
    """

    def __init__(self, ram_budget_mb=None, cold_dir=None,
                 demotion_idle_ms=50, warmup_iters=10):
        """
        Args:
            ram_budget_mb: Max RAM budget in MB. None = no limit (measure only).
            cold_dir: Directory for cold storage. None = auto temp dir.
            demotion_idle_ms: Demote to WARM after this many ms idle.
            warmup_iters: Number of iterations to observe before condensing.
        """
        self.ram_budget_bytes = int(ram_budget_mb * 1024 * 1024) if ram_budget_mb else None
        self.cold_dir = cold_dir or tempfile.mkdtemp(prefix="condensate_cold_")
        self.demotion_idle_ms = demotion_idle_ms
        self.warmup_iters = warmup_iters

        self.regions = {}           # path → MemoryRegion
        self.predictor = None
        self.graph = None

        # Metrics
        self.metrics = {
            "peak_ram_no_condensate": 0,
            "peak_ram_with_condensate": 0,
            "total_promotions": 0,
            "total_demotions": 0,
            "prediction_driven_promotions": 0,
            "reactive_promotions": 0,
            "total_ram_saved_bytes": 0,
            "access_latencies_ns": [],
            "cold_accesses_avoided": 0,
            "cold_accesses_hit": 0,
        }

    def register(self, path, data):
        """Register a memory region for management."""
        self.regions[path] = MemoryRegion(path, data)

    def _current_ram(self):
        """Total current RAM usage across all regions."""
        return sum(r.current_ram_usage for r in self.regions.values())

    def _demote_coldest(self, target_savings):
        """Demote regions to meet RAM budget. Coldest first."""
        now = time.monotonic_ns()
        saved = 0

        # Sort by last access time (oldest first)
        candidates = sorted(
            [r for r in self.regions.values() if r.tier == "HOT"],
            key=lambda r: r.last_access_ns
        )

        for region in candidates:
            if saved >= target_savings:
                break

            idle_ms = (now - region.last_access_ns) / 1_000_000
            if idle_ms < self.demotion_idle_ms * 0.5:
                continue  # too recently accessed

            saved += region.compress_to_warm()
            self.metrics["total_demotions"] += 1

        # If still over budget, push WARM to COLD
        if saved < target_savings:
            warm_candidates = sorted(
                [r for r in self.regions.values() if r.tier == "WARM"],
                key=lambda r: r.last_access_ns
            )
            for region in warm_candidates:
                if saved >= target_savings:
                    break
                saved += region.compress_to_cold(self.cold_dir)
                self.metrics["total_demotions"] += 1

        return saved

    def _enforce_budget(self):
        """Enforce RAM budget by demoting as needed."""
        if self.ram_budget_bytes is None:
            return

        current = self._current_ram()
        if current > self.ram_budget_bytes:
            overage = current - self.ram_budget_bytes
            self._demote_coldest(overage)

    def _periodic_demotion(self):
        """Demote idle regions even without budget pressure."""
        now = time.monotonic_ns()

        for region in self.regions.values():
            if region.tier == "HOT":
                idle_ms = (now - region.last_access_ns) / 1_000_000
                if idle_ms > self.demotion_idle_ms:
                    region.compress_to_warm()
                    self.metrics["total_demotions"] += 1
            elif region.tier == "WARM":
                # Push long-idle WARM to COLD (disk) for real RAM savings
                idle_ms = (now - region.last_access_ns) / 1_000_000
                if idle_ms > self.demotion_idle_ms * 3:
                    region.compress_to_cold(self.cold_dir)
                    self.metrics["total_demotions"] += 1

    def access(self, path):
        """Access a region — promote if needed, record latency.

        Returns the data.
        """
        region = self.regions.get(path)
        if region is None:
            return None

        start = time.monotonic_ns()

        if region.tier != "HOT":
            # Need to promote — was this predicted?
            region.promote_to_hot()
            self.metrics["total_promotions"] += 1
            self.metrics["reactive_promotions"] += 1

            if region.tier != "HOT":
                # Still not hot — disk failure?
                return None

        elapsed_ns = time.monotonic_ns() - start
        self.metrics["access_latencies_ns"].append(elapsed_ns)
        region.touch()

        return region.hot_data

    def pre_promote(self, path):
        """Prediction-driven promotion — pre-stage before access.

        Called by the predictor when it predicts this path will be accessed.
        """
        region = self.regions.get(path)
        if region is None:
            return

        if region.tier != "HOT":
            region.promote_to_hot()
            self.metrics["total_promotions"] += 1
            self.metrics["prediction_driven_promotions"] += 1
            self.metrics["cold_accesses_avoided"] += 1
            region.prediction_hits += 1

    def run_benchmark(self, state, workload_fn, iterations=20,
                      name="benchmark"):
        """Full benchmark: measure RAM with and without condensation.

        Runs the workload twice:
        1. Baseline: no condensation, measure peak RAM
        2. Condensed: with prediction and tier management

        Args:
            state: dict of name → data (numpy arrays, dicts, etc.)
            workload_fn: function(wrapped_state) that accesses state
            iterations: how many times to run the workload
            name: label for the wrapped state

        Returns:
            dict with benchmark results
        """
        print(f"\n  Phase 1: Baseline measurement ({self.warmup_iters} iters)...")

        # --- BASELINE: No condensation ---
        total_state_size = 0
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                total_state_size += value.nbytes
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, np.ndarray):
                        total_state_size += v.nbytes

        baseline_ram = total_state_size
        self.metrics["peak_ram_no_condensate"] = baseline_ram

        # --- LEARN: Run workload with membrane to learn patterns ---
        Membrane.clear()
        wrapped = Membrane.wrap(
            {k: v.copy() if isinstance(v, np.ndarray) else
             {k2: v2.copy() if isinstance(v2, np.ndarray) else v2
              for k2, v2 in v.items()} if isinstance(v, dict) else v
             for k, v in state.items()},
            name
        )

        for _ in range(self.warmup_iters):
            workload_fn(wrapped)

        train_log = Membrane.get_log()

        # Build graph and predictor
        self.graph = GraphBuilder(causal_window_ns=3_000_000)
        self.graph.build(train_log)

        self.predictor = Predictor()
        self.predictor.learn(self.graph)

        # Score prediction accuracy on training data
        pred_result = self.predictor.score(train_log)
        pred_accuracy = pred_result["accuracy"]

        print(f"  Prediction accuracy on training data: {pred_accuracy}%")

        # --- CONDENSE: Register all regions, run with tier management ---
        print(f"\n  Phase 2: Condensed run ({iterations} iters)...")

        # Register all leaf data as regions
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                self.register(f"{name}.{key}", value.copy())
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    path = f"{name}.{key}.{k2}"
                    if isinstance(v2, np.ndarray):
                        self.register(path, v2.copy())
                    else:
                        self.register(path, v2)

        ram_snapshots = []
        promotion_log = []

        for iteration in range(iterations):
            # Periodic demotion of idle regions
            self._periodic_demotion()
            self._enforce_budget()

            # Run workload with condensation
            Membrane.clear()

            # We simulate the workload by tracking which paths get accessed
            # and using the predictor to pre-promote
            wrapped_sim = Membrane.wrap(
                {k: v.copy() if isinstance(v, np.ndarray) else
                 {k2: v2.copy() if isinstance(v2, np.ndarray) else v2
                  for k2, v2 in v.items()} if isinstance(v, dict) else v
                 for k, v in state.items()},
                name
            )

            workload_fn(wrapped_sim)
            iter_log = Membrane.get_log()

            # Process each access: predict → pre-promote → access
            for ts, event_type, path, size_bytes in sorted(iter_log, key=lambda e: e[0]):
                # Get predictions from this access
                predictions = self.predictor.predict(path, top_k=5)

                # Pre-promote predicted regions
                for pred in predictions:
                    if pred.confidence >= 0.5:
                        self.pre_promote(pred.path)

                # Access the region (may already be HOT from prediction)
                region = self.regions.get(path)
                if region:
                    if region.tier == "HOT":
                        region.touch()
                    else:
                        self.access(path)
                        self.metrics["cold_accesses_hit"] += 1

            # Snapshot RAM usage
            current_ram = self._current_ram()
            ram_snapshots.append(current_ram)

            hot_count = sum(1 for r in self.regions.values() if r.tier == "HOT")
            warm_count = sum(1 for r in self.regions.values() if r.tier == "WARM")
            cold_count = sum(1 for r in self.regions.values() if r.tier == "COLD")

            promotion_log.append({
                "iter": iteration,
                "ram_bytes": current_ram,
                "hot": hot_count,
                "warm": warm_count,
                "cold": cold_count,
            })

        # Final metrics
        min_ram = min(ram_snapshots) if ram_snapshots else baseline_ram
        avg_ram = np.mean(ram_snapshots) if ram_snapshots else baseline_ram
        self.metrics["peak_ram_with_condensate"] = max(ram_snapshots) if ram_snapshots else baseline_ram

        saved_bytes = baseline_ram - avg_ram
        saved_pct = (saved_bytes / baseline_ram * 100) if baseline_ram > 0 else 0
        self.metrics["total_ram_saved_bytes"] = int(saved_bytes)

        return {
            "baseline_ram_mb": baseline_ram / (1024 * 1024),
            "avg_condensed_ram_mb": avg_ram / (1024 * 1024),
            "min_condensed_ram_mb": min_ram / (1024 * 1024),
            "peak_condensed_ram_mb": self.metrics["peak_ram_with_condensate"] / (1024 * 1024),
            "saved_mb": saved_bytes / (1024 * 1024),
            "saved_pct": saved_pct,
            "prediction_accuracy": pred_accuracy,
            "prediction_promotions": self.metrics["prediction_driven_promotions"],
            "reactive_promotions": self.metrics["reactive_promotions"],
            "cold_accesses_avoided": self.metrics["cold_accesses_avoided"],
            "total_regions": len(self.regions),
            "ram_snapshots": ram_snapshots,
            "promotion_log": promotion_log,
        }

    def print_results(self, results):
        """Print benchmark results."""
        print(f"\n{'='*60}")
        print(f"  CONDENSATE — Layer 3 Benchmark Results")
        print(f"{'='*60}")

        print(f"\n  RAM Usage:")
        print(f"    Baseline (no condensation): {results['baseline_ram_mb']:>8.2f} MB")
        print(f"    Average condensed:          {results['avg_condensed_ram_mb']:>8.2f} MB")
        print(f"    Minimum condensed:          {results['min_condensed_ram_mb']:>8.2f} MB")
        print(f"    Peak condensed:             {results['peak_condensed_ram_mb']:>8.2f} MB")
        print(f"")
        print(f"    *** RAM SAVED: {results['saved_mb']:.2f} MB ({results['saved_pct']:.1f}%) ***")

        print(f"\n  Prediction Performance:")
        print(f"    Accuracy:                   {results['prediction_accuracy']}%")
        print(f"    Pre-staged (predicted):     {results['prediction_promotions']}")
        print(f"    Reactive (cache miss):      {results['reactive_promotions']}")
        print(f"    Cold accesses avoided:      {results['cold_accesses_avoided']}")

        print(f"\n  Region Management:")
        print(f"    Total regions:              {results['total_regions']}")

        if results.get("promotion_log"):
            last = results["promotion_log"][-1]
            print(f"    Final state:  HOT={last['hot']}  WARM={last['warm']}  COLD={last['cold']}")

        # Per-region breakdown
        print(f"\n  Per-Region Breakdown:")
        print(f"  {'Region':<35} {'Tier':>5} {'Size':>8} {'Accesses':>8} {'Promos':>6}")
        print(f"  {'-'*35} {'-'*5} {'-'*8} {'-'*8} {'-'*6}")

        sorted_regions = sorted(self.regions.values(),
                                key=lambda r: -r.access_count)
        for region in sorted_regions[:20]:
            short = region.path if len(region.path) <= 35 else "..." + region.path[-32:]
            size_kb = region.original_size / 1024
            print(f"  {short:<35} {region.tier:>5} {size_kb:>7.1f}K "
                  f"{region.access_count:>8} {region.promotions:>6}")

        if len(sorted_regions) > 20:
            print(f"  ... and {len(sorted_regions) - 20} more regions")

        # Compression ratios
        warm_regions = [r for r in self.regions.values() if r.tier == "WARM"]
        if warm_regions:
            ratios = [r.original_size / max(r.compressed_size, 1) for r in warm_regions]
            avg_ratio = np.mean(ratios)
            print(f"\n  Compression: {len(warm_regions)} WARM regions, "
                  f"avg ratio {avg_ratio:.1f}:1")

        print(f"\n{'='*60}\n")

    def cleanup(self):
        """Remove cold storage temp files."""
        import shutil
        if os.path.exists(self.cold_dir) and self.cold_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.cold_dir, ignore_errors=True)
