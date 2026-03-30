"""
Condensate: Rust-backed Predictor

Drop-in replacement for the Python predictor using the Rust core.
Falls back to Python if the Rust module isn't built.

Usage:
    from rust_predictor import get_predictor

    # Returns RustPredictor if available, Python Predictor if not
    predictor = get_predictor()
    predictor.learn(graph)
    predictions = predictor.predict("model.layer_0")
"""

import sys
import os

# Try to import the Rust module
_RUST_AVAILABLE = False
_rust_module = None

try:
    import condensate_core
    _RUST_AVAILABLE = True
    _rust_module = condensate_core
except ImportError:
    pass


class RustPredictorWrapper:
    """Wraps the Rust predictor with the same API as the Python Predictor.

    Translates between the Python GraphBuilder's data format and
    the Rust AccessGraph's format. The Rust core handles the heavy
    computation (graph build, prediction, scoring).
    """

    def __init__(self):
        if not _RUST_AVAILABLE:
            raise ImportError("condensate_core not built. Run: cd rust_core && maturin develop --release")

        self._graph = _rust_module.AccessGraph()
        self._predictor = _rust_module.RustPredictor()
        self._learned = False
        self.backend = "rust"

    def learn(self, graph):
        """Learn from a Python GraphBuilder's output.

        Extracts the raw access log and rebuilds the graph in Rust.
        This is faster than using the Python graph.
        """
        # We need the raw access log to feed the Rust graph.
        # The Python graph has it in its edges/nodes, but the Rust
        # graph wants raw events. We reconstruct them from the
        # Python graph's node access times.
        #
        # Alternative: learn directly from the Python membrane's log.
        # That's the better path — see learn_from_log().
        raise NotImplementedError(
            "Use learn_from_log() with raw membrane data instead. "
            "The Rust graph builds from raw events, not from a Python graph."
        )

    def learn_from_log(self, log_entries, causal_window_ns=5_000_000,
                       cluster_threshold=0.7):
        """Learn from raw membrane access log entries.

        This is the preferred path — feed raw events directly to
        the Rust graph builder. No Python graph needed.

        Args:
            log_entries: list of (timestamp_ns, event_type, path, size_bytes)
                         from Membrane.get_log()
            causal_window_ns: causal window for edge detection
            cluster_threshold: co-access ratio for clustering
        """
        # Convert membrane log format to Rust format
        # Membrane: (timestamp_ns, event_type, path, size_bytes)
        # Rust:     (timestamp_ns, path, size_bytes)
        events = [
            (int(ts), path, int(size))
            for ts, event_type, path, size in log_entries
        ]

        # Build Rust graph
        self._graph = _rust_module.AccessGraph(
            causal_window_ns=int(causal_window_ns),
            cluster_threshold=float(cluster_threshold),
        )
        self._graph.build(events)

        # Learn predictor from graph
        self._predictor = _rust_module.RustPredictor()
        self._predictor.learn(self._graph)
        self._learned = True

    def predict(self, path, top_k=10):
        """Predict what will be accessed next.

        Returns list of Prediction objects (from Rust) with:
          .path, .confidence, .expected_delta_ms, .source_path, .chain_depth
        """
        if not self._learned:
            return []
        return self._predictor.predict(path, top_k=top_k)

    def score(self, log_entries, verbose=False):
        """Score prediction accuracy against an access log.

        Args:
            log_entries: membrane log format (timestamp_ns, event_type, path, size_bytes)

        Returns dict with accuracy metrics.
        """
        if not self._learned:
            return {"error": "Not learned yet"}

        events = [
            (int(ts), path, int(size))
            for ts, event_type, path, size in log_entries
        ]

        result = self._predictor.score(events)

        return {
            "predictions_made": result.predictions_made,
            "hits": result.hits,
            "misses": result.misses,
            "accuracy": result.accuracy,
            "direct_hits": result.direct_hits,
            "chain_hits": result.chain_hits,
            "cluster_hits": result.cluster_hits,
        }

    def print_score(self, log_entries, verbose=False):
        """Score and print results."""
        result = self.score(log_entries, verbose)

        print(f"\n{'='*60}")
        print(f"  CONDENSATE — Rust Predictor Score")
        print(f"{'='*60}")
        print(f"  Backend: RUST (condensate_core)")
        print(f"  Predictions made:  {result['predictions_made']}")
        print(f"  Hits:              {result['hits']}")
        print(f"  Misses:            {result['misses']}")
        print(f"  Accuracy:          {result['accuracy']}%")
        print(f"  Hit breakdown:")
        print(f"    Direct successor:  {result['direct_hits']}")
        print(f"    Chain propagation: {result['chain_hits']}")
        print(f"    Cluster co-access: {result['cluster_hits']}")
        print(f"{'='*60}\n")

        return result


def get_predictor():
    """Get the best available predictor.

    Returns RustPredictorWrapper if the Rust core is built,
    falls back to Python Predictor otherwise.
    """
    if _RUST_AVAILABLE:
        return RustPredictorWrapper()
    else:
        from predictor import Predictor
        p = Predictor()
        p.backend = "python"
        return p


def is_rust_available():
    """Check if the Rust backend is available."""
    return _RUST_AVAILABLE
