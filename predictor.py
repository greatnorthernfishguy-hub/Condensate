"""
Condensate Layer 2: The Predictor

Takes the graph from Layer 1 and predicts future memory accesses
based on what was just accessed. This is the proto-SNN — causal
spike propagation through learned topology.

No real SNN yet — this is a weighted graph walk that proves the
PRINCIPLE of causal prediction. The Rust/NeuroGraph SNN replaces
this with real spike dynamics later.

Usage:
    from predictor import Predictor

    predictor = Predictor()
    predictor.learn(graph)  # from GraphBuilder

    # Live prediction
    predictions = predictor.predict("model.layer_0.q")
    # Returns: [("model.layer_0.k", 0.95, 0.02), ...]
    #          (path, confidence, expected_delta_ms)

    # Score against actual access log
    predictor.score(log_entries)
"""

import numpy as np
from collections import defaultdict
import time


class PredictionEntry:
    """A single prediction: what will be accessed, when, and how sure."""

    __slots__ = ['path', 'confidence', 'expected_delta_ms', 'source_path',
                 'chain_depth']

    def __init__(self, path, confidence, expected_delta_ms, source_path,
                 chain_depth=1):
        self.path = path
        self.confidence = confidence
        self.expected_delta_ms = expected_delta_ms
        self.source_path = source_path
        self.chain_depth = chain_depth

    def __repr__(self):
        return (f"Predict({self.path}, conf={self.confidence:.2f}, "
                f"Δt={self.expected_delta_ms:.2f}ms, depth={self.chain_depth})")


class SpikeChain:
    """A learned causal chain with timing.
    Proto-SNN: spike enters at head, propagates through chain.
    """

    def __init__(self, chain_id, links):
        """
        Args:
            chain_id: unique identifier
            links: list of (path, delta_ms) tuples
                   first entry has delta_ms=0 (chain head)
        """
        self.chain_id = chain_id
        self.links = links  # [(path, cumulative_delta_ms), ...]
        self.hit_count = 0
        self.miss_count = 0

    @property
    def accuracy(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.5

    @property
    def head(self):
        return self.links[0][0] if self.links else None

    def predictions_from(self, trigger_path):
        """If trigger_path is in this chain, return predictions for what follows."""
        predictions = []
        found = False
        cumulative_ms = 0.0

        for i, (path, delta_ms) in enumerate(self.links):
            if found:
                cumulative_ms += delta_ms
                # Confidence decays with chain depth
                depth = i - trigger_idx
                confidence = self.accuracy * (0.9 ** depth)
                predictions.append(PredictionEntry(
                    path=path,
                    confidence=confidence,
                    expected_delta_ms=cumulative_ms,
                    source_path=trigger_path,
                    chain_depth=depth,
                ))
            elif path == trigger_path:
                found = True
                trigger_idx = i
                cumulative_ms = 0.0

        return predictions


class Predictor:
    """Predicts future memory accesses from learned access topology.

    This is the proto-SNN. It learns:
    1. Direct successors: A is usually followed by B (with timing)
    2. Causal chains: A → B → C (multi-hop prediction)
    3. Cluster co-activation: if any member of cluster X fires, all will

    The real SNN (NeuroGraph) replaces this with spike propagation
    through learned synapses. This proves the principle.
    """

    def __init__(self):
        # Direct successor predictions: path → [(target, weight, delta_ms)]
        self.successors = defaultdict(list)

        # Learned chains
        self.chains = []

        # Cluster membership: path → cluster_id
        self.cluster_map = {}

        # Cluster members: cluster_id → set of paths
        self.cluster_members = {}

        # Statistics
        self._total_predictions = 0
        self._hits = 0
        self._misses = 0
        self._false_positives = 0

        # Prediction window for scoring (ms)
        self.score_window_ms = 10.0

        self._learned = False

    def learn(self, graph):
        """Learn prediction model from a GraphBuilder's output.

        Args:
            graph: a built GraphBuilder instance
        """
        if not graph._built:
            raise ValueError("Graph must be built first")

        # 1. Learn direct successors from strong edges
        max_weight = max((e.weight for e in graph.edges.values()), default=1.0)

        for (src, tgt), edge in graph.edges.items():
            if edge.weight < 1.0:
                continue
            norm_weight = edge.weight / max_weight
            self.successors[src].append((
                tgt,
                norm_weight,
                edge.mean_delta_ns / 1_000_000,  # ns → ms
            ))

        # Sort successors by weight descending
        for path in self.successors:
            self.successors[path].sort(key=lambda x: -x[1])
            # Keep top 10 to avoid noise
            self.successors[path] = self.successors[path][:10]

        # 2. Learn chains
        raw_chains = graph.get_causal_chains(min_weight=2.0)
        for i, chain in enumerate(raw_chains):
            spike_chain = SpikeChain(chain_id=i, links=chain)
            self.chains.append(spike_chain)

        # 3. Learn cluster membership
        for cluster in graph.clusters:
            cid = cluster.cluster_id
            self.cluster_members[cid] = set(cluster.members)
            for member in cluster.members:
                self.cluster_map[member] = cid

        self._learned = True

    def predict(self, accessed_path, top_k=10):
        """Predict what will be accessed next, given that accessed_path was just accessed.

        Returns list of PredictionEntry, sorted by confidence descending.
        """
        if not self._learned:
            return []

        predictions = {}  # path → best PredictionEntry

        def _add(pred):
            existing = predictions.get(pred.path)
            if existing is None or pred.confidence > existing.confidence:
                predictions[pred.path] = pred

        # Source 1: Direct successors
        for target, weight, delta_ms in self.successors.get(accessed_path, []):
            _add(PredictionEntry(
                path=target,
                confidence=weight,
                expected_delta_ms=delta_ms,
                source_path=accessed_path,
                chain_depth=1,
            ))

        # Source 2: Chain propagation
        for chain in self.chains:
            chain_preds = chain.predictions_from(accessed_path)
            for pred in chain_preds:
                _add(pred)

        # Source 3: Cluster co-activation
        cluster_id = self.cluster_map.get(accessed_path)
        if cluster_id is not None:
            members = self.cluster_members[cluster_id]
            for member in members:
                if member != accessed_path:
                    _add(PredictionEntry(
                        path=member,
                        confidence=0.85,  # high confidence for cluster members
                        expected_delta_ms=0.1,  # near-immediate
                        source_path=accessed_path,
                        chain_depth=1,
                    ))

        # Sort by confidence, return top_k
        result = sorted(predictions.values(), key=lambda p: -p.confidence)
        return result[:top_k]

    def score(self, log_entries, verbose=False):
        """Score prediction accuracy against an actual access log.

        For each access in the log:
        1. Generate predictions based on current access
        2. Check if the NEXT access was predicted
        3. Track hit/miss rates

        Returns dict with accuracy metrics.
        """
        if not self._learned:
            return {"error": "Not learned yet"}

        sorted_log = sorted(log_entries, key=lambda e: e[0])

        hits = 0
        misses = 0
        predictions_made = 0
        chain_hits = 0
        cluster_hits = 0
        direct_hits = 0
        timing_errors_ms = []
        hit_details = []

        window_ns = self.score_window_ms * 1_000_000

        for i in range(len(sorted_log) - 1):
            ts_i, _, path_i, _ = sorted_log[i]

            # Generate predictions for what comes after path_i
            preds = self.predict(path_i)
            if not preds:
                continue

            predictions_made += 1
            predicted_paths = {p.path: p for p in preds}

            # Check what actually came next (within scoring window)
            hit = False
            for j in range(i + 1, len(sorted_log)):
                ts_j, _, path_j, _ = sorted_log[j]
                delta_ns = ts_j - ts_i

                if delta_ns > window_ns:
                    break

                if path_j in predicted_paths:
                    hit = True
                    pred = predicted_paths[path_j]

                    # Track timing accuracy
                    actual_delta_ms = delta_ns / 1_000_000
                    timing_error = abs(actual_delta_ms - pred.expected_delta_ms)
                    timing_errors_ms.append(timing_error)

                    # Track prediction source
                    if pred.chain_depth > 1:
                        chain_hits += 1
                    elif pred.path in self.cluster_map:
                        cluster_hits += 1
                    else:
                        direct_hits += 1

                    if verbose and len(hit_details) < 20:
                        hit_details.append({
                            "trigger": path_i,
                            "predicted": path_j,
                            "confidence": pred.confidence,
                            "expected_ms": pred.expected_delta_ms,
                            "actual_ms": actual_delta_ms,
                            "depth": pred.chain_depth,
                        })

                    break  # count first hit only

            if hit:
                hits += 1
            else:
                misses += 1

        # Update running stats
        self._total_predictions += predictions_made
        self._hits += hits
        self._misses += misses

        accuracy = hits / predictions_made if predictions_made > 0 else 0.0
        mean_timing_error = (np.mean(timing_errors_ms)
                             if timing_errors_ms else float('nan'))

        result = {
            "predictions_made": predictions_made,
            "hits": hits,
            "misses": misses,
            "accuracy": round(accuracy * 100, 1),
            "direct_hits": direct_hits,
            "chain_hits": chain_hits,
            "cluster_hits": cluster_hits,
            "mean_timing_error_ms": round(mean_timing_error, 3),
            "hit_details": hit_details if verbose else [],
        }

        return result

    def print_score(self, log_entries, verbose=False):
        """Score and print results."""
        result = self.score(log_entries, verbose=verbose)

        print(f"\n{'='*60}")
        print(f"  CONDENSATE — Layer 2 Prediction Score")
        print(f"{'='*60}")
        print(f"  Predictions made:  {result['predictions_made']}")
        print(f"  Hits:              {result['hits']}")
        print(f"  Misses:            {result['misses']}")
        print(f"  Accuracy:          {result['accuracy']}%")
        print(f"")
        print(f"  Hit breakdown:")
        print(f"    Direct successor:  {result['direct_hits']}")
        print(f"    Chain propagation: {result['chain_hits']}")
        print(f"    Cluster co-access: {result['cluster_hits']}")
        print(f"")
        print(f"  Timing precision:")
        print(f"    Mean error:        {result['mean_timing_error_ms']:.3f} ms")

        if result.get("hit_details"):
            print(f"\n  Sample hits:")
            for h in result["hit_details"][:10]:
                trig = h['trigger'].split('.')[-1]
                pred = h['predicted'].split('.')[-1]
                print(f"    {trig:<15} → {pred:<15} "
                      f"conf={h['confidence']:.2f}  "
                      f"Δt={h['actual_ms']:.2f}ms "
                      f"(predicted {h['expected_ms']:.2f}ms)")

        print(f"{'='*60}\n")

        return result

    def print_model(self):
        """Print what the predictor learned."""
        print(f"\n{'='*60}")
        print(f"  CONDENSATE — Layer 2 Learned Model")
        print(f"{'='*60}")

        print(f"\n  Direct successors: {len(self.successors)} source paths")
        top_sources = sorted(self.successors.items(),
                             key=lambda x: -len(x[1]))[:5]
        for path, succs in top_sources:
            short = path if len(path) <= 30 else "..." + path[-27:]
            print(f"    {short:<30} → {len(succs)} targets")
            for target, weight, delta in succs[:3]:
                t_short = target.split(".")[-1]
                print(f"      → {t_short:<20} w={weight:.2f}  Δt={delta:.2f}ms")

        print(f"\n  Causal chains: {len(self.chains)}")
        for chain in self.chains[:5]:
            parts = [p.split(".")[-1] for p, _ in chain.links]
            print(f"    Chain {chain.chain_id}: {' → '.join(parts[:6])}"
                  + (" → ..." if len(parts) > 6 else ""))

        print(f"\n  Clusters: {len(self.cluster_members)}")
        for cid, members in sorted(self.cluster_members.items()):
            short_members = [m.split(".")[-1] for m in sorted(members)]
            if len(short_members) > 6:
                display = ", ".join(short_members[:6]) + f" +{len(short_members)-6}"
            else:
                display = ", ".join(short_members)
            print(f"    Cluster {cid}: {{{display}}}")

        print(f"{'='*60}\n")
