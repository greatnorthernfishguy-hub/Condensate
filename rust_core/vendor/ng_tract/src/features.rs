//! Feature extraction from TopologyEntry for the encoder mesh.
//!
//! Replicates the exact feature vectors that encoder_v2.py and
//! graph_encoder.py compute from Python dicts — but reads BTF
//! TopologyEntry structs directly. No Python. No dicts. No GIL.
//!
//! Two outputs:
//! - Per-hyperedge features: (score, [f32; 8]) ranked for top-K
//! - Global summary features: [f32; 8]
//!
//! The numerical results must be identical to the Python encoders.
//!
//! # ---- Changelog ----
//! # [2026-04-04] Claude Code (Opus 4.6) — Punchlist #119 Step 4
//! #   What: Feature extraction from BTF TopologyEntry
//! #   Why:  Encoder reads BTF binary directly instead of Python dicts.
//! #         Eliminates serialization boundary. Full resolution.
//! #   How:  Replicates encoder_v2.py lines 227-236 and 320-330 exactly.
//! # -------------------

use std::collections::HashSet;

use crate::format::{
    TopologyEntry, SALIENCE_HOT_ELIGIBILITY,
};

/// Per-hyperedge feature vector (8 floats) with a ranking score.
/// Score determines top-K selection.
pub struct ScoredFeatures {
    pub score: f32,
    pub features: [f32; 8],
}

/// Extract per-hyperedge features from a TopologyEntry.
///
/// Returns (score, features) per hyperedge, plus per fired-node entries
/// when hyperedges are absent or sparse. Sorted by score descending.
///
/// Features per entry (matches encoder_v2.py lines 227-236):
/// - [0] fired flag (1.0 if fired, 0.0 otherwise)
/// - [1] member_activity (fraction of members that fired)
/// - [2] member count normalized (len / 100, clamped to 1.0)
/// - [3] output targets normalized (len / 50, clamped to 1.0)
/// - [4] activation count log normalized (log1p / 10, clamped to 1.0)
/// - [5] synapse connectivity normalized (count / 50, clamped to 1.0)
/// - [6] mean synapse weight
/// - [7] max eligibility trace clamped to 1.0
///
/// Score = fired_flag + log1p(activation_count)/10 (capped at 1.0)
/// For the current BTF format, all hyperedges in fired_hyperedges fired,
/// so fired_flag is always 1.0. Individual fired nodes get score 0.8.
pub fn extract_he_features(entry: &TopologyEntry) -> Vec<ScoredFeatures> {
    // Build fired node ID set for member_activity calculation
    let fired_node_ids: HashSet<uuid::Uuid> = entry.fired_nodes.iter()
        .map(|n| n.node_id)
        .collect();

    // Build node_id -> (connectivity, weight_sum, syn_count, max_eligibility) from fired nodes
    // For each fired node, we know its outgoing synapses from the BTF data
    let mut node_synapse_stats: std::collections::HashMap<uuid::Uuid, (u32, f32, u32, f32)> =
        std::collections::HashMap::new();
    for node in &entry.fired_nodes {
        let mut connectivity: u32 = 0;
        let mut weight_sum: f32 = 0.0;
        let mut syn_count: u32 = 0;
        let mut max_trace: f32 = 0.0;
        for syn in &node.outgoing {
            connectivity += 1;
            weight_sum += syn.weight;
            syn_count += 1;
            if syn.eligibility_trace > max_trace {
                max_trace = syn.eligibility_trace;
            }
        }
        node_synapse_stats.insert(node.node_id, (connectivity, weight_sum, syn_count, max_trace));
    }

    // Also check salience signals for hot eligibility traces
    let mut node_salience_hot: std::collections::HashMap<uuid::Uuid, f32> =
        std::collections::HashMap::new();
    for sig in &entry.salience {
        if sig.signal_type == SALIENCE_HOT_ELIGIBILITY {
            let current = node_salience_hot.entry(sig.node_id).or_insert(0.0);
            if sig.value > *current {
                *current = sig.value;
            }
        }
    }

    let mut scored: Vec<ScoredFeatures> = Vec::new();

    // Hyperedges from the topology entry
    for he in &entry.fired_hyperedges {
        let member_set: HashSet<uuid::Uuid> = he.member_node_ids.iter().copied().collect();
        let num_members = member_set.len();

        // Member activity — fraction of members that fired this step
        let member_activity = if num_members > 0 {
            let members_fired = member_set.intersection(&fired_node_ids).count();
            members_fired as f32 / num_members as f32
        } else {
            0.0
        };

        // Connectivity and weight stats from fired member nodes
        let mut connectivity: u32 = 0;
        let mut weight_sum: f32 = 0.0;
        let mut syn_count: u32 = 0;
        let mut hot_eligibility: f32 = 0.0;

        for member_id in member_set.intersection(&fired_node_ids) {
            if let Some(&(c, ws, sc, mt)) = node_synapse_stats.get(member_id) {
                connectivity += c;
                weight_sum += ws;
                syn_count += sc;
                if mt > hot_eligibility {
                    hot_eligibility = mt;
                }
            }
            // Also check salience-based hot eligibility
            if let Some(&sal_hot) = node_salience_hot.get(member_id) {
                if sal_hot > hot_eligibility {
                    hot_eligibility = sal_hot;
                }
            }
        }

        let mean_weight = if syn_count > 0 {
            weight_sum / syn_count as f32
        } else {
            0.0
        };

        let features = [
            1.0,  // fired — all hyperedges in fired_hyperedges are fired
            member_activity,
            f32::min(num_members as f32 / 100.0, 1.0),
            f32::min(he.output_target_ids.len() as f32 / 50.0, 1.0),
            f32::min((he.activation_count as f32).ln_1p() / 10.0, 1.0),
            f32::min(connectivity as f32 / 50.0, 1.0),
            mean_weight,
            f32::min(hot_eligibility, 1.0),
        ];

        // Score: fired + member_activity * 0.5 (matches encoder_v2.py line 242)
        let score = 1.0 + member_activity * 0.5;

        scored.push(ScoredFeatures { score, features });
    }

    // If few or no hyperedges, fill with individual fired nodes (matches graph_encoder.py)
    if scored.len() < entry.fired_nodes.len() {
        for node in &entry.fired_nodes {
            let outgoing = &node.outgoing;
            let mean_w = if !outgoing.is_empty() {
                outgoing.iter().map(|s| s.weight).sum::<f32>() / outgoing.len() as f32
            } else {
                0.0
            };

            let mut hot: f32 = 0.0;
            if let Some(&sal_hot) = node_salience_hot.get(&node.node_id) {
                hot = sal_hot;
            }
            for syn in outgoing {
                if syn.eligibility_trace > hot {
                    hot = syn.eligibility_trace;
                }
            }

            let features = [
                1.0,  // fired
                1.0,  // individual node = 100% member activity
                0.01, // single node
                0.0,
                0.0,  // no activation count on individual nodes
                f32::min(outgoing.len() as f32 / 50.0, 1.0),
                mean_w,
                f32::min(hot, 1.0),
            ];

            scored.push(ScoredFeatures { score: 0.8, features });
        }
    }

    // Sort by score descending
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    scored
}

/// Extract global summary features from a TopologyEntry.
///
/// Matches encoder_v2.py lines 320-330 exactly:
/// - [0] fired nodes count / 100, clamped to 1.0
/// - [1] fired hyperedges count / 20, clamped to 1.0
/// - [2] predictions confirmed / 10, clamped to 1.0
/// - [3] predictions surprised / 5, clamped to 1.0
/// - [4] synapses pruned / 10, clamped to 1.0
/// - [5] synapses sprouted / 10, clamped to 1.0
/// - [6] surprise ratio: surprised / max(confirmed + surprised, 1), clamped to 1.0
/// - [7] timestep / 10000, clamped to 1.0
pub fn extract_global_features(entry: &TopologyEntry) -> [f32; 8] {
    let confirmed = entry.predictions_confirmed as f32;
    let surprised = entry.predictions_surprised as f32;

    [
        f32::min(entry.fired_nodes.len() as f32 / 100.0, 1.0),
        f32::min(entry.fired_hyperedges.len() as f32 / 20.0, 1.0),
        f32::min(confirmed / 10.0, 1.0),
        f32::min(surprised / 5.0, 1.0),
        f32::min(entry.synapses_pruned as f32 / 10.0, 1.0),
        f32::min(entry.synapses_sprouted as f32 / 10.0, 1.0),
        f32::min(surprised / f32::max(confirmed + surprised, 1.0), 1.0),
        f32::min(entry.timestep as f32 / 10000.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::*;
    use uuid::Uuid;

    fn make_test_entry() -> TopologyEntry {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let node3 = Uuid::new_v4();

        TopologyEntry {
            timestamp: 1000.0,
            timestep: 42,
            predictions_confirmed: 3,
            predictions_surprised: 1,
            synapses_pruned: 2,
            synapses_sprouted: 5,
            fired_nodes: vec![
                FiredNode {
                    node_id: node1,
                    label: "test1".into(),
                    embedding_dim: 0,
                    embedding: vec![],
                    outgoing: vec![
                        OutgoingSynapse {
                            post_node_id: node2,
                            weight: 0.7,
                            eligibility_trace: 0.3,
                        },
                    ],
                },
                FiredNode {
                    node_id: node2,
                    label: "test2".into(),
                    embedding_dim: 0,
                    embedding: vec![],
                    outgoing: vec![],
                },
            ],
            fired_hyperedges: vec![
                FiredHyperedge {
                    hyperedge_id: Uuid::new_v4(),
                    label: "he1".into(),
                    activation_count: 10,
                    member_node_ids: vec![node1, node2],
                    output_target_ids: vec![node3],
                },
            ],
            salience: vec![],
        }
    }

    #[test]
    fn test_global_features() {
        let entry = make_test_entry();
        let global = extract_global_features(&entry);

        assert_eq!(global.len(), 8);
        // 2 fired nodes / 100 = 0.02
        assert!((global[0] - 0.02).abs() < 1e-6);
        // 1 fired hyperedge / 20 = 0.05
        assert!((global[1] - 0.05).abs() < 1e-6);
        // confirmed=3 / 10 = 0.3
        assert!((global[2] - 0.3).abs() < 1e-6);
        // surprised=1 / 5 = 0.2
        assert!((global[3] - 0.2).abs() < 1e-6);
        // pruned=2 / 10 = 0.2
        assert!((global[4] - 0.2).abs() < 1e-6);
        // sprouted=5 / 10 = 0.5
        assert!((global[5] - 0.5).abs() < 1e-6);
        // surprise ratio: 1/max(3+1,1) = 0.25
        assert!((global[6] - 0.25).abs() < 1e-6);
        // timestep=42 / 10000 = 0.0042
        assert!((global[7] - 0.0042).abs() < 1e-6);
    }

    #[test]
    fn test_he_features() {
        let entry = make_test_entry();
        let scored = extract_he_features(&entry);

        assert!(!scored.is_empty());
        // First entry should be the hyperedge (score > 0.8)
        assert!(scored[0].score > 0.8);
        // fired flag = 1.0
        assert!((scored[0].features[0] - 1.0).abs() < 1e-6);
        // member_activity: both node1 and node2 are in fired_nodes and members
        assert!((scored[0].features[1] - 1.0).abs() < 1e-6);
        // member count: 2/100 = 0.02
        assert!((scored[0].features[2] - 0.02).abs() < 1e-6);
        // output targets: 1/50 = 0.02
        assert!((scored[0].features[3] - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_empty_entry() {
        let entry = TopologyEntry {
            timestamp: 0.0,
            timestep: 0,
            predictions_confirmed: 0,
            predictions_surprised: 0,
            synapses_pruned: 0,
            synapses_sprouted: 0,
            fired_nodes: vec![],
            fired_hyperedges: vec![],
            salience: vec![],
        };

        let scored = extract_he_features(&entry);
        assert!(scored.is_empty());

        let global = extract_global_features(&entry);
        assert_eq!(global, [0.0; 8]);
    }
}
