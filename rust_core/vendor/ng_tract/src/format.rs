//! Binary Tract Format v0.1 — core types and constants.
//!
//! Every entry on a tract starts with a 24-byte envelope.
//! Embeddings are 64-byte aligned. Everything else packs normally.
//! Node IDs are raw binary UUIDs (16 bytes).

use uuid::Uuid;

// --- Constants ---

/// Magic bytes: "BT" (Binary Tract)
pub const MAGIC: [u8; 2] = [0x42, 0x54];

/// Format version
pub const VERSION: u8 = 1;

/// Entry types
pub const ENTRY_OUTCOME: u8 = 1;
pub const ENTRY_TOPOLOGY: u8 = 2;
pub const ENTRY_EXPERIENCE: u8 = 3;

/// Endianness markers
pub const ENDIAN_LE: u8 = 0x01;
pub const ENDIAN_BE: u8 = 0x02;

/// Embedding format
pub const EMB_F32: u8 = 0x01;

/// Cache-line alignment for embeddings
pub const EMBEDDING_ALIGN: usize = 64;

/// Salience signal types
pub const SALIENCE_HOT_ELIGIBILITY: u8 = 1;
pub const SALIENCE_STRUCTURAL_CHANGE: u8 = 2;
pub const SALIENCE_FIRING_RATE_DEVIATION: u8 = 3;

// --- Envelope ---

/// 24-byte entry envelope. Starts every entry on the tract.
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    pub magic: [u8; 2],        // 0x4254
    pub version: u8,           // 1
    pub entry_type: u8,        // 1=outcome, 2=topology
    pub total_length: u32,     // bytes including envelope
    pub timestamp: f64,        // Unix seconds, microsecond precision
    pub checksum: u32,         // CRC32 of payload
    pub endianness: u8,        // 0x01=LE, 0x02=BE
    pub reserved: [u8; 3],     // must be 0
}

impl Envelope {
    pub const SIZE: usize = 24;

    pub fn new(entry_type: u8, total_length: u32, timestamp: f64, checksum: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            entry_type,
            total_length,
            timestamp,
            checksum,
            endianness: if cfg!(target_endian = "little") { ENDIAN_LE } else { ENDIAN_BE },
            reserved: [0; 3],
        }
    }

    /// Check if raw bytes start with BTF magic (vs JSONL which starts with '{')
    pub fn is_btf(bytes: &[u8]) -> bool {
        bytes.len() >= 2 && bytes[0] == MAGIC[0] && bytes[1] == MAGIC[1]
    }
}

// --- Outcome Entry ---

/// An outcome entry — what a module deposits when it learns something.
/// The bulk of River traffic.
#[derive(Debug, Clone)]
pub struct OutcomeEntry {
    pub timestamp: f64,
    pub module_id: String,
    pub target_id: String,
    pub success: bool,
    pub embedding_dim: u16,
    pub embedding: Vec<f32>,
    pub metadata: Vec<u8>,  // MessagePack encoded
}

// --- Topology Entry ---

/// A topology entry — raw output of graph.step() flowing downstream.
/// This is Tier 3 data.
#[derive(Debug, Clone)]
pub struct TopologyEntry {
    pub timestamp: f64,
    pub timestep: u32,
    pub predictions_confirmed: u16,
    pub predictions_surprised: u16,
    pub synapses_pruned: u16,
    pub synapses_sprouted: u16,
    pub fired_nodes: Vec<FiredNode>,
    pub fired_hyperedges: Vec<FiredHyperedge>,
    pub salience: Vec<SalienceSignal>,
}

/// A node that fired this step, with its causal context.
#[derive(Debug, Clone)]
pub struct FiredNode {
    pub node_id: Uuid,
    pub label: String,
    pub embedding_dim: u16,
    pub embedding: Vec<f32>,       // 768 f32s typical, empty if absent
    pub outgoing: Vec<OutgoingSynapse>,
}

/// An outgoing synapse from a fired node — the causal connection.
/// 24 bytes fixed: 16 (uuid) + 4 (weight) + 4 (trace)
#[derive(Debug, Clone, Copy)]
pub struct OutgoingSynapse {
    pub post_node_id: Uuid,
    pub weight: f32,
    pub eligibility_trace: f32,
}

/// A hyperedge that activated this step.
#[derive(Debug, Clone)]
pub struct FiredHyperedge {
    pub hyperedge_id: Uuid,
    pub label: String,
    pub activation_count: u32,
    pub member_node_ids: Vec<Uuid>,
    pub output_target_ids: Vec<Uuid>,
}

/// A salience signal — raw, the bucket interprets.
/// 28 bytes fixed.
#[derive(Debug, Clone, Copy)]
pub struct SalienceSignal {
    pub signal_type: u8,
    pub node_id: Uuid,  // zeroed if N/A
    pub value: f32,
    pub reference: f32,
}

// --- Experience Entry ---

/// Raw experience — text/file/url content flowing from feeders to the
/// topology owner.  No embedding, no classification, no transformation.
/// The content enters as raw bytes and stays that way until extraction.
#[derive(Debug, Clone)]
pub struct ExperienceEntry {
    pub timestamp: f64,
    pub source: String,        // "gui", "feed-syl", "mcp", etc.
    pub content_type: String,  // "text", "file", "url"
    pub content: Vec<u8>,      // raw content bytes — the experience itself
}

// --- Entry enum for the drain API ---

/// A typed entry read from a tract.
#[derive(Debug, Clone)]
pub enum TractEntry {
    Outcome(OutcomeEntry),
    Topology(TopologyEntry),
    Experience(ExperienceEntry),
}
