//! BTF writer — serializes entries to binary tract format.
//!
//! Embeddings are padded to 64-byte alignment (hot data).
//! Everything else packs normally (cold metadata).

use crate::format::*;
// write module — no std::io::Write needed, we build Vec<u8> directly

/// Deposit raw bytes to a tract file with flock(LOCK_EX) + append.
///
/// Replicates the exact semantics of Python's `_deposit_to_tract()`:
/// open with O_WRONLY|O_CREAT|O_APPEND, exclusive flock, write, unlock.
pub fn deposit_to_file(path: &str, data: &[u8]) -> std::io::Result<()> {
    use std::ffi::CString;

    let c_path = CString::new(path)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid path"))?;

    unsafe {
        let fd = libc::open(
            c_path.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND,
            0o664,
        );
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // Acquire exclusive lock
        if libc::flock(fd, libc::LOCK_EX) != 0 {
            let err = std::io::Error::last_os_error();
            libc::close(fd);
            return Err(err);
        }

        // Write all data
        let mut written = 0usize;
        while written < data.len() {
            let n = libc::write(
                fd,
                data[written..].as_ptr() as *const libc::c_void,
                data.len() - written,
            );
            if n < 0 {
                let err = std::io::Error::last_os_error();
                libc::flock(fd, libc::LOCK_UN);
                libc::close(fd);
                return Err(err);
            }
            written += n as usize;
        }

        // Unlock and close
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }

    Ok(())
}

/// Calculate padding needed to reach target alignment from current position.
fn align_padding(current_offset: usize, alignment: usize) -> usize {
    let remainder = current_offset % alignment;
    if remainder == 0 { 0 } else { alignment - remainder }
}

/// Write an outcome entry to a byte buffer.
pub fn write_outcome(entry: &OutcomeEntry) -> Vec<u8> {
    // Pre-calculate payload size to determine total_length
    let payload = build_outcome_payload(entry);
    let total_length = (Envelope::SIZE + payload.len()) as u32;
    let checksum = crc32fast::hash(&payload);

    let envelope = Envelope::new(ENTRY_OUTCOME, total_length, entry.timestamp, checksum);

    let mut buf = Vec::with_capacity(total_length as usize);
    write_envelope(&mut buf, &envelope);
    buf.extend_from_slice(&payload);
    buf
}

/// Write a topology entry to a byte buffer.
pub fn write_topology(entry: &TopologyEntry) -> Vec<u8> {
    let payload = build_topology_payload(entry);
    let total_length = (Envelope::SIZE + payload.len()) as u32;
    let checksum = crc32fast::hash(&payload);

    let envelope = Envelope::new(ENTRY_TOPOLOGY, total_length, entry.timestamp, checksum);

    let mut buf = Vec::with_capacity(total_length as usize);
    write_envelope(&mut buf, &envelope);
    buf.extend_from_slice(&payload);
    buf
}

// --- Envelope ---

fn write_envelope(buf: &mut Vec<u8>, env: &Envelope) {
    buf.extend_from_slice(&env.magic);
    buf.push(env.version);
    buf.push(env.entry_type);
    buf.extend_from_slice(&env.total_length.to_le_bytes());
    buf.extend_from_slice(&env.timestamp.to_le_bytes());
    buf.extend_from_slice(&env.checksum.to_le_bytes());
    buf.push(env.endianness);
    buf.extend_from_slice(&env.reserved);
}

// --- Outcome payload ---

fn build_outcome_payload(entry: &OutcomeEntry) -> Vec<u8> {
    let mut buf = Vec::new();

    // module_id (len-prefixed string)
    write_len_string(&mut buf, &entry.module_id);

    // target_id (len-prefixed string)
    write_len_string(&mut buf, &entry.target_id);

    // success
    buf.push(if entry.success { 1 } else { 0 });

    // embedding_dim
    buf.extend_from_slice(&entry.embedding_dim.to_le_bytes());

    // embedding_format
    buf.push(EMB_F32);

    // Pad to 64-byte alignment for the embedding
    // Current offset within payload is buf.len()
    // The embedding starts at Envelope::SIZE + buf.len() in the full entry
    let abs_offset = Envelope::SIZE + buf.len();
    let padding = align_padding(abs_offset, EMBEDDING_ALIGN);
    buf.extend(std::iter::repeat_n(0u8, padding));

    // embedding (contiguous f32s, now 64-byte aligned)
    for &val in &entry.embedding {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    // metadata (MessagePack, len-prefixed)
    buf.extend_from_slice(&(entry.metadata.len() as u32).to_le_bytes());
    buf.extend_from_slice(&entry.metadata);

    buf
}

// --- Topology payload ---

fn build_topology_payload(entry: &TopologyEntry) -> Vec<u8> {
    let mut buf = Vec::new();

    // Fixed header (24 bytes)
    buf.extend_from_slice(&entry.timestep.to_le_bytes());
    buf.extend_from_slice(&(entry.fired_nodes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(entry.fired_hyperedges.len() as u32).to_le_bytes());
    buf.extend_from_slice(&entry.predictions_confirmed.to_le_bytes());
    buf.extend_from_slice(&entry.predictions_surprised.to_le_bytes());
    buf.extend_from_slice(&entry.synapses_pruned.to_le_bytes());
    buf.extend_from_slice(&entry.synapses_sprouted.to_le_bytes());
    buf.extend_from_slice(&(entry.salience.len() as u16).to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // reserved

    // Fired nodes
    for node in &entry.fired_nodes {
        write_fired_node(&mut buf, node);
    }

    // Fired hyperedges
    for he in &entry.fired_hyperedges {
        write_fired_hyperedge(&mut buf, he);
    }

    // Salience signals
    for sig in &entry.salience {
        write_salience(&mut buf, sig);
    }

    buf
}

fn write_fired_node(buf: &mut Vec<u8>, node: &FiredNode) {
    // node_id (16 bytes raw UUID)
    buf.extend_from_slice(node.node_id.as_bytes());

    // label (len-prefixed string)
    write_len_string(buf, &node.label);

    // embedding_dim
    buf.extend_from_slice(&node.embedding_dim.to_le_bytes());

    // outgoing_count
    buf.extend_from_slice(&(node.outgoing.len() as u16).to_le_bytes());

    // Pad to 64-byte alignment for embedding
    if node.embedding_dim > 0 {
        let abs_offset = Envelope::SIZE + buf.len();
        let padding = align_padding(abs_offset, EMBEDDING_ALIGN);
        buf.extend(std::iter::repeat_n(0u8, padding));

        // embedding
        for &val in &node.embedding {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    // outgoing synapses (24 bytes each, fixed)
    for syn in &node.outgoing {
        buf.extend_from_slice(syn.post_node_id.as_bytes());
        buf.extend_from_slice(&syn.weight.to_le_bytes());
        buf.extend_from_slice(&syn.eligibility_trace.to_le_bytes());
    }
}

fn write_fired_hyperedge(buf: &mut Vec<u8>, he: &FiredHyperedge) {
    // hyperedge_id (16 bytes)
    buf.extend_from_slice(he.hyperedge_id.as_bytes());

    // label
    write_len_string(buf, &he.label);

    // activation_count
    buf.extend_from_slice(&he.activation_count.to_le_bytes());

    // member_count, output_count
    buf.extend_from_slice(&(he.member_node_ids.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(he.output_target_ids.len() as u16).to_le_bytes());

    // member node IDs (16 bytes each)
    for id in &he.member_node_ids {
        buf.extend_from_slice(id.as_bytes());
    }

    // output target IDs (16 bytes each)
    for id in &he.output_target_ids {
        buf.extend_from_slice(id.as_bytes());
    }
}

fn write_salience(buf: &mut Vec<u8>, sig: &SalienceSignal) {
    buf.push(sig.signal_type);
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(sig.node_id.as_bytes());
    buf.extend_from_slice(&sig.value.to_le_bytes());
    buf.extend_from_slice(&sig.reference.to_le_bytes());
}

// --- Experience payload ---

/// Write an experience entry to a byte buffer.
pub fn write_experience(entry: &ExperienceEntry) -> Vec<u8> {
    let payload = build_experience_payload(entry);
    let total_length = (Envelope::SIZE + payload.len()) as u32;
    let checksum = crc32fast::hash(&payload);

    let envelope = Envelope::new(ENTRY_EXPERIENCE, total_length, entry.timestamp, checksum);

    let mut buf = Vec::with_capacity(total_length as usize);
    write_envelope(&mut buf, &envelope);
    buf.extend_from_slice(&payload);
    buf
}

fn build_experience_payload(entry: &ExperienceEntry) -> Vec<u8> {
    let mut buf = Vec::new();

    // source (len-prefixed string)
    write_len_string(&mut buf, &entry.source);

    // content_type (len-prefixed string)
    write_len_string(&mut buf, &entry.content_type);

    // content (len-prefixed raw bytes)
    buf.extend_from_slice(&(entry.content.len() as u32).to_le_bytes());
    buf.extend_from_slice(&entry.content);

    buf
}

// --- Helpers ---

fn write_len_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_outcome_round_trip_size() {
        let entry = OutcomeEntry {
            timestamp: 1711270234.123,
            module_id: "neurograph".to_string(),
            target_id: "scan:a1b2c3d4e5f6".to_string(),
            success: true,
            embedding_dim: 768,
            embedding: vec![0.1; 768],
            metadata: vec![],
        };
        let bytes = write_outcome(&entry);
        // Envelope (24) + fields + padding + 768*4 (3072) + metadata (4+0)
        assert!(bytes.len() > 3072 + 24, "outcome too small: {}", bytes.len());
        assert!(bytes.len() < 3300, "outcome too large: {}", bytes.len());

        // Starts with magic
        assert_eq!(&bytes[0..2], &MAGIC);
        assert_eq!(bytes[3], ENTRY_OUTCOME);
    }

    #[test]
    fn test_topology_round_trip_size() {
        let entry = TopologyEntry {
            timestamp: 1711270234.123,
            timestep: 6365,
            predictions_confirmed: 0,
            predictions_surprised: 0,
            synapses_pruned: 0,
            synapses_sprouted: 0,
            fired_nodes: vec![
                FiredNode {
                    node_id: Uuid::new_v4(),
                    label: "test_node".to_string(),
                    embedding_dim: 768,
                    embedding: vec![0.1; 768],
                    outgoing: vec![],
                },
            ],
            fired_hyperedges: vec![],
            salience: vec![
                SalienceSignal {
                    signal_type: SALIENCE_HOT_ELIGIBILITY,
                    node_id: Uuid::new_v4(),
                    value: 0.75,
                    reference: 0.05,
                },
            ],
        };
        let bytes = write_topology(&entry);
        assert_eq!(&bytes[0..2], &MAGIC);
        assert_eq!(bytes[3], ENTRY_TOPOLOGY);
        // 1 node with 768 embedding = ~3100+, plus envelope + header + salience
        assert!(bytes.len() > 3100, "topology too small: {}", bytes.len());
    }

    #[test]
    fn test_magic_distinguishes_from_json() {
        let btf = write_outcome(&OutcomeEntry {
            timestamp: 0.0,
            module_id: String::new(),
            target_id: String::new(),
            success: false,
            embedding_dim: 0,
            embedding: vec![],
            metadata: vec![],
        });
        let json = b"{\"type\":\"topology_delta\"}";

        assert!(Envelope::is_btf(&btf));
        assert!(!Envelope::is_btf(json));
    }

    #[test]
    fn test_embedding_alignment() {
        let entry = OutcomeEntry {
            timestamp: 1.0,
            module_id: "test".to_string(),
            target_id: "target".to_string(),
            success: true,
            embedding_dim: 4,
            embedding: vec![1.0, 2.0, 3.0, 4.0],
            metadata: vec![],
        };
        let bytes = write_outcome(&entry);

        // Find where the embedding starts by looking for our known float pattern
        // 1.0 as f32 LE = 0x00 0x00 0x80 0x3f
        let f32_1_0 = 1.0_f32.to_le_bytes();
        let mut emb_offset = None;
        for i in Envelope::SIZE..bytes.len() - 3 {
            if bytes[i..i + 4] == f32_1_0 {
                emb_offset = Some(i);
                break;
            }
        }
        let offset = emb_offset.expect("embedding not found in output");
        assert_eq!(
            offset % EMBEDDING_ALIGN, 0,
            "embedding at offset {} is not {}-byte aligned",
            offset, EMBEDDING_ALIGN
        );
    }
}
