//! BTF reader — deserializes binary tract entries.
//!
//! Reads entries from a byte slice (mmap or file contents).
//! Supports scanning: read envelope, skip or parse payload.
//! Detects JSONL vs BTF by first byte for migration compatibility.

use crate::format::*;
use std::io;
use uuid::Uuid;

/// A cursor over a byte slice for reading BTF entries.
pub struct TractReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> TractReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// How many bytes have been consumed from the input.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Read the next entry. Returns None at end of data.
    /// Returns Err if the data is malformed BTF.
    /// Skips JSONL lines (returns them as raw bytes for legacy handling).
    pub fn next_entry(&mut self) -> Option<Result<ReadResult<'a>, io::Error>> {
        if self.pos >= self.data.len() {
            return None;
        }

        // Check first byte: BTF magic vs JSONL '{'
        let first = self.data[self.pos];
        if first == 0x7B {
            // JSONL line — find newline, return raw bytes
            return Some(self.read_jsonl_line());
        }
        if first != MAGIC[0] {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown entry start byte: 0x{:02x} at offset {}", first, self.pos),
            )));
        }

        Some(self.read_btf_entry())
    }

    fn read_jsonl_line(&mut self) -> Result<ReadResult<'a>, io::Error> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        let line = &self.data[start..self.pos];
        if self.pos < self.data.len() {
            self.pos += 1; // skip newline
        }
        Ok(ReadResult::JsonlLine(line))
    }

    fn read_btf_entry(&mut self) -> Result<ReadResult<'a>, io::Error> {
        let start = self.pos;

        if self.data.len() - self.pos < Envelope::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "not enough data for envelope",
            ));
        }

        // Parse envelope
        let envelope = read_envelope(&self.data[self.pos..])?;
        let total = envelope.total_length as usize;

        if self.data.len() - start < total {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("entry claims {} bytes but only {} available", total, self.data.len() - start),
            ));
        }

        let payload_start = start + Envelope::SIZE;
        let payload_end = start + total;
        let payload = &self.data[payload_start..payload_end];

        // Verify checksum (copy from packed struct to avoid unaligned ref)
        let expected_checksum = envelope.checksum;
        let computed = crc32fast::hash(payload);
        if computed != expected_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "checksum mismatch: expected 0x{:08x}, got 0x{:08x}",
                    expected_checksum, computed
                ),
            ));
        }

        self.pos = payload_end;

        match envelope.entry_type {
            ENTRY_OUTCOME => {
                let entry = parse_outcome(payload, envelope.timestamp)?;
                Ok(ReadResult::Entry(TractEntry::Outcome(entry)))
            }
            ENTRY_TOPOLOGY => {
                let entry = parse_topology(payload, envelope.timestamp)?;
                Ok(ReadResult::Entry(TractEntry::Topology(entry)))
            }
            ENTRY_EXPERIENCE => {
                let entry = parse_experience(payload, envelope.timestamp)?;
                Ok(ReadResult::Entry(TractEntry::Experience(entry)))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown entry type: {}", other),
            )),
        }
    }
}

/// Result from reading one item off the tract.
#[derive(Debug)]
pub enum ReadResult<'a> {
    /// A parsed BTF entry.
    Entry(TractEntry),
    /// A raw JSONL line (legacy, for Python to handle).
    JsonlLine(&'a [u8]),
}

// --- Envelope parsing ---

fn read_envelope(data: &[u8]) -> Result<Envelope, io::Error> {
    if data[0] != MAGIC[0] || data[1] != MAGIC[1] {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    Ok(Envelope {
        magic: MAGIC,
        version: data[2],
        entry_type: data[3],
        total_length: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
        timestamp: f64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]),
        checksum: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
        endianness: data[20],
        reserved: [data[21], data[22], data[23]],
    })
}

// --- Payload parsers ---

fn parse_outcome(data: &[u8], timestamp: f64) -> Result<OutcomeEntry, io::Error> {
    let mut pos = 0;

    let module_id = read_len_string(data, &mut pos)?;
    let target_id = read_len_string(data, &mut pos)?;

    let success = read_u8(data, &mut pos)? != 0;
    let embedding_dim = read_u16(data, &mut pos)?;
    let _embedding_format = read_u8(data, &mut pos)?; // currently always f32

    // Skip alignment padding to find embedding
    let abs_offset = Envelope::SIZE + pos;
    let padding = {
        let rem = abs_offset % EMBEDDING_ALIGN;
        if rem == 0 { 0 } else { EMBEDDING_ALIGN - rem }
    };
    pos += padding;

    let embedding = read_f32_slice(data, &mut pos, embedding_dim as usize)?;

    let metadata_len = read_u32(data, &mut pos)? as usize;
    let metadata = if metadata_len > 0 && pos + metadata_len <= data.len() {
        data[pos..pos + metadata_len].to_vec()
    } else {
        vec![]
    };

    Ok(OutcomeEntry {
        timestamp,
        module_id,
        target_id,
        success,
        embedding_dim,
        embedding,
        metadata,
    })
}

fn parse_topology(data: &[u8], timestamp: f64) -> Result<TopologyEntry, io::Error> {
    let mut pos = 0;

    let timestep = read_u32(data, &mut pos)?;
    let fired_node_count = read_u32(data, &mut pos)? as usize;
    let fired_he_count = read_u32(data, &mut pos)? as usize;
    let predictions_confirmed = read_u16(data, &mut pos)?;
    let predictions_surprised = read_u16(data, &mut pos)?;
    let synapses_pruned = read_u16(data, &mut pos)?;
    let synapses_sprouted = read_u16(data, &mut pos)?;
    let salience_count = read_u16(data, &mut pos)? as usize;
    let _reserved = read_u16(data, &mut pos)?;

    let mut fired_nodes = Vec::with_capacity(fired_node_count);
    for _ in 0..fired_node_count {
        fired_nodes.push(parse_fired_node(data, &mut pos)?);
    }

    let mut fired_hyperedges = Vec::with_capacity(fired_he_count);
    for _ in 0..fired_he_count {
        fired_hyperedges.push(parse_fired_hyperedge(data, &mut pos)?);
    }

    let mut salience = Vec::with_capacity(salience_count);
    for _ in 0..salience_count {
        salience.push(parse_salience(data, &mut pos)?);
    }

    Ok(TopologyEntry {
        timestamp,
        timestep,
        predictions_confirmed,
        predictions_surprised,
        synapses_pruned,
        synapses_sprouted,
        fired_nodes,
        fired_hyperedges,
        salience,
    })
}

fn parse_fired_node(data: &[u8], pos: &mut usize) -> Result<FiredNode, io::Error> {
    let node_id = read_uuid(data, pos)?;
    let label = read_len_string(data, pos)?;
    let embedding_dim = read_u16(data, pos)?;
    let outgoing_count = read_u16(data, pos)? as usize;

    let embedding = if embedding_dim > 0 {
        // Skip alignment padding
        let abs_offset = Envelope::SIZE + *pos;
        let rem = abs_offset % EMBEDDING_ALIGN;
        let padding = if rem == 0 { 0 } else { EMBEDDING_ALIGN - rem };
        *pos += padding;
        read_f32_slice(data, pos, embedding_dim as usize)?
    } else {
        vec![]
    };

    let mut outgoing = Vec::with_capacity(outgoing_count);
    for _ in 0..outgoing_count {
        let post_node_id = read_uuid(data, pos)?;
        let weight = read_f32(data, pos)?;
        let eligibility_trace = read_f32(data, pos)?;
        outgoing.push(OutgoingSynapse { post_node_id, weight, eligibility_trace });
    }

    Ok(FiredNode { node_id, label, embedding_dim, embedding, outgoing })
}

fn parse_fired_hyperedge(data: &[u8], pos: &mut usize) -> Result<FiredHyperedge, io::Error> {
    let hyperedge_id = read_uuid(data, pos)?;
    let label = read_len_string(data, pos)?;
    let activation_count = read_u32(data, pos)?;
    let member_count = read_u16(data, pos)? as usize;
    let output_count = read_u16(data, pos)? as usize;

    let mut member_node_ids = Vec::with_capacity(member_count);
    for _ in 0..member_count {
        member_node_ids.push(read_uuid(data, pos)?);
    }

    let mut output_target_ids = Vec::with_capacity(output_count);
    for _ in 0..output_count {
        output_target_ids.push(read_uuid(data, pos)?);
    }

    Ok(FiredHyperedge {
        hyperedge_id, label, activation_count,
        member_node_ids, output_target_ids,
    })
}

fn parse_salience(data: &[u8], pos: &mut usize) -> Result<SalienceSignal, io::Error> {
    let signal_type = read_u8(data, pos)?;
    *pos += 3; // reserved
    let node_id = read_uuid(data, pos)?;
    let value = read_f32(data, pos)?;
    let reference = read_f32(data, pos)?;
    Ok(SalienceSignal { signal_type, node_id, value, reference })
}

fn parse_experience(data: &[u8], timestamp: f64) -> Result<ExperienceEntry, io::Error> {
    let mut pos = 0usize;

    let source = read_len_string(data, &mut pos)?;
    let content_type = read_len_string(data, &mut pos)?;

    // content (len-prefixed raw bytes)
    let content_len = read_u32(data, &mut pos)? as usize;
    if pos + content_len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "experience content"));
    }
    let content = data[pos..pos + content_len].to_vec();

    Ok(ExperienceEntry {
        timestamp,
        source,
        content_type,
        content,
    })
}

// --- Primitive readers ---

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, io::Error> {
    if *pos >= data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u8"));
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, io::Error> {
    if *pos + 2 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u16"));
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, io::Error> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_u32"));
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, io::Error> {
    if *pos + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_f32"));
    }
    let v = f32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_uuid(data: &[u8], pos: &mut usize) -> Result<Uuid, io::Error> {
    if *pos + 16 > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_uuid"));
    }
    let bytes: [u8; 16] = data[*pos..*pos + 16].try_into().unwrap();
    *pos += 16;
    Ok(Uuid::from_bytes(bytes))
}

fn read_f32_slice(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<f32>, io::Error> {
    let byte_len = count * 4;
    if *pos + byte_len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_f32_slice"));
    }
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(read_f32(data, pos)?);
    }
    Ok(v)
}

fn read_len_string(data: &[u8], pos: &mut usize) -> Result<String, io::Error> {
    let len = read_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read_len_string"));
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    *pos += len;
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write;

    #[test]
    fn test_outcome_round_trip() {
        let original = OutcomeEntry {
            timestamp: 1711270234.123456,
            module_id: "immunis".to_string(),
            target_id: "scan:abcdef123456".to_string(),
            success: true,
            embedding_dim: 768,
            embedding: (0..768).map(|i| i as f32 * 0.001).collect(),
            metadata: {
                use std::collections::HashMap;
                let mut m = HashMap::new();
                m.insert("source".to_string(), "conversation".to_string());
                rmp_serde::to_vec(&m).unwrap_or_default()
            },
        };

        let bytes = write::write_outcome(&original);
        let mut reader = TractReader::new(&bytes);
        let result = reader.next_entry().unwrap().unwrap();

        match result {
            ReadResult::Entry(TractEntry::Outcome(parsed)) => {
                assert_eq!(parsed.module_id, "immunis");
                assert_eq!(parsed.target_id, "scan:abcdef123456");
                assert!(parsed.success);
                assert_eq!(parsed.embedding_dim, 768);
                assert_eq!(parsed.embedding.len(), 768);
                // Check first and last embedding values
                assert!((parsed.embedding[0] - 0.0).abs() < 1e-6);
                assert!((parsed.embedding[767] - 0.767).abs() < 1e-3);
                assert!(!parsed.metadata.is_empty());
            }
            other => panic!("expected Outcome, got {:?}", other),
        }

        // Verify end of data
        assert!(reader.next_entry().is_none());
    }

    #[test]
    fn test_topology_round_trip() {
        let node_id = Uuid::new_v4();
        let post_id = Uuid::new_v4();
        let he_id = Uuid::new_v4();
        let member1 = Uuid::new_v4();
        let member2 = Uuid::new_v4();
        let sal_node = Uuid::new_v4();

        let original = TopologyEntry {
            timestamp: 1711270234.0,
            timestep: 6365,
            predictions_confirmed: 3,
            predictions_surprised: 1,
            synapses_pruned: 2,
            synapses_sprouted: 5,
            fired_nodes: vec![FiredNode {
                node_id,
                label: "test_concept".to_string(),
                embedding_dim: 4,
                embedding: vec![1.0, 2.0, 3.0, 4.0],
                outgoing: vec![OutgoingSynapse {
                    post_node_id: post_id,
                    weight: 0.75,
                    eligibility_trace: 0.3,
                }],
            }],
            fired_hyperedges: vec![FiredHyperedge {
                hyperedge_id: he_id,
                label: "threat_cluster".to_string(),
                activation_count: 42,
                member_node_ids: vec![member1, member2],
                output_target_ids: vec![],
            }],
            salience: vec![SalienceSignal {
                signal_type: SALIENCE_HOT_ELIGIBILITY,
                node_id: sal_node,
                value: 0.85,
                reference: 0.05,
            }],
        };

        let bytes = write::write_topology(&original);
        let mut reader = TractReader::new(&bytes);
        let result = reader.next_entry().unwrap().unwrap();

        match result {
            ReadResult::Entry(TractEntry::Topology(parsed)) => {
                assert_eq!(parsed.timestep, 6365);
                assert_eq!(parsed.predictions_confirmed, 3);
                assert_eq!(parsed.predictions_surprised, 1);
                assert_eq!(parsed.synapses_pruned, 2);
                assert_eq!(parsed.synapses_sprouted, 5);

                assert_eq!(parsed.fired_nodes.len(), 1);
                let n = &parsed.fired_nodes[0];
                assert_eq!(n.node_id, node_id);
                assert_eq!(n.label, "test_concept");
                assert_eq!(n.embedding, vec![1.0, 2.0, 3.0, 4.0]);
                assert_eq!(n.outgoing.len(), 1);
                assert_eq!(n.outgoing[0].post_node_id, post_id);
                assert!((n.outgoing[0].weight - 0.75).abs() < 1e-6);

                assert_eq!(parsed.fired_hyperedges.len(), 1);
                let h = &parsed.fired_hyperedges[0];
                assert_eq!(h.hyperedge_id, he_id);
                assert_eq!(h.member_node_ids, vec![member1, member2]);

                assert_eq!(parsed.salience.len(), 1);
                assert_eq!(parsed.salience[0].signal_type, SALIENCE_HOT_ELIGIBILITY);
                assert_eq!(parsed.salience[0].node_id, sal_node);
            }
            other => panic!("expected Topology, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_entry_stream() {
        let o1 = write::write_outcome(&OutcomeEntry {
            timestamp: 1.0, module_id: "a".into(), target_id: "b".into(),
            success: true, embedding_dim: 2, embedding: vec![0.1, 0.2],
            metadata: vec![],
        });
        let o2 = write::write_outcome(&OutcomeEntry {
            timestamp: 2.0, module_id: "c".into(), target_id: "d".into(),
            success: false, embedding_dim: 2, embedding: vec![0.3, 0.4],
            metadata: vec![],
        });

        let mut stream = Vec::new();
        stream.extend_from_slice(&o1);
        stream.extend_from_slice(&o2);

        let mut reader = TractReader::new(&stream);
        let r1 = reader.next_entry().unwrap().unwrap();
        let r2 = reader.next_entry().unwrap().unwrap();
        assert!(reader.next_entry().is_none());

        match (r1, r2) {
            (ReadResult::Entry(TractEntry::Outcome(e1)),
             ReadResult::Entry(TractEntry::Outcome(e2))) => {
                assert_eq!(e1.module_id, "a");
                assert_eq!(e2.module_id, "c");
                assert!(!e2.success);
            }
            _ => panic!("expected two outcomes"),
        }
    }

    #[test]
    fn test_jsonl_passthrough() {
        let json_line = b"{\"type\":\"topology_delta\",\"version\":1}\n";
        let btf = write::write_outcome(&OutcomeEntry {
            timestamp: 1.0, module_id: "x".into(), target_id: "y".into(),
            success: true, embedding_dim: 0, embedding: vec![],
            metadata: vec![],
        });

        let mut stream = Vec::new();
        stream.extend_from_slice(json_line);
        stream.extend_from_slice(&btf);

        let mut reader = TractReader::new(&stream);

        // First entry should be JSONL passthrough
        match reader.next_entry().unwrap().unwrap() {
            ReadResult::JsonlLine(line) => {
                let s = std::str::from_utf8(line).unwrap();
                assert!(s.contains("topology_delta"));
            }
            other => panic!("expected JsonlLine, got {:?}", other),
        }

        // Second entry should be BTF
        match reader.next_entry().unwrap().unwrap() {
            ReadResult::Entry(TractEntry::Outcome(e)) => {
                assert_eq!(e.module_id, "x");
            }
            other => panic!("expected Outcome, got {:?}", other),
        }
    }
}
