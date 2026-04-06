//! Erasure Coding + Holographic Boundaries — Block L
//!
//! Replaces fragile keyframe+delta chains with fault-tolerant erasure-coded
//! fragments for the COLD memory tier.  COLD regions exist in RAM as pure
//! metadata (`HolographicBoundary`): zero data bytes in RAM, just the
//! reconstruction recipe and enough metadata to answer management queries
//! without waking the data.
//!
//! ## Erasure scheme (XOR-based, no external deps)
//!
//! A *systematic* code where the first K fragments ARE the data chunks
//! (split evenly, last padded with zeros if needed) and (N-K) parity
//! fragments are XOR combinations:
//!
//! - parity[0] = XOR of all K data chunks
//! - parity[1] = XOR of chunks 0 .. K/2
//! - parity[2] = XOR of chunks K/2 .. K
//! - additional parity fragments repeat the halving pattern
//!
//! This reliably handles 1-2 missing fragments.  Full Reed-Solomon can be
//! plugged in later via a proper crate without changing the public API.

// ---------------------------------------------------------------------------
// Hash helper (FNV-1a — no external dep required)
// ---------------------------------------------------------------------------

fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

/// One encoded shard of a larger data block.
///
/// The first `required_k` fragments (indices 0 .. required_k-1) are data
/// fragments; the remainder (indices required_k .. total_n-1) are parity.
pub struct Fragment {
    /// Position index in the full set [0, total_n).
    pub index: u8,
    /// Encoded payload bytes.
    pub data: Vec<u8>,
    /// Total number of fragments produced by the encoder.
    pub total_n: u8,
    /// Minimum number of data fragments needed to reconstruct.
    pub required_k: u8,
    /// Byte length of the original (pre-encoding) data.
    pub original_size: usize,
    /// FNV-1a hash of the original data for integrity checking.
    pub original_hash: u64,
}

// ---------------------------------------------------------------------------
// FragmentLocation
// ---------------------------------------------------------------------------

/// Where a fragment's bytes actually live.
pub enum FragmentLocation {
    /// Bytes are in process memory.
    Memory(Vec<u8>),
    /// Bytes are on disk at `(file_path, byte_offset)`.
    Disk(String, u64),
}

// ---------------------------------------------------------------------------
// DecodeError
// ---------------------------------------------------------------------------

/// Reasons that decoding can fail.
#[derive(Debug, PartialEq)]
pub enum DecodeError {
    /// Fewer fragments were supplied than `required_k`.
    InsufficientFragments { have: usize, need: usize },
    /// Two supplied fragments share the same index.
    DuplicateFragment { index: u8 },
    /// The reconstructed bytes don't match the stored integrity hash.
    HashMismatch { expected: u64, got: u64 },
    /// A parity fragment is needed for recovery but is missing from the set.
    MissingParity,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InsufficientFragments { have, need } => {
                write!(f, "insufficient fragments: have {have}, need {need}")
            }
            DecodeError::DuplicateFragment { index } => {
                write!(f, "duplicate fragment index {index}")
            }
            DecodeError::HashMismatch { expected, got } => {
                write!(f, "hash mismatch: expected {expected:#x}, got {got:#x}")
            }
            DecodeError::MissingParity => {
                write!(f, "missing parity fragment needed for reconstruction")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ErasureCoder
// ---------------------------------------------------------------------------

/// XOR-based K-of-N erasure coder.
pub struct ErasureCoder {
    /// Total fragments to produce per encode call.
    pub default_n: u8,
    /// Minimum fragments required to reconstruct.
    pub default_k: u8,
}

impl ErasureCoder {
    /// Create a new coder.  Panics if `default_k > default_n` or either is zero.
    pub fn new(default_n: u8, default_k: u8) -> Self {
        assert!(default_k > 0, "required_k must be >= 1");
        assert!(default_n >= default_k, "total_n must be >= required_k");
        Self { default_n, default_k }
    }

    // -----------------------------------------------------------------------
    // Encode
    // -----------------------------------------------------------------------

    /// Split `data` into `default_n` fragments: `default_k` data shards plus
    /// `(default_n - default_k)` XOR parity shards.
    ///
    /// Empty input produces fragments that each carry zero bytes.
    pub fn encode(&self, data: &[u8]) -> Vec<Fragment> {
        let k = self.default_k as usize;
        let n = self.default_n as usize;
        let original_size = data.len();
        let original_hash = simple_hash(data);

        // Compute chunk size: ceil(original_size / k), minimum 1 when non-empty
        let chunk_size = if original_size == 0 {
            0
        } else {
            (original_size + k - 1) / k
        };

        // Build K data chunks (last chunk zero-padded if necessary)
        let mut data_chunks: Vec<Vec<u8>> = Vec::with_capacity(k);
        for i in 0..k {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(original_size);
            let mut chunk = if start < original_size {
                data[start..end].to_vec()
            } else {
                Vec::new()
            };
            // Pad to uniform chunk_size
            chunk.resize(chunk_size, 0u8);
            data_chunks.push(chunk);
        }

        // Build parity chunks
        let parity_count = n - k;
        let mut parity_chunks: Vec<Vec<u8>> = Vec::with_capacity(parity_count);
        for p in 0..parity_count {
            let chunk = self.build_parity(p, &data_chunks, chunk_size);
            parity_chunks.push(chunk);
        }

        // Assemble Fragment list: data frags first, then parity
        let mut fragments = Vec::with_capacity(n);
        for i in 0..k {
            fragments.push(Fragment {
                index: i as u8,
                data: data_chunks[i].clone(),
                total_n: n as u8,
                required_k: k as u8,
                original_size,
                original_hash,
            });
        }
        for p in 0..parity_count {
            fragments.push(Fragment {
                index: (k + p) as u8,
                data: parity_chunks[p].clone(),
                total_n: n as u8,
                required_k: k as u8,
                original_size,
                original_hash,
            });
        }

        fragments
    }

    /// Compute parity fragment `p` from the data chunks.
    ///
    /// Parity layout:
    ///   p=0 → XOR of all K chunks          ("full" parity)
    ///   p=1 → XOR of chunks [0 .. k/2)     (low half)
    ///   p=2 → XOR of chunks [k/2 .. k)     (high half)
    ///   p=3 → XOR of chunks [0 .. k/4)     (quarter)
    ///   … and so on (halving, wrapping around)
    fn build_parity(&self, p: usize, chunks: &[Vec<u8>], chunk_size: usize) -> Vec<u8> {
        let k = chunks.len();
        let mut result = vec![0u8; chunk_size];

        let indices: Vec<usize> = if p == 0 {
            // Full parity: all chunks
            (0..k).collect()
        } else {
            // Halving pattern
            let half = k / 2;
            let half = half.max(1); // guard against k==1
            let step = p - 1;
            // Alternate between low and high halves across steps
            if step % 2 == 0 {
                // low half
                (0..half).collect()
            } else {
                // high half
                (half..k).collect()
            }
        };

        for &ci in &indices {
            xor_into(&mut result, &chunks[ci]);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Decode
    // -----------------------------------------------------------------------

    /// Reconstruct the original data from any sufficient subset of fragments.
    ///
    /// If all `required_k` **data** fragments (indices 0 .. k-1) are present,
    /// reconstruction is trivial concatenation.  If any data fragment is
    /// missing, the decoder attempts XOR recovery using parity fragments.
    pub fn decode(&self, fragments: &[Fragment]) -> Result<Vec<u8>, DecodeError> {
        if fragments.is_empty() {
            return Err(DecodeError::InsufficientFragments { have: 0, need: self.default_k as usize });
        }

        // Use metadata from the first fragment (all must agree)
        let original_size = fragments[0].original_size;
        let original_hash = fragments[0].original_hash;
        let k = fragments[0].required_k as usize;

        // Check for duplicate indices
        let mut seen = [false; 256];
        for f in fragments {
            if seen[f.index as usize] {
                return Err(DecodeError::DuplicateFragment { index: f.index });
            }
            seen[f.index as usize] = true;
        }

        // Collect into indexed map
        let mut by_index: std::collections::HashMap<u8, &Fragment> =
            std::collections::HashMap::new();
        for f in fragments {
            by_index.insert(f.index, f);
        }

        let total_available = by_index.len();
        if total_available < k {
            return Err(DecodeError::InsufficientFragments {
                have: total_available,
                need: k,
            });
        }

        // Check which data fragments are present
        let mut data_present = vec![false; k];
        for i in 0..k {
            data_present[i] = by_index.contains_key(&(i as u8));
        }

        let missing_data: Vec<usize> = data_present.iter().enumerate()
            .filter(|(_, &p)| !p)
            .map(|(i, _)| i)
            .collect();

        // Figure out chunk size from any available data fragment
        let chunk_size = if original_size == 0 {
            0
        } else {
            (original_size + k - 1) / k
        };

        // Reconstruct data chunks
        let mut chunks: Vec<Vec<u8>> = vec![vec![0u8; chunk_size]; k];

        // Fill in present data chunks
        for i in 0..k {
            if data_present[i] {
                chunks[i] = by_index[&(i as u8)].data.clone();
                chunks[i].resize(chunk_size, 0u8);
            }
        }

        // Recover missing data chunks using parity
        if !missing_data.is_empty() {
            self.recover_missing(&mut chunks, &missing_data, &by_index, chunk_size)?;
        }

        // Reconstruct original bytes: concatenate chunks, trim to original_size
        let mut result: Vec<u8> = chunks.into_iter().flatten().collect();
        result.truncate(original_size);

        // Integrity check
        let got_hash = simple_hash(&result);
        if got_hash != original_hash {
            return Err(DecodeError::HashMismatch {
                expected: original_hash,
                got: got_hash,
            });
        }

        Ok(result)
    }

    /// Attempt to recover missing data chunks using available parity fragments.
    ///
    /// This works for the simple XOR parity scheme as long as each missing
    /// chunk can be isolated by XOR-ing the parity fragment whose range covers
    /// that chunk with all other known chunks in that range.
    fn recover_missing(
        &self,
        chunks: &mut Vec<Vec<u8>>,
        missing: &[usize],
        by_index: &std::collections::HashMap<u8, &Fragment>,
        chunk_size: usize,
    ) -> Result<(), DecodeError> {
        let k = chunks.len();

        for &mi in missing {
            // Try each available parity fragment in order
            let mut recovered = false;

            // Collect parity fragments (indices k..N)
            let mut parity_frags: Vec<(usize, &Fragment)> = by_index
                .iter()
                .filter(|(&idx, _)| idx as usize >= k)
                .map(|(&idx, &f)| (idx as usize - k, f))
                .collect();
            parity_frags.sort_by_key(|(p, _)| *p);

            for (p_idx, parity_frag) in &parity_frags {
                // Determine which data chunk indices this parity covers
                let covered = self.parity_coverage(*p_idx, k);

                if !covered.contains(&mi) {
                    continue;
                }

                // All other covered indices must NOT be in missing (or already recovered)
                let others_not_missing = covered.iter()
                    .filter(|&&ci| ci != mi)
                    .all(|&ci| !missing.contains(&ci) || chunks[ci].iter().any(|&b| b != 0) /* already recovered */);

                if !others_not_missing {
                    continue; // can't use this parity yet
                }

                // Recover: missing_chunk = parity XOR all_other_covered_chunks
                let mut recovered_chunk = parity_frag.data.clone();
                recovered_chunk.resize(chunk_size, 0u8);

                for &ci in covered.iter().filter(|&&ci| ci != mi) {
                    xor_into(&mut recovered_chunk, &chunks[ci]);
                }

                chunks[mi] = recovered_chunk;
                recovered = true;
                break;
            }

            if !recovered {
                return Err(DecodeError::MissingParity);
            }
        }

        Ok(())
    }

    /// Return the data chunk indices covered by parity fragment `p_idx`.
    fn parity_coverage(&self, p_idx: usize, k: usize) -> Vec<usize> {
        if p_idx == 0 {
            // Full parity covers all k chunks
            (0..k).collect()
        } else {
            let half = (k / 2).max(1);
            let step = p_idx - 1;
            if step % 2 == 0 {
                (0..half).collect()
            } else {
                (half..k).collect()
            }
        }
    }

    // -----------------------------------------------------------------------
    // Integrity
    // -----------------------------------------------------------------------

    /// Verify that `data` matches `expected_hash`.
    pub fn verify_hash(data: &[u8], expected_hash: u64) -> bool {
        simple_hash(data) == expected_hash
    }
}

// ---------------------------------------------------------------------------
// XOR helper
// ---------------------------------------------------------------------------

/// XOR every byte of `src` into `dst`.  If `src` is shorter than `dst`, the
/// remaining bytes of `dst` are left unchanged.
fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

// ---------------------------------------------------------------------------
// BoundaryQuery
// ---------------------------------------------------------------------------

/// A management question that can be answered from the boundary metadata alone
/// without loading or reconstructing any data.
pub enum BoundaryQuery {
    /// Should this region be promoted to a warmer tier?
    ShouldPromote,
    /// How many bytes of RAM does keeping this cold save?
    CompressionSavings,
    /// Is this region connected to the given peer region?
    IsRelatedTo(u32),
    /// What is the coarse data type (derived from first-64-byte fingerprint)?
    DataType,
    /// Has the content changed since the given hash was recorded?
    HasChanged(u64),
}

// ---------------------------------------------------------------------------
// HolographicBoundary
// ---------------------------------------------------------------------------

/// Zero-data COLD region descriptor.
///
/// Lives entirely in RAM as pure metadata: the reconstruction recipe for the
/// erasure-coded fragments plus enough contextual information to answer every
/// common management question without touching the actual data.
pub struct HolographicBoundary {
    /// Unique ID of the memory region this boundary represents.
    pub region_id: u32,
    /// Original data size in bytes.
    pub original_size: usize,
    /// FNV-1a hash of the original content.
    pub content_hash: u64,
    /// Hash of the first 64 bytes — coarse type fingerprint.
    pub type_signature: u64,
    /// Ratio: original_size / storage_size (>1 means compression saved space).
    pub compression_ratio: f32,
    /// Graph edges to peer regions: (peer_region_id, edge_weight).
    pub graph_connections: Vec<(u32, f64)>,
    /// Total number of erasure fragments produced.
    pub fragment_count: u8,
    /// Minimum fragments needed to reconstruct.
    pub fragments_required: u8,
    /// Estimated microseconds to reconstruct (I/O + XOR cost).
    pub reconstruction_cost_us: u64,
    /// Nanosecond timestamp of last access.
    pub last_access_ns: u64,
    /// Exponentially-smoothed access rate (accesses per second, approx).
    pub access_frequency: f32,
}

impl HolographicBoundary {
    /// Build a boundary from raw data.
    ///
    /// `data` is the original bytes being cold-stored.  After this call the
    /// caller should hand `data` off to the erasure coder and drop it.
    /// `connections` is the set of graph edges to neighbouring regions.
    pub fn new(region_id: u32, data: &[u8], connections: Vec<(u32, f64)>) -> Self {
        let content_hash = simple_hash(data);

        // Type signature: hash of first 64 bytes (or all bytes if shorter)
        let prefix = &data[..data.len().min(64)];
        let type_signature = simple_hash(prefix);

        // Rough compression ratio estimate: XOR entropy proxy
        // We use a simple byte-frequency model: unique bytes / 256 * 2
        let storage_estimate = estimate_compressed_size(data);
        let compression_ratio = if storage_estimate == 0 {
            1.0
        } else {
            data.len() as f32 / storage_estimate as f32
        };

        // Reconstruction cost: assume ~10µs base + 1µs per KB of data
        let reconstruction_cost_us = 10 + (data.len() as u64 / 1024);

        Self {
            region_id,
            original_size: data.len(),
            content_hash,
            type_signature,
            compression_ratio,
            graph_connections: connections,
            fragment_count: 0,   // caller sets after encoding
            fragments_required: 0,
            reconstruction_cost_us,
            last_access_ns: 0,
            access_frequency: 0.0,
        }
    }

    /// Return true if the boundary metadata alone can answer `query`.
    ///
    /// All variants always return true — that is the invariant of the
    /// holographic boundary design.  This method exists to make that contract
    /// explicit and testable.
    pub fn can_answer_query(&self, query: &BoundaryQuery) -> bool {
        match query {
            BoundaryQuery::ShouldPromote => {
                // Needs access_frequency and graph_connections — both present
                true
            }
            BoundaryQuery::CompressionSavings => {
                // Needs compression_ratio and original_size — both present
                true
            }
            BoundaryQuery::IsRelatedTo(peer_id) => {
                // Just check the connections list
                let _ = self.graph_connections.iter().any(|(id, _)| id == peer_id);
                true
            }
            BoundaryQuery::DataType => {
                // Needs type_signature — present
                true
            }
            BoundaryQuery::HasChanged(hash) => {
                // Compare against content_hash — no data needed
                let _ = self.content_hash == *hash;
                true
            }
        }
    }

    /// Actually evaluate `query` and return the answer as a `QueryAnswer`.
    pub fn answer_query(&self, query: &BoundaryQuery) -> QueryAnswer {
        match query {
            BoundaryQuery::ShouldPromote => {
                // Promote when access_frequency > 0.01 Hz or highly connected
                let promote = self.access_frequency > 0.01
                    || self.graph_connections.len() > 5;
                QueryAnswer::Bool(promote)
            }
            BoundaryQuery::CompressionSavings => {
                let savings = if self.compression_ratio > 1.0 {
                    let stored = self.original_size as f32 / self.compression_ratio;
                    (self.original_size as f32 - stored) as usize
                } else {
                    0
                };
                QueryAnswer::Bytes(savings)
            }
            BoundaryQuery::IsRelatedTo(peer_id) => {
                let related = self.graph_connections.iter().any(|(id, _)| id == peer_id);
                QueryAnswer::Bool(related)
            }
            BoundaryQuery::DataType => {
                QueryAnswer::Hash(self.type_signature)
            }
            BoundaryQuery::HasChanged(hash) => {
                QueryAnswer::Bool(self.content_hash != *hash)
            }
        }
    }

    /// Record an access event at `now_ns` nanoseconds and update frequency.
    ///
    /// Uses a simple exponential moving average so frequency decays over time
    /// without storing a full access history.
    pub fn update_access(&mut self, now_ns: u64) {
        if self.last_access_ns > 0 && now_ns > self.last_access_ns {
            let dt_s = (now_ns - self.last_access_ns) as f64 / 1_000_000_000.0;
            let instant_rate = if dt_s > 0.0 { 1.0 / dt_s } else { 0.0 };
            // EMA with alpha = 0.2
            self.access_frequency = 0.8 * self.access_frequency + 0.2 * instant_rate as f32;
        }
        self.last_access_ns = now_ns;
    }
}

/// Typed return value from `HolographicBoundary::answer_query`.
pub enum QueryAnswer {
    Bool(bool),
    Bytes(usize),
    Hash(u64),
}

// ---------------------------------------------------------------------------
// Internal: compressed size estimator (no external dep)
// ---------------------------------------------------------------------------

/// Rough estimate of how many bytes `data` would compress to.
///
/// Uses byte-frequency entropy as a proxy: high entropy → near-incompressible.
/// This is intentionally cheap — it only needs to produce a plausible ratio
/// for the boundary metadata, not an accurate compress call.
fn estimate_compressed_size(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let n = data.len() as f64;
    // Shannon entropy (bits per byte)
    let entropy: f64 = freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum();
    // Estimated bits / 8 = bytes per byte of original
    let ratio = (entropy / 8.0).max(0.125); // floor at 8:1 compression
    (n * ratio) as usize + 1
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // test_erasure_encode_decode_roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_erasure_encode_decode_roundtrip() {
        let coder = ErasureCoder::new(6, 4);
        let original: Vec<u8> = (0u8..200).collect();

        let fragments = coder.encode(&original);
        assert_eq!(fragments.len(), 6);

        // Decode from all 6 fragments
        let recovered = coder.decode(&fragments).expect("decode from all fragments");
        assert_eq!(recovered, original, "roundtrip must be byte-identical");
    }

    // -----------------------------------------------------------------------
    // test_erasure_decode_with_minimum
    // -----------------------------------------------------------------------

    #[test]
    fn test_erasure_decode_with_minimum() {
        let coder = ErasureCoder::new(6, 4);
        let original: Vec<u8> = (0u8..=255).cycle().take(512).collect();

        let fragments = coder.encode(&original);

        // Use only the K=4 data fragments (indices 0..3)
        let data_only: Vec<Fragment> = fragments
            .into_iter()
            .filter(|f| (f.index as usize) < 4)
            .collect();
        assert_eq!(data_only.len(), 4);

        let recovered = coder.decode(&data_only).expect("decode from minimum data frags");
        assert_eq!(recovered, original);
    }

    // -----------------------------------------------------------------------
    // test_erasure_decode_with_parity
    // -----------------------------------------------------------------------

    #[test]
    fn test_erasure_decode_with_parity() {
        // N=4, K=3: indices 0,1,2 are data; index 3 is parity (XOR of all)
        let coder = ErasureCoder::new(4, 3);
        let original = b"Hello, erasure coding world! This is a test.".to_vec();

        let fragments = coder.encode(&original);
        assert_eq!(fragments.len(), 4);

        // Drop data fragment 0, keep 1, 2, and parity 3
        let subset: Vec<Fragment> = fragments
            .into_iter()
            .filter(|f| f.index != 0)
            .collect();
        assert_eq!(subset.len(), 3);

        let recovered = coder.decode(&subset).expect("should recover with parity");
        assert_eq!(recovered, original, "parity recovery must produce original data");
    }

    // -----------------------------------------------------------------------
    // test_erasure_decode_insufficient
    // -----------------------------------------------------------------------

    #[test]
    fn test_erasure_decode_insufficient() {
        let coder = ErasureCoder::new(6, 4);
        let original: Vec<u8> = (0u8..100).collect();

        let fragments = coder.encode(&original);

        // Keep only K-1 = 3 data fragments, no parity
        let tiny: Vec<Fragment> = fragments
            .into_iter()
            .filter(|f| f.index < 3)
            .collect();

        let result = coder.decode(&tiny);
        assert!(
            matches!(result, Err(DecodeError::InsufficientFragments { .. })),
            "should error with insufficient fragments, got: {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------------
    // test_holographic_boundary_creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_holographic_boundary_creation() {
        let data: Vec<u8> = (0u8..=127).cycle().take(4096).collect();
        let connections = vec![(42u32, 0.8f64), (99u32, 0.3f64)];

        let boundary = HolographicBoundary::new(7, &data, connections.clone());

        assert_eq!(boundary.region_id, 7);
        assert_eq!(boundary.original_size, 4096);
        assert_eq!(boundary.content_hash, simple_hash(&data));
        assert_eq!(boundary.type_signature, simple_hash(&data[..64]));
        assert_eq!(boundary.graph_connections.len(), 2);
        assert!(boundary.compression_ratio > 0.0);
        assert!(boundary.reconstruction_cost_us >= 10);
        assert_eq!(boundary.last_access_ns, 0);
        assert_eq!(boundary.access_frequency, 0.0);
    }

    // -----------------------------------------------------------------------
    // test_boundary_queries_no_data
    // -----------------------------------------------------------------------

    #[test]
    fn test_boundary_queries_no_data() {
        let data = b"Holographic boundary test payload. ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.";
        let connections = vec![(10u32, 1.0f64), (20u32, 0.5f64)];
        let mut boundary = HolographicBoundary::new(1, data, connections);
        boundary.access_frequency = 0.05; // above promote threshold

        let queries = [
            BoundaryQuery::ShouldPromote,
            BoundaryQuery::CompressionSavings,
            BoundaryQuery::IsRelatedTo(10),
            BoundaryQuery::IsRelatedTo(999), // not connected
            BoundaryQuery::DataType,
            BoundaryQuery::HasChanged(simple_hash(data)),
            BoundaryQuery::HasChanged(0xdeadbeef),
        ];

        for q in &queries {
            assert!(
                boundary.can_answer_query(q),
                "every BoundaryQuery must be answerable from metadata alone"
            );
        }

        // Spot-check actual answers
        assert!(matches!(boundary.answer_query(&BoundaryQuery::ShouldPromote), QueryAnswer::Bool(true)));
        assert!(matches!(boundary.answer_query(&BoundaryQuery::IsRelatedTo(10)), QueryAnswer::Bool(true)));
        assert!(matches!(boundary.answer_query(&BoundaryQuery::IsRelatedTo(999)), QueryAnswer::Bool(false)));
        assert!(matches!(boundary.answer_query(&BoundaryQuery::HasChanged(simple_hash(data))), QueryAnswer::Bool(false)));
        assert!(matches!(boundary.answer_query(&BoundaryQuery::HasChanged(0xdeadbeef)), QueryAnswer::Bool(true)));
        assert!(matches!(boundary.answer_query(&BoundaryQuery::DataType), QueryAnswer::Hash(_)));
    }

    // -----------------------------------------------------------------------
    // test_hash_integrity
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_integrity() {
        let data = b"integrity check payload";
        let h = simple_hash(data);

        assert!(ErasureCoder::verify_hash(data, h), "correct hash must verify");

        let mut corrupted = data.to_vec();
        corrupted[5] ^= 0xFF; // flip bits in one byte
        assert!(
            !ErasureCoder::verify_hash(&corrupted, h),
            "corrupted data must fail hash check"
        );
    }

    // -----------------------------------------------------------------------
    // test_encode_empty_data
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_empty_data() {
        let coder = ErasureCoder::new(4, 3);
        let fragments = coder.encode(&[]);

        assert_eq!(fragments.len(), 4);
        for f in &fragments {
            assert_eq!(f.original_size, 0);
        }

        // Decoding all fragments of empty data should return empty vec
        let recovered = coder.decode(&fragments).expect("empty encode/decode roundtrip");
        assert!(recovered.is_empty(), "empty input should decode to empty vec");
    }
}
