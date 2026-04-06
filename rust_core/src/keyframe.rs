//! Keyframe/Delta Encoding — video codec model applied to memory.
//!
//! Instead of storing full snapshots repeatedly, store one compressed
//! keyframe + tiny sparse diffs (deltas).  A 64KB region where only
//! 200 bytes changed produces a ~200-byte delta, not another 64KB copy.
//!
//! Design:
//!   - Keyframes are LZ4-compressed full snapshots.
//!   - Deltas are sparse: (offset, changed_bytes) pairs produced by
//!     XOR-walking the current data against the keyframe baseline.
//!   - Reconstruction applies all deltas in sequence.
//!   - After enough deltas (or enough idle observation cycles), the
//!     store can consolidate or mark a frame read-only.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Simple FNV-1a-style hash — no external dep required
// ---------------------------------------------------------------------------

fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Delta
// ---------------------------------------------------------------------------

/// A sparse record of bytes that changed relative to the keyframe baseline.
///
/// `changed_ranges` is a list of `(offset, changed_bytes)` pairs.
/// Only non-zero XOR regions are stored, so a 64KB region with 10
/// changed bytes results in roughly 10 bytes of delta payload.
pub struct Delta {
    pub id: u32,
    pub timestamp_ns: u64,
    /// Sparse changed ranges: (byte offset into original, changed bytes)
    pub changed_ranges: Vec<(usize, Vec<u8>)>,
    /// Total payload bytes across all ranges (useful for budgeting)
    pub cumulative_change_bytes: usize,
}

impl Delta {
    /// Apply this delta onto a mutable buffer (which must be at least as
    /// large as the keyframe's original data).
    fn apply(&self, buf: &mut [u8]) {
        for (offset, bytes) in &self.changed_ranges {
            let end = offset + bytes.len();
            if end <= buf.len() {
                buf[*offset..end].copy_from_slice(bytes);
            }
        }
    }

    /// Does this delta touch the half-open byte range `[range_start, range_end)`?
    fn touches_range(&self, range_start: usize, range_end: usize) -> bool {
        for (offset, bytes) in &self.changed_ranges {
            let end = offset + bytes.len();
            // Ranges overlap when start < other_end && end > other_start
            if *offset < range_end && end > range_start {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A compressed full snapshot with an attached chain of sparse deltas.
pub struct Keyframe {
    pub id: u32,
    /// LZ4-compressed bytes of the original snapshot
    compressed_data: Vec<u8>,
    /// Byte length before compression (needed for decompression)
    original_size: usize,
    /// Integrity hash over the original uncompressed bytes
    original_hash: u64,
    /// Ordered chain of deltas recorded after this keyframe was taken
    deltas: Vec<Delta>,
    /// When true, no further deltas are expected (memory went cold)
    pub is_read_only: bool,
    /// How many `mark_observation_cycle` calls have fired with no new delta
    observation_cycles: u32,
}

impl Keyframe {
    fn new(id: u32, data: &[u8]) -> Self {
        let original_hash = hash_bytes(data);
        let compressed_data = lz4_flex::compress_prepend_size(data);
        Self {
            id,
            compressed_data,
            original_size: data.len(),
            original_hash,
            deltas: Vec::new(),
            is_read_only: false,
            observation_cycles: 0,
        }
    }

    /// Decompress the keyframe back to its original bytes.
    fn decompress(&self) -> Option<Vec<u8>> {
        lz4_flex::decompress_size_prepended(&self.compressed_data).ok()
    }

    /// Reconstruct the full data by decompressing then replaying all deltas.
    fn reconstruct(&self) -> Option<Vec<u8>> {
        let mut buf = self.decompress()?;
        for delta in &self.deltas {
            delta.apply(&mut buf);
        }
        Some(buf)
    }

    /// Reconstruct only the slice `[offset, offset+length)`.
    ///
    /// We still have to decompress the whole keyframe because LZ4 is not
    /// randomly-accessible, but we only apply deltas that actually touch
    /// the requested range, which is cheaper for large delta chains.
    fn reconstruct_range(&self, offset: usize, length: usize) -> Option<Vec<u8>> {
        let range_end = offset.checked_add(length)?;
        if range_end > self.original_size {
            return None;
        }

        let mut buf = self.decompress()?;

        // Only replay deltas that overlap the requested range
        for delta in &self.deltas {
            if delta.touches_range(offset, range_end) {
                delta.apply(&mut buf);
            }
        }

        Some(buf[offset..range_end].to_vec())
    }

    /// Build a sparse delta from `current_data` vs the keyframe baseline.
    ///
    /// XOR walk: collect contiguous runs where XOR != 0 into
    /// (offset, actual_bytes_from_current) pairs.
    /// Returns `None` when there are no changes at all.
    fn build_delta(&self, id: u32, timestamp_ns: u64, current_data: &[u8]) -> Option<Delta> {
        let baseline = self.decompress()?;
        // Apply existing deltas so we diff against the *current* logical state,
        // not just the raw keyframe bytes.
        let mut logical = baseline;
        for d in &self.deltas {
            d.apply(&mut logical);
        }

        let cmp_len = logical.len().min(current_data.len());
        let mut changed_ranges: Vec<(usize, Vec<u8>)> = Vec::new();

        let mut i = 0;
        while i < cmp_len {
            if logical[i] != current_data[i] {
                // Start of a changed run
                let run_start = i;
                let mut run: Vec<u8> = Vec::new();
                while i < cmp_len && logical[i] != current_data[i] {
                    run.push(current_data[i]);
                    i += 1;
                }
                changed_ranges.push((run_start, run));
            } else {
                i += 1;
            }
        }

        // Handle the case where current_data is longer than logical
        if current_data.len() > logical.len() {
            let tail = current_data[logical.len()..].to_vec();
            changed_ranges.push((logical.len(), tail));
        }

        if changed_ranges.is_empty() {
            return None;
        }

        let cumulative_change_bytes = changed_ranges.iter().map(|(_, v)| v.len()).sum();
        Some(Delta {
            id,
            timestamp_ns,
            changed_ranges,
            cumulative_change_bytes,
        })
    }
}

// ---------------------------------------------------------------------------
// KeyframeStore
// ---------------------------------------------------------------------------

/// Central store for all keyframes and their delta chains.
pub struct KeyframeStore {
    frames: HashMap<u32, Keyframe>,
    next_id: u32,
    /// Maximum number of deltas before `record_delta` auto-consolidates
    pub consolidation_threshold: usize,
    /// Number of observation cycles with no deltas before marking read-only
    pub read_only_threshold: u32,
}

impl KeyframeStore {
    pub fn new(consolidation_threshold: usize, read_only_threshold: u32) -> Self {
        Self {
            frames: HashMap::new(),
            next_id: 0,
            consolidation_threshold,
            read_only_threshold,
        }
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /// Compress `data` as a new keyframe and return its ID.
    pub fn take_keyframe(&mut self, data: &[u8]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.frames.insert(id, Keyframe::new(id, data));
        id
    }

    /// Record a delta for keyframe `id` vs `current_data`.
    ///
    /// Only the changed bytes are stored (sparse).  If nothing changed,
    /// `None` is returned and nothing is stored.  When the delta chain
    /// reaches `consolidation_threshold`, the frame is automatically
    /// consolidated before the new delta is appended.
    ///
    /// Returns the delta ID on success.
    pub fn record_delta(&mut self, id: u32, current_data: &[u8]) -> Option<u32> {
        // Build the delta first (immutable borrow ends before we mutate)
        let (delta_id, delta) = {
            let frame = self.frames.get(&id)?;
            if frame.is_read_only {
                return None;
            }

            let delta_id = frame.deltas.len() as u32;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);

            let delta = frame.build_delta(delta_id, ts, current_data)?;
            (delta_id, delta)
        };

        // Auto-consolidate if we hit the threshold
        {
            let frame = self.frames.get(&id)?;
            if frame.deltas.len() >= self.consolidation_threshold {
                // We need to consolidate; do it before appending
                let _ = frame; // end borrow (drop reference, not value)
                self.consolidate(id);
            }
        }

        let frame = self.frames.get_mut(&id)?;
        frame.observation_cycles = 0; // activity resets the counter
        frame.deltas.push(delta);
        Some(delta_id)
    }

    /// Reconstruct the full logical data for keyframe `id`.
    pub fn reconstruct(&self, id: u32) -> Option<Vec<u8>> {
        self.frames.get(&id)?.reconstruct()
    }

    /// Reconstruct only `length` bytes starting at `offset` for keyframe `id`.
    pub fn reconstruct_range(&self, id: u32, offset: usize, length: usize) -> Option<Vec<u8>> {
        self.frames.get(&id)?.reconstruct_range(offset, length)
    }

    /// Fold the full delta chain back into a fresh compressed keyframe,
    /// resetting the delta chain to empty.
    pub fn consolidate(&mut self, id: u32) {
        let reconstructed = match self.frames.get(&id).and_then(|f| f.reconstruct()) {
            Some(data) => data,
            None => return,
        };

        if let Some(frame) = self.frames.get_mut(&id) {
            let hash_before = frame.original_hash;
            // Rebuild from scratch: fresh LZ4 + empty delta chain
            let new_compressed = lz4_flex::compress_prepend_size(&reconstructed);
            frame.compressed_data = new_compressed;
            frame.original_size = reconstructed.len();
            frame.original_hash = hash_bytes(&reconstructed);
            frame.deltas.clear();
            let _ = hash_before; // hash of original keyframe no longer relevant
        }
    }

    /// Check (and apply) the read-only transition for keyframe `id`.
    ///
    /// Returns `true` if the frame is now (or was already) read-only.
    pub fn check_read_only(&mut self, id: u32) -> bool {
        if let Some(frame) = self.frames.get_mut(&id) {
            if !frame.is_read_only
                && frame.deltas.is_empty()
                && frame.observation_cycles >= self.read_only_threshold
            {
                frame.is_read_only = true;
            }
            frame.is_read_only
        } else {
            false
        }
    }

    /// Increment the observation counter for keyframe `id`.
    ///
    /// Call this on every "tick" or scan cycle.  The counter only advances
    /// when there are no new deltas (activity resets it to zero in
    /// `record_delta`).  After `read_only_threshold` idle cycles the frame
    /// transitions to read-only via `check_read_only`.
    pub fn mark_observation_cycle(&mut self, id: u32) {
        if let Some(frame) = self.frames.get_mut(&id) {
            if !frame.is_read_only {
                frame.observation_cycles += 1;
                // Automatically apply the transition check each cycle
                if frame.deltas.is_empty()
                    && frame.observation_cycles >= self.read_only_threshold
                {
                    frame.is_read_only = true;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors / diagnostics
    // -----------------------------------------------------------------------

    pub fn delta_count(&self, id: u32) -> usize {
        self.frames.get(&id).map(|f| f.deltas.len()).unwrap_or(0)
    }

    pub fn is_read_only(&self, id: u32) -> bool {
        self.frames.get(&id).map(|f| f.is_read_only).unwrap_or(false)
    }

    pub fn original_hash(&self, id: u32) -> Option<u64> {
        self.frames.get(&id).map(|f| f.original_hash)
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> KeyframeStore {
        KeyframeStore::new(10, 3)
    }

    // -----------------------------------------------------------------------
    // test_keyframe_roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_keyframe_roundtrip() {
        let mut store = make_store();
        let original: Vec<u8> = (0..=255u8).cycle().take(4096).collect();

        let id = store.take_keyframe(&original);
        let restored = store.reconstruct(id).expect("reconstruct should succeed");

        assert_eq!(restored, original, "Roundtrip must be byte-identical");
    }

    // -----------------------------------------------------------------------
    // test_delta_captures_changes
    // -----------------------------------------------------------------------

    #[test]
    fn test_delta_captures_changes() {
        let mut store = make_store();

        // 64KB baseline of 0xAA bytes
        let baseline = vec![0xAAu8; 65_536];
        let id = store.take_keyframe(&baseline);

        // Modify exactly 10 bytes near offset 1000
        let mut modified = baseline.clone();
        for i in 0..10 {
            modified[1000 + i] = 0xFF;
        }

        let delta_id = store.record_delta(id, &modified)
            .expect("Should store a non-empty delta");
        assert_eq!(delta_id, 0);

        // Inspect the delta payload size — must be ≈ 10 bytes, not 64KB
        let frame = &store.frames[&id];
        let delta = &frame.deltas[0];
        assert_eq!(delta.cumulative_change_bytes, 10,
            "Delta payload must be sparse (~10 bytes), got {}",
            delta.cumulative_change_bytes);

        // Reconstruction must match the modified data
        let restored = store.reconstruct(id).expect("reconstruct");
        assert_eq!(restored, modified);
    }

    // -----------------------------------------------------------------------
    // test_multi_delta_reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_delta_reconstruction() {
        let mut store = make_store();

        let mut data: Vec<u8> = vec![0u8; 8192];
        let id = store.take_keyframe(&data);

        // Apply 5 successive mutations, recording a delta after each
        for step in 0u8..5 {
            let offset = (step as usize) * 100;
            data[offset] = step + 1;
            store.record_delta(id, &data)
                .expect("non-empty delta expected");
        }

        assert_eq!(store.delta_count(id), 5);

        let restored = store.reconstruct(id).expect("reconstruct");
        assert_eq!(restored, data, "Multi-delta reconstruction must match final state");
    }

    // -----------------------------------------------------------------------
    // test_consolidation_resets_deltas
    // -----------------------------------------------------------------------

    #[test]
    fn test_consolidation_resets_deltas() {
        let mut store = make_store();

        let mut data = vec![0u8; 4096];
        let id = store.take_keyframe(&data);

        // Record a few deltas
        for i in 0u8..3 {
            data[i as usize * 50] = i + 10;
            store.record_delta(id, &data).unwrap();
        }
        assert_eq!(store.delta_count(id), 3);

        store.consolidate(id);

        assert_eq!(store.delta_count(id), 0, "Consolidation must clear the delta chain");

        // Reconstruction after consolidation must still produce the correct data
        let restored = store.reconstruct(id).expect("reconstruct after consolidate");
        assert_eq!(restored, data, "Data must survive consolidation");
    }

    // -----------------------------------------------------------------------
    // test_read_only_detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_only_detection() {
        // read_only_threshold = 3 cycles
        let mut store = KeyframeStore::new(10, 3);
        let data = vec![42u8; 1024];
        let id = store.take_keyframe(&data);

        assert!(!store.is_read_only(id));

        // Fewer than threshold cycles — not yet read-only
        store.mark_observation_cycle(id);
        store.mark_observation_cycle(id);
        assert!(!store.is_read_only(id));

        // Third cycle crosses the threshold
        store.mark_observation_cycle(id);
        assert!(store.is_read_only(id), "Should be read-only after threshold cycles with no deltas");

        // check_read_only should also return true
        assert!(store.check_read_only(id));
    }

    // -----------------------------------------------------------------------
    // test_selective_reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_selective_reconstruction() {
        let mut store = make_store();

        // 64KB baseline — every byte equals its index mod 256
        let original: Vec<u8> = (0u8..=255).cycle().take(65_536).collect();
        let id = store.take_keyframe(&original);

        // Modify bytes far outside our target range
        let mut modified = original.clone();
        modified[40_000] = 0xFF;
        modified[50_000] = 0xEE;
        store.record_delta(id, &modified).unwrap();

        // Reconstruct a 100-byte slice at offset 0 (unaffected by the deltas)
        let slice = store.reconstruct_range(id, 0, 100)
            .expect("selective reconstruct");

        assert_eq!(slice.len(), 100);
        assert_eq!(&slice[..], &modified[0..100],
            "Selective range must match full reconstruction for same slice");

        // Also verify a range that DOES include a changed byte
        let changed_slice = store.reconstruct_range(id, 39_999, 3)
            .expect("reconstruct around changed byte");
        assert_eq!(changed_slice[1], 0xFF, "Changed byte must be visible in range reconstruct");
    }

    // -----------------------------------------------------------------------
    // test_empty_delta
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_delta() {
        let mut store = make_store();
        let data = vec![7u8; 2048];
        let id = store.take_keyframe(&data);

        // Record the identical data — nothing changed
        let result = store.record_delta(id, &data);

        assert!(result.is_none(), "Identical data must produce no delta");
        assert_eq!(store.delta_count(id), 0);
    }
}
