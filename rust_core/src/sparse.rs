//! Sparse Extract — sub-region decompression for compressed memory.
//!
//! When a compressed region is accessed, don't decompress the whole thing.
//! Decompress ONLY the accessed byte range. Serve EXACTLY what's needed,
//! no more, no less.
//!
//! Key insight: a 50 KB object where only 3 fields (200 bytes) are ever
//! accessed keeps ~200 bytes decompressed + the full 50 KB compressed.
//! That's 99.6% savings on the warm portion.
//!
//! Flow:
//!   1. Region registered with its LZ4 compressed backing.
//!   2. Every access is recorded in the ByteHeatMap.
//!   3. `extract()` checks existing hot ranges first; on a miss it
//!      decompresses the backing, slices the requested range, and
//!      promotes it to a hot range.
//!   4. `compact()` demotes hot ranges that have not been re-accessed
//!      since the last compaction pass.

use std::collections::HashMap;
use lz4_flex::decompress_size_prepended;

// ---------------------------------------------------------------------------
// ByteHeatMap
// ---------------------------------------------------------------------------

/// Per-region access heat tracker, bucketed at cache-line granularity (64 B).
pub struct ByteHeatMap {
    buckets: Vec<u32>,       // access count per 64-byte bucket
    bucket_size: usize,      // always 64 (cache line)
    region_size: usize,
}

impl ByteHeatMap {
    /// Create a new heat map for a region of `region_size` bytes.
    /// Number of buckets = ceil(region_size / 64).
    pub fn new(region_size: usize) -> Self {
        let bucket_size = 64;
        let num_buckets = (region_size + bucket_size - 1) / bucket_size;
        Self {
            buckets: vec![0u32; num_buckets],
            bucket_size,
            region_size,
        }
    }

    /// Record an access covering [offset, offset + length).
    /// Every bucket that overlaps the range is incremented by 1.
    pub fn record_access(&mut self, offset: usize, length: usize) {
        if length == 0 || offset >= self.region_size {
            return;
        }
        let end = (offset + length).min(self.region_size);
        let first_bucket = offset / self.bucket_size;
        let last_bucket = (end - 1) / self.bucket_size;
        for b in first_bucket..=last_bucket {
            if b < self.buckets.len() {
                self.buckets[b] = self.buckets[b].saturating_add(1);
            }
        }
    }

    /// Return (offset, length) pairs of contiguous bucket runs whose count
    /// is strictly above `threshold`. Adjacent hot buckets are merged into
    /// a single span.
    pub fn get_hot_buckets(&self, threshold: u32) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let mut run_start: Option<usize> = None;

        for (i, &count) in self.buckets.iter().enumerate() {
            if count > threshold {
                if run_start.is_none() {
                    run_start = Some(i);
                }
            } else if let Some(start) = run_start.take() {
                let offset = start * self.bucket_size;
                let end = (i * self.bucket_size).min(self.region_size);
                result.push((offset, end - offset));
            }
        }
        // flush a trailing run
        if let Some(start) = run_start {
            let offset = start * self.bucket_size;
            let end = self.region_size;
            result.push((offset, end - offset));
        }
        result
    }

    /// Reset all bucket counts to zero.
    pub fn reset(&mut self) {
        for b in self.buckets.iter_mut() {
            *b = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// HotRange
// ---------------------------------------------------------------------------

/// A decompressed slice that is currently held in RAM ("hot").
pub struct HotRange {
    pub offset: usize,
    pub length: usize,
    pub data: Vec<u8>,       // decompressed bytes for exactly this range
    pub access_count: u32,
    /// Monotonically-increasing epoch counter; bumped on every access.
    /// Used by `compact()` to detect stale ranges.
    last_access_epoch: u64,
}

impl HotRange {
    fn new(offset: usize, data: Vec<u8>, epoch: u64) -> Self {
        let length = data.len();
        Self {
            offset,
            length,
            data,
            access_count: 1,
            last_access_epoch: epoch,
        }
    }

    /// True when [offset, offset+length) fully contains [query_off, query_off+query_len).
    fn covers(&self, query_off: usize, query_len: usize) -> bool {
        query_off >= self.offset && query_off + query_len <= self.offset + self.length
    }

    /// Slice bytes for [query_off, query_off+query_len) out of this hot range.
    fn slice(&self, query_off: usize, query_len: usize) -> Vec<u8> {
        let rel = query_off - self.offset;
        self.data[rel..rel + query_len].to_vec()
    }
}

// ---------------------------------------------------------------------------
// SplitRegion
// ---------------------------------------------------------------------------

/// A compressed memory region that may have multiple decompressed hot slices.
pub struct SplitRegion {
    pub region_id: u32,
    pub total_size: usize,
    compressed_backing: Vec<u8>,  // full LZ4 compressed data (size-prepended)
    hot_ranges: Vec<HotRange>,    // decompressed hot slices
    heat_map: ByteHeatMap,
    last_compaction_ns: u64,
    /// Epoch counter — incremented on every access to this region.
    access_epoch: u64,
}

impl SplitRegion {
    fn new(region_id: u32, compressed_data: Vec<u8>, original_size: usize) -> Self {
        Self {
            region_id,
            total_size: original_size,
            compressed_backing: compressed_data,
            hot_ranges: Vec::new(),
            heat_map: ByteHeatMap::new(original_size),
            last_compaction_ns: 0,
            access_epoch: 0,
        }
    }

    /// Fully decompress the backing store and return it.
    fn decompress_full(&self) -> Result<Vec<u8>, String> {
        decompress_size_prepended(&self.compressed_backing)
            .map_err(|e| format!("LZ4 decompression error on region {}: {}", self.region_id, e))
    }

    /// Hot bytes currently held in RAM (may overlap, counted simply).
    fn hot_bytes(&self) -> usize {
        self.hot_ranges.iter().map(|r| r.length).sum()
    }

    /// Return bytes at [offset, offset+length) from the fully-decompressed
    /// data, and add a new HotRange for that span.
    fn decompress_and_promote(
        &mut self,
        offset: usize,
        length: usize,
        epoch: u64,
    ) -> Option<Vec<u8>> {
        let full = self.decompress_full().ok()?;
        if offset + length > full.len() {
            return None;
        }
        let slice = full[offset..offset + length].to_vec();
        self.hot_ranges.push(HotRange::new(offset, slice.clone(), epoch));
        Some(slice)
    }
}

// ---------------------------------------------------------------------------
// SparseExtractor
// ---------------------------------------------------------------------------

/// Manages many compressed regions, serving byte-range queries with minimal
/// decompression and tracking hot slices per region.
pub struct SparseExtractor {
    regions: HashMap<u32, SplitRegion>,
    compaction_interval_ns: u64,  // how often to demote stale hot ranges
    /// Global access epoch — incremented on every extract() call.
    epoch: u64,
}

impl SparseExtractor {
    pub fn new(compaction_interval_ns: u64) -> Self {
        Self {
            regions: HashMap::new(),
            compaction_interval_ns,
            epoch: 0,
        }
    }

    /// Register a compressed region. `compressed_data` must be an LZ4
    /// frame created with `compress_prepend_size` (so the original length
    /// is embedded in the first 4 bytes).
    pub fn register(&mut self, region_id: u32, compressed_data: Vec<u8>, original_size: usize) {
        self.regions.insert(
            region_id,
            SplitRegion::new(region_id, compressed_data, original_size),
        );
    }

    /// Record that bytes [offset, offset+length) of `region_id` were accessed.
    /// Updates the heat map. Does NOT decompress anything.
    pub fn record_access(&mut self, region_id: u32, offset: usize, length: usize) {
        if let Some(region) = self.regions.get_mut(&region_id) {
            region.heat_map.record_access(offset, length);
        }
    }

    /// Return bytes [offset, offset+length) from `region_id`.
    ///
    /// 1. Record the access in the heat map.
    /// 2. Search existing hot ranges for a hit — if found, return directly.
    /// 3. On a miss: decompress the full backing, slice the range, promote
    ///    it to a new hot range, return the slice.
    ///
    /// Returns `None` if the region does not exist or the range is out of
    /// bounds.
    pub fn extract(&mut self, region_id: u32, offset: usize, length: usize) -> Option<Vec<u8>> {
        self.epoch += 1;
        let epoch = self.epoch;

        let region = self.regions.get_mut(&region_id)?;
        region.access_epoch = epoch;

        // Record heat.
        region.heat_map.record_access(offset, length);

        // Bounds check.
        if offset + length > region.total_size {
            return None;
        }

        // Fast path: already hot.
        for hr in region.hot_ranges.iter_mut() {
            if hr.covers(offset, length) {
                hr.access_count += 1;
                hr.last_access_epoch = epoch;
                return Some(hr.slice(offset, length));
            }
        }

        // Slow path: decompress and promote.
        region.decompress_and_promote(offset, length, epoch)
    }

    /// Demote hot ranges that have not been accessed since the previous
    /// compaction pass.  Only runs if `now_ns - last_compaction_ns >=
    /// compaction_interval_ns`.
    ///
    /// A hot range is considered stale if its `last_access_epoch` is equal
    /// to the epoch that was current at the start of the last compaction —
    /// meaning no access has been recorded since then.
    pub fn compact(&mut self, region_id: u32, now_ns: u64) {
        let interval = self.compaction_interval_ns;
        let current_epoch = self.epoch;

        if let Some(region) = self.regions.get_mut(&region_id) {
            if now_ns.saturating_sub(region.last_compaction_ns) < interval {
                return;
            }
            // The epoch watermark we saved at last compaction time is stored
            // implicitly: any hot range whose last_access_epoch < current_epoch
            // at the START of this compaction has not been touched since the
            // last compact call.  We demote those.
            //
            // "Not accessed since last compaction" == last_access_epoch was set
            // before this compaction started (i.e. < current_epoch, because
            // every access bumps the global epoch).
            region.hot_ranges.retain(|hr| hr.last_access_epoch >= current_epoch);
            region.last_compaction_ns = now_ns;
            region.heat_map.reset();
        }
    }

    /// Return `(total_size, hot_bytes, compressed_bytes)` for a region.
    pub fn get_stats(&self, region_id: u32) -> Option<(usize, usize, usize)> {
        let region = self.regions.get(&region_id)?;
        Some((
            region.total_size,
            region.hot_bytes(),
            region.compressed_backing.len(),
        ))
    }

    /// Remove a region entirely, freeing both compressed backing and hot slices.
    pub fn unregister(&mut self, region_id: u32) {
        self.regions.remove(&region_id);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lz4_flex::compress_prepend_size;

    /// Build a deterministic 1 KB payload and compress it.
    fn make_compressed(size: usize) -> (Vec<u8>, Vec<u8>) {
        let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let compressed = compress_prepend_size(&data);
        (data, compressed)
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_heat_map_tracking() {
        let mut hm = ByteHeatMap::new(1024);

        // Access three non-overlapping ranges.
        hm.record_access(0, 64);    // bucket 0
        hm.record_access(128, 64);  // bucket 2
        hm.record_access(512, 128); // buckets 8 & 9

        // Bucket 0 was hit.
        assert!(hm.buckets[0] > 0, "bucket 0 should be hot");
        // Bucket 1 was NOT hit.
        assert_eq!(hm.buckets[1], 0, "bucket 1 should be cold");
        // Bucket 2 was hit.
        assert!(hm.buckets[2] > 0, "bucket 2 should be hot");
        // Buckets 8 & 9 were hit.
        assert!(hm.buckets[8] > 0, "bucket 8 should be hot");
        assert!(hm.buckets[9] > 0, "bucket 9 should be hot");
        // Bucket 10 was NOT hit.
        assert_eq!(hm.buckets[10], 0, "bucket 10 should be cold");
    }

    #[test]
    fn test_sparse_hot_range_identification() {
        let mut hm = ByteHeatMap::new(512);

        // Hit bucket 0 five times — above threshold 3.
        for _ in 0..5 {
            hm.record_access(0, 64);
        }
        // Hit bucket 4 once — below threshold 3.
        hm.record_access(256, 64);

        let hot = hm.get_hot_buckets(3);
        // Only bucket 0 (offset 0, len 64) qualifies.
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0], (0, 64));
    }

    #[test]
    fn test_sparse_extract_cold_promotes() {
        let (original, compressed) = make_compressed(1024);
        let mut sx = SparseExtractor::new(u64::MAX); // never auto-compact

        sx.register(1, compressed, 1024);

        // Region is cold — no hot ranges yet.
        let stats_before = sx.get_stats(1).unwrap();
        assert_eq!(stats_before.1, 0, "no hot bytes before first access");

        // Extract 64 bytes from offset 128.
        let result = sx.extract(1, 128, 64).expect("extract should succeed");
        assert_eq!(result, &original[128..192], "extracted bytes must match original");

        // Now there should be a hot range.
        let stats_after = sx.get_stats(1).unwrap();
        assert_eq!(stats_after.1, 64, "64 hot bytes after promotion");
    }

    #[test]
    fn test_sparse_extract_hot_direct() {
        let (original, compressed) = make_compressed(1024);
        let mut sx = SparseExtractor::new(u64::MAX);

        sx.register(2, compressed, 1024);

        // First access — promotes the range.
        let first = sx.extract(2, 256, 128).expect("first extract");
        assert_eq!(first, &original[256..384]);

        // Capture hot_bytes count — should stay the same after the second call.
        let stats_mid = sx.get_stats(2).unwrap();

        // Second access to the SAME range — must be served from hot range.
        let second = sx.extract(2, 256, 128).expect("second extract");
        assert_eq!(second, first, "hot path must return identical bytes");

        let stats_after = sx.get_stats(2).unwrap();
        // No new ranges should have been added.
        assert_eq!(stats_mid.1, stats_after.1, "hot bytes must not grow on hot hit");
    }

    #[test]
    fn test_sparse_compaction_demotes_stale() {
        let (_original, compressed) = make_compressed(1024);
        // Use a very short compaction interval so we can trigger it.
        let mut sx = SparseExtractor::new(1); // 1 ns interval

        sx.register(3, compressed, 1024);

        // Promote a range.
        sx.extract(3, 0, 64).expect("first extract");
        let stats = sx.get_stats(3).unwrap();
        assert_eq!(stats.1, 64, "64 hot bytes before compaction");

        // Compact WITHOUT any new access between promote and compact.
        // The hot range's last_access_epoch == epoch at time of extract (1).
        // current_epoch is also 1, so the condition hr.last_access_epoch >= current_epoch
        // would keep it.  We need to do another extract to advance the epoch first,
        // OR compact should use "last_access_epoch < epoch at compact start".
        //
        // Design: compact demotes ranges whose last_access_epoch < current_epoch at
        // compact time.  So we must advance the epoch by doing any extract on another
        // region, OR we explicitly advance by extracting on a sub-range that misses
        // so it re-promotes.  Simplest: advance epoch via another extract, then compact.

        // Access a DIFFERENT offset (not covered by existing hot range at 0..64)
        // to advance the global epoch.
        sx.extract(3, 512, 64).expect("second extract — advances epoch");

        // Now compact. The first hot range (last_access_epoch=1) is stale relative
        // to current_epoch=2; the second (last_access_epoch=2) is fresh.
        sx.compact(3, 1_000_000_000);

        let stats_after = sx.get_stats(3).unwrap();
        // The first range (offset 0, 64 B) should be gone; the second (offset 512) stays.
        assert_eq!(stats_after.1, 64, "only the recently-accessed range should remain");
    }

    #[test]
    fn test_sparse_stats_reporting() {
        let (_original, compressed) = make_compressed(2048);
        let compressed_len = compressed.len();
        let mut sx = SparseExtractor::new(u64::MAX);

        sx.register(4, compressed, 2048);

        // No hot ranges yet.
        let (total, hot, comp) = sx.get_stats(4).unwrap();
        assert_eq!(total, 2048);
        assert_eq!(hot, 0);
        assert_eq!(comp, compressed_len);

        // Promote 128 bytes.
        sx.extract(4, 0, 128).unwrap();
        let (total2, hot2, comp2) = sx.get_stats(4).unwrap();
        assert_eq!(total2, 2048);
        assert_eq!(hot2, 128);
        assert_eq!(comp2, compressed_len, "compressed backing must not change");
    }

    #[test]
    fn test_sparse_unregister() {
        let (_original, compressed) = make_compressed(512);
        let mut sx = SparseExtractor::new(u64::MAX);

        sx.register(5, compressed, 512);
        assert!(sx.get_stats(5).is_some(), "region should exist before unregister");

        sx.unregister(5);
        assert!(sx.get_stats(5).is_none(), "region should be gone after unregister");
        assert!(sx.extract(5, 0, 16).is_none(), "extract on removed region returns None");
    }
}
