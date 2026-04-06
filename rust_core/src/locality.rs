//! Block H — Manufactured Spatial Locality + Software Prefetch
//!
//! The SNN knows causal chains A→B→C. This module places those nodes in
//! adjacent cache lines so the hardware prefetcher succeeds by construction,
//! then emits software prefetch instructions timed to spike propagation.

use std::collections::HashMap;
use libc;

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// A causally ordered sequence of memory regions with predicted inter-access
/// timings. Produced by the SNN's spike propagation layer.
pub struct CausalChain {
    pub nodes: Vec<u32>,        // region IDs in causal order
    pub timings_ms: Vec<f64>,   // predicted inter-access times (len == nodes.len() - 1)
    pub total_confidence: f64,
}

/// A spatial layout plan: arena offsets chosen so causally related regions
/// land in adjacent cache lines.
pub struct LayoutPlan {
    placements: HashMap<u32, usize>,   // region_id → arena byte offset
    chain_groups: Vec<Vec<u32>>,       // groups of co-located region IDs
}

/// Which cache level to target with a software prefetch instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrefetchHint {
    L1,   // predicted access < 1 ms away
    L2,   // 1 – 5 ms
    L3,   // 5 – 20 ms
    None, // > 20 ms — not worth prefetching
}

/// A single prefetch instruction to be issued.
pub struct PrefetchInstruction {
    pub address: usize,
    pub hint: PrefetchHint,
    pub predicted_ms: f64,
}

/// A contiguous mmap-backed arena. Allocations are 64-byte (cache-line) aligned.
/// The arena can be reorganised during sleep consolidation via `relocate`.
pub struct CondensateArena {
    base: *mut u8,
    size: usize,
    free_list: Vec<(usize, usize)>,              // (offset, size) sorted by offset
    allocations: HashMap<u32, (usize, usize)>,   // region_id → (offset, size)
    cache_line_size: usize,                      // always 64
}

// ────────────────────────────────────────────────────────────────────────────
// CausalChain
// ────────────────────────────────────────────────────────────────────────────

impl CausalChain {
    pub fn new(nodes: Vec<u32>, timings_ms: Vec<f64>, total_confidence: f64) -> Self {
        // timings_ms should have (nodes.len() - 1) entries, but we don't panic
        // on bad input — callers might build chains incrementally.
        Self { nodes, timings_ms, total_confidence }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LayoutPlan
// ────────────────────────────────────────────────────────────────────────────

impl LayoutPlan {
    pub fn new() -> Self {
        Self {
            placements: HashMap::new(),
            chain_groups: Vec::new(),
        }
    }

    /// Assign contiguous arena offsets to regions so that members of the same
    /// causal chain are spatially adjacent.
    ///
    /// Strategy:
    /// 1. Sort chains by descending `total_confidence` so the most trusted
    ///    chains claim their preferred layout first.
    /// 2. For each chain, walk its nodes in order. If a node has already been
    ///    placed (because it appeared in a higher-confidence chain), keep that
    ///    placement; otherwise assign the next available slot.
    /// 3. Slots are one cache line (64 bytes) wide for the purposes of the
    ///    plan. Actual allocation sizes are determined by `CondensateArena`.
    pub fn compute(chains: &[CausalChain]) -> Self {
        const CACHE_LINE: usize = 64;

        let mut plan = Self::new();

        // Work on a sorted copy (by descending confidence).
        let mut order: Vec<usize> = (0..chains.len()).collect();
        order.sort_by(|&a, &b| {
            chains[b]
                .total_confidence
                .partial_cmp(&chains[a].total_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_offset: usize = 0;

        for chain_idx in order {
            let chain = &chains[chain_idx];
            let mut group: Vec<u32> = Vec::new();

            for &node in &chain.nodes {
                if !plan.placements.contains_key(&node) {
                    plan.placements.insert(node, next_offset);
                    next_offset += CACHE_LINE;
                }
                group.push(node);
            }

            if !group.is_empty() {
                plan.chain_groups.push(group);
            }
        }

        plan
    }

    /// Get the planned arena offset for a region.
    pub fn get_placement(&self, region_id: u32) -> Option<usize> {
        self.placements.get(&region_id).copied()
    }

    /// Get the chain group that contains a region (first match wins).
    pub fn get_chain_group(&self, region_id: u32) -> Option<&Vec<u32>> {
        self.chain_groups
            .iter()
            .find(|group| group.contains(&region_id))
    }
}

impl Default for LayoutPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PrefetchHint
// ────────────────────────────────────────────────────────────────────────────

impl PrefetchHint {
    /// Map a predicted inter-access time to the appropriate cache level.
    pub fn from_timing(predicted_ms: f64) -> Self {
        if predicted_ms < 1.0 {
            PrefetchHint::L1
        } else if predicted_ms < 5.0 {
            PrefetchHint::L2
        } else if predicted_ms <= 20.0 {
            PrefetchHint::L3
        } else {
            PrefetchHint::None
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CondensateArena
// ────────────────────────────────────────────────────────────────────────────

// Mark as Send so it can cross thread boundaries in the pipeline.
// SAFETY: The arena owns its memory exclusively; access must be serialised by
// the caller (the pipeline uses a Mutex<CondensateArena>).
unsafe impl Send for CondensateArena {}

impl CondensateArena {
    /// Allocate a contiguous anonymous private mapping of `size` bytes.
    pub fn new(size: usize) -> Self {
        // SAFETY: mmap with MAP_ANON | MAP_PRIVATE creates a fresh zero-filled
        // mapping. We check for MAP_FAILED before using the pointer.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        assert_ne!(
            base,
            libc::MAP_FAILED,
            "CondensateArena: mmap({size}) failed"
        );

        Self {
            base: base as *mut u8,
            size,
            free_list: vec![(0, size)],
            allocations: HashMap::new(),
            cache_line_size: 64,
        }
    }

    /// Round `offset` up to the next multiple of `align`.
    #[inline]
    fn align_up(offset: usize, align: usize) -> usize {
        (offset + align - 1) & !(align - 1)
    }

    /// Allocate `size` bytes for `region_id`, aligned to `cache_line_size`.
    /// Returns a raw pointer into the arena on success.
    pub fn allocate(&mut self, region_id: u32, size: usize) -> Option<*mut u8> {
        if self.allocations.contains_key(&region_id) {
            return None; // already allocated
        }

        let align = self.cache_line_size;
        let aligned_size = Self::align_up(size, align);

        // Find the first free block that fits after alignment.
        let mut chosen: Option<usize> = None;
        for (i, &(blk_off, blk_size)) in self.free_list.iter().enumerate() {
            let aligned_start = Self::align_up(blk_off, align);
            let padding = aligned_start - blk_off;
            if blk_size >= aligned_size + padding {
                chosen = Some(i);
                break;
            }
        }

        let idx = chosen?;
        let (blk_off, blk_size) = self.free_list[idx];
        let start = Self::align_up(blk_off, align);
        let padding = start - blk_off;
        let consumed = aligned_size + padding;

        self.free_list.remove(idx);

        // Return any leading padding as a free fragment.
        if padding > 0 {
            self.free_list.push((blk_off, padding));
        }
        // Return any trailing space.
        let trailing_off = start + aligned_size;
        let trailing_size = blk_size - consumed;
        if trailing_size > 0 {
            self.free_list.push((trailing_off, trailing_size));
        }

        self.free_list.sort_by_key(|&(off, _)| off);
        self.allocations.insert(region_id, (start, aligned_size));

        // SAFETY: `start` is within [0, self.size) because we checked blk_size
        // above. base is a valid mmap pointer for at least `self.size` bytes.
        Some(unsafe { self.base.add(start) })
    }

    /// Attempt to allocate at a specific byte offset (used by LayoutPlan).
    /// The requested range must lie entirely within a single free block.
    pub fn allocate_at(
        &mut self,
        region_id: u32,
        offset: usize,
        size: usize,
    ) -> Option<*mut u8> {
        if self.allocations.contains_key(&region_id) {
            return None;
        }

        let align = self.cache_line_size;
        let aligned_start = Self::align_up(offset, align);
        let aligned_size = Self::align_up(size, align);

        if aligned_start + aligned_size > self.size {
            return None;
        }

        // Find a free block that fully contains [aligned_start, aligned_start + aligned_size).
        let found = self.free_list.iter().enumerate().find(|(_, &(blk_off, blk_size))| {
            blk_off <= aligned_start && aligned_start + aligned_size <= blk_off + blk_size
        });

        let (idx, &(blk_off, blk_size)) = found?;
        self.free_list.remove(idx);

        // Return leading fragment.
        if aligned_start > blk_off {
            self.free_list.push((blk_off, aligned_start - blk_off));
        }
        // Return trailing fragment.
        let end = aligned_start + aligned_size;
        let blk_end = blk_off + blk_size;
        if end < blk_end {
            self.free_list.push((end, blk_end - end));
        }

        self.free_list.sort_by_key(|&(off, _)| off);
        self.allocations.insert(region_id, (aligned_start, aligned_size));

        // SAFETY: aligned_start is within the mmap'd region (checked above).
        Some(unsafe { self.base.add(aligned_start) })
    }

    /// Return a region's allocation to the free list, then coalesce adjacent
    /// free blocks so fragmentation doesn't grow unboundedly.
    pub fn free(&mut self, region_id: u32) {
        if let Some((offset, size)) = self.allocations.remove(&region_id) {
            self.free_list.push((offset, size));
            self.free_list.sort_by_key(|&(off, _)| off);
            self.coalesce();
        }
    }

    /// Merge adjacent free blocks. Called after every `free`.
    fn coalesce(&mut self) {
        if self.free_list.len() < 2 {
            return;
        }

        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free_list.len());
        let mut iter = self.free_list.drain(..);
        let (mut cur_off, mut cur_size) = iter.next().unwrap();

        for (off, sz) in iter {
            if off == cur_off + cur_size {
                // Adjacent — extend current block.
                cur_size += sz;
            } else {
                merged.push((cur_off, cur_size));
                cur_off = off;
                cur_size = sz;
            }
        }
        merged.push((cur_off, cur_size));
        self.free_list = merged;
    }

    /// Move a region's data to `new_offset` within the arena (memcpy).
    /// Used by the sleep consolidation pass to tighten the layout.
    /// Returns `true` on success, `false` if the move isn't possible.
    pub fn relocate(&mut self, region_id: u32, new_offset: usize) -> bool {
        let (old_offset, size) = match self.allocations.get(&region_id).copied() {
            Some(v) => v,
            None => return false,
        };

        let aligned_new = Self::align_up(new_offset, self.cache_line_size);

        if aligned_new == old_offset {
            return true; // already there
        }

        if aligned_new + size > self.size {
            return false;
        }

        // The destination range must be free (or be the source itself).
        // We check by temporarily freeing the source and trying allocate_at.
        // To avoid double-borrow, we do it manually.

        // Check destination is free.
        let dest_free = self.free_list.iter().any(|&(blk_off, blk_size)| {
            blk_off <= aligned_new && aligned_new + size <= blk_off + blk_size
        });
        if !dest_free {
            return false;
        }

        // SAFETY: Both source and destination are within [base, base+size).
        // We checked all offsets above. src and dst may not overlap — if they
        // do, memmove semantics are required; we use copy_nonoverlapping only
        // when the ranges are disjoint, which is guaranteed because aligned_new
        // comes from the free list (i.e., it does not overlap old_offset..old_offset+size).
        unsafe {
            let src = self.base.add(old_offset);
            let dst = self.base.add(aligned_new);
            std::ptr::copy(src, dst, size); // copy handles overlap correctly
        }

        // Update the free list: old range becomes free, new range consumed.
        // We already verified new range is free, so remove it from free list.
        let dest_idx = self
            .free_list
            .iter()
            .position(|&(blk_off, blk_size)| {
                blk_off <= aligned_new && aligned_new + size <= blk_off + blk_size
            })
            .unwrap();
        let (blk_off, blk_size) = self.free_list.remove(dest_idx);

        if blk_off < aligned_new {
            self.free_list.push((blk_off, aligned_new - blk_off));
        }
        let blk_end = blk_off + blk_size;
        let dest_end = aligned_new + size;
        if dest_end < blk_end {
            self.free_list.push((dest_end, blk_end - dest_end));
        }

        // Old range is now free.
        self.free_list.push((old_offset, size));
        self.free_list.sort_by_key(|&(off, _)| off);
        self.coalesce();

        self.allocations.insert(region_id, (aligned_new, size));
        true
    }

    /// Get the current pointer for a region.
    pub fn get_ptr(&self, region_id: u32) -> Option<*mut u8> {
        self.allocations.get(&region_id).map(|&(off, _)| {
            // SAFETY: offset was validated at allocation time and is within
            // the mmap'd region.
            unsafe { self.base.add(off) }
        })
    }

    /// Returns `(total_size, allocated_bytes, free_bytes)`.
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let allocated: usize = self.allocations.values().map(|&(_, sz)| sz).sum();
        let free: usize = self.free_list.iter().map(|&(_, sz)| sz).sum();
        (self.size, allocated, free)
    }

    /// For each node that follows `current_node` in `chain`, emit a
    /// `PrefetchInstruction` based on cumulative timing from the current node.
    ///
    /// The prefetch addresses come from the arena's allocation map so they
    /// point at actual data — regions not yet allocated are skipped.
    pub fn prefetch_chain(
        &self,
        chain: &CausalChain,
        current_node: u32,
    ) -> Vec<PrefetchInstruction> {
        let mut instructions = Vec::new();

        // Find the position of current_node in the chain.
        let pos = match chain.nodes.iter().position(|&n| n == current_node) {
            Some(p) => p,
            None => return instructions,
        };

        // Accumulate timing from current_node outward.
        let mut cumulative_ms = 0.0_f64;

        for i in (pos + 1)..chain.nodes.len() {
            // timing[i-1] is the gap between node[i-1] and node[i].
            if let Some(&gap) = chain.timings_ms.get(i - 1) {
                cumulative_ms += gap;
            } else {
                break;
            }

            let next_node = chain.nodes[i];

            if let Some(&(offset, _)) = self.allocations.get(&next_node) {
                let address = offset; // offset into arena; caller adds base if needed
                let hint = PrefetchHint::from_timing(cumulative_ms);

                // Emit the actual x86_64 prefetch instruction when possible.
                #[cfg(target_arch = "x86_64")]
                {
                    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2};
                    // SAFETY: The pointer is within the mmap'd arena and the
                    // data is valid memory. Prefetch faults are suppressed by
                    // the CPU; worst case it's a no-op.
                    unsafe {
                        let ptr = self.base.add(offset) as *const i8;
                        match hint {
                            PrefetchHint::L1 => _mm_prefetch(ptr, _MM_HINT_T0),
                            PrefetchHint::L2 => _mm_prefetch(ptr, _MM_HINT_T1),
                            PrefetchHint::L3 => _mm_prefetch(ptr, _MM_HINT_T2),
                            PrefetchHint::None => {} // not worth it
                        }
                    }
                }

                instructions.push(PrefetchInstruction {
                    address,
                    hint,
                    predicted_ms: cumulative_ms,
                });
            }
        }

        instructions
    }
}

impl Drop for CondensateArena {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: `self.base` was obtained from `libc::mmap` with
            // `self.size` bytes. We own this mapping exclusively and are now
            // releasing it. No references into the arena can outlive `self`
            // because the raw pointers returned by `allocate`/`get_ptr` are
            // not lifetime-tracked — callers must ensure they don't outlive
            // the arena.
            unsafe {
                libc::munmap(self.base as *mut libc::c_void, self.size);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PrefetchHint ─────────────────────────────────────────────────────────

    #[test]
    fn locality_test_prefetch_hint_mapping() {
        assert_eq!(PrefetchHint::from_timing(0.5), PrefetchHint::L1);
        assert_eq!(PrefetchHint::from_timing(3.0), PrefetchHint::L2);
        assert_eq!(PrefetchHint::from_timing(10.0), PrefetchHint::L3);
        assert_eq!(PrefetchHint::from_timing(50.0), PrefetchHint::None);

        // Boundary checks
        assert_eq!(PrefetchHint::from_timing(0.999), PrefetchHint::L1);
        assert_eq!(PrefetchHint::from_timing(1.0), PrefetchHint::L2);
        assert_eq!(PrefetchHint::from_timing(5.0), PrefetchHint::L3);
        assert_eq!(PrefetchHint::from_timing(20.0), PrefetchHint::L3);
        assert_eq!(PrefetchHint::from_timing(20.001), PrefetchHint::None);
    }

    // ── LayoutPlan ───────────────────────────────────────────────────────────

    #[test]
    fn locality_test_layout_chain_adjacency() {
        // Chain A→B→C should produce consecutive offsets 64 bytes apart.
        let chain = CausalChain::new(
            vec![1, 2, 3],
            vec![0.5, 0.5],
            0.9,
        );
        let plan = LayoutPlan::compute(&[chain]);

        let a = plan.get_placement(1).expect("A not placed");
        let b = plan.get_placement(2).expect("B not placed");
        let c = plan.get_placement(3).expect("C not placed");

        // Each slot is one cache line (64 bytes).
        assert_eq!(b, a + 64, "B should be one cache line after A");
        assert_eq!(c, a + 128, "C should be two cache lines after A");

        // All three should be in the same group.
        let group = plan.get_chain_group(1).expect("no group for A");
        assert!(group.contains(&1));
        assert!(group.contains(&2));
        assert!(group.contains(&3));
    }

    #[test]
    fn locality_test_layout_shared_node() {
        // Node 2 appears in both chains; it should get a stable placement.
        let chain1 = CausalChain::new(vec![1, 2, 3], vec![1.0, 1.0], 0.9);
        let chain2 = CausalChain::new(vec![4, 2, 5], vec![1.0, 1.0], 0.5);
        let plan = LayoutPlan::compute(&[chain1, chain2]);

        // All five nodes should have placements.
        for id in [1u32, 2, 3, 4, 5] {
            assert!(plan.get_placement(id).is_some(), "node {id} not placed");
        }
        // Node 2 should be in a group.
        assert!(plan.get_chain_group(2).is_some());
    }

    // ── CondensateArena ──────────────────────────────────────────────────────

    #[test]
    fn locality_test_arena_allocate_aligned() {
        let mut arena = CondensateArena::new(4096);
        for id in 0u32..8 {
            let ptr = arena.allocate(id, 100).expect("allocation failed");
            assert_eq!(
                ptr as usize % 64,
                0,
                "allocation for region {id} is not 64-byte aligned"
            );
        }
    }

    #[test]
    fn locality_test_arena_allocate_free_reuse() {
        let mut arena = CondensateArena::new(4096);

        let ptr1 = arena.allocate(1, 64).expect("first alloc");
        let off1 = ptr1 as usize;

        arena.free(1);

        let ptr2 = arena.allocate(2, 64).expect("second alloc after free");
        let off2 = ptr2 as usize;

        // After a free + coalesce, the same offset should be reused.
        assert_eq!(off1, off2, "freed space should be reused");

        let (total, allocated, free) = arena.get_stats();
        assert_eq!(total, 4096);
        assert!(allocated > 0);
        assert_eq!(total, allocated + free);
    }

    #[test]
    fn locality_test_arena_relocate() {
        let mut arena = CondensateArena::new(4096);

        // Allocate region 1 and write a known pattern.
        let ptr = arena.allocate(1, 64).expect("alloc");
        // SAFETY: ptr is valid for 64 bytes — we just allocated it.
        unsafe {
            for i in 0..64usize {
                ptr.add(i).write(i as u8);
            }
        }

        // Allocate and free region 2 to open a gap at a higher offset.
        let ptr2 = arena.allocate(2, 64).expect("alloc 2");
        let new_offset = ptr2 as usize - arena.base as usize;
        arena.free(2);

        // Relocate region 1 into that gap.
        assert!(arena.relocate(1, new_offset), "relocate failed");

        // Verify data integrity.
        let moved_ptr = arena.get_ptr(1).expect("ptr after relocate");
        // SAFETY: moved_ptr is valid for 64 bytes after a successful relocate.
        unsafe {
            for i in 0..64usize {
                assert_eq!(
                    moved_ptr.add(i).read(),
                    i as u8,
                    "data corruption at byte {i} after relocate"
                );
            }
        }
    }

    #[test]
    fn locality_test_arena_coalesce() {
        let mut arena = CondensateArena::new(4096);

        // Fill arena with three adjacent regions.
        arena.allocate(1, 64).unwrap();
        arena.allocate(2, 64).unwrap();
        arena.allocate(3, 64).unwrap();

        // Free all three — they should coalesce into one big block.
        arena.free(1);
        arena.free(2);
        arena.free(3);

        // After coalescing we should be able to allocate a region larger than
        // one slot (e.g., 192 bytes spanning the three former slots).
        let big = arena.allocate(99, 192);
        assert!(big.is_some(), "coalesced free space should satisfy 192-byte alloc");
    }

    // ── Prefetch chain ───────────────────────────────────────────────────────

    #[test]
    fn locality_test_prefetch_chain_generation() {
        // Chain: A(0) →0.5ms→ B(1) →3ms→ C(2)
        // From A: expect prefetch for B (L1, 0.5ms) and C (L2, 3.5ms cumulative).
        let chain = CausalChain::new(
            vec![10, 11, 12],
            vec![0.5, 3.0],
            0.95,
        );

        let mut arena = CondensateArena::new(4096);
        // Allocate all nodes so addresses are available.
        arena.allocate(10, 64).unwrap();
        arena.allocate(11, 64).unwrap();
        arena.allocate(12, 64).unwrap();

        let instrs = arena.prefetch_chain(&chain, 10);
        assert_eq!(instrs.len(), 2, "should emit prefetch for B and C");

        // First instruction: B, 0.5ms → L1
        assert_eq!(instrs[0].hint, PrefetchHint::L1);
        assert!((instrs[0].predicted_ms - 0.5).abs() < 1e-9);

        // Second instruction: C, 3.5ms cumulative → L2
        assert_eq!(instrs[1].hint, PrefetchHint::L2);
        assert!((instrs[1].predicted_ms - 3.5).abs() < 1e-9);

        // From B: only C should be prefetched.
        let instrs_b = arena.prefetch_chain(&chain, 11);
        assert_eq!(instrs_b.len(), 1);
        // 3.0ms is in [1.0, 5.0) → L2
        assert_eq!(instrs_b[0].hint, PrefetchHint::L2);

        // From C (tail): no prefetch.
        let instrs_c = arena.prefetch_chain(&chain, 12);
        assert!(instrs_c.is_empty());

        // From a node not in chain: no prefetch.
        let instrs_x = arena.prefetch_chain(&chain, 99);
        assert!(instrs_x.is_empty());
    }
}
