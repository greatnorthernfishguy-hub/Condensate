//! Gaussian Splat Field Geometry — Block K
//!
//! Regions in the thermal field are not points — they are overlapping
//! Gaussian influence zones. Each splat has a position (size-class
//! centroid), opacity (temperature), and covariance (how far its
//! influence radiates). Splats adaptively split when internally diverse
//! and merge when redundantly similar. A tiled scan prioritises hot
//! regions so the field evolves efficiently at scale.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single Gaussian splat — one managed memory region.
#[derive(Clone, Debug)]
pub struct Splat {
    pub id: u32,
    /// Size-class centroid (log-space address / size class index).
    pub position: f64,
    /// Temperature / opacity: 0.0 (cold) → 1.0 (hot).
    pub opacity: f64,
    /// Correlation spread — how far this splat's influence reaches.
    pub covariance: f64,
    /// Total bytes managed by this splat.
    pub mass: usize,
    pub process_id: u32,
    pub access_count: u64,
    /// Child splat IDs when this splat has been split.
    pub child_ids: Vec<u32>,
    /// Parent splat ID when this splat was produced by a merge.
    pub parent_id: Option<u32>,
}

/// A tile — a contiguous position-range bucket of splats scanned together.
#[derive(Clone, Debug)]
pub struct Tile {
    pub id: u32,
    pub splat_ids: Vec<u32>,
    /// Average opacity of member splats.
    pub heat: f64,
    /// Hot tiles are scanned more often than cold ones.
    pub scan_priority: f64,
    pub last_scan_ns: u64,
}

/// The field: a collection of splats partitioned into tiles.
pub struct SplatField {
    splats: HashMap<u32, Splat>,
    tiles: Vec<Tile>,
    next_splat_id: u32,
    tile_scan_cursor: usize,
    /// Coefficient-of-variation threshold above which a splat is split.
    split_threshold: f64,
    /// Similarity threshold above which two splats are merged.
    merge_threshold: f64,
    /// Maximum total (opacity × mass) in bytes.
    ram_budget_bytes: usize,
}

/// Per-cycle summary produced by [`SplatField::summary`].
#[derive(Clone, Debug)]
pub struct SplatSummary {
    pub total_splats: usize,
    pub splits_this_cycle: usize,
    pub merges_this_cycle: usize,
    pub tiles_scanned: usize,
    pub total_opacity: f64,
    pub hottest_splat: Option<(u32, f64)>,
    pub coldest_splat: Option<(u32, f64)>,
}

// ---------------------------------------------------------------------------
// SplatField implementation
// ---------------------------------------------------------------------------

impl SplatField {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new `SplatField`.
    ///
    /// * `ram_budget_bytes` — maximum total weighted energy (opacity × mass).
    /// * `split_threshold`  — coefficient of variation above which a splat splits.
    /// * `merge_threshold`  — similarity above which two splats merge.
    pub fn new(
        ram_budget_bytes: usize,
        split_threshold: f64,
        merge_threshold: f64,
    ) -> Self {
        Self {
            splats: HashMap::new(),
            tiles: Vec::new(),
            next_splat_id: 0,
            tile_scan_cursor: 0,
            split_threshold,
            merge_threshold,
            ram_budget_bytes,
        }
    }

    // -----------------------------------------------------------------------
    // Splat lifecycle
    // -----------------------------------------------------------------------

    /// Add a splat to the field and return its assigned ID.
    pub fn add_splat(
        &mut self,
        position: f64,
        opacity: f64,
        covariance: f64,
        mass: usize,
        process_id: u32,
    ) -> u32 {
        let id = self.next_splat_id;
        self.next_splat_id += 1;
        self.splats.insert(
            id,
            Splat {
                id,
                position,
                opacity: opacity.clamp(0.0, 1.0),
                covariance,
                mass,
                process_id,
                access_count: 0,
                child_ids: Vec::new(),
                parent_id: None,
            },
        );
        id
    }

    /// Remove a splat from the field.
    pub fn remove_splat(&mut self, id: u32) {
        self.splats.remove(&id);
        // Purge the id from any tile that still references it.
        for tile in self.tiles.iter_mut() {
            tile.splat_ids.retain(|&s| s != id);
        }
    }

    // -----------------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------------

    /// Mark a splat as accessed: push opacity toward 1.0 and increment counter.
    pub fn access(&mut self, id: u32) {
        if let Some(splat) = self.splats.get_mut(&id) {
            // Heat injection: strong enough to overcome per-step decay.
            let heat = 0.5 * (1.0 - splat.opacity) + 0.1;
            splat.opacity = (splat.opacity + heat).min(1.0);
            splat.access_count += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Gaussian influence
    // -----------------------------------------------------------------------

    /// Compute the Gaussian influence the source splat exerts on the target.
    ///
    /// `influence = opacity_source × exp(-0.5 × ((Δpos / covariance_source)²))`
    ///
    /// Returns 0.0 if either splat does not exist or if covariance is zero.
    pub fn compute_influence(&self, source_id: u32, target_id: u32) -> f64 {
        let source = match self.splats.get(&source_id) {
            Some(s) => s,
            None => return 0.0,
        };
        let target = match self.splats.get(&target_id) {
            Some(t) => t,
            None => return 0.0,
        };
        if source.covariance == 0.0 {
            return 0.0;
        }
        let delta = (source.position - target.position) / source.covariance;
        source.opacity * (-0.5 * delta * delta).exp()
    }

    // -----------------------------------------------------------------------
    // Field evolution
    // -----------------------------------------------------------------------

    /// Advance the field by one step.
    ///
    /// 1. For each splat, accumulate Gaussian-weighted influence from every
    ///    other splat (activation = weighted sum).
    /// 2. Apply the Lenia-style Gaussian growth function to that activation.
    /// 3. Apply natural decay (opacity × 0.98).
    /// 4. Enforce mass conservation: if total (opacity × mass) exceeds the RAM
    ///    budget, scale all opacities down proportionally.
    pub fn step(&mut self, _dt: f64) {
        // Collect all current splat IDs to avoid borrow issues.
        let ids: Vec<u32> = self.splats.keys().copied().collect();

        // Phase 1: compute new opacities.
        let mut new_opacities: HashMap<u32, f64> = HashMap::new();

        for &id in &ids {
            let old_opacity = match self.splats.get(&id) {
                Some(s) => s.opacity,
                None => continue,
            };

            // Accumulate influence from all other splats.
            let mut activation = 0.0f64;
            for &other_id in &ids {
                if other_id == id {
                    continue;
                }
                activation += self.compute_influence(other_id, id);
            }

            // Growth function: Gaussian bump centred at 0.5, sigma = 0.15.
            // Returns a value in [0, 1].  We treat it as a growth delta.
            let growth = growth_fn(activation);

            // New opacity: apply growth bump then decay.
            let new_opacity = ((old_opacity + growth * 0.1) * 0.98).clamp(0.0, 1.0);
            new_opacities.insert(id, new_opacity);
        }

        // Phase 2: write back new opacities.
        for (&id, &new_op) in &new_opacities {
            if let Some(splat) = self.splats.get_mut(&id) {
                splat.opacity = new_op;
            }
        }

        // Phase 3: mass conservation.
        let total_energy: f64 = self
            .splats
            .values()
            .map(|s| s.opacity * s.mass as f64)
            .sum();

        if total_energy > self.ram_budget_bytes as f64 && total_energy > 0.0 {
            let scale = self.ram_budget_bytes as f64 / total_energy;
            for splat in self.splats.values_mut() {
                splat.opacity = (splat.opacity * scale).clamp(0.0, 1.0);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Adaptive split / merge
    // -----------------------------------------------------------------------

    /// Attempt to split a splat into children.
    ///
    /// `sub_opacities` is a slice of per-sub-region opacity samples inside the
    /// splat.  If the coefficient of variation of those samples exceeds
    /// `split_threshold`, the splat is split into `sub_opacities.len()`
    /// children and their IDs are returned.  The parent's `child_ids` are
    /// updated; each child's `parent_id` is set to `None` (they are new roots).
    /// Returns `None` if the splat does not exist, has fewer than two
    /// sub-opacities, or the internal diversity is below the threshold.
    pub fn try_split(&mut self, id: u32, sub_opacities: &[f64]) -> Option<Vec<u32>> {
        if sub_opacities.len() < 2 {
            return None;
        }

        // Read parent data first (immutable borrow).
        let (parent_pos, parent_cov, parent_mass, parent_pid) = {
            let parent = self.splats.get(&id)?;
            (
                parent.position,
                parent.covariance,
                parent.mass,
                parent.process_id,
            )
        };

        // Compute coefficient of variation.
        let n = sub_opacities.len() as f64;
        let mean: f64 = sub_opacities.iter().sum::<f64>() / n;
        if mean == 0.0 {
            return None;
        }
        let variance: f64 =
            sub_opacities.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let cv = variance.sqrt() / mean;

        if cv <= self.split_threshold {
            return None;
        }

        // Create one child per sub-region, spread evenly around parent position.
        let spread = parent_cov;
        let n_children = sub_opacities.len();
        let child_mass = parent_mass / n_children.max(1);
        let child_cov = parent_cov / 2.0;

        let mut child_ids = Vec::with_capacity(n_children);
        for (i, &sub_op) in sub_opacities.iter().enumerate() {
            // Spread children symmetrically around parent position.
            let offset = (i as f64 - (n_children as f64 - 1.0) / 2.0)
                * spread
                / n_children as f64;
            let child_id = self.next_splat_id;
            self.next_splat_id += 1;
            self.splats.insert(
                child_id,
                Splat {
                    id: child_id,
                    position: parent_pos + offset,
                    opacity: sub_op.clamp(0.0, 1.0),
                    covariance: child_cov,
                    mass: child_mass,
                    process_id: parent_pid,
                    access_count: 0,
                    child_ids: Vec::new(),
                    parent_id: Some(id),
                },
            );
            child_ids.push(child_id);
        }

        // Update parent's child list.
        if let Some(parent) = self.splats.get_mut(&id) {
            parent.child_ids = child_ids.clone();
        }

        Some(child_ids)
    }

    /// Attempt to merge a set of splats into one.
    ///
    /// Merges if every pair in `ids` has opacity within 10% of each other
    /// AND the Gaussian influence between all pairs exceeds `merge_threshold`.
    /// Returns the ID of the new merged splat, or `None` if the conditions are
    /// not met or fewer than two IDs are provided.
    pub fn try_merge(&mut self, ids: &[u32]) -> Option<u32> {
        if ids.len() < 2 {
            return None;
        }

        // Gather splat snapshots.
        let splats: Vec<Splat> = ids
            .iter()
            .filter_map(|&id| self.splats.get(&id).cloned())
            .collect();

        if splats.len() < 2 {
            return None;
        }

        // Check temperature similarity: all opacities within 10% of the mean.
        let mean_opacity: f64 = splats.iter().map(|s| s.opacity).sum::<f64>()
            / splats.len() as f64;
        let all_similar = splats
            .iter()
            .all(|s| (s.opacity - mean_opacity).abs() <= 0.1);
        if !all_similar {
            return None;
        }

        // Check pairwise Gaussian correlation (use compute_influence proxy):
        // influence between two splats must exceed merge_threshold.
        for i in 0..splats.len() {
            for j in (i + 1)..splats.len() {
                let influence =
                    self.compute_influence(splats[i].id, splats[j].id);
                if influence < self.merge_threshold {
                    return None;
                }
            }
        }

        // Build the merged splat.
        let merged_position =
            splats.iter().map(|s| s.position).sum::<f64>() / splats.len() as f64;
        let merged_opacity = mean_opacity;
        let merged_covariance =
            splats.iter().map(|s| s.covariance).sum::<f64>() / splats.len() as f64;
        let merged_mass: usize = splats.iter().map(|s| s.mass).sum();
        let merged_pid = splats[0].process_id;
        let merged_access: u64 = splats.iter().map(|s| s.access_count).sum();

        let merged_id = self.next_splat_id;
        self.next_splat_id += 1;
        self.splats.insert(
            merged_id,
            Splat {
                id: merged_id,
                position: merged_position,
                opacity: merged_opacity.clamp(0.0, 1.0),
                covariance: merged_covariance,
                mass: merged_mass,
                process_id: merged_pid,
                access_count: merged_access,
                child_ids: Vec::new(),
                parent_id: None,
            },
        );

        // Remove the source splats.
        for id in ids {
            self.remove_splat(*id);
        }

        Some(merged_id)
    }

    // -----------------------------------------------------------------------
    // Tiled scanning
    // -----------------------------------------------------------------------

    /// Partition all current splats into `num_tiles` tiles by position range.
    ///
    /// Tiles are rebuilt from scratch each call.  After partitioning, each
    /// tile's `heat` and `scan_priority` are recomputed.
    pub fn partition_tiles(&mut self, num_tiles: usize) {
        if num_tiles == 0 || self.splats.is_empty() {
            self.tiles.clear();
            return;
        }

        // Find position range.
        let min_pos = self
            .splats
            .values()
            .map(|s| s.position)
            .fold(f64::INFINITY, f64::min);
        let max_pos = self
            .splats
            .values()
            .map(|s| s.position)
            .fold(f64::NEG_INFINITY, f64::max);

        let range = (max_pos - min_pos).max(1e-12);
        let tile_width = range / num_tiles as f64;

        // Build tiles.
        let mut tiles: Vec<Tile> = (0..num_tiles)
            .map(|i| Tile {
                id: i as u32,
                splat_ids: Vec::new(),
                heat: 0.0,
                scan_priority: 0.0,
                last_scan_ns: 0,
            })
            .collect();

        for splat in self.splats.values() {
            let idx = ((splat.position - min_pos) / tile_width) as usize;
            let idx = idx.min(num_tiles - 1);
            tiles[idx].splat_ids.push(splat.id);
        }

        // Compute per-tile heat and scan priority.
        for tile in tiles.iter_mut() {
            if tile.splat_ids.is_empty() {
                tile.heat = 0.0;
                tile.scan_priority = 0.0;
                continue;
            }
            let total_opacity: f64 = tile
                .splat_ids
                .iter()
                .filter_map(|&id| self.splats.get(&id))
                .map(|s| s.opacity)
                .sum();
            tile.heat = total_opacity / tile.splat_ids.len() as f64;
            tile.scan_priority = tile.heat; // hot tiles scan more
        }

        self.tiles = tiles;
        // Reset cursor so iteration starts from a fresh position.
        self.tile_scan_cursor = 0;
    }

    /// Advance the round-robin tile cursor and return the next tile to scan.
    ///
    /// The cursor is biased toward hot tiles: after returning a tile it bumps
    /// `scan_priority` by 1.0 for hot tiles so they rise to the top of
    /// future natural ordering, but the cursor itself is a simple modular
    /// advance for predictability.  `last_scan_ns` is updated on the returned
    /// tile.
    ///
    /// Returns `None` if there are no tiles.
    pub fn scan_next_tile(&mut self, now_ns: u64) -> Option<&Tile> {
        if self.tiles.is_empty() {
            return None;
        }

        // Find the tile with the highest scan_priority, using the cursor as a
        // tiebreaker (prefer tiles that haven't been scanned recently in order).
        // This gives hot tiles more frequent visits while still cycling through all.
        let n = self.tiles.len();

        // Pick the tile with maximum scan_priority; ties broken by cursor order.
        let mut best_idx = self.tile_scan_cursor % n;
        let mut best_priority = self.tiles[best_idx].scan_priority;
        for i in 1..n {
            let idx = (self.tile_scan_cursor + i) % n;
            if self.tiles[idx].scan_priority > best_priority {
                best_priority = self.tiles[idx].scan_priority;
                best_idx = idx;
            }
        }

        // Update the chosen tile.
        self.tiles[best_idx].last_scan_ns = now_ns;
        // Reduce its scan_priority so it won't monopolise — decay toward heat baseline.
        self.tiles[best_idx].scan_priority =
            self.tiles[best_idx].heat; // reset; will grow again next partition

        // Advance cursor.
        self.tile_scan_cursor = (best_idx + 1) % n;

        Some(&self.tiles[best_idx])
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return IDs of all splats whose opacity is below `threshold`.
    pub fn get_cold_splats(&self, threshold: f64) -> Vec<u32> {
        self.splats
            .values()
            .filter(|s| s.opacity < threshold)
            .map(|s| s.id)
            .collect()
    }

    /// Return IDs of all splats whose opacity is above `threshold`.
    pub fn get_hot_splats(&self, threshold: f64) -> Vec<u32> {
        self.splats
            .values()
            .filter(|s| s.opacity > threshold)
            .map(|s| s.id)
            .collect()
    }

    /// Summarise the current field state.
    pub fn summary(&self) -> SplatSummary {
        let total_opacity: f64 = self.splats.values().map(|s| s.opacity).sum();

        let hottest = self
            .splats
            .values()
            .max_by(|a, b| a.opacity.partial_cmp(&b.opacity).unwrap())
            .map(|s| (s.id, s.opacity));

        let coldest = self
            .splats
            .values()
            .min_by(|a, b| a.opacity.partial_cmp(&b.opacity).unwrap())
            .map(|s| (s.id, s.opacity));

        SplatSummary {
            total_splats: self.splats.len(),
            splits_this_cycle: 0, // caller tracks across calls
            merges_this_cycle: 0,
            tiles_scanned: 0,
            total_opacity,
            hottest_splat: hottest,
            coldest_splat: coldest,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Lenia-style Gaussian growth function.
///
/// Returns a value in [0, 1]: peaks when `activation` ≈ 0.5, falls toward 0
/// for very low or very high activation.
#[inline]
fn growth_fn(activation: f64) -> f64 {
    let x = (activation - 0.5) / 0.15;
    (-0.5 * x * x).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field() -> SplatField {
        SplatField::new(
            1_000_000_000, // 1 GB budget — generous for tests
            0.3,           // split_threshold: CV > 0.3 → split
            0.05,          // merge_threshold: influence > 0.05 → eligible for merge
        )
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_gaussian_influence_falloff() {
        let mut field = make_field();

        // Source at position 0.0, covariance 1.0, full opacity.
        let src = field.add_splat(0.0, 1.0, 1.0, 1024, 1);
        // Near target: position 0.5
        let near = field.add_splat(0.5, 0.5, 1.0, 1024, 1);
        // Far target: position 5.0
        let far = field.add_splat(5.0, 0.5, 1.0, 1024, 1);

        let near_inf = field.compute_influence(src, near);
        let far_inf = field.compute_influence(src, far);

        assert!(
            near_inf > far_inf,
            "Closer target must receive more influence: near={near_inf:.4} far={far_inf:.4}"
        );
        assert!(near_inf > 0.0, "Near influence must be positive");
        assert!(far_inf >= 0.0, "Far influence must be non-negative");
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_mass_conservation() {
        // Tight budget: 100 000 bytes.  Five splats each with 50 000-byte mass
        // and opacity 1.0 → total = 250 000 > budget, must be scaled down.
        let mut field = SplatField::new(100_000, 0.5, 0.05);

        for i in 0..5 {
            field.add_splat(i as f64, 1.0, 1.0, 50_000, 1);
        }

        field.step(0.1);

        let total_energy: f64 = field
            .splats
            .values()
            .map(|s| s.opacity * s.mass as f64)
            .sum();

        assert!(
            total_energy <= 100_000.0 * 1.001, // tiny float tolerance
            "Energy must be within budget after step(): {total_energy:.1}"
        );
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_access_heats_splat() {
        let mut field = make_field();
        let id = field.add_splat(0.0, 0.1, 1.0, 1024, 1);

        let before = field.splats[&id].opacity;
        field.access(id);
        let after = field.splats[&id].opacity;

        assert!(
            after > before,
            "Access must raise opacity: {before:.4} → {after:.4}"
        );
        assert_eq!(field.splats[&id].access_count, 1);
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_decay_cools_splat() {
        let mut field = make_field();
        // Start hot; no access; no neighbours.
        let id = field.add_splat(0.0, 1.0, 1.0, 1024, 1);

        for _ in 0..50 {
            field.step(0.1);
        }

        let final_opacity = field.splats[&id].opacity;
        assert!(
            final_opacity < 1.0,
            "Splat must cool down over 50 steps without access: opacity={final_opacity:.4}"
        );
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_split_creates_children() {
        let mut field = make_field();
        let parent_id = field.add_splat(5.0, 0.5, 2.0, 8192, 42);

        // Sub-opacities with high coefficient of variation → forces a split.
        let sub_ops = [0.05, 0.95, 0.1, 0.9];
        let children = field
            .try_split(parent_id, &sub_ops)
            .expect("Split should succeed with high CV");

        assert_eq!(children.len(), 4, "Should create one child per sub-opacity");

        // Each child must point back to the parent.
        for &child_id in &children {
            let child = &field.splats[&child_id];
            assert_eq!(
                child.parent_id,
                Some(parent_id),
                "Child {child_id} must reference parent {parent_id}"
            );
        }

        // Parent must record the children.
        let parent = &field.splats[&parent_id];
        assert_eq!(
            parent.child_ids, children,
            "Parent child_ids must match returned IDs"
        );
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_combines_splats() {
        let mut field = make_field();

        // Two nearly identical splats at close positions so influence is high.
        let a = field.add_splat(0.0, 0.5, 10.0, 512, 1);
        let b = field.add_splat(0.1, 0.5, 10.0, 512, 1);

        let merged = field
            .try_merge(&[a, b])
            .expect("Merge should succeed for similar, close splats");

        // Originals must be gone.
        assert!(
            !field.splats.contains_key(&a),
            "Source splat A must be removed after merge"
        );
        assert!(
            !field.splats.contains_key(&b),
            "Source splat B must be removed after merge"
        );

        // Merged splat must exist and have combined mass.
        let m = &field.splats[&merged];
        assert_eq!(m.mass, 1024, "Merged mass must be sum of sources");
        assert!(
            (m.opacity - 0.5).abs() < 0.05,
            "Merged opacity must be approximately the mean"
        );
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_tiled_scan_priority() {
        let mut field = make_field();

        // Cold cluster: positions 0-2, low opacity.
        for i in 0..3 {
            field.add_splat(i as f64, 0.05, 1.0, 512, 1);
        }
        // Hot cluster: positions 10-12, high opacity.
        for i in 0..3 {
            field.add_splat(10.0 + i as f64, 0.95, 1.0, 512, 1);
        }

        field.partition_tiles(2);

        assert_eq!(field.tiles.len(), 2, "Should have exactly 2 tiles");

        // The hot tile should have higher scan_priority.
        let max_priority = field
            .tiles
            .iter()
            .map(|t| t.scan_priority)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_priority = field
            .tiles
            .iter()
            .map(|t| t.scan_priority)
            .fold(f64::INFINITY, f64::min);

        assert!(
            max_priority > min_priority,
            "Hot tile must have higher priority than cold tile: max={max_priority:.3} min={min_priority:.3}"
        );

        // Repeatedly scanning must always pick the hot tile first (it has higher
        // initial priority and resets to heat baseline after each scan).
        let first = field.scan_next_tile(1_000).unwrap().clone();
        assert!(
            first.heat > 0.5,
            "First scanned tile should be the hot one: heat={:.3}",
            first.heat
        );
    }

    // -----------------------------------------------------------------------

    #[test]
    fn test_cold_hot_identification() {
        let mut field = make_field();

        // Cold cluster at positions 0-2, hot cluster at positions 100-102.
        // The 100-unit gap with covariance=1.0 makes cross-cluster Gaussian
        // influence vanishingly small (≈ exp(-0.5 × 100²) ≈ 0), so the cold
        // splats cannot be warmed by the hot ones over a handful of steps.
        let c0 = field.add_splat(0.0, 0.05, 1.0, 512, 1);
        let c1 = field.add_splat(1.0, 0.08, 1.0, 512, 1);
        let c2 = field.add_splat(2.0, 0.12, 1.0, 512, 1);
        // Three hot splats well separated from cold cluster.
        let h0 = field.add_splat(100.0, 0.85, 1.0, 512, 1);
        let h1 = field.add_splat(101.0, 0.90, 1.0, 512, 1);
        let h2 = field.add_splat(102.0, 0.95, 1.0, 512, 1);

        // Evolve a few steps to exercise the pipeline end-to-end.
        for _ in 0..5 {
            field.step(0.1);
        }

        let cold = field.get_cold_splats(0.2);
        let hot = field.get_hot_splats(0.7);

        // Original cold set must still be cold.
        for &id in &[c0, c1, c2] {
            assert!(
                cold.contains(&id),
                "Splat {id} should be in the cold list"
            );
        }
        // Original hot set must still be hot.
        for &id in &[h0, h1, h2] {
            assert!(
                hot.contains(&id),
                "Splat {id} should be in the hot list"
            );
        }
    }
}
