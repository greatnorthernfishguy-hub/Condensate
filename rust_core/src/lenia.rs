//! Lenia Field — continuous thermal dynamics for memory management.
//!
//! Replaces the hard HOT/WARM/COLD tiers with a continuous field
//! that evolves according to Lenia dynamics. Each memory region
//! has a temperature (activation level) that flows smoothly between
//! fully materialized and fully compressed.
//!
//! The Gaussian splat connection:
//!   - Each managed region is a Gaussian in memory space
//!   - Position = address/size class
//!   - Opacity = access temperature (high = hot, low = cold)
//!   - Covariance = how spread the access pattern is
//!   - Adaptive density: hot regions split (finer tracking),
//!     cold regions merge (coarser, save overhead)
//!
//! Lenia dynamics:
//!   - Growth function: how temperature spreads from accessed regions
//!   - Kernel: neighborhood function (which regions influence each other)
//!   - Mass conservation: total "heat" bounded by RAM budget
//!   - Continuous: no discrete tiers, smooth gradient

use std::collections::HashMap;

/// A region in the Lenia field
#[derive(Clone, Debug)]
pub struct FieldRegion {
    /// Unique identifier (size-class path from pipeline)
    pub id: u32,
    /// Process that owns this region
    pub process_id: u32,
    /// Current temperature: 0.0 (frozen/cold) to 1.0 (fully hot)
    pub temperature: f64,
    /// Temperature at last step (for delta computation)
    pub prev_temperature: f64,
    /// Access weight: accumulated access intensity
    pub access_weight: f64,
    /// Decay rate: how fast this region cools when not accessed
    pub decay_rate: f64,
    /// Size in bytes (for mass conservation weighting)
    pub size_bytes: u64,
    /// Number of times accessed
    pub access_count: u64,
    /// Whether this region is priority (temperature floor at 0.5)
    pub priority: bool,
}

impl FieldRegion {
    pub fn new(id: u32, size_bytes: u64) -> Self {
        Self {
            id,
            process_id: 0,
            temperature: 1.0, // start hot (just allocated)
            prev_temperature: 1.0,
            access_weight: 1.0,
            decay_rate: 0.05, // 5% decay per step
            size_bytes,
            access_count: 1,
            priority: false,
        }
    }

    /// Temperature delta since last step
    pub fn delta(&self) -> f64 {
        self.temperature - self.prev_temperature
    }

    /// Is this region effectively cold? (below materialization threshold)
    pub fn is_cold(&self, threshold: f64) -> bool {
        self.temperature < threshold
    }

    /// Is this region effectively hot? (above full-materialization threshold)
    pub fn is_hot(&self, threshold: f64) -> bool {
        self.temperature > threshold
    }
}

/// Lenia growth function — how temperature responds to neighborhood activation
#[derive(Clone, Debug)]
pub enum GrowthFunction {
    /// Gaussian bump: peaks at `center`, width `sigma`
    /// Temperature grows when neighborhood activation is near `center`
    Gaussian { center: f64, sigma: f64 },
    /// Step function: grows if activation > threshold
    Step { threshold: f64 },
}

impl GrowthFunction {
    /// Evaluate the growth function
    pub fn evaluate(&self, activation: f64) -> f64 {
        match self {
            GrowthFunction::Gaussian { center, sigma } => {
                let x = (activation - center) / sigma;
                (-(x * x) / 2.0).exp() * 2.0 - 1.0
                // Returns [-1, 1]: positive = grow, negative = shrink
            }
            GrowthFunction::Step { threshold } => {
                if activation > *threshold { 1.0 } else { -1.0 }
            }
        }
    }
}

/// The Lenia field engine
pub struct LeniaField {
    /// All regions in the field
    regions: HashMap<u32, FieldRegion>,

    /// Neighborhood connections: region_id → [(neighbor_id, coupling_weight)]
    /// Built from the AccessGraph's edges
    neighbors: HashMap<u32, Vec<(u32, f64)>>,

    /// Growth function
    growth: GrowthFunction,

    /// Global decay rate (cooling)
    decay_rate: f64,

    /// Mass conservation: maximum total weighted temperature
    /// (RAM budget expressed as field energy)
    max_total_energy: f64,

    /// RAM budget in MB (kept in sync with max_total_energy)
    ram_budget_mb: usize,

    /// Current total energy
    total_energy: f64,

    /// Materialization threshold: below this, compress
    cold_threshold: f64,

    /// Full materialization threshold: above this, fully hot
    hot_threshold: f64,

    /// Step count
    steps: u64,

    /// Time step size (controls how fast the field evolves)
    dt: f64,

    /// Accumulated page fault count since last tune
    page_fault_count: u64,

    /// Steps since last adaptive tune
    steps_since_tune: u64,

    /// How many steps between adaptive tuning checks
    tune_interval: u64,
}

impl LeniaField {
    pub fn new(ram_budget_mb: f64) -> Self {
        // Convert RAM budget to field energy units
        // 1 MB = 1.0 energy unit
        let max_energy = ram_budget_mb;

        Self {
            regions: HashMap::new(),
            neighbors: HashMap::new(),
            growth: GrowthFunction::Gaussian {
                center: 0.5,  // optimal neighborhood activation
                sigma: 0.15,  // width of the growth peak
            },
            decay_rate: 0.02,   // 2% cooling per step
            max_total_energy: max_energy,
            ram_budget_mb: ram_budget_mb as usize,
            total_energy: 0.0,
            cold_threshold: 0.2,  // below 20% = compress
            hot_threshold: 0.7,   // above 70% = fully materialized
            steps: 0,
            dt: 0.1,  // time step
            page_fault_count: 0,
            steps_since_tune: 0,
            tune_interval: 100,
        }
    }

    /// Add a region to the field with explicit process ownership
    pub fn add_region(&mut self, id: u32, size_bytes: usize, process_id: u32) {
        let mut region = FieldRegion::new(id, size_bytes as u64);
        region.process_id = process_id;
        let energy = region.temperature * (size_bytes as f64 / (1024.0 * 1024.0));
        self.total_energy += energy;
        self.regions.insert(id, region);
    }

    /// Remove a region from the field — called when an allocation is freed.
    /// Reclaims the energy and removes from primary tracking.
    /// Stale neighbor references are left in place — step() already handles
    /// missing regions gracefully (skips them). Eager neighbor cleanup was
    /// O(N × avg_neighbors) on every free, which killed throughput.
    /// Sleep consolidation prunes stale references in batch.
    pub fn remove_region(&mut self, id: u32) {
        if let Some(region) = self.regions.remove(&id) {
            let energy = region.temperature * (region.size_bytes as f64 / (1024.0 * 1024.0));
            self.total_energy -= energy;
            if self.total_energy < 0.0 {
                self.total_energy = 0.0;
            }
        }
        self.neighbors.remove(&id);
        // Stale references in OTHER regions' neighbor lists are harmless —
        // step() checks regions.contains_key() before using a neighbor.
        // Batch cleanup happens during sleep consolidation.
    }

    /// Prune stale neighbor references — call during sleep consolidation.
    /// Removes references to regions that no longer exist.
    pub fn prune_stale_neighbors(&mut self) {
        for (_rid, nbrs) in self.neighbors.iter_mut() {
            nbrs.retain(|(nid, _)| self.regions.contains_key(nid));
        }
    }

    /// Set neighborhood connections from graph edges
    pub fn set_neighbors(&mut self, id: u32, neighbors: Vec<(u32, f64)>) {
        self.neighbors.insert(id, neighbors);
    }

    /// Update the RAM budget directly (in MB)
    pub fn set_budget(&mut self, budget_mb: usize) {
        self.ram_budget_mb = budget_mb;
        self.max_total_energy = budget_mb as f64;
    }

    /// Read /proc/meminfo and update budget from MemAvailable
    /// Silently no-ops if the file cannot be read or parsed
    pub fn update_budget_from_system(&mut self) {
        let contents = match std::fs::read_to_string("/proc/meminfo") {
            Ok(c) => c,
            Err(_) => return,
        };
        for line in contents.lines() {
            if line.starts_with("MemAvailable:") {
                // Format: "MemAvailable:   12345678 kB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        let mb = kb / 1024;
                        self.set_budget(mb);
                    }
                }
                break;
            }
        }
    }

    /// Record a page fault event for adaptive growth tuning
    pub fn record_page_fault(&mut self) {
        self.page_fault_count += 1;
    }

    /// Set whether a region is priority (temperature clamped to >= 0.5)
    pub fn set_priority(&mut self, id: u32, priority: bool) {
        if let Some(region) = self.regions.get_mut(&id) {
            region.priority = priority;
        }
    }

    /// Record an access — heats up the region
    pub fn access(&mut self, id: u32) {
        if let Some(region) = self.regions.get_mut(&id) {
            // Heat injection: access pushes temperature toward 1.0
            // Strong enough to overcome decay — accessed regions STAY hot
            let heat = 0.5 * (1.0 - region.temperature) + 0.1;
            region.temperature = (region.temperature + heat).min(1.0);
            region.access_count += 1;
            region.access_weight += 1.0;
        }
    }

    /// Step the field forward — the core Lenia dynamics
    ///
    /// For each region:
    /// 1. Compute neighborhood activation (weighted avg of neighbor temps)
    /// 2. Apply growth function (determines if region heats or cools)
    /// 3. Apply natural decay (everything cools)
    /// 4. Enforce mass conservation (total energy bounded)
    /// 5. Clamp priority regions to >= 0.5
    /// 6. Adaptive growth tuning every tune_interval steps
    pub fn step(&mut self) {
        self.steps += 1;
        self.steps_since_tune += 1;

        // Phase 1: Compute new temperatures
        let mut new_temps: HashMap<u32, f64> = HashMap::new();

        for (&id, region) in &self.regions {
            // Save previous temperature
            let old_temp = region.temperature;

            // Compute neighborhood activation
            let neighborhood_activation = self.compute_neighborhood(id);

            // Apply growth function
            let growth = self.growth.evaluate(neighborhood_activation);

            // New temperature = old + growth * dt - decay
            let decay = self.decay_rate * old_temp;
            let new_temp = (old_temp + growth * self.dt - decay)
                .max(0.0)
                .min(1.0);

            new_temps.insert(id, new_temp);
        }

        // Phase 2: Apply new temperatures and clamp priority regions
        self.total_energy = 0.0;
        for (&id, region) in self.regions.iter_mut() {
            region.prev_temperature = region.temperature;
            if let Some(&new_temp) = new_temps.get(&id) {
                region.temperature = new_temp;
            }

            // Priority floor: if priority and dropped below 0.5, clamp up
            if region.priority && region.temperature < 0.5 {
                region.temperature = 0.5;
            }

            // Accumulate energy (temperature * size in MB)
            self.total_energy += region.temperature
                * (region.size_bytes as f64 / (1024.0 * 1024.0));

            // Decay access weight over time
            region.access_weight *= 0.95;
        }

        // Phase 3: Mass conservation — if over budget, cool everything proportionally
        if self.total_energy > self.max_total_energy && self.total_energy > 0.0 {
            let scale = self.max_total_energy / self.total_energy;
            for region in self.regions.values_mut() {
                region.temperature *= scale;
                // Re-apply priority floor after scaling
                if region.priority && region.temperature < 0.5 {
                    region.temperature = 0.5;
                }
            }
            self.total_energy = self.max_total_energy;
        }

        // Phase 4: Adaptive growth tuning (Gaussian only)
        if self.steps_since_tune >= self.tune_interval {
            let fault_rate = if self.steps_since_tune > 0 {
                self.page_fault_count as f64 / self.steps_since_tune as f64
            } else {
                0.0
            };

            if let GrowthFunction::Gaussian { ref mut center, ref mut sigma } = self.growth {
                if fault_rate > 0.01 {
                    // Over-cooling: too many faults — widen sigma, raise center
                    *sigma = (*sigma * 1.05).min(0.5);
                    *center = (*center * 1.02).min(0.8);
                } else if fault_rate < 0.001 {
                    // Under-cooling: check if usage > 80% budget
                    let usage_pct = if self.max_total_energy > 0.0 {
                        self.total_energy / self.max_total_energy
                    } else {
                        0.0
                    };
                    if usage_pct > 0.80 {
                        *sigma = (*sigma * 0.95).max(0.05);
                        *center = (*center * 0.98).max(0.2);
                    }
                }
            }

            // Reset counters
            self.page_fault_count = 0;
            self.steps_since_tune = 0;
        }
    }

    /// Compute neighborhood activation for a region
    fn compute_neighborhood(&self, id: u32) -> f64 {
        let neighbors = match self.neighbors.get(&id) {
            Some(n) => n,
            None => return 0.0,
        };

        if neighbors.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for &(neighbor_id, coupling) in neighbors {
            if let Some(neighbor) = self.regions.get(&neighbor_id) {
                weighted_sum += neighbor.temperature * coupling;
                weight_total += coupling;
            }
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }

    /// Get regions that should be compressed (below cold threshold)
    pub fn get_cold_regions(&self) -> Vec<(u32, f64)> {
        self.regions.iter()
            .filter(|(_, r)| r.is_cold(self.cold_threshold))
            .map(|(&id, r)| (id, r.temperature))
            .collect()
    }

    /// Get regions that should be fully materialized (above hot threshold)
    pub fn get_hot_regions(&self) -> Vec<(u32, f64)> {
        self.regions.iter()
            .filter(|(_, r)| r.is_hot(self.hot_threshold))
            .map(|(&id, r)| (id, r.temperature))
            .collect()
    }

    /// Get a summary of the field state
    pub fn summary(&self) -> LeniaSummary {
        let mut hot = 0u32;
        let mut warm = 0u32;
        let mut cold = 0u32;
        let mut hot_mb = 0.0f64;
        let mut warm_mb = 0.0f64;
        let mut cold_mb = 0.0f64;

        for region in self.regions.values() {
            let mb = region.size_bytes as f64 / (1024.0 * 1024.0);
            if region.is_hot(self.hot_threshold) {
                hot += 1;
                hot_mb += mb;
            } else if region.is_cold(self.cold_threshold) {
                cold += 1;
                cold_mb += mb;
            } else {
                warm += 1;
                warm_mb += mb;
            }
        }

        LeniaSummary {
            total_regions: self.regions.len() as u32,
            hot, warm, cold,
            hot_mb, warm_mb, cold_mb,
            total_energy: self.total_energy,
            max_energy: self.max_total_energy,
            energy_pct: if self.max_total_energy > 0.0 {
                self.total_energy / self.max_total_energy * 100.0
            } else { 0.0 },
            steps: self.steps,
            cold_threshold: self.cold_threshold,
            hot_threshold: self.hot_threshold,
        }
    }

    /// Serialize the field state to bytes.
    ///
    /// Format: 4-byte region count (u32 LE), then per region:
    ///   u32 id, u32 process_id, f32 temperature, u64 size_bytes,
    ///   f32 decay_rate, u8 priority
    /// = 25 bytes per region + 4 header
    pub fn serialize(&self) -> Vec<u8> {
        let count = self.regions.len() as u32;
        let mut buf = Vec::with_capacity(4 + count as usize * 25);

        buf.extend_from_slice(&count.to_le_bytes());

        // Sort by id for deterministic output
        let mut ids: Vec<u32> = self.regions.keys().copied().collect();
        ids.sort_unstable();

        for id in ids {
            let r = &self.regions[&id];
            buf.extend_from_slice(&r.id.to_le_bytes());
            buf.extend_from_slice(&r.process_id.to_le_bytes());
            buf.extend_from_slice(&(r.temperature as f32).to_le_bytes());
            buf.extend_from_slice(&r.size_bytes.to_le_bytes());
            buf.extend_from_slice(&(r.decay_rate as f32).to_le_bytes());
            buf.push(if r.priority { 1u8 } else { 0u8 });
        }

        buf
    }

    /// Deserialize a field from bytes produced by `serialize`.
    /// Returns None if the data is malformed or truncated.
    pub fn deserialize(data: &[u8], ram_budget_mb: usize) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        let count = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let expected_len = 4 + count * 25;
        if data.len() < expected_len {
            return None;
        }

        let mut field = LeniaField::new(ram_budget_mb as f64);

        let mut offset = 4usize;
        for _ in 0..count {
            let id         = u32::from_le_bytes(data[offset..offset+4].try_into().ok()?);
            let process_id = u32::from_le_bytes(data[offset+4..offset+8].try_into().ok()?);
            let temperature = f32::from_le_bytes(data[offset+8..offset+12].try_into().ok()?) as f64;
            let size_bytes  = u64::from_le_bytes(data[offset+12..offset+20].try_into().ok()?);
            let decay_rate  = f32::from_le_bytes(data[offset+20..offset+24].try_into().ok()?) as f64;
            let priority    = data[offset+24] != 0;
            offset += 25;

            let mut region = FieldRegion::new(id, size_bytes);
            region.process_id = process_id;
            region.temperature = temperature;
            region.prev_temperature = temperature;
            region.decay_rate = decay_rate;
            region.priority = priority;

            let energy = temperature * (size_bytes as f64 / (1024.0 * 1024.0));
            field.total_energy += energy;
            field.regions.insert(id, region);
        }

        Some(field)
    }
}

/// Field summary
#[derive(Clone, Debug)]
pub struct LeniaSummary {
    pub total_regions: u32,
    pub hot: u32,
    pub warm: u32,
    pub cold: u32,
    pub hot_mb: f64,
    pub warm_mb: f64,
    pub cold_mb: f64,
    pub total_energy: f64,
    pub max_energy: f64,
    pub energy_pct: f64,
    pub steps: u64,
    pub cold_threshold: f64,
    pub hot_threshold: f64,
}

impl LeniaSummary {
    pub fn print(&self) {
        eprintln!("\n{}", "=".repeat(55));
        eprintln!("  CONDENSATE — Lenia Thermal Field");
        eprintln!("{}", "=".repeat(55));
        eprintln!("  Regions:  {}", self.total_regions);
        eprintln!("  Steps:    {}", self.steps);
        eprintln!("  Energy:   {:.1} / {:.1} ({:.1}% of budget)",
                 self.total_energy, self.max_energy, self.energy_pct);
        eprintln!();
        eprintln!("  HOT  (>{:.0}%): {} regions, {:.1} MB",
                 self.hot_threshold * 100.0, self.hot, self.hot_mb);
        eprintln!("  WARM ({:.0}%-{:.0}%): {} regions, {:.1} MB",
                 self.cold_threshold * 100.0, self.hot_threshold * 100.0,
                 self.warm, self.warm_mb);
        eprintln!("  COLD (<{:.0}%): {} regions, {:.1} MB",
                 self.cold_threshold * 100.0, self.cold, self.cold_mb);
        eprintln!("{}\n", "=".repeat(55));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── existing tests (unchanged behaviour) ─────────────────────────────────

    #[test]
    fn test_field_creation() {
        let mut field = LeniaField::new(100.0); // 100MB budget

        field.add_region(0, 1_048_576, 0);
        field.add_region(1, 1_048_576, 0);
        field.add_region(2, 1_048_576, 0);

        assert_eq!(field.regions.len(), 3);

        let summary = field.summary();
        assert_eq!(summary.hot, 3); // all start hot
    }

    #[test]
    fn test_decay_makes_cold() {
        let mut field = LeniaField::new(100.0);

        field.add_region(0, 1_048_576, 0);

        // Step many times without access — should cool down
        for _ in 0..100 {
            field.step();
        }

        let summary = field.summary();
        assert_eq!(summary.cold, 1, "Region should be cold after 100 steps without access");
    }

    #[test]
    fn test_access_keeps_hot() {
        let mut field = LeniaField::new(100.0);

        field.add_region(0, 1_048_576, 0);
        field.add_region(1, 1_048_576, 0);

        // Step and access region 0, ignore region 1
        for _ in 0..50 {
            field.access(0);
            field.step();
        }

        let region_0 = &field.regions[&0];
        let region_1 = &field.regions[&1];

        assert!(region_0.temperature > region_1.temperature,
                "Accessed region should be hotter: {} vs {}",
                region_0.temperature, region_1.temperature);
        assert!(region_0.is_hot(0.7), "Accessed region should be hot");
        assert!(region_1.is_cold(0.2), "Ignored region should be cold");
    }

    #[test]
    fn test_mass_conservation() {
        let mut field = LeniaField::new(2.0); // Only 2MB budget

        // Add 5 x 1MB regions — 5MB total, budget is 2MB
        for i in 0..5 {
            field.add_region(i, 1_048_576, 0);
            field.access(i);
        }

        // Step to enforce conservation
        field.step();

        let summary = field.summary();
        assert!(summary.total_energy <= 2.1, // small float tolerance
                "Energy should be bounded by budget: {} > 2.0", summary.total_energy);
    }

    #[test]
    fn test_neighborhood_spreading() {
        let mut field = LeniaField::new(100.0);

        field.add_region(0, 1_048_576, 0);
        field.add_region(1, 1_048_576, 0);
        field.add_region(2, 1_048_576, 0);

        // Region 0 neighbors region 1 and 2
        field.set_neighbors(0, vec![(1, 1.0), (2, 1.0)]);
        field.set_neighbors(1, vec![(0, 1.0)]);
        field.set_neighbors(2, vec![(0, 1.0)]);

        // Let all cool down
        for _ in 0..50 {
            field.step();
        }

        // Now heat region 0 — neighbors should warm up through spreading
        for _ in 0..20 {
            field.access(0);
            field.step();
        }

        let t0 = field.regions[&0].temperature;
        let t1 = field.regions[&1].temperature;
        let t2 = field.regions[&2].temperature;

        assert!(t0 > t1, "Source should be hottest: {} vs {}", t0, t1);
        // Neighbors might warm up if the growth function responds
        // to neighborhood activation
        let summary = field.summary();
        summary.print();
    }

    #[test]
    fn test_splat_analogy() {
        // Gaussian splatting: low opacity → prune
        // Condensate Lenia: low temperature → compress
        let mut field = LeniaField::new(50.0);

        // 10 regions, access only 3
        for i in 0..10 {
            field.add_region(i, 5_242_880, 0); // 5MB each = 50MB total = at budget
        }

        // Hot set: regions 0, 1, 2
        for _ in 0..100 {
            field.access(0);
            field.access(1);
            field.access(2);
            field.step();
        }

        let cold = field.get_cold_regions();
        let hot = field.get_hot_regions();

        assert!(hot.len() >= 2, "Should have hot regions: {}", hot.len());
        assert!(cold.len() >= 5, "Should have cold regions: {}", cold.len());

        let summary = field.summary();
        summary.print();

        // Mass conservation: with budget = 50MB and 50MB total,
        // energy should be at or below budget
        assert!(summary.total_energy <= 50.1);
    }

    // ── new tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lenia_process_tagged() {
        let mut field = LeniaField::new(100.0);

        field.add_region(10, 1_048_576, 42);
        field.add_region(11, 1_048_576, 42);
        field.add_region(12, 1_048_576, 99);

        assert_eq!(field.regions[&10].process_id, 42);
        assert_eq!(field.regions[&11].process_id, 42);
        assert_eq!(field.regions[&12].process_id, 99);

        // Default process_id is 0 for regions added with process_id=0
        field.add_region(13, 1_048_576, 0);
        assert_eq!(field.regions[&13].process_id, 0);
    }

    #[test]
    fn test_lenia_set_budget() {
        let mut field = LeniaField::new(10.0); // 10MB budget

        // Fill to just above the original budget
        for i in 0..5 {
            field.add_region(i, 2_097_152, 0); // 2MB each = 10MB
            field.access(i);
        }
        field.step();

        let energy_at_10mb = field.summary().total_energy;
        assert!(energy_at_10mb <= 10.1, "Energy should be at most 10MB: {}", energy_at_10mb);

        // Expand budget — next step should allow more energy
        field.set_budget(20);
        assert_eq!(field.ram_budget_mb, 20);
        assert!((field.max_total_energy - 20.0).abs() < 0.001,
                "max_total_energy should be 20.0 after set_budget(20)");

        // Re-heat everything and step — conservation limit is now 20MB
        for i in 0..5 {
            field.access(i);
        }
        field.step();

        let energy_at_20mb = field.summary().total_energy;
        assert!(energy_at_20mb <= 20.1, "Energy should be within new 20MB budget: {}", energy_at_20mb);
    }

    #[test]
    fn test_lenia_adaptive_overcooling() {
        // tune_interval is 100; record many faults then step 100 times
        // fault_rate = faults / steps_since_tune
        // We want fault_rate > 0.01 → record > 1 fault per 100 steps
        let mut field = LeniaField::new(100.0);
        field.add_region(0, 1_048_576, 0);

        // Capture initial sigma
        let initial_sigma = match &field.growth {
            GrowthFunction::Gaussian { sigma, .. } => *sigma,
            _ => panic!("Expected Gaussian growth function"),
        };

        // Record 50 page faults before the 100-step tune interval fires
        for _ in 0..50 {
            field.record_page_fault();
        }

        // Step exactly tune_interval times to trigger one tuning cycle
        for _ in 0..100 {
            field.step();
        }

        let new_sigma = match &field.growth {
            GrowthFunction::Gaussian { sigma, .. } => *sigma,
            _ => panic!("Expected Gaussian growth function"),
        };

        assert!(new_sigma > initial_sigma,
            "Sigma should have widened due to over-cooling (fault_rate=0.5): initial={}, new={}",
            initial_sigma, new_sigma);
    }

    #[test]
    fn test_lenia_priority_exempt() {
        let mut field = LeniaField::new(100.0);

        // Add two regions: one priority, one not
        field.add_region(0, 1_048_576, 0);
        field.add_region(1, 1_048_576, 0);
        field.set_priority(0, true);

        // Let both cool for many steps without any access
        for _ in 0..200 {
            field.step();
        }

        let priority_temp = field.regions[&0].temperature;
        let normal_temp   = field.regions[&1].temperature;

        assert!(priority_temp >= 0.5,
            "Priority region must not drop below 0.5: {}", priority_temp);
        assert!(normal_temp < 0.5,
            "Normal region should cool below 0.5: {}", normal_temp);
    }

    #[test]
    fn test_lenia_serialize_roundtrip() {
        let mut field = LeniaField::new(64.0);

        field.add_region(1, 1_048_576, 7);
        field.add_region(2, 2_097_152, 13);
        field.add_region(3, 4_194_304, 0);

        field.set_priority(1, true);
        field.access(2);
        field.step();

        let bytes = field.serialize();

        // Header: 4 bytes + 3 regions * 25 bytes = 79 bytes
        assert_eq!(bytes.len(), 4 + 3 * 25);

        let restored = LeniaField::deserialize(&bytes, 64)
            .expect("deserialize should succeed");

        assert_eq!(restored.regions.len(), field.regions.len());

        for id in [1u32, 2, 3] {
            let orig = &field.regions[&id];
            let rest = &restored.regions[&id];

            assert_eq!(rest.id, orig.id, "id mismatch for region {}", id);
            assert_eq!(rest.process_id, orig.process_id, "process_id mismatch for {}", id);
            assert_eq!(rest.size_bytes, orig.size_bytes, "size_bytes mismatch for {}", id);
            assert_eq!(rest.priority, orig.priority, "priority mismatch for {}", id);

            // f32 round-trip loses a tiny bit of precision
            let temp_diff = (rest.temperature - orig.temperature).abs();
            assert!(temp_diff < 1e-5,
                "temperature mismatch for region {}: {} vs {}", id, orig.temperature, rest.temperature);

            let decay_diff = (rest.decay_rate - orig.decay_rate).abs();
            assert!(decay_diff < 1e-5,
                "decay_rate mismatch for region {}: {} vs {}", id, orig.decay_rate, rest.decay_rate);
        }
    }

    #[test]
    fn test_lenia_cross_process_energy() {
        // Two process groups: PIDs 1 and 2, three regions each
        let mut field = LeniaField::new(6.0); // exactly 6MB budget

        // Process 1: regions 10, 11, 12 (1MB each)
        field.add_region(10, 1_048_576, 1);
        field.add_region(11, 1_048_576, 1);
        field.add_region(12, 1_048_576, 1);

        // Process 2: regions 20, 21, 22 (1MB each)
        field.add_region(20, 1_048_576, 2);
        field.add_region(21, 1_048_576, 2);
        field.add_region(22, 1_048_576, 2);

        // Repeatedly access process 1's regions only
        for _ in 0..50 {
            field.access(10);
            field.access(11);
            field.access(12);
            field.step();
        }

        // Process 1 regions should be hotter than process 2 regions
        let p1_avg = [10u32, 11, 12].iter()
            .map(|id| field.regions[id].temperature)
            .sum::<f64>() / 3.0;
        let p2_avg = [20u32, 21, 22].iter()
            .map(|id| field.regions[id].temperature)
            .sum::<f64>() / 3.0;

        assert!(p1_avg > p2_avg,
            "Process 1 (accessed) should be hotter than process 2: {:.3} vs {:.3}",
            p1_avg, p2_avg);

        // Mass conservation still holds across both process groups
        let summary = field.summary();
        assert!(summary.total_energy <= 6.1,
            "Total energy must stay within 6MB budget: {}", summary.total_energy);
    }
}
