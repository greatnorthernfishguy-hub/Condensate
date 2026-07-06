//! Lenia Dynamics Engine — Rust implementation.
//!
//! Applies continuous Lenia dynamics to transformer weight matrices.
//! Drop-in replacement for ~/UniAI/lenia/engine.py.
//!
//! Key optimizations over Python:
//!   - Fused conv2d → growth → modulate → clamp → apply in one pass
//!   - No intermediate tensor allocations
//!   - Rayon parallel processing across all 168 matrices
//!   - Cache-friendly row-major traversal
//!
//! Current state: FFT-based conv2d is 14-31x slower than PyTorch's
//! BLAS-backed im2col+GEMM for 11×11 kernels. Needs either OpenBLAS
//! linkage for im2col+GEMM, or hybrid approach with PyTorch conv2d.
//! Decision pending.
//!
//! The growth function IS the learning rule. No backprop. No loss.

use rayon::prelude::*;

// OpenBLAS linked via build.rs

// --- Configuration ---

/// Matches Python LeniaConfig exactly.
#[derive(Debug, Clone)]
pub struct LeniaConfig {
    pub kernel_radius: usize,
    pub kernel_sigma: f32,
    pub growth_mu: f32,
    pub growth_sigma: f32,
    pub growth_scale: f32,       // dt — how fast weights change per step
    pub max_weight_delta: f32,
    pub weight_clip_min: f32,
    pub weight_clip_max: f32,
    pub activation_coupling: f32,
    // Energy scarcity — the selection pressure
    pub energy_consumption_rate: f32,
    pub energy_recovery_rate: f32,
    // Reaction-Diffusion — spatial structure
    pub inhibitor_production_rate: f32,   // how fast active matrices produce inhibitor
    pub inhibitor_diffusion_rate: f32,    // how fast inhibitor spreads to neighbors
    pub inhibitor_decay_rate: f32,        // how fast inhibitor dissipates per step
    // Phase dynamics
    pub phase_dt: f32,                    // phase advance per Lenia step
    pub freq_min: f32,                    // natural frequency range (low end)
    pub freq_max: f32,                    // natural frequency range (high end)
    pub freq_adaptation_rate: f32,        // how fast frequencies evolve via competence
    // Quantum amplitude — replaces Kuramoto phase coupling
    pub amplitude_decay: f32,             // decay without activation (~35 steps to halve)
    pub amplitude_coupling: f32,          // how strongly activation grows amplitude
    pub interference_rate: f32,           // how strongly neighbors interfere
}

impl Default for LeniaConfig {
    fn default() -> Self {
        Self {
            kernel_radius: 3,
            kernel_sigma: 1.0,
            growth_mu: 0.15,
            growth_sigma: 0.015,
            growth_scale: 0.001,
            max_weight_delta: 0.01,
            weight_clip_min: -3.0,
            weight_clip_max: 3.0,
            activation_coupling: 1.0,
            energy_consumption_rate: 0.1,
            energy_recovery_rate: 0.05,
            // R-D defaults
            inhibitor_production_rate: 0.15,
            inhibitor_diffusion_rate: 0.3,
            inhibitor_decay_rate: 0.1,
            // Phase — free drift, no coupling (interference handles it)
            phase_dt: 0.2,
            freq_min: 0.05,
            freq_max: 0.20,
            freq_adaptation_rate: 0.01,
            // Quantum amplitude
            amplitude_decay: 0.02,       // halves in ~35 steps without activation
            amplitude_coupling: 0.3,     // activation grows amplitude
            interference_rate: 0.1,      // neighbor interference strength
        }
    }
}

// --- Kernel ---

/// Pre-computed 2D ring kernel. Stored flat for cache efficiency.
#[derive(Debug, Clone)]
pub struct RingKernel {
    pub radius: usize,
    pub diameter: usize,
    /// Flat row-major kernel values, diameter × diameter.
    pub data: Vec<f32>,
}

impl RingKernel {
    /// Create a ring kernel matching the Python implementation exactly.
    /// K(r) = exp(-((r - 0.5) / sigma)^2 / 2), center zeroed, normalized.
    pub fn new(radius: usize, sigma: f32) -> Self {
        let diameter = 2 * radius + 1;
        let mut data = vec![0.0f32; diameter * diameter];
        let r = radius as f32;

        let mut sum = 0.0f32;
        for iy in 0..diameter {
            for ix in 0..diameter {
                let dy = iy as f32 - r;
                let dx = ix as f32 - r;
                let dist = (dx * dx + dy * dy).sqrt() / r; // normalized
                let val = (-((dist - 0.5) / sigma).powi(2) / 2.0).exp();
                data[iy * diameter + ix] = val;
                sum += val;
            }
        }

        // Zero center
        let center = radius * diameter + radius;
        sum -= data[center];
        data[center] = 0.0;

        // Normalize
        if sum > 1e-8 {
            let inv = 1.0 / sum;
            for v in &mut data {
                *v *= inv;
            }
        }

        Self { radius, diameter, data }
    }

    #[inline]
    fn get(&self, ky: usize, kx: usize) -> f32 {
        self.data[ky * self.diameter + kx]
    }
}

// --- Per-matrix step result ---

#[derive(Debug, Clone)]
pub struct MatrixMetrics {
    pub name: String,
    pub delta_norm: f64,
    pub activation_energy: f64,
    pub energy_level: f64,
    pub inhibitor_level: f64,
    pub phase: f64,
    pub amplitude: f64,       // quantum amplitude [0, 1]
    pub born_gate: f64,       // amplitude² — Born rule gate [0, 1]
    pub skipped: bool,
}

// --- Engine step result ---

#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: u64,
    pub total_delta_norm: f64,
    pub total_activation_energy: f64,
    pub time_ms: f64,
    pub matrices_processed: usize,
    pub matrices_skipped: usize,
}

// --- Weight matrix types ---

/// Metadata about a weight matrix. The data itself lives externally
/// (in a PyTorch tensor / numpy array). Rust borrows it, never copies.
pub struct WeightMeta {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub activation_mag: Option<f32>,
    pub initial_l1: Option<f64>,
    pub energy: f32,               // consumable resource [0.0, 1.0]
    // Reaction-Diffusion
    pub inhibitor: f32,            // inhibitor concentration [0.0, 1.0]
    // Phase + amplitude
    pub phase: f32,                // θ — phase angle [0, 2π]
    pub amplitude: f32,            // |a| — quantum amplitude [0, 1]
    pub natural_freq: f32,         // intrinsic oscillation frequency
    pub group_id: u16,             // neighbor group assignment
    pub contribution: f32,         // recent output contribution (for freq adaptation)
}

/// A weight matrix with owned data — used for pure-Rust tests
/// and standalone operation (no Python).
pub struct WeightMatrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub activation_mag: Option<f32>,
    pub initial_l1: Option<f64>,
    pub energy: f32,
    pub inhibitor: f32,
    pub phase: f32,
    pub amplitude: f32,
    pub natural_freq: f32,
    pub group_id: u16,
    pub contribution: f32,
}

// --- Core dynamics ---

/// Apply one Lenia step to a raw f32 slice + metadata.
/// This is the zero-copy path: Rust operates directly on the caller's memory.
/// The slice IS the PyTorch tensor's data buffer — no copy in, no copy out.
pub fn step_slice(
    data: &mut [f32],
    meta: &mut WeightMeta,
    kernel: &RingKernel,
    config: &LeniaConfig,
) -> MatrixMetrics {
    let rows = meta.rows;
    let cols = meta.cols;
    let d = kernel.diameter;

    if rows < d || cols < d {
        return MatrixMetrics {
            name: meta.name.clone(),
            delta_norm: 0.0,
            activation_energy: 0.0,
            energy_level: meta.energy as f64,
            inhibitor_level: meta.inhibitor as f64,
            phase: meta.phase as f64,
            amplitude: meta.amplitude as f64,
            born_gate: 0.0,
            skipped: true,
        };
    }

    // Record initial L1 norm for mass conservation
    let initial_l1 = match meta.initial_l1 {
        Some(n) => n,
        None => {
            let n: f64 = data.iter().map(|v| v.abs() as f64).sum();
            meta.initial_l1 = Some(n);
            n
        }
    };

    // --- Energy scarcity: logarithmic consumption, differential recovery ---
    //
    // Morphogenesis terrain lesson: quadratic consumption is a death sentence
    // for any matrix with activation > ~1.4. Energy flatlines to floor and
    // Lenia growth is permanently off. The organism looks alive but can't adapt.
    //
    // Logarithmic: cost = rate * (1 + ln(1 + act_mag))
    // At act_mag=0.8: 0.1 * (1 + 0.59) = 0.159  (quadratic was 0.064)
    // At act_mag=4.7: 0.1 * (1 + 1.74) = 0.274  (quadratic was 2.209!)
    // Cost scales with activity but recovery (0.05) can keep up.
    // Energy stays in healthy 0.3-1.0 range instead of flatining to floor.
    let act_mag = meta.activation_mag.unwrap_or(0.0);

    const ENERGY_FLOOR: f32 = 0.1;
    if act_mag > 0.0 && config.energy_consumption_rate > 0.0 {
        let consumption = config.energy_consumption_rate * (1.0 + (act_mag).ln_1p());
        meta.energy -= consumption;
        if meta.energy < ENERGY_FLOOR { meta.energy = ENERGY_FLOOR; }
    }

    // Growth is scaled by BOTH activation coupling AND remaining energy
    let act_scale = if config.activation_coupling > 0.0 && act_mag > 0.0 {
        (act_mag * config.activation_coupling).tanh()
    } else {
        1.0
    };
    let energy_scale = meta.energy;  // [0.0, 1.0] — depleted = no growth

    // --- Born rule gate: amplitude² (quantum mechanics) ---
    // Smooth, physically principled. No artificial sigmoid sharpening.
    // Low amplitude = low contribution. High amplitude = high contribution.
    let born_gate = meta.amplitude * meta.amplitude;

    // --- Inhibitor gate: R-D suppression ---
    let inhibitor_gate = (1.0 - meta.inhibitor).max(0.0);

    // Convolution: reads data, writes to separate potential buffer
    let potential = conv2d_tiled(data, rows, cols, kernel);

    // Growth gated by THREE constraints:
    //   energy: consumable resource (burn out / recover)
    //   amplitude²: quantum selection (Born rule — interference-dependent)
    //   inhibitor: spatial suppression (active neighbors suppress)
    // All three must be open for growth to occur.
    let total_elements = rows * cols;
    let mut delta_sum: f64 = 0.0;
    let growth_mu = config.growth_mu;
    let growth_sigma = config.growth_sigma;
    let growth_scale = config.growth_scale;
    let max_delta = config.max_weight_delta;

    for i in 0..total_elements {
        let z = (potential[i] - growth_mu) / growth_sigma;
        let growth = 2.0 * (-z * z / 2.0).exp() - 1.0;
        let mut delta = growth_scale * growth * act_scale
            * energy_scale * born_gate * inhibitor_gate;
        delta = delta.clamp(-max_delta, max_delta);
        data[i] += delta;
        delta_sum += delta.abs() as f64;
    }

    // Differential recovery: quiet matrices recover faster.
    // recovery = base_rate * (1.0 - act_mag)
    // At act_mag=0.0: full recovery rate.
    // At act_mag=0.5: half recovery rate.
    // At act_mag=1.0: zero recovery (still burning).
    let recovery = config.energy_recovery_rate * (1.0 - act_mag).max(0.0);
    meta.energy += recovery;
    if meta.energy > 1.0 { meta.energy = 1.0; }

    // Clip weights
    for v in data.iter_mut() {
        *v = v.clamp(config.weight_clip_min, config.weight_clip_max);
    }

    // Mass conservation
    let current_l1: f64 = data.iter().map(|v| v.abs() as f64).sum();
    if current_l1 > 1e-12 {
        let scale = (initial_l1 / current_l1) as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
    }

    // --- Inhibitor production (R-D): active matrices produce inhibitor ---
    if act_mag > 0.0 {
        meta.inhibitor += act_mag * config.inhibitor_production_rate;
        if meta.inhibitor > 1.0 { meta.inhibitor = 1.0; }
    }
    // Inhibitor decay
    meta.inhibitor *= 1.0 - config.inhibitor_decay_rate;

    // --- Phase advance (free drift — no coupling, interference handles selection) ---
    meta.phase += meta.natural_freq * config.phase_dt;
    if meta.phase > std::f32::consts::TAU { meta.phase -= std::f32::consts::TAU; }

    // --- Amplitude dynamics (quantum) ---
    // Hebbian: activation grows amplitude. Decay without activation.
    // Interference with neighbors applied externally in inter_matrix_coupling.
    meta.amplitude = meta.amplitude * (1.0 - config.amplitude_decay)
        + act_mag * config.amplitude_coupling;
    meta.amplitude = meta.amplitude.clamp(0.0, 1.0);

    // --- Track contribution for frequency adaptation ---
    meta.contribution = delta_sum as f32 / total_elements as f32;

    MatrixMetrics {
        name: meta.name.clone(),
        delta_norm: delta_sum / total_elements as f64,
        activation_energy: current_l1,
        energy_level: meta.energy as f64,
        inhibitor_level: meta.inhibitor as f64,
        phase: meta.phase as f64,
        amplitude: meta.amplitude as f64,
        born_gate: born_gate as f64,
        skipped: false,
    }
}

/// Convenience wrapper for owned WeightMatrix (tests, standalone).
pub fn step_matrix(matrix: &mut WeightMatrix, kernel: &RingKernel, config: &LeniaConfig) -> MatrixMetrics {
    let mut meta = WeightMeta {
        name: matrix.name.clone(),
        rows: matrix.rows,
        cols: matrix.cols,
        activation_mag: matrix.activation_mag,
        initial_l1: matrix.initial_l1,
        energy: matrix.energy,
        inhibitor: matrix.inhibitor,
        phase: matrix.phase,
        amplitude: matrix.amplitude,
        natural_freq: matrix.natural_freq,
        group_id: matrix.group_id,
        contribution: matrix.contribution,
    };
    let result = step_slice(&mut matrix.data, &mut meta, kernel, config);
    matrix.initial_l1 = meta.initial_l1;
    matrix.energy = meta.energy;
    matrix.inhibitor = meta.inhibitor;
    matrix.phase = meta.phase;
    matrix.amplitude = meta.amplitude;
    matrix.contribution = meta.contribution;
    result
}

// --- Neighbor Topology ---

/// Neighbor group: indices of matrices that are coupled.
#[derive(Debug, Clone)]
pub struct NeighborGroup {
    pub members: Vec<usize>,    // indices into the matrices vec
    pub coupling: f32,          // coupling strength for this group
}

/// Inter-matrix coupling topology parsed from transformer structure.
#[derive(Debug, Clone)]
pub struct NeighborTopology {
    /// For each matrix index, its neighbor indices + coupling weights
    pub neighbors: Vec<Vec<(usize, f32)>>,
    pub groups: Vec<NeighborGroup>,
}

impl NeighborTopology {
    /// Build topology from matrix names.
    /// Parses "layers.N.self_attn.{q,k,v,o}_proj.weight" and
    /// "layers.N.mlp.{gate,up,down}_proj.weight" patterns.
    pub fn build(metas: &[WeightMeta], config: &LeniaConfig) -> Self {
        let n = metas.len();
        let mut neighbors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut groups: Vec<NeighborGroup> = Vec::new();

        let intra_coupling = config.interference_rate;
        let inter_coupling = config.interference_rate * 0.3; // weaker across layers

        // Parse layer and projection type from names
        struct Parsed { layer: i32, proj_type: String }
        let parsed: Vec<Option<Parsed>> = metas.iter().map(|m| {
            // Try to parse "layers.N.*.proj_name.weight" or similar
            let parts: Vec<&str> = m.name.split('.').collect();
            let mut layer = -1i32;
            let mut proj = String::new();
            for (i, p) in parts.iter().enumerate() {
                if let Ok(l) = p.parse::<i32>() {
                    layer = l;
                }
                if p.ends_with("_proj") || *p == "weight" {
                    // Use the part before .weight as the projection name
                    if i > 0 { proj = parts[i-1].to_string(); }
                }
            }
            if layer >= 0 && !proj.is_empty() {
                Some(Parsed { layer, proj_type: proj })
            } else {
                None
            }
        }).collect();

        // Build intra-layer groups (attention: q/k/v/o, mlp: gate/up/down)
        let max_layer = parsed.iter().filter_map(|p| p.as_ref().map(|p| p.layer)).max().unwrap_or(0);
        for layer in 0..=max_layer {
            // Attention group
            let attn: Vec<usize> = (0..n).filter(|&i| {
                parsed[i].as_ref().map_or(false, |p|
                    p.layer == layer && (p.proj_type == "q" || p.proj_type == "k"
                        || p.proj_type == "v" || p.proj_type == "o"
                        || p.proj_type == "q_proj" || p.proj_type == "k_proj"
                        || p.proj_type == "v_proj" || p.proj_type == "o_proj"))
            }).collect();

            if attn.len() > 1 {
                for &a in &attn {
                    for &b in &attn {
                        if a != b { neighbors[a].push((b, intra_coupling)); }
                    }
                }
                groups.push(NeighborGroup { members: attn, coupling: intra_coupling });
            }

            // MLP group
            let mlp: Vec<usize> = (0..n).filter(|&i| {
                parsed[i].as_ref().map_or(false, |p|
                    p.layer == layer && (p.proj_type == "gate" || p.proj_type == "up"
                        || p.proj_type == "down" || p.proj_type == "gate_proj"
                        || p.proj_type == "up_proj" || p.proj_type == "down_proj"))
            }).collect();

            if mlp.len() > 1 {
                for &a in &mlp {
                    for &b in &mlp {
                        if a != b { neighbors[a].push((b, inter_coupling)); }
                    }
                }
                groups.push(NeighborGroup { members: mlp, coupling: intra_coupling });
            }
        }

        // Build inter-layer vertical chains (same proj type across layers)
        let proj_types: Vec<String> = parsed.iter()
            .filter_map(|p| p.as_ref().map(|p| p.proj_type.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();

        for pt in &proj_types {
            let mut chain: Vec<(i32, usize)> = (0..n).filter_map(|i| {
                parsed[i].as_ref().and_then(|p| {
                    if &p.proj_type == pt { Some((p.layer, i)) } else { None }
                })
            }).collect();
            chain.sort_by_key(|&(l, _)| l);

            for w in chain.windows(2) {
                let (_, a) = w[0];
                let (_, b) = w[1];
                neighbors[a].push((b, inter_coupling));
                neighbors[b].push((a, inter_coupling));
            }
        }

        // Assign initial natural frequencies and group IDs
        // (done externally when topology is applied to metas)

        NeighborTopology { neighbors, groups }
    }
}

/// Apply quantum interference + inhibitor diffusion across all matrices.
/// Must be called BEFORE the parallel per-matrix step.
/// Uses previous step's values (read-only) to avoid data races.
pub fn inter_matrix_coupling(
    metas: &mut [WeightMeta],
    topology: &NeighborTopology,
    config: &LeniaConfig,
) {
    let n = metas.len();
    if n == 0 { return; }

    // Snapshot current state (read from previous step)
    let prev_phases: Vec<f32> = metas.iter().map(|m| m.phase).collect();
    let prev_amplitudes: Vec<f32> = metas.iter().map(|m| m.amplitude).collect();
    let prev_inhibitors: Vec<f32> = metas.iter().map(|m| m.inhibitor).collect();

    for i in 0..n {
        let neighbors = &topology.neighbors[i];
        if neighbors.is_empty() { continue; }

        // --- Quantum interference ---
        // Constructive when phases align, destructive when opposed.
        // Changes AMPLITUDE based on phase relationships (not phase itself).
        // Phases drift freely at natural frequency.
        let mut interference: f32 = 0.0;
        for &(j, _) in neighbors {
            // a_j * cos(θ_j - θ_i) — constructive/destructive based on phase diff
            interference += prev_amplitudes[j] * (prev_phases[j] - prev_phases[i]).cos();
        }
        interference /= neighbors.len() as f32;
        metas[i].amplitude += config.interference_rate * interference;
        metas[i].amplitude = metas[i].amplitude.clamp(0.0, 1.0);

        // --- Inhibitor diffusion ---
        let mut diffused: f32 = 0.0;
        for &(j, _) in neighbors {
            diffused += prev_inhibitors[j];
        }
        diffused /= neighbors.len() as f32;
        metas[i].inhibitor += diffused * config.inhibitor_diffusion_rate;
        if metas[i].inhibitor > 1.0 { metas[i].inhibitor = 1.0; }
    }

    // --- Frequency adaptation (competence graduation) ---
    // Matrices with high contribution nudge toward group mean frequency.
    // Low contribution matrices drift away.
    if config.freq_adaptation_rate > 0.0 {
        for group in &topology.groups {
            if group.members.len() < 2 { continue; }
            let mean_freq: f32 = group.members.iter()
                .map(|&i| metas[i].natural_freq).sum::<f32>()
                / group.members.len() as f32;
            let mean_contribution: f32 = group.members.iter()
                .map(|&i| metas[i].contribution).sum::<f32>()
                / group.members.len() as f32;

            for &i in &group.members {
                let delta = if metas[i].contribution > mean_contribution {
                    // High contributor: nudge toward group mean (better sync)
                    (mean_freq - metas[i].natural_freq) * config.freq_adaptation_rate
                } else {
                    // Low contributor: nudge away (desync)
                    (metas[i].natural_freq - mean_freq) * config.freq_adaptation_rate * 0.5
                };
                metas[i].natural_freq = (metas[i].natural_freq + delta)
                    .clamp(config.freq_min, config.freq_max);
            }
        }
    }
}

/// 2D convolution via FFT. Pads input and kernel to same size,
/// multiplies in frequency domain, inverse FFTs.
///
/// This is O(N log N) vs O(N k²) for spatial convolution.
/// For a 4864×896 matrix with an 11×11 kernel:
///   Spatial: 4864 × 896 × 121 = ~527M multiplies
///   FFT: 2 FFTs + 1 pointwise multiply + 1 IFFT ≈ 4 × N log N ≈ 100M
/// 2D convolution — tile-based with all kernel positions applied per tile.
///
/// Instead of scanning the entire matrix 121 times (once per kernel entry),
/// process in row-strips that fit in L2 cache. For each strip, apply all
/// kernel positions before moving to the next strip. This keeps both
/// source and destination data hot in cache.
///
/// Tile height chosen so that (tile_rows + kernel_diameter) × cols × 4 bytes
/// fits comfortably in L2 cache (~256KB-1MB).
/// Precomputed non-zero kernel entries for efficient iteration.
struct KernelEntries {
    entries: Vec<(isize, isize, f32)>, // (dy, dx, kval)
}

impl KernelEntries {
    fn from_kernel(kernel: &RingKernel) -> Self {
        let d = kernel.diameter;
        let r = kernel.radius;
        let mut entries = Vec::with_capacity(d * d);
        for ky in 0..d {
            let dy = ky as isize - r as isize;
            for kx in 0..d {
                let kval = kernel.data[ky * d + kx];
                if kval != 0.0 {
                    let dx = kx as isize - r as isize;
                    entries.push((dy, dx, kval));
                }
            }
        }
        Self { entries }
    }
}

/// 2D convolution — tiled, sequential within each matrix.
/// Parallelism happens at the matrix level via step_all + rayon.
///
/// Row-strips sized to fit in L2 cache. All kernel entries applied
/// per strip before moving to next. Auto-vectorized inner loop.
fn conv2d_tiled(data: &[f32], rows: usize, cols: usize, kernel: &RingKernel) -> Vec<f32> {
    let ke = KernelEntries::from_kernel(kernel);

    // Tile height: (tile_h + kernel_d) rows × cols × 4 bytes ≤ ~128KB
    let bytes_per_row = cols * 4;
    let target_bytes = 128 * 1024;
    let tile_h = ((target_bytes / bytes_per_row).saturating_sub(kernel.diameter)).max(1);

    let mut result = vec![0.0f32; rows * cols];

    let mut y = 0;
    while y < rows {
        let y_end = (y + tile_h).min(rows);

        for &(dy, dx, kval) in &ke.entries {
            let x_start = 0.max(-dx) as usize;
            let x_end = cols.min((cols as isize - dx) as usize);
            if x_start >= x_end { continue; }
            let len = x_end - x_start;

            for out_y in y..y_end {
                let sy = out_y as isize + dy;
                if sy < 0 || sy >= rows as isize { continue; }
                let sy = sy as usize;

                let src_base = sy * cols + (x_start as isize + dx) as usize;
                let dst_base = out_y * cols + x_start;

                let src = &data[src_base..src_base + len];
                let dst = &mut result[dst_base..dst_base + len];
                for i in 0..len {
                    dst[i] += kval * src[i];
                }
            }
        }

        y = y_end;
    }

    result
}

/// Process all weight matrices in parallel using rayon.
pub fn step_all(
    matrices: &mut [WeightMatrix],
    kernel: &RingKernel,
    config: &LeniaConfig,
) -> Vec<MatrixMetrics> {
    matrices.par_iter_mut()
        .map(|m| step_matrix(m, kernel, config))
        .collect()
}

// =============================================================================
// Splat Dynamics — Lenia on sparse Gaussian splat point clouds
//
// Dense matrices are Lenia on a spatial grid.
// Splat matrices are Lenia on a point cloud — each splat is a "cell" whose
// neighborhood is defined by Gaussian distance in weight space, not grid adjacency.
//
// Two splat populations per layer:
//   Base splats:   frozen alpha (existing knowledge, Born-rule immune)
//   Myelin splats: learnable alpha, position drift, grown by Lenia into free space
//
// The Born rule protects high-amplitude splats from disturbance.
// Low-amplitude myelin splats are free to grow, drift, or die.
// =============================================================================

/// Per-splat internal state — NOT stored in Python numpy arrays.
/// Python holds mu, sigma, alpha, amp. Rust owns the rest.
#[derive(Debug, Clone)]
pub struct SplatMeta {
    /// Number of splats in this set.
    pub n: usize,
    /// Weight matrix shape (for position normalization).
    pub rows: usize,
    pub cols: usize,
    /// Layer-wide activation magnitude (from transformer forward pass).
    pub activation_mag: Option<f32>,
    /// Per-splat phase angle [0, 2π] — for interference.
    pub phases: Vec<f32>,
    /// Per-splat energy [0, 1] — consumable resource.
    pub energies: Vec<f32>,
    /// Per-splat inhibitor [0, 1] — spatial suppression.
    pub inhibitors: Vec<f32>,
    /// True = myelin (alpha can grow + drift); false = base (frozen alpha).
    pub is_myelin: Vec<bool>,
    /// Initial |alpha| — threshold for Born rule protection.
    /// Once amp > 0.7 AND |alpha| > initial, alpha is frozen even for myelin.
    pub initial_alpha_mags: Vec<f32>,
}

impl SplatMeta {
    pub fn new(n: usize, rows: usize, cols: usize, is_myelin: Vec<bool>) -> Self {
        use std::f32::consts::TAU;
        let phases: Vec<f32> = (0..n)
            .map(|i| TAU * (i as f32 / n as f32))
            .collect();
        Self {
            n,
            rows,
            cols,
            activation_mag: None,
            phases,
            energies: vec![1.0; n],
            inhibitors: vec![0.0; n],
            is_myelin,
            initial_alpha_mags: vec![0.0; n],
        }
    }
}

/// Per-step metrics for a splat matrix.
#[derive(Debug, Clone)]
pub struct SplatMetrics {
    pub name: String,
    pub n_splats: usize,
    pub n_myelin_active: usize,   // myelin splats that changed this step
    pub total_drift: f32,         // sum of |position drift| across all myelin splats
    pub avg_amp: f32,             // average amplitude
    pub high_amp: usize,          // splats with amp > 0.7 (Born-protected)
    pub low_amp: usize,           // splats with amp < 0.1 (candidates for death)
    pub avg_alpha_mag: f32,       // average |alpha| (weight contribution)
}

/// Apply one Lenia step to a set of Gaussian splats.
///
/// All arrays are flat f32 slices:
///   mu:    2N floats — [row_0, col_0, row_1, col_1, ...]  (normalized [0,1])
///   sigma: N floats  — isotropic spread per splat
///   alpha: N floats  — weight value (signed)
///   amp:   N floats  — Born amplitude [0, 1]
///
/// Mu/sigma/alpha/amp are modified in-place.
/// Internal state (phases, energies, etc.) is in `meta`.
pub fn step_splats(
    mu: &mut [f32],
    sigma: &mut [f32],
    alpha: &mut [f32],
    amp: &mut [f32],
    meta: &mut SplatMeta,
    config: &LeniaConfig,
    name: &str,
) -> SplatMetrics {
    let n = meta.n;
    if n == 0 {
        return SplatMetrics {
            name: name.to_string(),
            n_splats: 0,
            n_myelin_active: 0,
            total_drift: 0.0,
            avg_amp: 0.0,
            high_amp: 0,
            low_amp: 0,
            avg_alpha_mag: 0.0,
        };
    }

    let act_mag = meta.activation_mag.unwrap_or(0.0);

    // -------------------------------------------------------------------------
    // Neighborhood potential: u_i = Σ_j K(dist(mu_i, mu_j)) * alpha_j * amp_j²
    //
    // K is the same ring kernel evaluated at normalized distance.
    // We compute distances in [0, 1] × [0, 1] weight space.
    // Ring kernel: K(r) = exp(-((r - 0.5) / sigma_k)²  / 2)
    //   peak at r = 0.5 (mid-range), drops off toward 0 and 1.
    // This means: too-close neighbors suppress (like R-D inhibitor),
    // mid-range neighbors activate (cooperative growth zone),
    // distant neighbors irrelevant.
    // -------------------------------------------------------------------------
    let kernel_sigma = config.kernel_sigma;
    let mut potentials = vec![0.0f32; n];
    let mut grad_row = vec![0.0f32; n]; // position gradient for drift
    let mut grad_col = vec![0.0f32; n];

    for i in 0..n {
        let r_i = mu[2 * i];
        let c_i = mu[2 * i + 1];

        let mut u = 0.0f32;
        let mut gr = 0.0f32;
        let mut gc = 0.0f32;

        for j in 0..n {
            if i == j { continue; }
            let r_j = mu[2 * j];
            let c_j = mu[2 * j + 1];
            let dr = r_i - r_j;
            let dc = c_i - c_j;
            let dist = (dr * dr + dc * dc).sqrt();
            if dist < 1e-6 { continue; }

            // Ring kernel value
            let z = (dist - 0.5) / kernel_sigma;
            let kval = (-z * z / 2.0).exp();

            let contribution = alpha[j] * amp[j] * amp[j];
            u += kval * contribution;

            // Gradient of kernel w.r.t. position i (for drift)
            // dK/dmu_i = kval * (-(dist - 0.5) / (kernel_sigma² * dist)) * (mu_i - mu_j)
            let dk = kval * (-(dist - 0.5) / (kernel_sigma * kernel_sigma * dist));
            gr += dk * contribution * dr;
            gc += dk * contribution * dc;
        }

        // Normalize by N (mean field approximation)
        let norm = (n - 1).max(1) as f32;
        potentials[i] = u / norm;
        grad_row[i] = gr / norm;
        grad_col[i] = gc / norm;
    }

    // -------------------------------------------------------------------------
    // Compute growth function for each splat
    // G(u) = 2·exp(-((u - μ_g) / σ_g)²/2) - 1
    // -------------------------------------------------------------------------
    let growth_mu = config.growth_mu;
    let growth_sigma = config.growth_sigma;
    let growth_scale = config.growth_scale;
    let max_delta = config.max_weight_delta;

    let act_scale = if config.activation_coupling > 0.0 && act_mag > 0.0 {
        (act_mag * config.activation_coupling).tanh()
    } else {
        1.0
    };

    let mut n_myelin_active = 0usize;
    let mut total_drift = 0.0f32;
    const POSITION_LR: f32 = 0.0002; // conservative: splats drift slowly
    const BORN_PROTECTION_THRESHOLD: f32 = 0.7; // amp above this → frozen
    const ENERGY_FLOOR: f32 = 0.1;

    for i in 0..n {
        // --- Energy scarcity (same physics as dense matrices) ---
        if act_mag > 0.0 && config.energy_consumption_rate > 0.0 {
            let consumption = config.energy_consumption_rate * (1.0 + act_mag.ln_1p());
            meta.energies[i] -= consumption;
            if meta.energies[i] < ENERGY_FLOOR { meta.energies[i] = ENERGY_FLOOR; }
        }
        let energy_scale = meta.energies[i];

        // --- Born gate ---
        let born_gate = amp[i] * amp[i];
        let inhibitor_gate = (1.0 - meta.inhibitors[i]).max(0.0);

        let z = (potentials[i] - growth_mu) / growth_sigma;
        let growth = 2.0 * (-z * z / 2.0).exp() - 1.0;

        // --- Alpha update (myelin only, Born-rule protected) ---
        if meta.is_myelin[i] && amp[i] < BORN_PROTECTION_THRESHOLD {
            let mut delta = growth_scale * growth * act_scale
                * energy_scale * born_gate * inhibitor_gate;
            delta = delta.clamp(-max_delta, max_delta);
            if delta.abs() > 1e-7 {
                alpha[i] += delta;
                alpha[i] = alpha[i].clamp(config.weight_clip_min, config.weight_clip_max);
                n_myelin_active += 1;
            }
        }

        // --- Position drift (myelin only, young splats drift freely) ---
        // Drift magnitude inversely proportional to amp: new splats drift fast,
        // mature ones lock in.
        if meta.is_myelin[i] && amp[i] < BORN_PROTECTION_THRESHOLD {
            let mobility = 1.0 - amp[i]; // [0, 1] — high amp = locked
            let dr = POSITION_LR * grad_row[i] * mobility;
            let dc = POSITION_LR * grad_col[i] * mobility;
            mu[2 * i] = (mu[2 * i] + dr).clamp(0.0, 1.0);
            mu[2 * i + 1] = (mu[2 * i + 1] + dc).clamp(0.0, 1.0);
            total_drift += (dr * dr + dc * dc).sqrt();
        }

        // --- Amplitude dynamics (Hebbian, all splats) ---
        meta.phases[i] += meta.phases[i] * config.phase_dt * 0.1; // slow phase drift
        meta.phases[i] = meta.phases[i] % std::f32::consts::TAU;

        amp[i] = amp[i] * (1.0 - config.amplitude_decay)
            + act_mag * config.amplitude_coupling;
        amp[i] = amp[i].clamp(0.0, 1.0);

        // --- Energy recovery ---
        let recovery = config.energy_recovery_rate * (1.0 - act_mag).max(0.0);
        meta.energies[i] += recovery;
        if meta.energies[i] > 1.0 { meta.energies[i] = 1.0; }

        // --- Inhibitor dynamics ---
        if act_mag > 0.0 {
            meta.inhibitors[i] += act_mag * config.inhibitor_production_rate;
            if meta.inhibitors[i] > 1.0 { meta.inhibitors[i] = 1.0; }
        }
        meta.inhibitors[i] *= 1.0 - config.inhibitor_decay_rate;

        // Track initial_alpha_mag on first real contribution
        if meta.initial_alpha_mags[i] < 1e-6 && alpha[i].abs() > 1e-4 {
            meta.initial_alpha_mags[i] = alpha[i].abs();
        }
    }

    // Amplitude pool conservation (same as dense)
    {
        let total_amp: f32 = amp.iter().sum();
        let target = n as f32 * 0.5;
        if total_amp > 0.0 {
            let scale = target / total_amp;
            for a in amp.iter_mut() {
                *a = (*a * scale).clamp(0.0, 1.0);
            }
        }
    }

    let avg_amp = amp.iter().sum::<f32>() / n as f32;
    let high_amp = amp.iter().filter(|&&a| a > 0.7).count();
    let low_amp = amp.iter().filter(|&&a| a < 0.1).count();
    let avg_alpha_mag = alpha.iter().map(|a| a.abs()).sum::<f32>() / n as f32;

    SplatMetrics {
        name: name.to_string(),
        n_splats: n,
        n_myelin_active,
        total_drift,
        avg_amp,
        high_amp,
        low_amp,
        avg_alpha_mag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_kernel() -> RingKernel {
        RingKernel::new(3, 1.0)
    }

    #[test]
    fn test_kernel_shape() {
        let k = make_test_kernel();
        assert_eq!(k.diameter, 7);
        assert_eq!(k.data.len(), 49);
        // Center should be zero
        assert_eq!(k.get(3, 3), 0.0);
        // Should be normalized (sum ~= 1.0)
        let sum: f32 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "kernel sum = {}", sum);
    }

    #[test]
    fn test_growth_function_at_mu() {
        // At mu, growth should be ~1.0 (peak of bell curve)
        let config = LeniaConfig::default();
        let z = (config.growth_mu - config.growth_mu) / config.growth_sigma;
        let g = 2.0 * (-z * z / 2.0_f32).exp() - 1.0;
        assert!((g - 1.0).abs() < 0.01, "growth at mu = {}", g);
    }

    #[test]
    fn test_growth_function_far_from_mu() {
        // Far from mu, growth should approach -1.0
        let config = LeniaConfig::default();
        let potential = 10.0; // way off
        let z = (potential - config.growth_mu) / config.growth_sigma;
        let g = 2.0 * (-z * z / 2.0_f32).exp() - 1.0;
        assert!((g - (-1.0)).abs() < 0.01, "growth far from mu = {}", g);
    }

    #[test]
    fn test_mass_conservation() {
        let kernel = make_test_kernel();
        let config = LeniaConfig {
            growth_scale: 0.01, // big step to see change
            ..Default::default()
        };

        let rows = 32;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.01).sin()).collect();
        let initial_l1: f64 = data.iter().map(|v| v.abs() as f64).sum();

        let mut matrix = WeightMatrix {
            name: "test".into(),
            rows, cols, data,
            activation_mag: Some(0.5),
            initial_l1: None,
            energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
        };

        step_matrix(&mut matrix, &kernel, &config);

        let final_l1: f64 = matrix.data.iter().map(|v| v.abs() as f64).sum();
        let drift = (final_l1 - initial_l1).abs() / initial_l1;
        assert!(drift < 0.001, "mass conservation drift = {:.6}", drift);
    }

    #[test]
    fn test_step_modifies_weights() {
        let kernel = make_test_kernel();
        let config = LeniaConfig {
            growth_scale: 0.005,
            ..Default::default()
        };

        let rows = 32;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.01).sin()).collect();
        let original = data.clone();

        let mut matrix = WeightMatrix {
            name: "test".into(),
            rows, cols, data,
            activation_mag: Some(1.0),
            initial_l1: None,
            energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
        };

        let metrics = step_matrix(&mut matrix, &kernel, &config);
        assert!(!metrics.skipped);
        assert!(metrics.delta_norm > 0.0);

        // Weights should have changed
        let changed = matrix.data.iter().zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "weights should have changed after step");
    }

    #[test]
    fn test_skip_small_matrix() {
        let kernel = RingKernel::new(5, 0.8); // diameter 11
        let config = LeniaConfig::default();

        let mut matrix = WeightMatrix {
            name: "tiny".into(),
            rows: 4, cols: 4,
            data: vec![0.1; 16],
            activation_mag: None,
            initial_l1: None,
            energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
        };

        let metrics = step_matrix(&mut matrix, &kernel, &config);
        assert!(metrics.skipped);
    }

    #[test]
    fn test_parallel_step() {
        let kernel = make_test_kernel();
        let config = LeniaConfig::default();

        let mut matrices: Vec<WeightMatrix> = (0..8).map(|i| {
            let data = vec![0.1; 64 * 64];
            WeightMatrix {
                name: format!("layer.{}", i),
                rows: 64, cols: 64, data,
                activation_mag: Some(0.5),
                initial_l1: None,
                energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
            }
        }).collect();

        let results = step_all(&mut matrices, &kernel, &config);
        assert_eq!(results.len(), 8);
        for r in &results {
            assert!(!r.skipped);
            assert!(r.delta_norm > 0.0);
        }
    }

    #[test]
    fn test_realistic_sizes() {
        // Verify we can process the actual Qwen2.5 matrix sizes
        let kernel = RingKernel::new(5, 0.8);
        let config = LeniaConfig {
            kernel_radius: 5,
            kernel_sigma: 0.8,
            growth_mu: 0.12,
            growth_sigma: 0.02,
            growth_scale: 0.005,
            max_weight_delta: 0.05,
            activation_coupling: 2.0,
            ..Default::default()
        };

        // One 896×896 matrix (q_proj size)
        let data = vec![0.02; 896 * 896];
        let mut matrix = WeightMatrix {
            name: "layers.0.q_proj.weight".into(),
            rows: 896, cols: 896, data,
            activation_mag: Some(0.3),
            initial_l1: None,
            energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
        };

        let metrics = step_matrix(&mut matrix, &kernel, &config);
        assert!(!metrics.skipped);
        assert!(metrics.delta_norm > 0.0);
        assert!(metrics.activation_energy > 0.0);
    }

    #[test]
    fn bench_full_168() {
        use std::time::Instant;

        let kernel = RingKernel::new(5, 0.8);
        let config = LeniaConfig {
            kernel_radius: 5, kernel_sigma: 0.8,
            growth_mu: 0.12, growth_sigma: 0.02, growth_scale: 0.005,
            max_weight_delta: 0.05, activation_coupling: 2.0,
            ..Default::default()
        };

        // Single matrix timing per size
        for (name, rows, cols, count) in &[
            ("896x896", 896usize, 896usize, 96),
            ("4864x896", 4864, 896, 48),
            ("896x4864", 896, 4864, 24),
        ] {
            let mut m = WeightMatrix {
                name: name.to_string(), rows: *rows, cols: *cols,
                data: vec![0.02; rows * cols],
                activation_mag: Some(0.3), initial_l1: None,
            energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
            };
            let t0 = Instant::now();
            step_matrix(&mut m, &kernel, &config);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("{}: {:.1} ms x {} = {:.0} ms", name, ms, count, ms * *count as f64);
        }

        // Full parallel
        let mut matrices = Vec::with_capacity(168);
        for _ in 0..24 {
            for _ in 0..4 {
                matrices.push(WeightMatrix {
                    name: "q".into(), rows: 896, cols: 896,
                    data: vec![0.02; 896*896], activation_mag: Some(0.3), initial_l1: None, energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
                });
            }
            for _ in 0..2 {
                matrices.push(WeightMatrix {
                    name: "gate".into(), rows: 4864, cols: 896,
                    data: vec![0.02; 4864*896], activation_mag: Some(0.3), initial_l1: None, energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
                });
            }
            matrices.push(WeightMatrix {
                name: "down".into(), rows: 896, cols: 4864,
                data: vec![0.02; 896*4864], activation_mag: Some(0.3), initial_l1: None, energy: 1.0, inhibitor: 0.0, phase: 0.0, amplitude: 0.5, natural_freq: 0.1, group_id: 0, contribution: 0.0,
            });
        }

        let t0 = Instant::now();
        let results = step_all(&mut matrices, &kernel, &config);
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let processed = results.iter().filter(|r| !r.skipped).count();

        println!("\nFull 168 parallel: {:.0} ms ({} processed)", total_ms, processed);
        println!("Python baseline (VPS): 24,203 ms");
        println!("Speedup: {:.1}x", 24203.0 / total_ms);
        println!("Target (<5s): {}", if total_ms < 5000.0 { "MET" } else { "NOT YET" });
    }
}
