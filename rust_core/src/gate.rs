//! Prediction Gate — KISS overhead reduction for Condensate.
//!
//! Confirmed predictions don't get logged. Only surprises teach the substrate.
//! The cost of running Condensate decreases over time as the substrate learns.
//! Tighter timing tolerances mean better cache tier targeting.
//!
//! Mechanics:
//! - Each path gets a PathGate that tracks confirmed/surprise/miss counts.
//! - Timing tolerance starts at 50ms and tightens (×0.95) on each confirmation,
//!   loosens (×1.2) on each surprise, clamped to [2ms, 100ms].
//! - A ring buffer of recent outcomes drives a burst detector: if the surprise
//!   ratio exceeds `surprise_burst_threshold`, gating is disabled globally until
//!   the ratio drops below threshold × 0.5.

use std::collections::HashMap;

// ─── Public types ────────────────────────────────────────────────────────────

/// A raw memory-access event observed from the system.
pub struct AccessEvent {
    pub timestamp_ns: u64,
    pub path: String,
    pub size_bytes: u64,
}

/// A live prediction issued by the predictor for an upcoming access.
pub struct Prediction {
    pub id: u32,
    pub path: String,
    pub confidence: f64,
    pub predicted_at_ns: u64,
    pub expected_delta_ms: f64,
}

/// The outcome of running an AccessEvent through the gate.
pub enum GateOutcome {
    /// The event matched a prediction within timing tolerance.
    Confirmed {
        prediction_id: u32,
        timing_error_ms: f64,
    },
    /// The event was not predicted — teach the substrate.
    Surprise {
        event: AccessEvent,
    },
    /// A prediction window expired without a matching event.
    Miss {
        prediction_id: u32,
        expected_path: String,
    },
}

// ─── Per-path gate ────────────────────────────────────────────────────────────

const TOLERANCE_START_MS: f64 = 50.0;
const TOLERANCE_MIN_MS: f64 = 2.0;
const TOLERANCE_MAX_MS: f64 = 100.0;
const TIGHTEN_FACTOR: f64 = 0.95;
const LOOSEN_FACTOR: f64 = 1.2;

/// Per-path state: timing statistics and adaptive tolerance.
pub struct PathGate {
    pub path_id: u32,
    confirmed_count: u64,
    surprise_count: u64,
    miss_count: u64,
    timing_tolerance_ms: f64,
    gating_enabled: bool,
}

impl PathGate {
    fn new(path_id: u32) -> Self {
        Self {
            path_id,
            confirmed_count: 0,
            surprise_count: 0,
            miss_count: 0,
            timing_tolerance_ms: TOLERANCE_START_MS,
            gating_enabled: true,
        }
    }

    fn on_confirmed(&mut self) {
        self.confirmed_count += 1;
        self.timing_tolerance_ms =
            (self.timing_tolerance_ms * TIGHTEN_FACTOR).max(TOLERANCE_MIN_MS);
    }

    fn on_surprise(&mut self) {
        self.surprise_count += 1;
        self.timing_tolerance_ms =
            (self.timing_tolerance_ms * LOOSEN_FACTOR).min(TOLERANCE_MAX_MS);
    }

    fn on_miss(&mut self) {
        self.miss_count += 1;
        // Decay: treat miss like a mild surprise for tolerance purposes.
        self.timing_tolerance_ms =
            (self.timing_tolerance_ms * LOOSEN_FACTOR).min(TOLERANCE_MAX_MS);
    }
}

// ─── Global prediction gate ───────────────────────────────────────────────────

/// Global gate that routes events through per-path prediction windows.
pub struct PredictionGate {
    gates: HashMap<String, PathGate>,
    global_confirmed: u64,
    global_total: u64,
    surprise_burst_threshold: f64,
    window: Vec<bool>,   // ring buffer; true = surprise
    window_pos: usize,
    window_size: usize,
    next_path_id: u32,
}

impl PredictionGate {
    // ── Construction ─────────────────────────────────────────────────────────

    pub fn new(window_size: usize, surprise_burst_threshold: f64) -> Self {
        let window_size = window_size.max(1);
        Self {
            gates: HashMap::new(),
            global_confirmed: 0,
            global_total: 0,
            surprise_burst_threshold,
            window: vec![false; window_size],
            window_pos: 0,
            window_size,
            next_path_id: 0,
        }
    }

    // ── Core gate check ───────────────────────────────────────────────────────

    /// Route an event through the active prediction set.
    ///
    /// 1. Walk `active_predictions` looking for a path match within timing tolerance.
    ///    The first match with the smallest timing error wins → Confirmed.
    /// 2. If no match → Surprise.
    /// 3. Predictions whose window has expired and haven't fired → Miss (returned
    ///    separately; callers should call `record_outcome` for each Miss too, but
    ///    this function returns the first actionable outcome for the current event).
    ///
    /// Note: Miss detection for *stale* predictions is done inside this function
    /// and the returned outcome may be a Miss when `event`'s timestamp reveals that
    /// an earlier prediction has expired.  The caller should check the return type.
    pub fn check(&mut self, event: &AccessEvent, active_predictions: &[Prediction]) -> GateOutcome {
        // Look for any predictions that fired (path match + timing window).
        let event_time_ms = event.timestamp_ns as f64 / 1_000_000.0;

        // Find the best matching prediction for this event's path.
        let gate = self.get_or_create_gate(&event.path);
        let tolerance = gate.timing_tolerance_ms;
        let gating_ok = gate.gating_enabled;

        // If gating is disabled for this path, treat as surprise.
        if !gating_ok {
            return GateOutcome::Surprise {
                event: AccessEvent {
                    timestamp_ns: event.timestamp_ns,
                    path: event.path.clone(),
                    size_bytes: event.size_bytes,
                },
            };
        }

        // Scan predictions for a match on this path.
        let mut best_match: Option<(u32, f64)> = None; // (id, timing_error_ms)

        for pred in active_predictions {
            if pred.path != event.path {
                continue;
            }
            let predicted_fire_ns = pred.predicted_at_ns
                + (pred.expected_delta_ms * 1_000_000.0) as u64;
            let predicted_fire_ms = predicted_fire_ns as f64 / 1_000_000.0;
            let timing_error_ms = (event_time_ms - predicted_fire_ms).abs();

            if timing_error_ms <= tolerance {
                match best_match {
                    None => best_match = Some((pred.id, timing_error_ms)),
                    Some((_, best_err)) if timing_error_ms < best_err => {
                        best_match = Some((pred.id, timing_error_ms));
                    }
                    _ => {}
                }
            }
        }

        if let Some((pred_id, timing_error_ms)) = best_match {
            return GateOutcome::Confirmed {
                prediction_id: pred_id,
                timing_error_ms,
            };
        }

        // Check for stale predictions (overdue misses) before declaring Surprise.
        // Return the first expired prediction as a Miss; the event becomes a
        // subsequent call.  If none are stale, return Surprise for this event.
        for pred in active_predictions {
            let predicted_fire_ns = pred.predicted_at_ns
                + (pred.expected_delta_ms * 1_000_000.0) as u64;
            // Allow generous 2× tolerance window before calling a miss.
            let deadline_ns = predicted_fire_ns
                + (tolerance * 2.0 * 1_000_000.0) as u64;
            if event.timestamp_ns > deadline_ns {
                return GateOutcome::Miss {
                    prediction_id: pred.id,
                    expected_path: pred.path.clone(),
                };
            }
        }

        // Nothing matched — genuine surprise.
        GateOutcome::Surprise {
            event: AccessEvent {
                timestamp_ns: event.timestamp_ns,
                path: event.path.clone(),
                size_bytes: event.size_bytes,
            },
        }
    }

    // ── Outcome recording ─────────────────────────────────────────────────────

    /// Update internal state based on a gate outcome.
    ///
    /// - Confirmed → tighten timing tolerance for the path.
    /// - Surprise  → loosen tolerance, mark window slot.
    /// - Miss      → decay (loosen) tolerance for the expected path.
    pub fn record_outcome(&mut self, outcome: &GateOutcome) {
        match outcome {
            GateOutcome::Confirmed { prediction_id: _, timing_error_ms: _ } => {
                // We need the path for confirmed — look it up by scanning gates.
                // Since we can't get the path from the outcome alone, the caller
                // must ensure they call check() then record_outcome() in sequence
                // so the path gate was already touched.  We update global counters
                // and the ring buffer here; per-path update is done in
                // record_outcome_for_path().
                self.push_window(false);
                self.global_confirmed += 1;
                self.global_total += 1;
            }
            GateOutcome::Surprise { event } => {
                let gate = self.get_or_create_gate(&event.path);
                gate.on_surprise();
                self.push_window(true);
                self.global_total += 1;
                self.check_surprise_burst();
            }
            GateOutcome::Miss { prediction_id: _, expected_path } => {
                // Loosen the gate for the path that missed.
                let path = expected_path.clone();
                let gate = self.get_or_create_gate(&path);
                gate.on_miss();
                // Misses don't go into the surprise window (they're a different
                // signal), but they don't count as confirmations either.
            }
        }
    }

    /// Per-path confirmed update — call after record_outcome for Confirmed outcomes.
    ///
    /// Because GateOutcome::Confirmed doesn't carry the path, the caller must
    /// supply it.  This is a deliberate design: the gate is checked per-event and
    /// the path is known at the call site.
    pub fn record_confirmed_for_path(&mut self, path: &str) {
        let gate = self.get_or_create_gate(path);
        gate.on_confirmed();
    }

    // ── Ratio & burst ─────────────────────────────────────────────────────────

    /// Fraction of recent window events that were confirmed (1 − surprise_ratio).
    ///
    /// Returns 0.0 at cold start (all slots are false = confirmed, but
    /// global_total == 0 means nothing has happened yet).
    pub fn gate_ratio(&self) -> f64 {
        if self.global_total == 0 {
            return 0.0;
        }
        // Count surprises in the window.
        let surprises = self.window.iter().filter(|&&s| s).count();
        let filled = self.global_total.min(self.window_size as u64) as usize;
        if filled == 0 {
            return 0.0;
        }
        let surprise_ratio = surprises as f64 / filled as f64;
        1.0 - surprise_ratio
    }

    /// Is gating active for a specific path?
    pub fn is_gating_enabled(&self, path: &str) -> bool {
        match self.gates.get(path) {
            Some(g) => g.gating_enabled,
            None => true, // default: enabled (new paths start gated)
        }
    }

    /// Check the surprise window; disable gating if burst threshold is exceeded,
    /// re-enable if ratio drops below threshold × 0.5.
    ///
    /// Returns `true` if gating is currently in burst-disable mode.
    pub fn check_surprise_burst(&mut self) -> bool {
        let filled = self.global_total.min(self.window_size as u64) as usize;
        if filled == 0 {
            return false;
        }
        let surprises = self.window.iter().filter(|&&s| s).count();
        let ratio = surprises as f64 / filled as f64;

        let in_burst = ratio > self.surprise_burst_threshold;
        let recovered = ratio < self.surprise_burst_threshold * 0.5;

        for gate in self.gates.values_mut() {
            if in_burst {
                gate.gating_enabled = false;
            } else if recovered {
                gate.gating_enabled = true;
            }
        }

        in_burst
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Reset a specific path's gate — pattern changed, need to relearn.
    pub fn reset_gate(&mut self, path: &str) {
        if let Some(gate) = self.gates.get_mut(path) {
            gate.confirmed_count = 0;
            gate.surprise_count = 0;
            gate.miss_count = 0;
            gate.timing_tolerance_ms = TOLERANCE_START_MS;
            gate.gating_enabled = true;
        }
    }

    /// Return `(confirmed, surprise, miss, timing_tolerance_ms)` for a path.
    pub fn get_path_stats(&self, path: &str) -> Option<(u64, u64, u64, f64)> {
        self.gates.get(path).map(|g| {
            (g.confirmed_count, g.surprise_count, g.miss_count, g.timing_tolerance_ms)
        })
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    fn get_or_create_gate(&mut self, path: &str) -> &mut PathGate {
        if !self.gates.contains_key(path) {
            let id = self.next_path_id;
            self.next_path_id += 1;
            self.gates.insert(path.to_string(), PathGate::new(id));
        }
        self.gates.get_mut(path).unwrap()
    }

    fn push_window(&mut self, is_surprise: bool) {
        self.window[self.window_pos] = is_surprise;
        self.window_pos = (self.window_pos + 1) % self.window_size;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a prediction that fires at `fire_at_ns`.
    fn make_prediction(id: u32, path: &str, fire_at_ns: u64) -> Prediction {
        Prediction {
            id,
            path: path.to_string(),
            confidence: 0.9,
            predicted_at_ns: fire_at_ns,   // expected_delta_ms = 0 → fires immediately
            expected_delta_ms: 0.0,
        }
    }

    // Helper: build a prediction that fires `delta_ms` after `issued_at_ns`.
    fn make_prediction_delta(
        id: u32,
        path: &str,
        issued_at_ns: u64,
        delta_ms: f64,
    ) -> Prediction {
        Prediction {
            id,
            path: path.to_string(),
            confidence: 0.9,
            predicted_at_ns: issued_at_ns,
            expected_delta_ms: delta_ms,
        }
    }

    fn make_event(path: &str, timestamp_ns: u64) -> AccessEvent {
        AccessEvent {
            timestamp_ns,
            path: path.to_string(),
            size_bytes: 4096,
        }
    }

    // ── 1. Confirmed prediction is gated ─────────────────────────────────────

    #[test]
    fn test_gate_confirmed_prediction_gated() {
        let mut gate = PredictionGate::new(64, 0.3);
        // Prediction: /data/foo fires at t=1_000_000 ns (1 ms).
        // Event arrives at exactly t=1_000_000 ns → timing_error = 0 ms ≤ 50 ms.
        let preds = vec![make_prediction(1, "/data/foo", 1_000_000)];
        let event = make_event("/data/foo", 1_000_000);

        match gate.check(&event, &preds) {
            GateOutcome::Confirmed { prediction_id, timing_error_ms } => {
                assert_eq!(prediction_id, 1);
                assert!(timing_error_ms < 1.0, "Expected ~0 ms error, got {}", timing_error_ms);
            }
            other => panic!("Expected Confirmed, got {:?}", discriminant_name(&other)),
        }
    }

    // ── 2. Unpredicted event is a Surprise ────────────────────────────────────

    #[test]
    fn test_gate_surprise_event() {
        let mut gate = PredictionGate::new(64, 0.3);
        let preds: Vec<Prediction> = vec![];   // no predictions
        let event = make_event("/unexpected/path", 5_000_000);

        match gate.check(&event, &preds) {
            GateOutcome::Surprise { event: e } => {
                assert_eq!(e.path, "/unexpected/path");
            }
            other => panic!("Expected Surprise, got {:?}", discriminant_name(&other)),
        }
    }

    // ── 3. Miss detection ────────────────────────────────────────────────────

    #[test]
    fn test_gate_miss_detection() {
        let mut gate = PredictionGate::new(64, 0.3);

        // Prediction issued at t=0, expected in 10 ms.
        // Event arrives at t=200 ms (far past deadline).
        let preds = vec![make_prediction_delta(42, "/stale/path", 0, 10.0)];
        let late_event = make_event("/other/path", 200_000_000); // 200 ms

        match gate.check(&late_event, &preds) {
            GateOutcome::Miss { prediction_id, expected_path } => {
                assert_eq!(prediction_id, 42);
                assert_eq!(expected_path, "/stale/path");
            }
            other => panic!("Expected Miss, got {:?}", discriminant_name(&other)),
        }
    }

    // ── 4. Gate ratio climbs toward 0.9 over stable events ───────────────────

    #[test]
    fn test_gate_gate_ratio_increases() {
        let window = 200;
        let mut gate = PredictionGate::new(window, 0.3);

        // Feed 1000 confirmed events into the gate.
        for i in 0u64..1000 {
            let t = i * 1_000_000; // 1 ms apart
            let preds = vec![make_prediction(i as u32, "/stable/path", t)];
            let event = make_event("/stable/path", t);

            let outcome = gate.check(&event, &preds);
            gate.record_outcome(&outcome);
            gate.record_confirmed_for_path("/stable/path");
        }

        let ratio = gate.gate_ratio();
        assert!(
            ratio >= 0.85,
            "Expected gate ratio ≥ 0.85 after 1000 stable events, got {:.3}",
            ratio
        );
    }

    // ── 5. Timing tolerance tightens on repeated confirmations ───────────────

    #[test]
    fn test_gate_timing_tolerance_tightens() {
        let mut gate = PredictionGate::new(64, 0.3);
        let path = "/tight/path";

        // Force 40 confirmations via record_confirmed_for_path.
        for _ in 0..40 {
            gate.record_confirmed_for_path(path);
        }

        let (_, _, _, tol) = gate.get_path_stats(path).expect("gate should exist");
        // After 40 × 0.95: 50 × 0.95^40 ≈ 6.5 ms (above 2 ms floor).
        assert!(tol < 25.0, "Tolerance should have tightened, got {:.2} ms", tol);
        assert!(tol >= TOLERANCE_MIN_MS, "Tolerance must not go below {} ms", TOLERANCE_MIN_MS);
    }

    // ── 6. Timing tolerance loosens on surprises ──────────────────────────────

    #[test]
    fn test_gate_timing_tolerance_loosens() {
        let mut gate = PredictionGate::new(64, 0.3);
        let path = "/loose/path";

        // First tighten significantly.
        for _ in 0..30 {
            gate.record_confirmed_for_path(path);
        }
        let (_, _, _, tol_before) = gate.get_path_stats(path).unwrap();

        // Now inject surprises via record_outcome.
        for i in 0u64..10 {
            let event = AccessEvent {
                timestamp_ns: i * 1_000_000,
                path: path.to_string(),
                size_bytes: 4096,
            };
            gate.record_outcome(&GateOutcome::Surprise { event });
        }

        let (_, _, _, tol_after) = gate.get_path_stats(path).unwrap();
        assert!(
            tol_after > tol_before,
            "Tolerance should have loosened: before={:.2} after={:.2}",
            tol_before, tol_after
        );
    }

    // ── 7. Surprise burst disables gating ────────────────────────────────────

    #[test]
    fn test_gate_surprise_burst_disables_gating() {
        let window = 20;
        let threshold = 0.3;
        let mut gate = PredictionGate::new(window, threshold);
        let path = "/burst/path";

        // Prime the gate so it exists.
        gate.record_confirmed_for_path(path);

        // Fill window with surprises (> 30%).
        for i in 0u64..15 {
            let event = AccessEvent {
                timestamp_ns: i * 1_000_000,
                path: path.to_string(),
                size_bytes: 4096,
            };
            gate.record_outcome(&GateOutcome::Surprise { event });
        }

        // check_surprise_burst should disable gating.
        let burst = gate.check_surprise_burst();
        assert!(burst, "Burst should be detected");
        assert!(
            !gate.is_gating_enabled(path),
            "Gating should be disabled during burst"
        );
    }

    // ── 8. Gating re-enables after burst subsides ─────────────────────────────

    #[test]
    fn test_gate_recovery_re_enables_gating() {
        let window = 20;
        let threshold = 0.3;
        let mut gate = PredictionGate::new(window, threshold);
        let path = "/recovery/path";

        // Prime the gate.
        gate.record_confirmed_for_path(path);

        // Inject enough surprises to trigger burst.
        for i in 0u64..8 {
            let event = AccessEvent {
                timestamp_ns: i * 1_000_000,
                path: path.to_string(),
                size_bytes: 4096,
            };
            gate.record_outcome(&GateOutcome::Surprise { event });
        }
        gate.check_surprise_burst();

        // Now flood with confirmed outcomes to push ratio below threshold × 0.5.
        // We need to replace the surprise slots in the ring buffer.
        for i in 0u64..(window as u64) {
            let outcome = GateOutcome::Confirmed {
                prediction_id: i as u32,
                timing_error_ms: 0.5,
            };
            gate.record_outcome(&outcome);
        }

        let burst = gate.check_surprise_burst();
        assert!(!burst, "Burst should have subsided");
        assert!(
            gate.is_gating_enabled(path),
            "Gating should be re-enabled after recovery"
        );
    }

    // ── 9. Reset clears path stats ────────────────────────────────────────────

    #[test]
    fn test_gate_reset_gate() {
        let mut gate = PredictionGate::new(64, 0.3);
        let path = "/reset/path";

        // Build up some state.
        for _ in 0..20 {
            gate.record_confirmed_for_path(path);
        }
        for i in 0u64..5 {
            let event = AccessEvent {
                timestamp_ns: i * 1_000_000,
                path: path.to_string(),
                size_bytes: 4096,
            };
            gate.record_outcome(&GateOutcome::Surprise { event });
        }

        let (conf, surp, miss, tol) = gate.get_path_stats(path).unwrap();
        assert!(conf > 0 || surp > 0, "Should have accumulated counts");
        assert!(tol != TOLERANCE_START_MS || conf > 0, "Tolerance should have changed");
        let _ = (miss, tol); // suppress warnings

        // Reset.
        gate.reset_gate(path);

        let (conf2, surp2, miss2, tol2) = gate.get_path_stats(path).unwrap();
        assert_eq!(conf2, 0);
        assert_eq!(surp2, 0);
        assert_eq!(miss2, 0);
        assert!(
            (tol2 - TOLERANCE_START_MS).abs() < 0.001,
            "Tolerance should reset to {}ms, got {}ms",
            TOLERANCE_START_MS, tol2
        );
    }

    // ── Helper: enum variant name for error messages ──────────────────────────

    fn discriminant_name(outcome: &GateOutcome) -> &'static str {
        match outcome {
            GateOutcome::Confirmed { .. } => "Confirmed",
            GateOutcome::Surprise { .. } => "Surprise",
            GateOutcome::Miss { .. } => "Miss",
        }
    }
}
