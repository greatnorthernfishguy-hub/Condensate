//! Sleep Consolidation — Block I of the Condensate living-memory lifecycle.
//!
//! During idle periods the system enters a biological sleep cycle:
//!   Phase 1 (Replay)     — replay recent access patterns at high speed
//!   Phase 2 (Reorganize) — compute layout improvements
//!   Phase 3 (Prune)      — remove weak edges, compact
//!
//! The caller drives each phase with tick_* methods and is responsible for
//! applying the returned hints to the actual graph/layout structures.

// ─── ReplayEvent ────────────────────────────────────────────────────────────

/// A single recorded memory-access event stored in the replay buffer.
#[derive(Clone, Debug)]
pub struct ReplayEvent {
    pub timestamp_ns: u64,
    pub path_id: u32,
    pub size: u64,
    /// true = allocation, false = free
    pub is_alloc: bool,
}

// ─── ReplayBuffer ───────────────────────────────────────────────────────────

/// Fixed-capacity ring buffer of ReplayEvents.  Oldest events are silently
/// overwritten once the buffer is full.
pub struct ReplayBuffer {
    events: Vec<ReplayEvent>,
    capacity: usize,
    write_pos: usize,
    wrapped: bool,
}

impl ReplayBuffer {
    /// Allocate a ring buffer with `capacity` slots.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ReplayBuffer capacity must be > 0");
        Self {
            events: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            wrapped: false,
        }
    }

    /// Push one event.  If the buffer is full the oldest event is overwritten.
    pub fn push(&mut self, event: ReplayEvent) {
        if self.events.len() < self.capacity {
            // Still filling up — just append.
            self.events.push(event);
        } else {
            // Ring is full: overwrite at write_pos.
            self.events[self.write_pos] = event;
            self.wrapped = true;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    /// Return all stored events in chronological order (oldest → newest).
    pub fn drain(&self) -> Vec<&ReplayEvent> {
        let len = self.events.len();
        if len == 0 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(len);

        if !self.wrapped {
            // Buffer never overflowed — elements are already in order.
            for e in &self.events {
                out.push(e);
            }
        } else {
            // write_pos points to the *oldest* slot.
            for i in 0..len {
                let idx = (self.write_pos + i) % self.capacity;
                out.push(&self.events[idx]);
            }
        }

        out
    }

    /// Number of events currently stored.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Remove all stored events and reset internal state.
    pub fn clear(&mut self) {
        self.events.clear();
        self.write_pos = 0;
        self.wrapped = false;
    }
}

// ─── SleepPhase ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SleepPhase {
    Awake,
    /// Phase 1: replay recent patterns at high speed.
    Replay,
    /// Phase 2: compute layout improvements.
    Reorganize,
    /// Phase 3: remove weak edges, compact.
    Prune,
}

// ─── SleepReport ────────────────────────────────────────────────────────────

/// Summary produced at the end of a sleep cycle.
pub struct SleepReport {
    pub duration_ms: u64,
    pub events_replayed: usize,
    pub edges_strengthened: usize,
    pub edges_pruned: usize,
    pub regions_relocated: usize,
    pub keyframes_consolidated: usize,
    pub bytes_freed: usize,
    pub interrupted: bool,
    pub phase_reached: SleepPhase,
}

// ─── SleepController ────────────────────────────────────────────────────────

/// Drives the three-phase sleep cycle for Condensate.
///
/// # Lifecycle
/// ```text
/// (idle detected)
///   → enter_sleep()       [Awake → Replay]
///   → tick_replay()       [repeat until done]
///   → advance_phase()     [Replay → Reorganize]
///   → tick_reorganize()   [repeat until done]
///   → advance_phase()     [Reorganize → Prune]
///   → tick_prune()        [repeat until done]
///   → advance_phase() / wake()  [Prune → Awake]
/// ```
pub struct SleepController {
    state: SleepPhase,
    last_sleep_ns: u64,
    events_since_sleep: u64,
    idle_threshold_ns: u64,
    /// Adaptive threshold — updated from idle_gap_samples.
    learned_idle_gap_ns: u64,
    /// Rolling window of inter-event gaps (max 100).
    idle_gap_samples: Vec<u64>,
    replay_buffer: ReplayBuffer,
    /// Set to true to request an immediate wake.
    wake_interrupt: bool,
    current_report: Option<SleepReport>,
    /// Timestamp (ns) when the current sleep phase started.
    sleep_start_ns: u64,
    /// Snapshot of events replayed — used by tick_replay.
    replay_events_snapshot: Vec<ReplayEvent>,
    /// Replay cursor — how many events we have processed so far.
    replay_cursor: usize,
    /// Edge-strengthening counters: maps (src, dst) → count.
    edge_counts: std::collections::HashMap<(u32, u32), u64>,
}

const IDLE_GAP_WINDOW: usize = 100;

impl SleepController {
    /// Create a new controller.
    ///
    /// * `idle_threshold_ns` — baseline idle gap before the adaptive learner
    ///   kicks in.
    /// * `replay_capacity`   — maximum events held in the ring buffer.
    pub fn new(idle_threshold_ns: u64, replay_capacity: usize) -> Self {
        Self {
            state: SleepPhase::Awake,
            last_sleep_ns: 0,
            events_since_sleep: 0,
            idle_threshold_ns,
            learned_idle_gap_ns: idle_threshold_ns,
            idle_gap_samples: Vec::with_capacity(IDLE_GAP_WINDOW),
            replay_buffer: ReplayBuffer::new(replay_capacity),
            wake_interrupt: false,
            current_report: None,
            sleep_start_ns: 0,
            replay_events_snapshot: Vec::new(),
            replay_cursor: 0,
            edge_counts: std::collections::HashMap::new(),
        }
    }

    // ── Recording ───────────────────────────────────────────────────────────

    /// Record an access event: store it in the replay buffer and update
    /// the adaptive idle-gap learner.
    pub fn record_event(&mut self, event: ReplayEvent) {
        // Learn from the gap to the previous event (if any).
        if self.events_since_sleep > 0 {
            let last_ts = self
                .replay_buffer
                .drain()
                .last()
                .map(|e| e.timestamp_ns)
                .unwrap_or(0);
            if event.timestamp_ns > last_ts {
                let gap = event.timestamp_ns - last_ts;
                self.observe_gap(gap);
            }
        }

        self.events_since_sleep += 1;
        self.replay_buffer.push(event);
    }

    /// Feed one inter-event gap into the rolling window and recompute the
    /// adaptive threshold.
    fn observe_gap(&mut self, gap_ns: u64) {
        if self.idle_gap_samples.len() == IDLE_GAP_WINDOW {
            self.idle_gap_samples.remove(0);
        }
        self.idle_gap_samples.push(gap_ns);
        self.update_adaptive_threshold();
    }

    /// Recompute `learned_idle_gap_ns` = mean + 2 * stddev of the sample
    /// window.  Falls back to `idle_threshold_ns` when no samples exist.
    fn update_adaptive_threshold(&mut self) {
        let n = self.idle_gap_samples.len();
        if n == 0 {
            self.learned_idle_gap_ns = self.idle_threshold_ns;
            return;
        }

        let sum: u64 = self.idle_gap_samples.iter().sum();
        let mean = sum / n as u64;

        // Variance (integer arithmetic — sufficient precision for ns gaps).
        let variance: u64 = self
            .idle_gap_samples
            .iter()
            .map(|&g| {
                let d = if g > mean { g - mean } else { mean - g };
                d * d
            })
            .sum::<u64>()
            / n as u64;

        let stddev = integer_sqrt(variance);

        // threshold = mean + max(2 * stddev, 10 % of mean).
        //
        // The 10 % floor prevents the degenerate case where all gaps are
        // identical (stddev = 0) from producing a threshold exactly equal to
        // the mean.  A server with perfectly regular 2-second gaps must NOT
        // trigger sleep on those 2-second pauses, so the threshold must be
        // strictly above 2 s.
        let margin = (2 * stddev).max(mean / 10);
        let adaptive = mean.saturating_add(margin);
        self.learned_idle_gap_ns = adaptive.max(self.idle_threshold_ns);
    }

    // ── Idle detection ──────────────────────────────────────────────────────

    /// Returns true when the gap between `last_event_ns` and `now_ns` exceeds
    /// the adaptive idle threshold.
    pub fn is_idle(&self, now_ns: u64, last_event_ns: u64) -> bool {
        if now_ns <= last_event_ns {
            return false;
        }
        now_ns - last_event_ns >= self.learned_idle_gap_ns
    }

    // ── Phase management ────────────────────────────────────────────────────

    /// Transition from Awake into Replay, initialising a fresh report.
    /// Returns `SleepPhase::Replay`.
    pub fn enter_sleep(&mut self, now_ns: u64) -> SleepPhase {
        self.state = SleepPhase::Replay;
        self.sleep_start_ns = now_ns;
        self.wake_interrupt = false;
        self.edge_counts.clear();

        // Snapshot the replay buffer so that tick_replay can iterate it
        // without borrowing issues.
        self.replay_events_snapshot = self
            .replay_buffer
            .drain()
            .into_iter()
            .cloned()
            .collect();
        self.replay_cursor = 0;

        self.current_report = Some(SleepReport {
            duration_ms: 0,
            events_replayed: 0,
            edges_strengthened: 0,
            edges_pruned: 0,
            regions_relocated: 0,
            keyframes_consolidated: 0,
            bytes_freed: 0,
            interrupted: false,
            phase_reached: SleepPhase::Replay,
        });

        SleepPhase::Replay
    }

    /// Process a batch of replay events.
    ///
    /// Returns `(edges_strengthened, edges_weakened)`.
    ///
    /// For every sequential pair (A, B) in the replay stream, the A→B edge
    /// counter is incremented.  The caller is responsible for applying the
    /// returned counts to the actual graph.
    pub fn tick_replay(&mut self) -> (usize, usize) {
        let events = &self.replay_events_snapshot;
        let total = events.len();

        if self.replay_cursor >= total.saturating_sub(1) {
            // Nothing (more) to do.
            if let Some(ref mut r) = self.current_report {
                r.events_replayed = total;
            }
            return (0, 0);
        }

        // Process all remaining sequential pairs in one tick (callers can
        // chunk however they like by calling multiple times, but we keep it
        // simple here: process everything remaining).
        let mut strengthened = 0usize;

        while self.replay_cursor + 1 < total {
            let src = events[self.replay_cursor].path_id;
            let dst = events[self.replay_cursor + 1].path_id;
            let counter = self.edge_counts.entry((src, dst)).or_insert(0);
            *counter += 1;
            strengthened += 1;
            self.replay_cursor += 1;
        }
        // Advance past the last event.
        self.replay_cursor = total;

        if let Some(ref mut r) = self.current_report {
            r.events_replayed = total;
            r.edges_strengthened += strengthened;
        }

        (strengthened, 0)
    }

    /// Identify regions whose replay pattern suggests adjacency.
    ///
    /// Returns the count of regions that should be relocated.  The caller
    /// performs the actual relocation.
    ///
    /// Heuristic: any path_id pair that co-occurs in the replay stream with a
    /// count ≥ 2 is considered a relocation candidate; the number of *unique*
    /// such path_ids is reported.
    pub fn tick_reorganize(&mut self) -> usize {
        let hot_nodes: std::collections::HashSet<u32> = self
            .edge_counts
            .iter()
            .filter(|(_, &count)| count >= 2)
            .flat_map(|((src, dst), _)| [*src, *dst])
            .collect();

        let relocated = hot_nodes.len();

        if let Some(ref mut r) = self.current_report {
            r.regions_relocated = relocated;
            r.phase_reached = SleepPhase::Reorganize;
        }

        relocated
    }

    /// Given current edge weights, return edges whose weight is below
    /// `threshold`.  The caller removes them from the graph.
    pub fn tick_prune(
        &mut self,
        edge_weights: &[(u32, u32, f64)],
        threshold: f64,
    ) -> Vec<(u32, u32)> {
        let pruned: Vec<(u32, u32)> = edge_weights
            .iter()
            .filter(|&&(_, _, w)| w < threshold)
            .map(|&(src, dst, _)| (src, dst))
            .collect();

        if let Some(ref mut r) = self.current_report {
            r.edges_pruned = pruned.len();
            r.phase_reached = SleepPhase::Prune;
        }

        pruned
    }

    /// Advance to the next phase in the cycle.
    ///
    /// ```text
    /// Replay → Reorganize → Prune → Awake
    /// ```
    pub fn advance_phase(&mut self) -> SleepPhase {
        self.state = match self.state {
            SleepPhase::Awake => SleepPhase::Replay,
            SleepPhase::Replay => SleepPhase::Reorganize,
            SleepPhase::Reorganize => SleepPhase::Prune,
            SleepPhase::Prune => SleepPhase::Awake,
        };
        self.state
    }

    // ── Wake ────────────────────────────────────────────────────────────────

    /// Interrupt sleep immediately and return a finalised report.
    pub fn wake(&mut self) -> SleepReport {
        // We need a current timestamp — we do not have wall-clock access here,
        // so duration is computed as 0 when entered without a wall-clock tick.
        // Callers that want accurate duration should store the entry time and
        // subtract.  We store sleep_start_ns so the caller can do so.
        let now_ns = self.sleep_start_ns; // conservative — will be 0 if no real clock
        let duration_ms = now_ns.saturating_sub(self.sleep_start_ns) / 1_000_000;

        let interrupted = self.wake_interrupt || self.state != SleepPhase::Awake;
        let phase_reached = self.state;

        self.state = SleepPhase::Awake;
        self.wake_interrupt = false;
        self.events_since_sleep = 0;
        self.replay_buffer.clear();
        self.replay_events_snapshot.clear();
        self.replay_cursor = 0;

        let mut report = self
            .current_report
            .take()
            .unwrap_or_else(|| SleepReport {
                duration_ms: 0,
                events_replayed: 0,
                edges_strengthened: 0,
                edges_pruned: 0,
                regions_relocated: 0,
                keyframes_consolidated: 0,
                bytes_freed: 0,
                interrupted: false,
                phase_reached: SleepPhase::Awake,
            });

        report.duration_ms = duration_ms;
        report.interrupted = interrupted;
        report.phase_reached = phase_reached;

        report
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    /// True if `wake_interrupt` has been set.
    pub fn should_wake(&self) -> bool {
        self.wake_interrupt
    }

    /// Signal that an external event arrived and sleep should end.
    pub fn set_wake_interrupt(&mut self) {
        self.wake_interrupt = true;
    }

    pub fn get_phase(&self) -> SleepPhase {
        self.state
    }

    pub fn events_since_sleep(&self) -> u64 {
        self.events_since_sleep
    }
}

// ─── Utilities ──────────────────────────────────────────────────────────────

/// Integer square root (floor) — avoids pulling in floating-point for the
/// adaptive-threshold computation.
fn integer_sqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(ts: u64, path_id: u32) -> ReplayEvent {
        ReplayEvent {
            timestamp_ns: ts,
            path_id,
            size: 64,
            is_alloc: true,
        }
    }

    // ── ReplayBuffer ────────────────────────────────────────────────────────

    #[test]
    fn test_sleep_replay_buffer_ring() {
        let mut buf = ReplayBuffer::new(3);
        // Fill beyond capacity.
        for i in 0..6u32 {
            buf.push(make_event(i as u64 * 100, i));
        }
        // Only 3 events must be present (the last 3: ids 3, 4, 5).
        assert_eq!(buf.len(), 3);
        let drained = buf.drain();
        let ids: Vec<u32> = drained.iter().map(|e| e.path_id).collect();
        assert!(
            ids.contains(&3) && ids.contains(&4) && ids.contains(&5),
            "expected ids 3,4,5 but got {:?}",
            ids
        );
    }

    #[test]
    fn test_sleep_replay_buffer_drain_order() {
        let mut buf = ReplayBuffer::new(5);
        for i in 0..5u64 {
            buf.push(make_event(i * 10, i as u32));
        }
        let drained = buf.drain();
        let timestamps: Vec<u64> = drained.iter().map(|e| e.timestamp_ns).collect();
        // Must be monotonically non-decreasing (chronological).
        for w in timestamps.windows(2) {
            assert!(
                w[0] <= w[1],
                "drain order violated: {:?} > {:?}",
                w[0],
                w[1]
            );
        }

        // Also test after a wrap.
        let mut buf2 = ReplayBuffer::new(3);
        for i in 0..5u64 {
            buf2.push(make_event(i * 10, i as u32));
        }
        let drained2 = buf2.drain();
        let ts2: Vec<u64> = drained2.iter().map(|e| e.timestamp_ns).collect();
        for w in ts2.windows(2) {
            assert!(w[0] <= w[1], "wrapped drain order violated");
        }
    }

    // ── Idle detection ──────────────────────────────────────────────────────

    #[test]
    fn test_sleep_idle_detection() {
        let threshold_ns = 5_000_000_000u64; // 5 seconds
        let ctrl = SleepController::new(threshold_ns, 64);

        let last_event = 1_000_000_000u64; // 1 s
        // 4 s after last event — NOT idle.
        assert!(!ctrl.is_idle(last_event + 4_000_000_000, last_event));
        // 6 s after last event — idle.
        assert!(ctrl.is_idle(last_event + 6_000_000_000, last_event));
    }

    #[test]
    fn test_sleep_adaptive_idle_threshold() {
        let baseline_ns = 500_000_000u64; // 0.5 s baseline
        let mut ctrl = SleepController::new(baseline_ns, 64);

        // Simulate a server with regular ~2-second inter-event gaps.
        let gap_2s = 2_000_000_000u64;
        for _ in 0..50 {
            ctrl.observe_gap(gap_2s);
        }

        // The adaptive threshold must exceed 2 s so that normal 2-s pauses
        // do NOT trigger sleep.
        assert!(
            ctrl.learned_idle_gap_ns > gap_2s,
            "adaptive threshold ({}) should be above 2 s gap ({})",
            ctrl.learned_idle_gap_ns,
            gap_2s
        );

        let last_event = 0u64;
        // Exactly 2 s later should NOT be idle (normal pause).
        assert!(!ctrl.is_idle(gap_2s, last_event));
    }

    // ── Phase progression ───────────────────────────────────────────────────

    #[test]
    fn test_sleep_phases_advance() {
        let mut ctrl = SleepController::new(1_000_000_000, 16);

        let phase = ctrl.enter_sleep(0);
        assert_eq!(phase, SleepPhase::Replay);

        let p2 = ctrl.advance_phase();
        assert_eq!(p2, SleepPhase::Reorganize);

        let p3 = ctrl.advance_phase();
        assert_eq!(p3, SleepPhase::Prune);

        let p4 = ctrl.advance_phase();
        assert_eq!(p4, SleepPhase::Awake);
    }

    // ── Wake interrupt ──────────────────────────────────────────────────────

    #[test]
    fn test_sleep_wake_interrupts() {
        let mut ctrl = SleepController::new(1_000_000_000, 16);

        ctrl.enter_sleep(0);
        assert_eq!(ctrl.get_phase(), SleepPhase::Replay);
        assert!(!ctrl.should_wake());

        ctrl.set_wake_interrupt();
        assert!(ctrl.should_wake());

        let report = ctrl.wake();
        assert!(report.interrupted, "report should be marked as interrupted");
        assert_eq!(ctrl.get_phase(), SleepPhase::Awake);
    }

    // ── Replay strengthening ────────────────────────────────────────────────

    #[test]
    fn test_sleep_replay_strengthening() {
        let mut ctrl = SleepController::new(1_000_000_000, 64);

        // Push a pattern: A→B→A→B (paths 1, 2, 1, 2).
        ctrl.record_event(make_event(100, 1));
        ctrl.record_event(make_event(200, 2));
        ctrl.record_event(make_event(300, 1));
        ctrl.record_event(make_event(400, 2));

        ctrl.enter_sleep(500);

        let (strengthened, weakened) = ctrl.tick_replay();

        // Three sequential pairs: (1,2), (2,1), (1,2) → 3 edge increments.
        assert_eq!(strengthened, 3, "expected 3 strengthened edges");
        assert_eq!(weakened, 0);

        // The 1→2 edge should have been seen twice.
        assert_eq!(*ctrl.edge_counts.get(&(1, 2)).unwrap_or(&0), 2);
    }

    // ── Prune weak edges ────────────────────────────────────────────────────

    #[test]
    fn test_sleep_prune_weak_edges() {
        let mut ctrl = SleepController::new(1_000_000_000, 16);
        ctrl.enter_sleep(0);

        let edge_weights = vec![
            (1u32, 2u32, 0.9f64), // strong — keep
            (2u32, 3u32, 0.1f64), // weak — prune
            (3u32, 4u32, 0.05f64), // weak — prune
            (4u32, 5u32, 0.8f64), // strong — keep
        ];
        let threshold = 0.2;

        let pruned = ctrl.tick_prune(&edge_weights, threshold);

        assert_eq!(pruned.len(), 2, "expected 2 edges pruned");
        assert!(pruned.contains(&(2, 3)));
        assert!(pruned.contains(&(3, 4)));
    }
}
