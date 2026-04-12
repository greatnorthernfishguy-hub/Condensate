#!/usr/bin/env python3
"""
Condensate Observer — passive memory pattern watcher.

Polls /proc for memory snapshots of known VPS processes,
watches systemd journal for service lifecycle events, and
deposits observations into NeuroGraph via ng_tract.

Zero interference. Pure observation. Bunyan food later.

E-T Systems / Condensate
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ng_tract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [condensate-observer] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("condensate-observer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODULE_ID       = "condensate"
TRACTS_DIR      = Path(os.environ.get("ET_TRACTS_DIR", os.path.expanduser("~/.et_modules/tracts")))
TRACT_PEERS     = ["neurograph", "bunyan"]
POLL_INTERVAL   = 10    # seconds between memory polls
JOURNAL_INTERVAL = 60   # seconds between journal polls
EMBEDDING_DIM   = 768

# name → cmdline fragment to match in /proc/{pid}/cmdline
WATCHED_PROCS: Dict[str, str] = {
    "openclaw-gateway": "openclaw-gateway",
    "neurograph_rpc":   "neurograph_rpc.py",
    "oc-usage-shim":    "oc-usage-shim.py",
    "python3-tid":      "runserver",        # TID Django process
}

# systemd units to watch for start/stop events
WATCHED_SERVICES = [
    "openclaw-gateway.service",
    "condensate-watchdog.service",
    "condensate-observer.service",
]

# ---------------------------------------------------------------------------
# /proc helpers
# ---------------------------------------------------------------------------

def _read_kv_file(path: str) -> Dict[str, str]:
    try:
        with open(path) as f:
            d = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
            return d
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}

def _kb(s: Optional[str]) -> float:
    if not s:
        return 0.0
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return 0.0

def find_pid(fragment: str) -> Optional[int]:
    try:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                cmdline = Path(f"/proc/{entry.name}/cmdline").read_bytes()
                if fragment.encode() in cmdline:
                    return int(entry.name)
            except (FileNotFoundError, PermissionError):
                continue
    except Exception:
        pass
    return None

def take_snapshot(pid: int, name: str) -> Optional[Dict]:
    status = _read_kv_file(f"/proc/{pid}/status")
    if not status:
        return None
    smaps  = _read_kv_file(f"/proc/{pid}/smaps_rollup")
    mem    = _read_kv_file("/proc/meminfo")

    threads_raw = status.get("Threads", "1").split()
    threads = int(threads_raw[0]) if threads_raw else 1

    return {
        "pid":              pid,
        "name":             name,
        "rss_mb":           _kb(status.get("VmRSS"))  / 1024,
        "vsize_mb":         _kb(status.get("VmSize")) / 1024,
        "heap_mb":          _kb(smaps.get("Heap"))    / 1024,
        "stack_kb":         _kb(smaps.get("Stack")),
        "anon_mb":          _kb(smaps.get("Anonymous")) / 1024,
        "threads":          threads,
        "sys_mem_total_mb": _kb(mem.get("MemTotal"))     / 1024,
        "sys_mem_avail_mb": _kb(mem.get("MemAvailable")) / 1024,
        "sys_swap_total_mb":_kb(mem.get("SwapTotal"))    / 1024,
        "sys_swap_free_mb": _kb(mem.get("SwapFree"))     / 1024,
        "timestamp":        time.time(),
    }

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
#
# Layout (768 dims):
#   [0]  rss_mb / 16384           normalized RSS
#   [1]  vsize_mb / 65536         virtual size
#   [2]  threads / 256            thread count
#   [3]  heap_mb / 16384          heap
#   [4]  stack_kb / 65536         stack
#   [5]  anon_mb / 16384          anonymous mappings
#   [6]  rss_growth_mb_per_min    rate of change, clamped −1..1
#   [7]  is_new_pid               1.0 on first observation of this pid
#   [8]  is_restart               1.0 if pid changed since last poll
#   [9]  sys_mem_used_ratio       1 − avail/total
#  [10]  sys_swap_used_ratio      swap fraction used
#  [11−767] zeros (reserved for future Lenia / membrane features)

def encode_snapshot(snap: Dict,
                    prev: Optional[Dict] = None,
                    event: str = "poll") -> np.ndarray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[0]  = snap["rss_mb"]    / 16384.0
    v[1]  = snap["vsize_mb"]  / 65536.0
    v[2]  = snap["threads"]   / 256.0
    v[3]  = snap["heap_mb"]   / 16384.0
    v[4]  = snap["stack_kb"]  / 65536.0
    v[5]  = snap["anon_mb"]   / 16384.0

    if prev is not None:
        dt = max((snap["timestamp"] - prev["timestamp"]) / 60.0, 0.001)
        growth = (snap["rss_mb"] - prev["rss_mb"]) / dt
        v[6] = float(np.clip(growth / 100.0, -1.0, 1.0))

    v[7]  = 1.0 if event == "new_pid"  else 0.0
    v[8]  = 1.0 if event == "restart"  else 0.0

    total = snap["sys_mem_total_mb"]
    if total > 0:
        v[9]  = 1.0 - snap["sys_mem_avail_mb"] / total
    swap = snap["sys_swap_total_mb"]
    if swap > 0:
        v[10] = 1.0 - snap["sys_swap_free_mb"] / swap

    return v

# ---------------------------------------------------------------------------
# Tract deposit
# ---------------------------------------------------------------------------

def _tract_paths() -> List[str]:
    my_dir = TRACTS_DIR / MODULE_ID
    my_dir.mkdir(parents=True, exist_ok=True)
    return [str(my_dir / f"{peer}.tract") for peer in TRACT_PEERS]

def deposit(snap: Dict, emb: np.ndarray, event: str, anomaly: bool = False):
    # [2026-04-12] deposit_outcome — Rust owns the bytes end-to-end. Python never touches them.
    meta = {k: v for k, v in snap.items() if k != "timestamp"}
    meta["event"] = event
    try:
        ng_tract.deposit_outcome(
            snap["timestamp"],
            MODULE_ID,
            snap["name"],
            not anomaly,
            emb,            # numpy f32 array — zero-copy to Rust
            _tract_paths(),
        )
        # Sidecar JSONL — human-readable audit trail alongside the binary tracts
        sidecar = TRACTS_DIR / MODULE_ID / "observations.jsonl"
        with open(sidecar, "a") as f:
            f.write(json.dumps({"ts": snap["timestamp"], **meta}) + "\n")
    except Exception as exc:
        log.warning("deposit failed (%s/%s): %s", snap["name"], event, exc)

# ---------------------------------------------------------------------------
# Journal watcher
# ---------------------------------------------------------------------------

class JournalWatcher:
    def __init__(self):
        self._last = time.time() - 5.0

    def poll(self) -> List[Tuple[str, str, float]]:
        """Return [(unit, state, ts), ...] for lifecycle events since last call."""
        since = f"@{int(self._last)}"
        self._last = time.time()
        units: List[str] = []
        for svc in WATCHED_SERVICES:
            units += ["-u", svc]
        events: List[Tuple[str, str, float]] = []
        try:
            result = subprocess.run(
                ["journalctl", "--since", since, *units,
                 "--output=json", "--no-pager", "-q"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                try:
                    e = json.loads(line)
                    msg  = e.get("MESSAGE", "")
                    unit = e.get("_SYSTEMD_UNIT", e.get("UNIT", "unknown"))
                    ts   = int(e.get("__REALTIME_TIMESTAMP", 0)) / 1e6
                    if any(w in msg for w in ("Started", "start")):
                        events.append((unit, "started", ts))
                    elif any(w in msg for w in ("Stopped", "Failed", "stop")):
                        events.append((unit, "stopped", ts))
                except (json.JSONDecodeError, KeyError):
                    continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return events

# ---------------------------------------------------------------------------
# Process tracker
# ---------------------------------------------------------------------------

class ProcessTracker:
    def __init__(self):
        self._pids:  Dict[str, int]  = {}
        self._snaps: Dict[str, Dict] = {}

    def scan(self) -> List[Tuple[str, Optional[int], str, Optional[Dict], Optional[Dict]]]:
        """Scan watched processes. Returns (name, pid, event, snap, prev_snap)."""
        results = []
        for name, fragment in WATCHED_PROCS.items():
            pid  = find_pid(fragment)

            if pid is None:
                if name in self._pids:
                    log.info("gone: %s (was pid %d)", name, self._pids[name])
                    results.append((name, None, "gone", None, self._snaps.get(name)))
                    del self._pids[name]
                    self._snaps.pop(name, None)
                continue

            snap = take_snapshot(pid, name)
            if snap is None:
                continue   # exited mid-scan

            prev = self._snaps.get(name)

            if name not in self._pids:
                event = "new_pid"
                log.info("new: %s pid=%d rss=%.1f MB", name, pid, snap["rss_mb"])
            elif self._pids[name] != pid:
                event = "restart"
                log.info("restart: %s pid %d→%d rss=%.1f MB",
                         name, self._pids[name], pid, snap["rss_mb"])
            else:
                event = "poll"

            self._pids[name]  = pid
            self._snaps[name] = snap
            results.append((name, pid, event, snap, prev))

        return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _lifecycle_snap(unit: str, ts: float) -> Dict:
    """Minimal snapshot for a journal lifecycle event."""
    mem = _read_kv_file("/proc/meminfo")
    total = _kb(mem.get("MemTotal")) / 1024
    avail = _kb(mem.get("MemAvailable")) / 1024
    return {
        "pid": 0, "name": unit.replace(".service", ""),
        "rss_mb": 0, "vsize_mb": 0, "heap_mb": 0,
        "stack_kb": 0, "anon_mb": 0, "threads": 0,
        "sys_mem_total_mb": total, "sys_mem_avail_mb": avail,
        "sys_swap_total_mb": 0, "sys_swap_free_mb": 0,
        "timestamp": ts,
    }

def main():
    log.info("starting — module=%s peers=%s poll=%ds", MODULE_ID, TRACT_PEERS, POLL_INTERVAL)
    log.info("tract dir: %s", TRACTS_DIR / MODULE_ID)

    tracker = ProcessTracker()
    journal = JournalWatcher()
    last_journal = 0.0

    # Initial scan — birth weights for everything already running
    log.info("initial scan...")
    for name, pid, event, snap, prev in tracker.scan():
        if snap:
            emb = encode_snapshot(snap, prev, event)
            deposit(snap, emb, event)
            log.info("  %s pid=%s rss=%.1f MB threads=%d", name, pid, snap["rss_mb"], snap["threads"])

    log.info("observing.")

    while True:
        time.sleep(POLL_INTERVAL)
        now = time.time()

        for name, pid, event, snap, prev in tracker.scan():
            if snap is None:
                continue
            anomaly = snap["rss_mb"] > 12000
            emb = encode_snapshot(snap, prev, event)
            deposit(snap, emb, event, anomaly=anomaly)
            if event != "poll":
                log.info("lifecycle: %s %s pid=%s rss=%.1f MB", event, name, pid, snap["rss_mb"])
            if anomaly:
                log.warning("anomaly: %s rss=%.1f MB exceeds threshold", name, snap["rss_mb"])

        if now - last_journal >= JOURNAL_INTERVAL:
            last_journal = now
            for unit, state, ts in journal.poll():
                log.info("journal: %s → %s", unit, state)
                snap = _lifecycle_snap(unit, ts or now)
                emb  = encode_snapshot(snap, event=state)
                deposit(snap, emb, event=state)


if __name__ == "__main__":
    main()
