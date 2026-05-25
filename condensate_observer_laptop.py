#!/usr/bin/env python3
"""
Condensate Observer (Laptop) — system-wide passive memory watcher.

Unlike the VPS observer which targets named ecosystem processes, this
watches EVERY process above MIN_RSS_MB. Purpose: learn what a system-wide
LD_PRELOAD membrane would encounter on this machine before deploying it.

The unknown processes are where the surprises live. Cast wide.

Writes JSONL sidecar to OBSERVATIONS_PATH. No ng_tract — no ecosystem
River on the laptop. Pure stdlib, zero dependencies.

---- Changelog ----
[2026-05-25] CC — Initial version
  What: System-wide /proc scanner for laptop Condensate pre-deployment.
  Why:  VPS membrane kept hitting unexpected things from processes we
        didn't model. Laptop observer watches everything so the membrane
        knows the full landscape before it's deployed.
  How:  Scans ALL /proc PIDs every POLL_INTERVAL. Tracks RSS, vsize,
        threads, zram pressure. Emits new_pid/poll/gone events. JSONL only.
-------------------

E-T Systems / Condensate
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [condensate-laptop] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("condensate-laptop")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL    = 10     # seconds between full /proc scans
MIN_RSS_MB       = 20.0   # ignore processes with RSS below this (filter noise)
ANOMALY_RSS_MB   = 500.0  # single process RSS > this is notable on 8GB
SYS_WARN_AVAIL_MB = 1024  # system available < 1GB is notable

OBSERVATIONS_PATH = Path(
    os.environ.get(
        "CONDENSATE_LAPTOP_OBS",
        os.path.expanduser("~/.et_modules/tracts/condensate/laptop_observations.jsonl")
    )
)

# ---------------------------------------------------------------------------
# /proc helpers
# ---------------------------------------------------------------------------

def _read_kv(path: str) -> Dict[str, str]:
    try:
        with open(path) as f:
            d = {}
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    d[k.strip()] = v.strip()
            return d
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return {}

def _kb(val: Optional[str]) -> float:
    if not val:
        return 0.0
    try:
        return float(val.split()[0])
    except (ValueError, IndexError):
        return 0.0

def _read_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return "unknown"

def _read_cmdline_short(pid: int) -> str:
    """First two words of cmdline, space-joined. Enough to identify the process."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").strip()
        parts = raw.decode("utf-8", errors="replace").split()
        # Strip full paths — just the basename of argv[0] + first arg if short
        if parts:
            parts[0] = os.path.basename(parts[0])
        return " ".join(parts[:2])[:80]
    except (FileNotFoundError, PermissionError, OSError):
        return ""

def _zram_stats() -> Dict[str, float]:
    """Read zram0 stats if available. Returns compressed/uncompressed MB."""
    try:
        # /sys/block/zram0/mm_stat: orig_data_size compr_data_size mem_used_total...
        raw = Path("/sys/block/zram0/mm_stat").read_text().split()
        orig_mb  = int(raw[0]) / (1024 * 1024)
        compr_mb = int(raw[1]) / (1024 * 1024)
        ratio    = orig_mb / max(compr_mb, 0.001)
        return {"zram_orig_mb": orig_mb, "zram_compr_mb": compr_mb, "zram_ratio": ratio}
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return {}

def _sys_mem() -> Dict[str, float]:
    mem = _read_kv("/proc/meminfo")
    total = _kb(mem.get("MemTotal")) / 1024
    avail = _kb(mem.get("MemAvailable")) / 1024
    swap_total = _kb(mem.get("SwapTotal")) / 1024
    swap_free  = _kb(mem.get("SwapFree"))  / 1024
    return {
        "sys_mem_total_mb": total,
        "sys_mem_avail_mb": avail,
        "sys_mem_used_mb":  total - avail,
        "sys_swap_total_mb": swap_total,
        "sys_swap_free_mb":  swap_free,
    }

def snapshot_pid(pid: int) -> Optional[Dict]:
    """Take a memory snapshot of a single PID. Returns None if process is gone."""
    status = _read_kv(f"/proc/{pid}/status")
    if not status:
        return None

    rss_mb = _kb(status.get("VmRSS")) / 1024
    if rss_mb < MIN_RSS_MB:
        return None  # below floor — not worth tracking

    threads_raw = status.get("Threads", "1").split()
    threads = int(threads_raw[0]) if threads_raw else 1

    return {
        "pid":      pid,
        "comm":     _read_comm(pid),
        "cmd":      _read_cmdline_short(pid),
        "rss_mb":   rss_mb,
        "vsize_mb": _kb(status.get("VmSize")) / 1024,
        "threads":  threads,
        "ts":       time.time(),
    }

# ---------------------------------------------------------------------------
# System-wide scan
# ---------------------------------------------------------------------------

def scan_all_pids() -> Dict[int, Dict]:
    """Scan /proc for all PIDs above MIN_RSS_MB. Returns {pid: snapshot}."""
    found = {}
    try:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            snap = snapshot_pid(pid)
            if snap is not None:
                found[pid] = snap
    except (PermissionError, OSError):
        pass
    return found

# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def _write(record: Dict):
    try:
        OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OBSERVATIONS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.warning("write failed: %s", exc)

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SystemTracker:
    def __init__(self):
        # pid → last snapshot
        self._last: Dict[int, Dict] = {}
        # pid → first-seen timestamp (for lifetime tracking)
        self._born: Dict[int, float] = {}

    def update(self, current: Dict[int, Dict], sys_mem: Dict, zram: Dict) -> List[Dict]:
        """Compare current scan vs. last scan, emit events. Returns list of records."""
        records = []
        now = time.time()

        # New or updated PIDs
        for pid, snap in current.items():
            if pid not in self._last:
                # New PID — first time we've seen it above the floor
                self._born[pid] = now
                event = "new_pid"
                log.info("new: pid=%d comm=%s rss=%.1f MB cmd=%s",
                         pid, snap["comm"], snap["rss_mb"], snap["cmd"])
            else:
                event = "poll"

            prev = self._last.get(pid)
            rec = {
                "event": event,
                **snap,
                **sys_mem,
                **zram,
            }
            if prev is not None:
                dt = max(snap["ts"] - prev["ts"], 0.001)
                rec["rss_delta_mb"] = snap["rss_mb"] - prev["rss_mb"]
                rec["rss_rate_mb_s"] = rec["rss_delta_mb"] / dt

            if snap["rss_mb"] >= ANOMALY_RSS_MB:
                rec["anomaly"] = True
                log.warning("anomaly: pid=%d comm=%s rss=%.1f MB",
                            pid, snap["comm"], snap["rss_mb"])

            records.append(rec)
            self._last[pid] = snap

        # Gone PIDs — were above floor, now missing
        gone_pids = set(self._last) - set(current)
        for pid in gone_pids:
            prev = self._last.pop(pid)
            lifetime_s = now - self._born.pop(pid, now)
            rec = {
                "event": "gone",
                "pid":   pid,
                "comm":  prev["comm"],
                "cmd":   prev["cmd"],
                "last_rss_mb": prev["rss_mb"],
                "lifetime_s":  lifetime_s,
                "ts":    now,
                **sys_mem,
                **zram,
            }
            records.append(rec)
            if lifetime_s < 5:
                # Short-lived processes are interesting — membrane sees these too
                log.info("short-lived: pid=%d comm=%s lived=%.1fs rss_peak=%.1f MB",
                         pid, prev["comm"], lifetime_s, prev["rss_mb"])

        return records

# ---------------------------------------------------------------------------
# Summary stats (periodic console output)
# ---------------------------------------------------------------------------

def _summary_line(current: Dict[int, Dict], sys_mem: Dict, zram: Dict):
    if not current:
        return
    total_rss = sum(s["rss_mb"] for s in current.values())
    top5 = sorted(current.values(), key=lambda s: s["rss_mb"], reverse=True)[:5]
    avail = sys_mem.get("sys_mem_avail_mb", 0)
    log.info("system: %d procs tracked | total_rss=%.0f MB | sys_avail=%.0f MB",
             len(current), total_rss, avail)
    if zram:
        log.info("zram: orig=%.0f MB compressed=%.0f MB ratio=%.1f×",
                 zram.get("zram_orig_mb", 0), zram.get("zram_compr_mb", 0),
                 zram.get("zram_ratio", 1))
    log.info("top processes by RSS:")
    for s in top5:
        log.info("  pid=%-6d rss=%6.0f MB  %s", s["pid"], s["rss_mb"], s["cmd"] or s["comm"])
    if avail < SYS_WARN_AVAIL_MB:
        log.warning("LOW MEMORY: sys_avail=%.0f MB < %d MB threshold", avail, SYS_WARN_AVAIL_MB)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("starting system-wide observer")
    log.info("  floor: %.0f MB RSS  |  anomaly: %.0f MB  |  poll: %ds",
             MIN_RSS_MB, ANOMALY_RSS_MB, POLL_INTERVAL)
    log.info("  output: %s", OBSERVATIONS_PATH)

    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    tracker = SystemTracker()
    poll_count = 0

    # Initial scan
    log.info("initial scan...")
    sys_mem = _sys_mem()
    zram    = _zram_stats()
    current = scan_all_pids()
    records = tracker.update(current, sys_mem, zram)
    for rec in records:
        _write(rec)
    log.info("initial scan: %d processes above %.0f MB floor", len(current), MIN_RSS_MB)
    _summary_line(current, sys_mem, zram)

    log.info("observing. Ctrl-C to stop.")

    while True:
        time.sleep(POLL_INTERVAL)
        poll_count += 1

        sys_mem = _sys_mem()
        zram    = _zram_stats()
        current = scan_all_pids()
        records = tracker.update(current, sys_mem, zram)

        for rec in records:
            _write(rec)

        # Summary every 60 polls (~10 minutes)
        if poll_count % 60 == 0:
            _summary_line(current, sys_mem, zram)


if __name__ == "__main__":
    main()
