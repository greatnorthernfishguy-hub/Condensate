# engine.py — bridges the PyTorch sensor's output to the Rust condensate_core
# engine. Tiering + lz4 measurement + lossless verification all run through the
# real Rust calls; this module only orchestrates (Law: data path is Rust).
#
# ---- Changelog ----
# [2026-06-24] CC — Task 2 initial implementation
# What: Sensor output → Rust condensate_core bridge with sampled/full modes + lossless verify
# Why: demo-reimagining SDD Task 2 spec; required interface for viz.py and app.py (Task 3+4)
# How: classify_tier, lz4_compress_len, lz4_verify_roundtrip all delegated to Rust; Python only orchestrates
# -------------------
import condensate_core as cc


def condense(regions, mode="sampled", verify=True):
    """Classify regions HOT/WARM/COLD and measure lossless lz4 savings on COLD.

    mode="sampled": compress sample_bytes, extrapolate ratio to size_bytes (fast).
    mode="full":    compress full_bytes exactly.
    """
    max_access = max((int(r["access_count"]) for r in regions), default=0)
    out, total, cold_orig, cold_comp = [], 0, 0, 0
    lossless_ok = True

    for r in regions:
        size = int(r["size_bytes"])
        total += size
        tier = cc.classify_tier(int(r["access_count"]), int(max_access))
        entry = {"name": r["name"], "tier": tier, "size_bytes": size}

        if tier == "COLD":
            data = r["full_bytes"] if mode == "full" else r["sample_bytes"]
            if data is None or len(data) == 0:
                out.append(entry); continue
            comp = cc.lz4_compress_len(data)
            ratio = comp / len(data)
            if verify and not cc.lz4_verify_roundtrip(data):
                lossless_ok = False
            region_comp = int(size * ratio)
            cold_orig += size
            cold_comp += region_comp
            entry["compressed_bytes"] = region_comp
            entry["ratio"] = ratio
        out.append(entry)

    saved = cold_orig - cold_comp
    return {
        "mode": mode,
        "regions": out,
        "total_bytes": total,
        "condensed_bytes": total - saved,
        "saved_bytes": saved,
        "saved_pct": (100.0 * saved / total) if total else 0.0,
        "lossless_ok": lossless_ok,
    }
