import os, numpy as np
import engine

def _region(name, size, access, compressible=True):
    # compressible region = zeros (lz4 crushes it); size_bytes is the "real" size
    buf = (np.zeros(size, np.uint8) if compressible
           else np.random.randint(0, 256, size, np.uint8)).tobytes()
    return {"name": name, "size_bytes": size, "access_count": access,
            "sample_bytes": buf[: min(len(buf), 1 << 20)], "full_bytes": buf}

def test_cold_regions_save_and_verify_lossless():
    regions = [
        _region("layer0.q_proj", 1_000_000, 100),   # HOT (max access)
        _region("layer9.down_proj", 1_000_000, 2),   # COLD -> compressible
    ]
    r = engine.condense(regions, mode="full")
    tiers = {x["name"]: x["tier"] for x in r["regions"]}
    assert tiers["layer0.q_proj"] == "HOT"
    assert tiers["layer9.down_proj"] == "COLD"
    assert r["saved_bytes"] > 0
    assert r["lossless_ok"] is True
    assert 0 < r["saved_pct"] <= 100

def test_sampled_and_full_agree_on_uniform_region():
    regions = [_region("a", 4_000_000, 1), _region("hot", 4_000_000, 100)]
    full = engine.condense(regions, mode="full")["saved_bytes"]
    sampled = engine.condense(regions, mode="sampled")["saved_bytes"]
    # zeros compress identically per-byte, so sampled extrapolation ≈ full
    assert abs(full - sampled) / full < 0.05
