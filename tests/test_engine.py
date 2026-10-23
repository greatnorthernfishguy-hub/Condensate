import numpy as np
import engine


def _region(name, size_bytes, access_count, shape=None, weight_data=None):
    """Build a minimal region dict for engine.condense()."""
    r = {"name": name, "size_bytes": size_bytes, "access_count": access_count,
         "shape": shape or (), "weight_data": weight_data}
    return r


def _low_rank_weights(rows, cols, rank=2, dtype=np.float16):
    """Construct a near-rank-r weight matrix: easy pickings for SVD."""
    rng = np.random.default_rng(42)
    U = rng.standard_normal((rows, rank)).astype(np.float32)
    V = rng.standard_normal((cols, rank)).astype(np.float32)
    return (U @ V.T).astype(dtype)


def test_cold_region_svd_saves_and_high_fidelity():
    rows, cols = 128, 64
    w = _low_rank_weights(rows, cols)
    nbytes = rows * cols * w.itemsize
    regions = [
        _region("hot.weight",  nbytes, 100),        # HOT — max access
        _region("cold.weight", nbytes,   2,          # COLD — near-rank-2 matrix
                shape=(rows, cols), weight_data=w),
    ]
    r = engine.condense(regions, mode="full")
    tiers = {x["name"]: x["tier"] for x in r["regions"]}
    assert tiers["hot.weight"] == "HOT"
    assert tiers["cold.weight"] == "COLD"
    assert r["saved_bytes"] > 0, "SVD boundary should be smaller than raw fp16"
    assert r["mean_fidelity"] >= 0.90, f"fidelity={r['mean_fidelity']:.4f}"
    assert 0 < r["saved_pct"] <= 100


def test_warm_region_not_compressed():
    rows, cols = 64, 64
    w = _low_rank_weights(rows, cols)
    nbytes = rows * cols * w.itemsize
    regions = [
        _region("hot.weight",  nbytes, 100),
        _region("warm.weight", nbytes,  40),         # WARM — no weight_data
    ]
    r = engine.condense(regions, mode="sampled")
    entries = {x["name"]: x for x in r["regions"]}
    assert "compressed_bytes" not in entries["warm.weight"]
    assert r["saved_bytes"] == 0


def test_all_hot_zero_savings():
    nbytes = 1_000_000
    regions = [
        _region("a.weight", nbytes, 100),
        _region("b.weight", nbytes, 100),
    ]
    r = engine.condense(regions, mode="sampled")
    assert r["saved_bytes"] == 0
    assert r["saved_pct"] == 0.0
    assert r["mean_fidelity"] == 1.0


def test_condensed_never_exceeds_full():
    rows, cols = 32, 32
    rng = np.random.default_rng(7)
    w = rng.standard_normal((rows, cols)).astype(np.float16)  # full-rank noise
    nbytes = rows * cols * w.itemsize
    regions = [
        _region("hot.weight",       nbytes, 100),
        _region("noisy_cold.weight", nbytes,  1, shape=(rows, cols), weight_data=w),
    ]
    r = engine.condense(regions, mode="sampled")
    assert r["condensed_bytes"] <= r["total_bytes"], (
        f"condensed {r['condensed_bytes']} > full {r['total_bytes']}"
    )
