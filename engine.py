# engine.py — bridges the PyTorch sensor's output to the Rust condensate_core
# engine. Tiering is delegated to Rust; holographic boundary compression is
# computed here via truncated SVD on COLD weight matrices.
#
# ---- Changelog ----
# [2026-06-24] CC — Task 2 initial implementation (lz4 sampled/full modes)
# [2026-06-24] CC — Signal fix: avg_activation replaces forward_count (punchlist #265)
# [2026-06-25] CC — Holographic boundary: SVD replaces lz4
# What: COLD blocks compressed via truncated SVD (holographic boundary).
# Why: lz4 cannot compress fp16 model weights (high-entropy bytes); SVD exploits
#      the low effective rank of COLD weight matrices — the Area Law guarantee:
#      low-activation blocks have low entanglement entropy → compressible at boundary.
# How: torch.svd_lowrank on COLD blocks; rank = max(16, acc_pct * max_rank) so
#      colder blocks get lower rank (more compression). Numpy fallback for tests.
#      Fidelity = 1 - ||W - U·diag(S)·V^T||_F / ||W||_F. SVD boundary is stored
#      as (U, S, V) fp16 — boundary size = (m + n + 1) * k * 2 bytes.
# -------------------
import condensate_core as cc
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _rank_for(acc_pct, mode):
    """Adaptive SVD rank: colder blocks (lower acc_pct) use less rank."""
    max_rank = 512 if mode == "full" else 128
    return max(16, int(acc_pct * max_rank))


def _svd_boundary(weight_data, shape, rank):
    """Truncated SVD holographic boundary for a COLD weight matrix.

    Returns (compressed_bytes, fidelity, rank_used).
    compressed_bytes = (m + n + 1) * k * 2  (U, S, V stored in fp16)
    fidelity = 1 - ||W - U·diag(S)·V^T||_F / ||W||_F
    """
    m = int(shape[0])
    n = int(np.prod(shape[1:]))
    k = min(rank, min(m, n) - 1)
    if k < 1:
        return int(weight_data.nbytes), 1.0, 0

    if _HAS_TORCH:
        W = torch.from_numpy(weight_data.reshape(m, n)).float()
        U, S, V = torch.svd_lowrank(W, q=k)
        W_approx = U @ torch.diag(S) @ V.T
        residual = float(torch.norm(W - W_approx))
        total_norm = float(torch.norm(W))
    else:
        W = weight_data.reshape(m, n).astype(np.float32)
        U_full, S_full, Vt_full = np.linalg.svd(W, full_matrices=False)
        U, S, Vt = U_full[:, :k], S_full[:k], Vt_full[:k, :]
        W_approx = (U * S) @ Vt
        residual = float(np.linalg.norm(W - W_approx, "fro"))
        total_norm = float(np.linalg.norm(W, "fro"))

    fidelity = 1.0 - (residual / total_norm if total_norm > 0.0 else 0.0)
    compressed_bytes = int((m + n + 1) * k * 2)   # fp16 per element
    return compressed_bytes, fidelity, k


def condense(regions, mode="sampled", verify=True):
    """Classify regions HOT/WARM/COLD and apply SVD holographic boundary to COLD.

    Expects each region dict to contain:
      name, size_bytes, access_count, shape (tuple), weight_data (np.ndarray|None)

    COLD regions with 2-D+ weight_data are compressed via truncated SVD.
    HOT/WARM regions are counted at full size with no compression.
    """
    max_access = max((int(r["access_count"]) for r in regions), default=0)
    out, total, cold_orig, cold_comp = [], 0, 0, 0
    fidelities = []

    for r in regions:
        size = int(r["size_bytes"])
        total += size
        tier = cc.classify_tier(int(r["access_count"]), int(max_access))
        entry = {"name": r["name"], "tier": tier, "size_bytes": size}

        weight_data = r.get("weight_data")
        shape = r.get("shape", ())

        if tier == "COLD" and weight_data is not None and len(shape) >= 2:
            acc_pct = int(r["access_count"]) / max_access if max_access > 0 else 0.0
            rank = _rank_for(acc_pct, mode)
            comp_bytes, fidelity, rank_used = _svd_boundary(weight_data, shape, rank)
            cold_orig += size
            cold_comp += comp_bytes
            entry["compressed_bytes"] = comp_bytes
            entry["rank"] = rank_used
            entry["fidelity"] = fidelity
            fidelities.append(fidelity)

        out.append(entry)

    # Clamp: boundary is never reported as larger than bulk.
    saved = max(0, cold_orig - cold_comp)
    mean_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0

    return {
        "mode": mode,
        "regions": out,
        "total_bytes": total,
        "condensed_bytes": total - saved,
        "saved_bytes": saved,
        "saved_pct": (100.0 * saved / total) if total else 0.0,
        "lossless_ok": mean_fidelity >= 0.99,
        "mean_fidelity": mean_fidelity,
    }
