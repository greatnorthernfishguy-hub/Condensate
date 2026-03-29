---
title: Condensate
emoji: 🧊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# Condensate — Do the Same, or More, With Less

A living memory manager that uses neural substrate topology and continuous field dynamics to dynamically condense runtime memory usage.

**Try it:** Enter a prompt and see which model layers are HOT (needed for this input) vs COLD (condensable). The predictor learns access patterns from causal observation and pre-stages data before it's needed.

## How It Works

1. **Membrane** — Hooks into PyTorch model forward passes, records which layers activate per input
2. **Graph Builder** — Discovers clusters (proto-hyperedges), causal chains, and hot/cold patterns from access logs
3. **Predictor** — Predicts next memory access from learned causal topology (98.8% accuracy on inference workloads)
4. **Condenser** — Compresses cold regions, pages to disk, pre-promotes on prediction

## Key Results (PoC)

| Metric | Value |
|---|---|
| Prediction accuracy (inference) | 98.8% |
| RAM reduction (selective access) | 50-82% |
| Compression (structured data) | 3:1 LZ4 |
| Theoretical speedup (cold access) | 5x |

## Architecture

The production version uses:
- **NeuroGraph** SNN for causal spike propagation (temporal prediction)
- **Lenia/Flow-Lenia** continuous field dynamics (thermal gradient management)
- **Rust core** with Python bindings (cache-line aligned, software prefetch)
- **Erasure coding** for fault-tolerant distributed storage

This demo proves the principle with a Python prototype.

*E-T Systems / NeuroGraph Foundation*
*AGPL-3.0*
