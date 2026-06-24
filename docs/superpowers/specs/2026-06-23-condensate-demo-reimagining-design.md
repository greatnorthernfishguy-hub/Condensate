# Condensate Demo Space — Reimagining (Design Spec)

**Date:** 2026-06-23
**Author:** Claude (Opus 4.8) with Josh
**Status:** Design approved; pending spec review → implementation plan
**Repo:** `~/Condensate` (the public demo Space)
**Live Space:** `huggingface.co/spaces/Executor-Tyrant-Framework/Condensate` (Gradio SDK, ZeroGPU)

---

## 1. Purpose

The current demo is parked-and-stale: it runs the original Python proof-of-concept,
renders the inherently-visual HOT/COLD story as ASCII tables, gates value behind a
3-click flow, and the deployed Space is 27 commits behind. None of the Rust engine
that received all the real development is represented.

This is a ground-up reimagining: **visual + real engine + one flow**. It makes the
product promise — *"do the same, or more, with less"* — visceral in one screen, and
it does so on the actual `condensate_core` Rust engine for the first time.

## 2. Goals / Non-goals

**Goals**
- One action → instant visual payoff → a screenshot-worthy savings number.
- Drive the real Rust engine (`AccessGraph`, `RustPredictor`, and a newly-exposed
  `Condenser`), not the Python PoC.
- Honest-by-construction savings (lossless condensation → identical output).
- Keep free ZeroGPU (truly pay-per-use) via Gradio SDK + a prebuilt Rust wheel.

**Non-goals (this pass)**
- Live token-by-token "cooling" animation (stretch/v2; client-side CSS fade only).
- Paid GPU (revisit only if sustained traffic or a model too big for ZeroGPU).
- Fixing the half-migrated PoC files in place — the rebuild supersedes them.

## 3. The Experience (one flow)

1. User types a prompt (or picks an example chip). One button: **Run**.
2. Model generates — **real text shown** (the "same output" proof).
3. Rust engine classifies every model region HOT/WARM/COLD from real access patterns.
4. **Heatmap reveals on completion** (reveal-on-complete; optional client-side CSS
   cool-down fade for drama at zero GPU cost).
5. **Headline number** lands: `58% saved · 15.0 GB → 6.3 GB · same output`.

Load/Train happen **lazily on first Run** with a progress indicator — no manual steps.

**Model:** `Qwen2.5-7B-Instruct` (Apache-2.0; clean for a public AGPL demo; fits a
ZeroGPU A100 slice). Headline becomes "condensed ~9 GB off a live 7B, same answer."
Fallback to a 3B if ZeroGPU per-call time is too slow.

## 4. Visualization

Composition (single screen, top→bottom):
1. **Headline** savings banner (`gr.HTML`).
2. **Temperature heatmap** (centerpiece): rows = layers, cols = the 7 linear modules
   per layer (q/k/v/o + gate/up/down) → clean 28×7 grid for Qwen2.5-7B. COLD =
   deep blue/near-black, WARM = amber, HOT = bright orange-red. Hover = module name,
   access count, MB. Toggle → finer head-level view (layers × attention heads).
3. **Before/after RAM bars** (full vs condensed, delta annotated).
4. **Stats strip**: HOT/WARM/COLD counts + prediction accuracy (from `RustPredictor`).
5. **Generated text** (same-output proof).

**Tech:** `gr.Plot` + Plotly (interactive hover; ZeroGPU-safe). Data is arrays the
Rust engine already produces; map module names → grid coordinates.

**Render timing:** reveal-on-complete (GPU held only for inference) + a client-side
CSS fade for the "cooling" feel. True live-streaming is a deferred v2.

## 5. Engine Integration & Data Flow

Split: **PyTorch is the sensor, Rust `condensate_core` is the brain.**

```
[1] SENSE (GPU, Python)  torch_membrane forward-hooks every Linear during
                         model.generate() → access events (ts_ns, module, bytes)
                         + weight tensor refs. Output text captured.
[2] THINK (CPU, Rust)    AccessGraph.build(events) → causal topology
                         RustPredictor.learn/score → prediction accuracy
                         Condenser.tier() → HOT/WARM/COLD per region
                         Condenser lz4-compresses COLD bytes → real sizes
[3] SHOW (Python)        map module→grid → heatmap, RAM bars, stats, text
```

### Integrity backbone (must stay honest)
Condensation is **lossless lz4 of idle regions with decompress-on-access**, NOT
weight dropping/quantization. COLD bytes remain; they're just stored compactly while
untouched. Therefore:
- **RAM saved** = Σ(cold_region_size − lz4_compressed_size) — *measured*.
- **"Same output"** = guaranteed by the lossless round-trip; **assert in code**
  (decompress == original) and surface a green "✓ lossless verified" check.
- **Generated text** = the model's one real run. No re-run, no overclaim.

### Measurement modes (toggle; programmatically drivable)
The engine call takes a `mode` argument so one code path serves both:
- **Sampled (default):** lz4 a representative sample per COLD region, apply the
  measured ratio. Snappy. Labeled *"ratio measured on N MB/region sample."*
- **Full (compress-all):** lz4 every COLD byte; exact. Labeled *"full — exact"* with
  the wall-clock shown. The audit mode.

`mode` is a plain function argument (not UI-only), so it can be forced (`mode="full"`)
from a script or by a coding agent without touching the widget.

### Build items created here
1. **Expose `Condenser` via PyO3** — new `#[pyclass]` in `rust_core/src/lib.rs` +
   methods (`tier`, `compressed_sizes`/savings with `mode`, lossless-verify). Replaces
   the `condenser.py` placeholder. (Currently only `AccessGraph`, `RustPredictor`,
   `Prediction` are exposed.)
2. **Refit `torch_membrane`** as the sensor feeding the Rust path; drop the Python-side
   "potential" math in favor of the Rust Condenser's real numbers.

## 6. Packaging (wheel + Space)

Stays **Gradio SDK** (for ZeroGPU). Rust reaches a cargo-less runtime as a prebuilt wheel.

- **Wheel:** add PyO3 **`abi3-py310`** (one wheel works across Python ≥3.10) and build
  with **`maturin build --release --features python --zig --compatibility
  manylinux_2_28`** → `condensate_core-…-cp310-abi3-manylinux_2_28_x86_64.whl` (~1 MB),
  committed under `wheels/` and referenced from `requirements.txt`. No PyPI publish.
- **Structure:**
  ```
  README.md        sdk: gradio, sdk_version: 6.x, app_file: app.py
  requirements.txt gradio, spaces, torch, transformers, accelerate, plotly, ./wheels/<wheel>
  app.py           one-flow UI
  sensor.py        torch_membrane refit
  viz.py           plotly heatmap + RAM bars
  examples.py      prompt chips
  wheels/<wheel>   prebuilt Rust engine
  ```
- **ZeroGPU constraints:** `import spaces`; `@spaces.GPU(duration=…)`; load weights to
  CPU at import, move to CUDA *inside* the decorated fn; no `torch.compile`; CUDA-wheel torch.

**Operational risk:** a 7B → ~15 GB model download per cold start (ephemeral ZeroGPU
storage). First post-idle run is slow. Options: accept (session cache warms), attach
persistent storage, or choose a smaller-footprint model. Not a blocker.

## 7. Deployment

**Staging-first.** Build/validate on a throwaway `Condensate-dev` Space; promote to the
live Space when green → zero public downtime. As part of promotion, **fix the broken
auto-sync** (GitHub Action triggers on `main`; branch is `master`) so future pushes deploy.

## 8. Testing

The 8 GB laptop can't run Gradio, so test the parts that matter headlessly:
- Rust unit tests for the new `Condenser` binding.
- **Headless pipeline test on tiny gpt2**: sensor → `condensate_core` → savings +
  lossless-verify + plotly figure builds — proves the chain without Gradio or GPU.
- Full UI validated on the staging Space.

## 9. Known prior-state notes (do not lose)
- The demo's PoC files are a **half-finished migration**: `graph_builder.py` /
  `predictor.py` are real Rust shims; `membrane.py` is a stub; `condenser.py` is an
  explicit placeholder. `app.py` (Mar 30) predates the shim rewrite (Apr 5) and calls
  `Membrane.clear/wrap/get_log`, which no longer exist → the Synthetic tab would
  `AttributeError` locally. The live Space works only because it's 27 commits behind.
  The rebuild replaces this path entirely (LAW 3: restore, don't accrete shrapnel).

## 10. Decisions locked
- Architecture: Gradio SDK + ZeroGPU + prebuilt Rust wheel.
- Model: Qwen2.5-7B-Instruct (3B fallback).
- Visual: heatmap-centric, reveal-on-complete + client-side cool-down.
- Engine: real `condensate_core`; expose `Condenser` via PyO3; lossless integrity.
- Measurement: sampled default + full toggle (mode is a plain arg).
- Deploy: staging-first, then promote + fix auto-sync.
