# Condensate Demo Reimagining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Condensate demo Space as a one-flow, heatmap-centric demo driven by the real Rust `condensate_core` engine, with honest (lossless) savings on a live 7B model.

**Architecture:** PyTorch is the *sensor* (forward-hooks capture access patterns during generation); Rust `condensate_core` is the *brain* (real lz4 measurement + tiering + lossless verify); a Gradio/ZeroGPU UI renders a Plotly heatmap + before/after RAM. The Rust engine ships as a prebuilt abi3 maturin wheel pip-installed into the Gradio Space.

**Tech Stack:** Rust + PyO3 (abi3) + maturin/zig wheel; Python 3.10; Gradio 6.x; `spaces` (ZeroGPU); transformers + accelerate; Plotly; lz4_flex.

**Spec:** `~/docs/superpowers/specs/2026-06-23-condensate-demo-reimagining-design.md` (Vault canonical).

## Global Constraints

- Gradio **SDK** Space (not Docker) — required for ZeroGPU. `sdk: gradio`, `sdk_version: 6.x`, `app_file: app.py`.
- Rust reaches the runtime ONLY as a prebuilt wheel; no cargo at Space build time.
- PyO3 **abi3-py310** → one wheel for Python ≥3.10. manylinux_2_28 x86_64.
- lz4 calls MUST match `condenser.rs`: `lz4_flex::compress_prepend_size` / `decompress_size_prepended`.
- ZeroGPU: `import spaces`; `@spaces.GPU(duration=…)`; load weights to CPU at import, move to CUDA *inside* the decorated fn; **no `torch.compile`**.
- Model cache MUST be persistent: `HF_HOME=/data/hf-cache` (bucket `Condensate-storage` mounts at `/data`).
- Model: `Qwen2.5-7B-Instruct` (Apache-2.0); 3B fallback if ZeroGPU per-call too slow.
- Integrity: condensation = lossless lz4 of COLD regions; "same output" asserted via round-trip; never overclaim. Label sampled vs full measurement.
- Develop on a **staging** Space (`Condensate-dev`); promote to live only when green.
- Docs: any spec/plan/dev-log lands byte-identical in the `~/docs/` Vault (canonical) too.

---

## File Structure

**Rust (wheel):**
- `rust_core/Cargo.toml` — add `abi3-py310` to pyo3 features.
- `rust_core/pyproject.toml` (new) — maturin build config.
- `rust_core/src/pybind.rs` (new) — pure impls + `#[pyfunction]` wrappers (lz4 measure, lossless verify, tier classify).
- `rust_core/src/lib.rs` — `mod pybind;` + register the new functions.

**Python Space (`~/Condensate/` root, deployed to the Space):**
- `engine.py` (new) — bridges sensor output → `condensate_core` → savings/tiers/lossless result dict.
- `viz.py` (new) — Plotly heatmap + RAM bars from the engine result.
- `examples.py` (new) — prompt chips.
- `app.py` (rewrite) — one-flow Gradio/ZeroGPU UI.
- `requirements.txt` (rewrite) — gradio, spaces, torch, transformers, accelerate, plotly, the wheel.
- `README.md` (update frontmatter + copy).
- `wheels/condensate_core-*-abi3-*.whl` (new) — prebuilt engine.
- `tests/test_engine.py`, `tests/test_viz.py` (new) — headless tests.
- Keep `torch_membrane.py` as the sensor (reused as-is). Delete the dead PoC shims (`membrane.py`, `graph_builder.py`, `predictor.py`, `condenser.py`) — superseded.

---

## Task 1: Expose real lz4 measurement + tiering via PyO3 (abi3 wheel)

**Files:**
- Create: `rust_core/src/pybind.rs`
- Modify: `rust_core/src/lib.rs` (pymodule at lines 54-61), `rust_core/Cargo.toml` (pyo3 dep line 17)
- Create: `rust_core/pyproject.toml`

**Interfaces:**
- Produces (Rust pure, testable): `lz4_compress_len_impl(&[u8]) -> usize`, `lz4_verify_roundtrip_impl(&[u8]) -> bool`, `classify_tier_impl(u64, u64) -> &'static str`.
- Produces (Python module `condensate_core`): `lz4_compress_len(bytes) -> int`, `lz4_verify_roundtrip(bytes) -> bool`, `classify_tier(access_count:int, max_access:int) -> str` ∈ {"HOT","WARM","COLD"}.

- [ ] **Step 1: Write the failing Rust tests** in `rust_core/src/pybind.rs`

```rust
//! PyO3 bindings for the demo: real lz4 measurement + tiering.
//! Pure `*_impl` fns are always compiled and unit-tested; thin #[pyfunction]
//! wrappers (python feature) expose them to Python. lz4 calls mirror
//! condenser.rs::ManagedRegion (compress_prepend_size / decompress_size_prepended).

/// lz4-compressed length of `data` (same call as the engine's compressor).
pub fn lz4_compress_len_impl(data: &[u8]) -> usize {
    lz4_flex::compress_prepend_size(data).len()
}

/// Lossless round-trip: compress then decompress, must equal the original.
/// Backs the demo's "✓ lossless verified" claim.
pub fn lz4_verify_roundtrip_impl(data: &[u8]) -> bool {
    let c = lz4_flex::compress_prepend_size(data);
    matches!(lz4_flex::decompress_size_prepended(&c), Ok(d) if d == data)
}

/// HOT/WARM/COLD by access fraction vs the busiest region.
pub fn classify_tier_impl(access_count: u64, max_access: u64) -> &'static str {
    let f = if max_access == 0 { 0.0 } else { access_count as f64 / max_access as f64 };
    if f >= 0.50 { "HOT" } else if f >= 0.10 { "WARM" } else { "COLD" }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn lz4_compress_len(data: &[u8]) -> usize { super::lz4_compress_len_impl(data) }

    #[pyfunction]
    pub fn lz4_verify_roundtrip(data: &[u8]) -> bool { super::lz4_verify_roundtrip_impl(data) }

    #[pyfunction]
    pub fn classify_tier(access_count: u64, max_access: u64) -> String {
        super::classify_tier_impl(access_count, max_access).to_string()
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(lz4_compress_len, m)?)?;
        m.add_function(wrap_pyfunction!(lz4_verify_roundtrip, m)?)?;
        m.add_function(wrap_pyfunction!(classify_tier, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressible_data_shrinks() {
        let data = vec![7u8; 100_000];
        assert!(lz4_compress_len_impl(&data) < data.len());
    }

    #[test]
    fn roundtrip_is_lossless() {
        let data: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        assert!(lz4_verify_roundtrip_impl(&data));
    }

    #[test]
    fn tiers_classify_by_fraction() {
        assert_eq!(classify_tier_impl(100, 100), "HOT");
        assert_eq!(classify_tier_impl(20, 100), "WARM");
        assert_eq!(classify_tier_impl(1, 100), "COLD");
        assert_eq!(classify_tier_impl(5, 0), "COLD"); // no accesses anywhere
    }
}
```

- [ ] **Step 2: Wire the module + run tests to verify they pass**

In `rust_core/src/lib.rs`, add near the other `mod` declarations:

```rust
mod pybind;
```

Replace the pymodule body (lines 54-61) with:

```rust
#[cfg(feature = "python")]
#[pymodule]
fn condensate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core pipeline
    m.add_class::<graph::AccessGraph>()?;
    m.add_class::<predictor::RustPredictor>()?;
    m.add_class::<predictor::Prediction>()?;
    // Demo measurement + tiering
    pybind::py::register(m)?;
    Ok(())
}
```

Run: `cd ~/Condensate/rust_core && cargo test pybind`
Expected: PASS (3 tests: compressible_data_shrinks, roundtrip_is_lossless, tiers_classify_by_fraction)

- [ ] **Step 3: Add abi3 to pyo3** in `rust_core/Cargo.toml` (line 17)

```toml
pyo3 = { version = "0.24", features = ["extension-module", "abi3-py310"], optional = true }
```

- [ ] **Step 4: Create `rust_core/pyproject.toml`** (maturin build config)

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "condensate_core"
version = "0.1.0"
requires-python = ">=3.10"

[tool.maturin]
features = ["python"]
```

- [ ] **Step 5: Build the abi3 manylinux wheel**

```bash
cd ~/Condensate/rust_core
pip install -q 'maturin>=1.5,<2.0' ziglang
maturin build --release --zig --compatibility manylinux_2_28 -o ../wheels
ls -la ../wheels/*.whl
```
Expected: a file like `wheels/condensate_core-0.1.0-cp310-abi3-manylinux_2_28_x86_64.whl`.

- [ ] **Step 6: Smoke-test the built wheel in a clean venv**

```bash
cd ~/Condensate
python3 -m venv /tmp/whtest && /tmp/whtest/bin/pip install -q wheels/condensate_core-*.whl
/tmp/whtest/bin/python -c "import condensate_core as c; print(c.classify_tier(1,100), c.lz4_verify_roundtrip(b'x'*1000), c.lz4_compress_len(b'\0'*100000))"
```
Expected: `COLD True <small int <100000>`

- [ ] **Step 7: Commit**

```bash
cd ~/Condensate
git add rust_core/src/pybind.rs rust_core/src/lib.rs rust_core/Cargo.toml rust_core/Cargo.lock rust_core/pyproject.toml wheels/*.whl
git commit -m "feat(engine): expose lz4 measure + tier via PyO3 abi3; build wheel"
```

---

## Task 2: Engine bridge — sensor output → savings/tiers/lossless

**Files:**
- Create: `engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `condensate_core.{classify_tier, lz4_compress_len, lz4_verify_roundtrip}` (Task 1).
- Produces: `condense(regions, mode="sampled", verify=True) -> dict`.
  - `regions`: list of `{"name": str, "size_bytes": int, "access_count": int, "sample_bytes": bytes, "full_bytes": bytes|None}`.
  - returns `{"mode", "regions":[{name,tier,size_bytes,compressed_bytes?,ratio?}], "total_bytes", "condensed_bytes", "saved_bytes", "saved_pct", "lossless_ok"}`.

- [ ] **Step 1: Write the failing test** `tests/test_engine.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Condensate && pip install -q wheels/condensate_core-*.whl numpy pytest && pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine'`

- [ ] **Step 3: Implement `engine.py`**

```python
# engine.py — bridges the PyTorch sensor's output to the Rust condensate_core
# engine. Tiering + lz4 measurement + lossless verification all run through the
# real Rust calls; this module only orchestrates (Law: data path is Rust).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Condensate && pytest tests/test_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Condensate
git add engine.py tests/test_engine.py
git commit -m "feat(engine.py): sensor->condensate_core bridge with sampled/full + lossless verify"
```

---

## Task 3: Visualization — Plotly heatmap + RAM bars

**Files:**
- Create: `viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: the `condense()` result dict (Task 2); module names of form `model.layers.<L>.<sub>.<proj>`.
- Produces: `temperature_heatmap(result) -> plotly.graph_objects.Figure`, `ram_bars(result) -> plotly.graph_objects.Figure`, `headline(result) -> str` (HTML).

- [ ] **Step 1: Write the failing test** `tests/test_viz.py`

```python
import plotly.graph_objects as go
import engine, viz, numpy as np

def _r(name, size, access):
    buf = np.zeros(size, np.uint8).tobytes()
    return {"name": name, "size_bytes": size, "access_count": access,
            "sample_bytes": buf, "full_bytes": buf}

def _result():
    regs = [_r(f"model.layers.{l}.self_attn.q_proj", 100_000, 100 if l == 0 else 1)
            for l in range(4)]
    return engine.condense(regs, mode="full")

def test_heatmap_is_a_figure():
    fig = viz.temperature_heatmap(_result())
    assert isinstance(fig, go.Figure)

def test_ram_bars_is_a_figure():
    assert isinstance(viz.ram_bars(_result()), go.Figure)

def test_headline_has_pct_and_lossless():
    html = viz.headline(_result())
    assert "%" in html and ("lossless" in html.lower())
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/Condensate && pip install -q plotly && pytest tests/test_viz.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'viz'`

- [ ] **Step 3: Implement `viz.py`**

```python
# viz.py — Plotly renderers for the condense() result. No engine logic here.
import re
import plotly.graph_objects as go

_TIER_VAL = {"COLD": 0.0, "WARM": 0.5, "HOT": 1.0}
# COLD deep-blue/near-black -> WARM amber -> HOT bright orange-red
_COLORSCALE = [[0.0, "#0a1633"], [0.5, "#e0a000"], [1.0, "#ff3b1f"]]
_RX = re.compile(r"layers\.(\d+)\..*?\.(\w+_proj)$")
_COLS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def temperature_heatmap(result):
    """Grid: rows = layers, cols = the 7 linear modules, color = temperature."""
    layers = set()
    cell = {}
    hover = {}
    for r in result["regions"]:
        m = _RX.search(r["name"])
        if not m:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        layers.add(layer)
        cell[(layer, proj)] = _TIER_VAL.get(r["tier"], 0.0)
        mb = r["size_bytes"] / 1e6
        hover[(layer, proj)] = f"L{layer} {proj}<br>{r['tier']}<br>{mb:.1f} MB"
    layers = sorted(layers)
    z = [[cell.get((L, c), None) for c in _COLS] for L in layers]
    text = [[hover.get((L, c), "") for c in _COLS] for L in layers]
    fig = go.Figure(go.Heatmap(
        z=z, x=_COLS, y=[f"L{L}" for L in layers],
        text=text, hoverinfo="text",
        colorscale=_COLORSCALE, zmin=0.0, zmax=1.0, showscale=False,
    ))
    fig.update_layout(title="Model temperature — COLD is condensable",
                      margin=dict(l=40, r=10, t=40, b=10),
                      paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                      font=dict(color="#cdd6f4"))
    fig.update_yaxes(autorange="reversed")
    return fig


def ram_bars(result):
    full_gb = result["total_bytes"] / 1e9
    cond_gb = result["condensed_bytes"] / 1e9
    fig = go.Figure(go.Bar(
        x=[full_gb, cond_gb], y=["Full model", "Condensed"], orientation="h",
        marker_color=["#ff3b1f", "#3bd16f"],
        text=[f"{full_gb:.1f} GB", f"{cond_gb:.1f} GB"], textposition="auto",
    ))
    fig.update_layout(title="RAM: full vs condensed",
                      margin=dict(l=80, r=10, t=40, b=10),
                      paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                      font=dict(color="#cdd6f4"))
    return fig


def headline(result):
    pct = result["saved_pct"]
    full_gb = result["total_bytes"] / 1e9
    cond_gb = result["condensed_bytes"] / 1e9
    mode = result["mode"]
    check = ("✓ lossless verified" if result["lossless_ok"]
             else "⚠ lossless check FAILED")
    color = "#3bd16f" if result["lossless_ok"] else "#ff3b1f"
    return (
        f'<div style="font-size:1.6rem;font-weight:700">'
        f'{pct:.0f}% saved &middot; {full_gb:.1f} GB &rarr; {cond_gb:.1f} GB '
        f'&middot; same output</div>'
        f'<div style="color:{color}">{check} &middot; '
        f'measurement: {mode}</div>'
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/Condensate && pytest tests/test_viz.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Condensate
git add viz.py tests/test_viz.py
git commit -m "feat(viz.py): plotly temperature heatmap, RAM bars, headline"
```

---

## Task 4: One-flow Gradio/ZeroGPU app

**Files:**
- Create: `examples.py`
- Rewrite: `app.py`
- Delete (dead PoC shrapnel): `membrane.py`, `graph_builder.py`, `predictor.py`, `condenser.py`
- Keep: `torch_membrane.py` (sensor)

**Interfaces:**
- Consumes: `engine.condense`, `viz.{temperature_heatmap,ram_bars,headline}`, `torch_membrane.TorchMembrane` (`get_events`, `get_activation_map`, `reset`).
- Produces: a launchable Gradio `Blocks` on port 7860.

- [ ] **Step 1: Create `examples.py`**

```python
EXAMPLES = [
    "Explain why the sky is blue, in two sentences.",
    "Write a haiku about persistent memory.",
    "What is the capital of France, and one fact about it?",
    "Summarize the plot of Hamlet in three bullet points.",
]
```

- [ ] **Step 2: Write `app.py`** (one flow; lazy load; ZeroGPU; HF cache on the bucket)

```python
# app.py — Condensate demo: one-flow, real engine, visual.
# HF cache MUST live on the persistent bucket so the 7B downloads once.
import os
os.environ.setdefault("HF_HOME", "/data/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf-cache/hub")

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import engine, viz
from examples import EXAMPLES
from torch_membrane import TorchMembrane

MODEL_NAME = os.environ.get("CONDENSATE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_M = {"model": None, "tok": None, "sensor": None}


def _ensure_loaded():
    """Lazy CPU load (no CUDA at import — ZeroGPU rule). Cached on the bucket."""
    if _M["model"] is None:
        _M["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _M["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16)
        _M["model"].eval()
        _M["sensor"] = TorchMembrane(_M["model"])


def _regions_from(model, sensor):
    """Map sensor access stats + weight sizes → engine region dicts."""
    access = {a["name"]: a.get("forward_count", a.get("access_count", 0))
              for a in sensor.get_activation_map()}
    regions = []
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        mod = name[: -len(".weight")]
        nbytes = p.numel() * p.element_size()
        buf = p.detach().to(torch.float16).cpu().numpy().tobytes()
        regions.append({
            "name": mod,
            "size_bytes": nbytes,
            "access_count": int(access.get(mod, 0)),
            "sample_bytes": buf[: 4 << 20],   # 4 MB sample
            "full_bytes": buf,
        })
    return regions


@spaces.GPU(duration=120)
def run(prompt, max_tokens, measurement):
    _ensure_loaded()
    model, tok, sensor = _M["model"], _M["tok"], _M["sensor"]
    model.to("cuda")
    sensor.reset()

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(max_tokens),
                             do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)

    model.to("cpu")  # weights back on host for honest RAM measurement
    regions = _regions_from(model, sensor)
    result = engine.condense(regions, mode=measurement)

    return (viz.headline(result), viz.temperature_heatmap(result),
            viz.ram_bars(result), text)


with gr.Blocks(title="Condensate — Do More With Less", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Condensate\n### Watch a live model condense in RAM — same output.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=2, scale=4)
        run_btn = gr.Button("Run", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=prompt)
    with gr.Row():
        max_tokens = gr.Slider(10, 100, value=40, step=5, label="Max tokens")
        measurement = gr.Radio(["sampled", "full"], value="sampled",
                               label="Savings measurement")
    headline = gr.HTML()
    with gr.Row():
        heat = gr.Plot(label="Temperature map")
        bars = gr.Plot(label="RAM")
    text = gr.Textbox(label="Generated (same output)", lines=4)

    run_btn.click(run, [prompt, max_tokens, measurement],
                  [headline, heat, bars, text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 3: Remove the dead PoC shrapnel**

```bash
cd ~/Condensate
git rm membrane.py graph_builder.py predictor.py condenser.py
```

- [ ] **Step 4: Headless import check (no GPU/Gradio launch)**

Run:
```bash
cd ~/Condensate && python3 -c "import ast; ast.parse(open('app.py').read()); ast.parse(open('examples.py').read()); print('app.py + examples.py parse OK')"
```
Expected: `app.py + examples.py parse OK`

- [ ] **Step 5: Commit**

```bash
cd ~/Condensate
git add app.py examples.py
git commit -m "feat(app): one-flow ZeroGPU demo on real engine; drop PoC shrapnel"
```

---

## Task 5: Space packaging + staging deploy (`Condensate-dev`)

**Files:**
- Rewrite: `requirements.txt`, `README.md`
- Uses: `wheels/*.whl` (Task 1)

**Interfaces:** Produces a running staging Space at `Executor-Tyrant-Framework/Condensate-dev`.

- [ ] **Step 1: Write `requirements.txt`**

```
gradio==6.10.0
spaces
torch
transformers
accelerate
plotly
./wheels/condensate_core-0.1.0-cp310-abi3-manylinux_2_28_x86_64.whl
```
(Adjust the wheel filename to the exact artifact from Task 1 Step 5.)

- [ ] **Step 2: Update `README.md` frontmatter** (top of file)

```yaml
---
title: Condensate
emoji: 🧊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: agpl-3.0
---
```

- [ ] **Step 3: Create the private staging Space**

```bash
python3 - <<'PY'
from huggingface_hub import create_repo
url = create_repo("Executor-Tyrant-Framework/Condensate-dev", repo_type="space",
                  space_sdk="gradio", private=True, exist_ok=True)
print("space:", url)
PY
```
Expected: prints the Space URL.

- [ ] **Step 4: Attach the persistent bucket to `Condensate-dev`**

Manual (Josh / web UI or API): mount the `Condensate-storage` bucket at `/data` on `Condensate-dev` (same as the live Space) so `HF_HOME=/data/hf-cache` resolves. Confirm before first run.

- [ ] **Step 5: Push the working tree to the staging Space `main`**

```bash
cd ~/Condensate
TOKEN=$(cat ~/.cache/huggingface/token)
git push "https://Executor-Tyrant-Framework:$TOKEN@huggingface.co/spaces/Executor-Tyrant-Framework/Condensate-dev" HEAD:main 2>&1 | sed -E "s/$TOKEN/***/g" | tail -4
```
(If rejected as non-fast-forward, fetch that `main`, `git merge --allow-unrelated-histories -X ours FETCH_HEAD`, push again — the HF template seeds a README/.gitattributes.)

- [ ] **Step 6: Watch the build to a terminal stage**

```bash
python3 - <<'PY'
import time
from huggingface_hub import get_space_runtime
repo = "Executor-Tyrant-Framework/Condensate-dev"
for _ in range(90):
    s = get_space_runtime(repo).stage
    print(s, flush=True)
    if s in ("RUNNING","RUNTIME_ERROR","BUILD_ERROR"): break
    time.sleep(20)
PY
```
Expected: `RUNNING`. If `*_ERROR`, fetch logs:
`https://huggingface.co/api/spaces/<repo>/logs/{build,run}` with `Authorization: Bearer <token>` and fix.

- [ ] **Step 7: Validate the demo on staging**

Manual: open the staging Space, Run an example, confirm — heatmap renders, headline shows a savings %, "✓ lossless verified" is green, generated text appears, second run is fast (model cached on the bucket). Toggle `full` and confirm an exact number.

- [ ] **Step 8: Commit packaging**

```bash
cd ~/Condensate
git add requirements.txt README.md
git commit -m "build: gradio6 + wheel requirements; Space frontmatter"
```

---

## Task 6: Promote to live + fix auto-sync

**Files:**
- Modify: `.github/workflows/*.yml` (trigger branch)

**Interfaces:** Live Space `Executor-Tyrant-Framework/Condensate` serves the new demo; pushes to `master` auto-deploy.

- [ ] **Step 1: Fix the sync workflow trigger** (currently `main`; repo branch is `master`)

In `.github/workflows/<sync>.yml`:
```yaml
on:
  push:
    branches:
      - master
```

- [ ] **Step 2: Commit the workflow fix**

```bash
cd ~/Condensate
git add .github/workflows/
git commit -m "ci: auto-sync triggers on master (was main)"
```

- [ ] **Step 3: Promote — push the validated tree to the LIVE Space `main`**

```bash
cd ~/Condensate
TOKEN=$(cat ~/.cache/huggingface/token)
git push "https://Executor-Tyrant-Framework:$TOKEN@huggingface.co/spaces/Executor-Tyrant-Framework/Condensate" HEAD:main 2>&1 | sed -E "s/$TOKEN/***/g" | tail -4
```
(Confirm the live Space also has the bucket mounted at `/data` before first run.)

- [ ] **Step 4: Watch live build to RUNNING** (same watcher as Task 5 Step 6, repo `Condensate`).

- [ ] **Step 5: Push to GitHub origin** (after Josh's OK — pushing to default branch needs explicit authorization)

```bash
cd ~/Condensate && git push origin master
```

- [ ] **Step 6: Write the dev-log** (Vault canonical + repo, byte-identical)

Create `~/docs/dev-log/2026-06-24_condensate-demo-reimagining.md` summarizing what shipped, the wheel/abi3 approach, the staging→promote flow, and link `[[Condensate]]`, the spec, `[[laptop_home_project_condensate_lab]]`. Copy byte-identical to `~/Condensate/docs/superpowers/` if desired. `cd ~/docs && git add … && git commit`.

---

## Self-Review

**Spec coverage:** §3 experience → Task 4. §4 visuals → Task 3. §5 engine/sensor/lossless/modes → Tasks 1+2. §6 wheel/packaging/HF_HOME → Tasks 1+5. §7 staging→promote+auto-sync → Tasks 5+6. §8 testing → Tasks 1-3 tests. §9 shrapnel removal → Task 4 Step 3. All covered.

**Placeholder scan:** none — every code/test/command step carries real content.

**Type consistency:** `condense()` result keys (`regions/total_bytes/condensed_bytes/saved_bytes/saved_pct/lossless_ok/mode`) are produced in Task 2 and consumed verbatim in Task 3 (`viz`) and Task 4 (`app`). Rust `classify_tier`/`lz4_compress_len`/`lz4_verify_roundtrip` names match across Task 1 (def) and Task 2 (use). Region dict keys (`name/size_bytes/access_count/sample_bytes/full_bytes`) match between Task 2 tests, `engine.condense`, and Task 4 `_regions_from`.

**Open risks (carry into execution):**
- `torch_membrane.get_activation_map()` field name for access counts — code reads `forward_count` then `access_count`; verify against the live sensor and adjust the `.get` in `_regions_from` if neither matches.
- Materializing all weight bytes for `full_bytes` on a 7B is heavy; in `app.py` consider lazy `full_bytes` (only realize for COLD regions, or always sample and only realize full when `measurement=="full"`). Optimize in Task 4 if memory-bound.
