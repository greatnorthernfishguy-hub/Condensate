# app.py — Condensate demo: one-flow, real engine, visual.
# HF cache MUST live on the persistent bucket so the 7B downloads once.
#
# ---- Changelog ----
# [2026-06-24] CC — Task 4: rewrite as one-flow ZeroGPU app on real engine
# What: Single run() flow: inference → sensor → engine.condense → viz → UI.
#       Memory-safe _regions_from bounds materialization to ~4 MB sample bytes
#       per region; full bytes only for COLD regions when measurement="full".
# Why: SDD task-4-brief.md — replace multi-step PoC with production-quality demo.
# How: Lazy CPU load, CUDA inside @spaces.GPU, model back to CPU before
#      region building, condensate_core.classify_tier inline to gate full_bytes.
# [2026-06-24] CC — Task 4 review fixes (Finding 1 + Finding 2)
# What: F1: _access_count_for() helper — unobserved modules treated HOT (conservative),
#       with diagnostic stderr line. F2: pad_token guard after tokenizer load.
# Why: F1: savings % was overstated when model module names didn't join sensor names
#      (unobserved → default 0 → COLD → counted as condensable — wrong).
#      F2: generate() can fail with pad_token_id=None on some tokenizer configs.
# How: F1: Pure helper with max_access fallback; count+MB of unobserved emitted to
#      stderr before _regions_from returns. F2: Standard pad_token = eos_token guard.
# -------------------

import os
import sys
os.environ.setdefault("HF_HOME", "/data/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf-cache/hub")

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import engine
import viz
import condensate_core
from examples import EXAMPLES
from torch_membrane import TorchMembrane
from app_helpers import _access_count_for

MODEL_NAME = os.environ.get("CONDENSATE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_M = {"model": None, "tok": None, "sensor": None}

# 4 MB cap for sample path: fp16 = 2 bytes/elem → 4 MB / 2 = 2 M elements.
_SAMPLE_ELEMS = (4 << 20) // 2


def _ensure_loaded():
    """Lazy CPU load (no CUDA at import — ZeroGPU rule). Cached on the bucket."""
    if _M["model"] is None:
        _M["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _M["tok"].pad_token is None:
            _M["tok"].pad_token = _M["tok"].eos_token
        _M["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16)
        _M["model"].eval()
        _M["sensor"] = TorchMembrane(_M["model"])


def _regions_from(model, sensor, measurement):
    """Map sensor access stats + weight sizes → engine region dicts.

    Memory safety: sample_bytes is always bounded to ~4 MB of fp16 data
    (slice before .tobytes(), never materialise the whole tensor).
    full_bytes is only materialised for COLD regions when measurement=="full".
    Full mode on a large model is memory-intensive — acceptable v1 trade-off;
    streaming full_bytes is a future optimisation.
    """
    access = {a["name"]: a.get("forward_count", a.get("access_count", 0))
              for a in sensor.get_activation_map()}
    max_access = max(access.values(), default=0)

    regions = []
    unobserved_count = 0
    unobserved_bytes = 0

    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        mod = name[: -len(".weight")]
        nbytes = p.numel() * p.element_size()

        # Conservative: unobserved modules are treated as fully-hot so savings
        # are never overstated.  Observed modules use their real access count.
        acc = _access_count_for(mod, access, max_access)
        if mod not in access:
            unobserved_count += 1
            unobserved_bytes += nbytes

        # Bounded sample: slice first, then convert — never the whole tensor.
        flat = p.detach().flatten()
        sample = flat[:_SAMPLE_ELEMS].to(torch.float16).cpu().numpy().tobytes()

        # Determine tier inline; only pay for full materialisation on COLD.
        tier = condensate_core.classify_tier(acc, int(max_access))
        if measurement == "full" and tier == "COLD":
            full = p.detach().to(torch.float16).cpu().numpy().tobytes()
        else:
            full = None

        regions.append({
            "name": mod,
            "size_bytes": nbytes,
            "access_count": acc,
            "sample_bytes": sample,
            "full_bytes": full,
        })

    total_weight_params = len(regions)
    if unobserved_count:
        unobserved_mb = unobserved_bytes / (1024 * 1024)
        print(
            f"[regions] {unobserved_count}/{total_weight_params} weight modules "
            f"unobserved by sensor ({unobserved_mb:.1f} MB) -> treated HOT",
            file=sys.stderr,
        )
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

    # Weights back on host for honest RAM measurement.
    model.to("cpu")
    regions = _regions_from(model, sensor, measurement)
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
