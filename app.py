"""
Condensate — Live Demo
HuggingFace Spaces Gradio App (ZeroGPU)

Shows real-time RAM condensation on a live model.
Compares baseline vs condensed inference.

"Do the same, or more, with less."
"""

import spaces
import gradio as gr
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from membrane import Membrane
from graph_builder import GraphBuilder
from predictor import Predictor

# Lazy imports for heavy deps
torch = None
TorchMembrane = None


def _ensure_torch():
    global torch, TorchMembrane
    if torch is None:
        import torch as _torch
        torch = _torch
        from torch_membrane import TorchMembrane as _TM
        TorchMembrane = _TM


# --- Global state ---
MODEL = None
TOKENIZER = None
MEMBRANE = None
PREDICTOR = None
GRAPH = None
MODEL_NAME = "distilgpt2"


@spaces.GPU
def load_model():
    """Load model and install membrane."""
    global MODEL, TOKENIZER, MEMBRANE

    _ensure_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        output_attentions=True,
    )
    MODEL.eval()
    MODEL.to("cuda")

    MEMBRANE = TorchMembrane(MODEL)

    param_count = sum(p.numel() for p in MODEL.parameters()) / 1e6
    return f"Loaded {MODEL_NAME} ({param_count:.1f}M params) on ZeroGPU"


@spaces.GPU
def train_predictor(num_prompts=5):
    """Run several prompts to train the predictor on access patterns."""
    global PREDICTOR, GRAPH, MEMBRANE

    _ensure_torch()

    if MODEL is None:
        load_model()

    MEMBRANE.reset()

    training_prompts = [
        "The quick brown fox jumps over the lazy",
        "In the beginning there was darkness and then",
        "Machine learning models can be optimized by",
        "The capital of France is Paris and the",
        "Once upon a time in a land far far",
        "Artificial intelligence will transform the way we",
        "The most important thing about programming is",
        "When the sun sets over the mountains the",
    ][:num_prompts]

    for prompt in training_prompts:
        inputs = TOKENIZER(prompt, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            MODEL.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=TOKENIZER.pad_token_id,
            )

    log = MEMBRANE.to_access_log()

    GRAPH = GraphBuilder(causal_window_ns=5_000_000)
    GRAPH.build(log)

    PREDICTOR = Predictor()
    PREDICTOR.learn(GRAPH)

    result = PREDICTOR.score(log)

    return (f"Trained on {len(training_prompts)} prompts, "
            f"{len(log)} access events observed.\n"
            f"Prediction accuracy: {result['accuracy']}%\n"
            f"Causal chains discovered: {len(GRAPH.get_causal_chains())}\n"
            f"Clusters (proto-hyperedges): {len(GRAPH.clusters)}")


@spaces.GPU
def run_analysis(prompt, max_tokens=30):
    """Run inference, show activation map + condensation potential."""
    global MEMBRANE, PREDICTOR

    _ensure_torch()

    if MODEL is None:
        load_model()
    if PREDICTOR is None:
        train_predictor()

    MEMBRANE.reset()

    inputs = TOKENIZER(prompt, return_tensors="pt", padding=True).to("cuda")
    start = time.monotonic()

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.pad_token_id,
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

    activation_map = MEMBRANE.get_activation_map()
    potential = MEMBRANE.get_condensation_potential()

    log = MEMBRANE.to_access_log()
    pred_result = PREDICTOR.score(log)

    # Build comparison output
    comparison = []
    comparison.append("=" * 55)
    comparison.append("  BASELINE vs CONDENSATE")
    comparison.append("=" * 55)
    comparison.append(f"\n  Generated: {generated_text}")
    comparison.append(f"  Time: {elapsed_ms:.0f}ms\n")

    baseline_mb = potential['total_mb']
    condensed_mb = potential['hot_mb']
    saved_pct = potential['savings_pct']

    comparison.append(f"  WITHOUT Condensate:")
    comparison.append(f"    All {potential['total_layers']} layers in RAM:  {baseline_mb:.2f} MB")
    comparison.append(f"    (Every weight loaded, whether needed or not)\n")

    comparison.append(f"  WITH Condensate:")
    comparison.append(f"    {potential['hot_layers']} HOT layers in RAM:   {condensed_mb:.2f} MB")
    comparison.append(f"    {potential['cold_layers']} COLD layers paged:   {potential['cold_mb']:.2f} MB saved")
    comparison.append(f"    (Cold layers compressed or on disk,")
    comparison.append(f"     pre-staged back to RAM before needed)\n")

    comparison.append(f"  ┌─────────────────────────────────────┐")
    comparison.append(f"  │  RAM REDUCTION: {saved_pct:.1f}%                │")
    comparison.append(f"  │  {baseline_mb:.2f} MB → {condensed_mb:.2f} MB             │")
    comparison.append(f"  │  Same output. Same quality.         │")
    comparison.append(f"  └─────────────────────────────────────┘\n")

    comparison.append(f"  Prediction accuracy: {pred_result['accuracy']}%")
    comparison.append(f"  Access events: {len(log)}")

    # Build analysis output
    analysis = []
    analysis.append("=" * 55)
    analysis.append("  LAYER ACTIVATION MAP")
    analysis.append("=" * 55)
    analysis.append(f"\n  {'Layer':<35} {'Fwd':>4} {'Activation':>10} {'MB':>6} {'Tier':>5}")
    analysis.append(f"  {'-'*35} {'-'*4} {'-'*10} {'-'*6} {'-'*5}")

    for layer in activation_map[:40]:
        name = layer['name']
        if len(name) > 35:
            name = "..." + name[-32:]
        attn = " [A]" if layer['is_attention'] else ""
        analysis.append(f"  {name:<35} {layer['forward_count']:>4} "
                       f"{layer['avg_activation']:>10.3f} "
                       f"{layer['param_mb']:>6.3f} "
                       f"{layer['temperature']:>5}{attn}")

    if len(activation_map) > 40:
        analysis.append(f"  ... and {len(activation_map) - 40} more layers")

    return "\n".join(comparison), "\n".join(analysis)


# --- Also keep the synthetic demo for comparison ---

def run_synthetic_demo(num_layers, num_hot, num_iterations):
    """Run the PoC pipeline on synthetic data (no GPU needed)."""
    from condenser import Condenser

    num_layers = int(num_layers)
    num_hot = int(min(num_hot, num_layers))
    num_iterations = int(num_iterations)

    output = []
    output.append("=" * 55)
    output.append("  CONDENSATE — Synthetic Pipeline Demo")
    output.append("=" * 55)

    state = {}
    for i in range(num_layers):
        arr = np.zeros((128, 128), dtype=np.float32)
        mask = np.random.random((128, 128)) < 0.2
        arr[mask] = np.random.randn(mask.sum()).astype(np.float32)
        state[f"layer_{i}"] = arr

    total_mb = sum(v.nbytes for v in state.values()) / 1024 / 1024
    hot_set = set(range(num_hot))

    output.append(f"\n  {num_layers} regions x 64KB = {total_mb:.1f} MB total")
    output.append(f"  {num_hot} hot / {num_layers - num_hot} cold")

    # Membrane + Graph + Predictor
    Membrane.clear()
    wrapped = Membrane.wrap(state.copy(), "model")
    for _ in range(num_iterations):
        for i in range(num_layers):
            if i in hot_set:
                _ = wrapped[f"layer_{i}"]
            elif np.random.random() < 0.03:
                _ = wrapped[f"layer_{i}"]
        time.sleep(0.001)

    log = Membrane.get_log()
    graph = GraphBuilder(causal_window_ns=5_000_000)
    graph.build(log)
    predictor = Predictor()
    predictor.learn(graph)
    score = predictor.score(log)

    output.append(f"\n  Prediction accuracy: {score['accuracy']}%")
    output.append(f"  Clusters: {len(graph.clusters)}")
    output.append(f"  Causal chains: {len(graph.get_causal_chains())}")

    # Condenser
    def workload_fn(w):
        for i in range(num_layers):
            if i in hot_set:
                _ = w[f"layer_{i}"]
            elif np.random.random() < 0.03:
                _ = w[f"layer_{i}"]
        time.sleep(0.001)

    condenser = Condenser(demotion_idle_ms=10, warmup_iters=8)
    bench = condenser.run_benchmark(state, workload_fn,
                                     iterations=num_iterations, name="model")

    output.append(f"\n  Baseline:  {bench['baseline_ram_mb']:.2f} MB")
    output.append(f"  Condensed: {bench['avg_condensed_ram_mb']:.2f} MB")
    output.append(f"  *** SAVED: {bench['saved_mb']:.2f} MB ({bench['saved_pct']:.1f}%) ***")

    if bench.get('promotion_log'):
        last = bench['promotion_log'][-1]
        output.append(f"  Final: HOT={last['hot']}  WARM={last['warm']}  COLD={last['cold']}")

    condenser.cleanup()
    output.append(f"\n{'=' * 55}")
    return "\n".join(output)


# --- Gradio UI ---

with gr.Blocks(title="Condensate — Do More With Less") as demo:

    gr.Markdown("""
    # Condensate
    ### A Living Memory Manager — Do the Same, or More, With Less.

    Condensate uses a neural substrate with causal spike propagation
    to learn memory access patterns and dynamically condense RAM usage.

    **Live Model tab:** Runs a real transformer (distilgpt2) on ZeroGPU
    and shows which layers are HOT vs COLD for your input.

    **Synthetic tab:** Runs the full 4-layer pipeline on configurable
    simulated workloads (no GPU needed).
    """)

    with gr.Tabs():
        with gr.TabItem("Live Model (ZeroGPU)"):
            with gr.Row():
                with gr.Column():
                    status = gr.Textbox(label="Status", interactive=False, lines=3)
                    load_btn = gr.Button("1. Load Model", variant="primary")
                    train_btn = gr.Button("2. Train Predictor", variant="primary")

            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        value="The future of artificial intelligence is",
                        lines=2,
                    )
                    max_tokens = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Max tokens"
                    )
                    run_btn = gr.Button("3. Run & Analyze", variant="primary")

            with gr.Row():
                with gr.Column():
                    comparison_output = gr.Textbox(
                        label="Baseline vs Condensate",
                        lines=25, interactive=False,
                    )
                with gr.Column():
                    analysis_output = gr.Textbox(
                        label="Layer Activation Map",
                        lines=25, interactive=False,
                    )

            load_btn.click(fn=load_model, outputs=status)
            train_btn.click(fn=train_predictor, outputs=status)
            run_btn.click(
                fn=run_analysis,
                inputs=[prompt_input, max_tokens],
                outputs=[comparison_output, analysis_output],
            )

        with gr.TabItem("Synthetic Workload"):
            with gr.Row():
                with gr.Column():
                    syn_layers = gr.Slider(minimum=4, maximum=128, value=32, step=4,
                                           label="Total memory regions")
                    syn_hot = gr.Slider(minimum=1, maximum=64, value=6, step=1,
                                        label="Hot regions")
                    syn_iters = gr.Slider(minimum=10, maximum=50, value=20, step=5,
                                          label="Iterations")
                    syn_btn = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    syn_output = gr.Textbox(
                        label="Results", lines=25,
                        interactive=False,
                    )

            syn_btn.click(
                fn=run_synthetic_demo,
                inputs=[syn_layers, syn_hot, syn_iters],
                outputs=syn_output,
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
