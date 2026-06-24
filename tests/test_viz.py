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
