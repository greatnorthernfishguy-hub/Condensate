# viz.py — Plotly renderers for the condense() result. No engine logic here.
# ---- Changelog ----
# [2026-06-24] CC — Task 3: implement temperature_heatmap, ram_bars, headline
# What: Three public renderers consuming the condense() result dict.
# Why: SDD task-3-brief.md — visualization layer for the Condensate demo Space.
# How: Pure rendering; no tier/savings logic. Color semantics: COLD=deep-blue,
#      WARM=amber, HOT=bright orange-red. Heatmap maps layers×proj columns.
# -------------------
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
    """Horizontal bar chart: full model vs condensed RAM in GB."""
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
    """HTML summary string: savings %, GB delta, lossless status, mode."""
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
