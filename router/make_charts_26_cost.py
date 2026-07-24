#!/usr/bin/env python3
"""Cost charts for the 26-language router MLP.

Measured on NVIDIA H200 (Modal), scratchpad/bench_router26.py loading the real
router26_mlp.pt. Two charts, house style:
  1. router26_cost_bars.png  — per-request wall-clock vs the Qwen3-8B prefill
  2. router26_cost_batch.png — router latency/request vs batch size (fp32/bf16)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "results26" / "charts"

SURF = "#fcfcfb"
INK = "#111111"
INK2 = "#52514e"
MUTED = "#85837d"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#e07800"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.facecolor": SURF,
})

# ---- measured numbers (H200) ------------------------------------------------
TARGET_PREFILL_US = 46608.77          # Qwen3-8B, 256-tok prompt, bf16
ROUTER_B1_US = 48.47                  # fp32, batch 1
ROUTER_B64_US = 1.727                 # fp32, per request at batch 64

# per-request latency vs batch (fp32 total/B, bf16 total/B)
BATCH = [1, 4, 16, 64, 256]
FP32_PER_REQ = [48.47, 24.94, 4.85, 1.727, 1.157]
BF16_PER_REQ = [63.31, 13.22, 3.231, 0.817, 0.201]


def _clean(ax):
    ax.grid(axis="both", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def bars():
    labels = ["Qwen3-8B target\nprefill (256 tok)",
              "Router MLP\n(batch 1)",
              "Router MLP\n(batched, B=64)"]
    vals = [TARGET_PREFILL_US, ROUTER_B1_US, ROUTER_B64_US]
    colors = [MUTED, BLUE, ORANGE]
    notes = ["", f"{ROUTER_B1_US/TARGET_PREFILL_US*100:.2f}% of prefill",
             f"{ROUTER_B64_US/TARGET_PREFILL_US*100:.3f}% of prefill"]

    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=180)
    y = range(len(labels))
    ax.barh(y, vals, color=colors, zorder=3, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK2, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock per request (µs, log scale)", color=INK2)
    _clean(ax)

    def fmt(v):
        return f"{v/1000:.1f} ms" if v >= 1000 else f"{v:.1f} µs"
    for i, (v, n) in enumerate(zip(vals, notes)):
        lab = fmt(v) + (f"   ·   {n}" if n else "")
        ax.text(v * 1.25, i, lab, va="center", color=INK, fontsize=10,
                fontweight="bold")
    ax.set_xlim(1, TARGET_PREFILL_US * 6)

    fig.suptitle("Cost of the router MLP",
                 x=0.02, y=0.98, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.02, 0.905,
             "Wall-clock time on H200 using hidden features DFlash already "
             "extracts",
             ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    CHARTS.mkdir(parents=True, exist_ok=True)
    out = CHARTS / "router26_cost_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"chart -> {out}")


def batch():
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    ax.plot(BATCH, FP32_PER_REQ, color=BLUE, lw=2.2, marker="o", ms=5,
            zorder=3, label="fp32 (serving path)")
    ax.plot(BATCH, BF16_PER_REQ, color=ORANGE, lw=2.2, marker="s", ms=5,
            zorder=3, label="bf16")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(BATCH)
    ax.set_xticklabels([str(b) for b in BATCH])
    ax.set_xlabel("batch size (requests routed together)", color=INK2)
    ax.set_ylabel("latency per request (µs)", color=INK2)
    _clean(ax)
    ax.legend(frameon=False, loc="upper right",
              labelcolor=INK2, fontsize=10)

    fig.suptitle("Router latency per request amortizes to ~1 µs",
                 x=0.02, y=0.98, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.02, 0.905,
             "20480->512->26 MLP on H200 · batch-1 is launch-bound (~48 µs), not "
             "compute-bound (21 MFLOP)",
             ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out = CHARTS / "router26_cost_batch.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"chart -> {out}")


if __name__ == "__main__":
    bars()
    batch()
