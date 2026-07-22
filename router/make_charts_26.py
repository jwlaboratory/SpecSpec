#!/usr/bin/env python3
"""Charts for the clean 26-language router run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results26"
CHARTS = RESULTS / "charts"

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


def main() -> None:
    d = json.loads((RESULTS / "router26_confusion.json").read_text())
    rows = sorted(((c, d["per_class"][c] * 100) for c in d["classes"]),
                  key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10.2, 8.2), dpi=180)
    y = list(range(len(rows)))
    vals = [v for _, v in rows]
    colors = [ORANGE if v < 60 else BLUE for v in vals]
    ax.barh(y, vals, color=colors, height=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c for c, _ in rows], color=INK2)
    ax.set_xlim(0, 105)
    ax.set_xlabel("test accuracy (%)", color=INK2)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    for yy, v in zip(y, vals):
        ax.text(min(v + 1.2, 96), yy, f"{v:.1f}%", va="center",
                fontsize=8.5, color=INK2)
    fig.suptitle("26-way clean language router accuracy",
                 x=0.08, y=0.995, ha="left", color=INK,
                 fontsize=14, fontweight="bold")
    fig.text(0.08, 0.958,
             f"1,000-train-prompt languages · val {d['val_acc']*100:.1f}% · "
             f"test {d['test_acc']*100:.1f}% · {d['test_n']} test prompts",
             ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / "router26_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"charts -> {CHARTS}")


if __name__ == "__main__":
    main()
