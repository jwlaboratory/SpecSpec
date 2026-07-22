#!/usr/bin/env python3
"""Charts for the 40-language router run.

Reads router/results40/router40_confusion.json and writes:

    router/results40/charts/router40_accuracy.png
    router/results40/charts/router40_top_confusions.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results40"
CHARTS = RESULTS / "charts"

INK = "#111111"
INK2 = "#52514e"
MUTED = "#85837d"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#e07800"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.facecolor": SURF,
})


def load() -> dict:
    return json.loads((RESULTS / "router40_confusion.json").read_text())


def style_axis(ax, grid_axis: str = "x") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, lw=0.85)
    ax.set_axisbelow(True)


def accuracy_chart(d: dict) -> None:
    classes = d["classes"]
    per = d["per_class"]
    rows = sorted(((c, per[c] * 100.0) for c in classes), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10.8, 11.8))
    y = list(range(len(rows)))
    vals = [v for _, v in rows]
    colors = [ORANGE if v < 50 else BLUE for v in vals]
    ax.barh(y, vals, color=colors, height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c for c, _ in rows], color=INK2)
    ax.set_xlim(0, 105)
    ax.set_xlabel("test accuracy (%)", color=INK2, fontsize=10)
    style_axis(ax)
    for yy, v in zip(y, vals):
        ax.text(min(v + 1.2, 96), yy, f"{v:.1f}%", va="center",
                ha="left", fontsize=8.5, color=INK2)
    fig.suptitle(
        "40-way language router accuracy",
        x=0.08,
        y=0.990,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.955,
        f"Qwen3-8B hidden-state MLP · val {d['val_acc']*100:.1f}% · "
        f"test {d['test_acc']*100:.1f}% · {d['test_n']} test prompts",
        color=INK2,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(CHARTS / "router40_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def top_confusions_chart(d: dict, top_k: int = 22) -> None:
    classes = d["classes"]
    conf = d["confusion"]
    rows = []
    for i, true in enumerate(classes):
        total = max(sum(conf[i]), 1)
        for j, pred in enumerate(classes):
            if i == j or conf[i][j] == 0:
                continue
            count = conf[i][j]
            rows.append((count, 100.0 * count / total, true, pred))
    rows = sorted(rows, reverse=True)[:top_k]
    rows = list(reversed(rows))

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    y = list(range(len(rows)))
    counts = [r[0] for r in rows]
    labels = [f"{true} -> {pred}" for _, _, true, pred in rows]
    ax.barh(y, counts, color=ORANGE, height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK2)
    ax.set_xlabel("misrouted test prompts", color=INK2, fontsize=10)
    style_axis(ax)
    for yy, (count, pct, _, _) in zip(y, rows):
        ax.text(count + 0.4, yy, f"{count} ({pct:.0f}%)", va="center",
                ha="left", fontsize=8.5, color=INK2)
    ax.set_title(
        "Largest 40-way router confusions",
        loc="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0,
        1.01,
        "Rows are true WildChat language labels; arrows show predicted router labels",
        transform=ax.transAxes,
        color=INK2,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "router40_top_confusions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    d = load()
    accuracy_chart(d)
    top_confusions_chart(d)
    print(f"charts -> {CHARTS}")


if __name__ == "__main__":
    main()
