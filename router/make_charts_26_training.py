#!/usr/bin/env python3
"""Training curve (loss + val accuracy) for the clean 26-language router run."""
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
    hist = d["history"]
    epochs = [h["epoch"] for h in hist]
    loss = [h["train_loss"] for h in hist]

    fig, ax_loss = plt.subplots(figsize=(9.4, 5.4), dpi=180)

    # train loss (orange)
    ax_loss.plot(epochs, loss, color=ORANGE, lw=2.2, marker="o", ms=4,
                 zorder=3, label="train loss")
    ax_loss.set_xlabel("epoch", color=INK2)
    ax_loss.set_ylabel("train loss", color=ORANGE)
    ax_loss.tick_params(axis="y", colors=ORANGE, length=0)
    ax_loss.set_ylim(0, max(loss) * 1.08)

    ax_loss.grid(axis="both", color=GRID, zorder=0)
    ax_loss.set_axisbelow(True)
    ax_loss.tick_params(axis="x", colors=MUTED, length=0)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax_loss.spines[spine].set_color(AXIS)

    fig.suptitle("26-way language router — training curve",
                 x=0.09, y=0.985, ha="left", color=INK,
                 fontsize=14, fontweight="bold")
    fig.text(0.09, 0.93,
             f"MLP 20480->512->26 · early-stop on val acc · "
             f"test {d['test_acc']*100:.1f}% · {d['train_n']:,} train prompts",
             ha="left", color=INK2, fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    CHARTS.mkdir(parents=True, exist_ok=True)
    out = CHARTS / "router26_training.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
