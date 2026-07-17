#!/usr/bin/env python3
"""Charts for the adapter router: confusion-matrix heatmap + per-class accuracy.

    python router/make_charts.py          # reads results/router_confusion.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# sequential single-hue ramp (blue) for magnitude — light -> dark
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def main():
    d = json.loads((RESULTS / "router_confusion.json").read_text())
    classes, conf = d["classes"], d["confusion"]
    n = len(classes)
    row_tot = [max(sum(r), 1) for r in conf]
    frac = [[conf[i][j] / row_tot[i] for j in range(n)] for i in range(n)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9.5, color=INK2)
    ax.set_yticklabels(classes, fontsize=9.5, color=INK2)
    ax.set_xlabel("predicted", color=MUTED, fontsize=9.5)
    ax.set_ylabel("true", color=MUTED, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = conf[i][j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        fontweight="bold",
                        color="#ffffff" if frac[i][j] > 0.55 else INK)
    ax.set_title(f"Confusion matrix  ·  test accuracy {d['test_acc']*100:.1f}%",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    per = d["per_class"]
    ys = range(len(classes))
    vals = [per[c] * 100 for c in classes]
    ax2.barh(list(ys), vals, height=0.62, color="#2a78d6", zorder=3)
    ax2.set_yticks(list(ys)); ax2.set_yticklabels(classes, fontsize=10, color=INK2)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(AXIS)
    ax2.tick_params(colors=MUTED, labelsize=9, length=0)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for y, v in zip(ys, vals):
        ax2.text(min(v + 1.5, 96), y, f"{v:.1f}%", va="center", fontsize=9.5,
                 color=INK, fontweight="bold")
    ax2.set_title("Per-class accuracy", color=INK2, fontsize=11, loc="left", pad=10)

    fig.suptitle("Adapter router  ·  which LoRA should this request use?",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.02, "MLP on mean-pooled Qwen3-8B hidden states (layers 1/9/17/25/33) — "
             "the same states the DFlash drafter already consumes. 'other' routes to the "
             "base drafter.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "router.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {out}/router.png")


if __name__ == "__main__":
    main()
