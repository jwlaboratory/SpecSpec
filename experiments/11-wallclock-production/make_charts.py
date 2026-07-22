#!/usr/bin/env python3
"""Make summary charts for experiment 11."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "full" / "results"
OUT = RESULTS / "charts"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "target_only": "#898781",
    "base": "#52514e",
    "merged_combined": "#d97706",
    "merged_own": "#2a78d6",
    "hotswap_own": "#c23b21",
}
LABELS = {
    "target_only": "target only",
    "base": "base DFlash",
    "merged_combined": "merged combined",
    "merged_own": "N merged own",
    "hotswap_own": "hot-swap own",
}


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    modes = [m for m in ["target_only", "base", "merged_combined", "merged_own", "hotswap_own"]
             if m in summary["modes"]]
    vals = [summary["modes"][m].get("speedup_vs_target", 1.0) for m in modes]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=170)
    bars = ax.bar([LABELS[m] for m in modes], vals,
                  color=[COLORS[m] for m in modes], zorder=3)
    ax.axhline(1.0, color="#c3c2b7", lw=1)
    ax.grid(axis="y", color="#e1e0d9", zorder=0)
    ax.set_ylabel("speedup vs target-only")
    ax.set_title("Production wall-clock speedup by LoRA serving mode",
                 loc="left", fontsize=12, weight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "speedup_modes.png")
    plt.close(fig)

    # Per-language specialist quality: base vs merged own vs merged combined.
    rows = []
    for lang, row in summary["per_lang"].items():
        if all(m in row for m in ("base", "merged_combined", "merged_own")):
            rows.append((lang, row))
    rows.sort(key=lambda x: x[1]["base"]["mean_accept_length"])
    langs = [r[0] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8.4, 9.0), dpi=170)
    for mode, marker, dx in [
        ("base", "o", 0),
        ("merged_combined", "o", 0),
        ("merged_own", "o", 0),
    ]:
        xs = [r[1][mode]["mean_accept_length"] for r in rows]
        ax.scatter(xs, list(y), s=34, marker=marker, color=COLORS[mode],
                   label=LABELS[mode], zorder=3)
    for i, (_, row) in enumerate(rows):
        xs = [row[m]["mean_accept_length"] for m in ("base", "merged_combined", "merged_own")]
        ax.plot([min(xs), max(xs)], [i, i], color="#e1e0d9", lw=1.2, zorder=1)
    ax.set_yticks(list(y), langs)
    ax.set_xlabel("mean accept length")
    ax.set_title("Accepted length by language", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", color="#e1e0d9", zorder=0)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "mean_accept_length_by_language.png")
    plt.close(fig)
    print(f"charts -> {OUT}")


if __name__ == "__main__":
    main()

