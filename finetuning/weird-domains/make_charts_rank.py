#!/usr/bin/env python3
"""Rank ladder for the weird domains: base vs r4 vs r16 vs r64 own-LoRA (DFlash).

    python finetuning/weird-domains/make_charts_rank.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOMAINS = ["translation", "roleplay", "poetry"]
SERIES = [("base", "base", "#898781"),
          ("own_r4", "own r4 (α8)", "#f5c9ae"),
          ("own", "own r16 (α32)", "#f0a175"),
          ("own_r64", "own r64 (α128)", "#eb6834")]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def pooled(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    tp = sum(r["proposed_draft_tokens"] for r in recs)
    return 100 * sum(r["accepted_draft_tokens"] for r in recs) / tp if tp else 0.0


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    xs = range(len(DOMAINS))
    w = 0.8 / len(SERIES)
    vals = {}
    for d in DOMAINS:
        for key, _, _ in SERIES:
            f = RESULTS / f"dflash_{d}_{key}.jsonl"
            vals[(d, key)] = pooled(f) if f.exists() else 0.0
    top = max(vals.values())
    for i, (key, label, color) in enumerate(SERIES):
        offs = [x + (i - 1.5) * w for x in xs]
        hs = [vals[(d, key)] for d in DOMAINS]
        ax.bar(offs, hs, width=w * 0.86, color=color, zorder=3, label=label)
        for x, h in zip(offs, hs):
            ax.text(x, h + top * 0.015, f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d.capitalize() for d in DOMAINS])
    ax.set_ylim(0, top * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("acceptance rate (%)", color=INK2, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=4)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Rank ladder on heterogeneous domains: even rank 4 captures most of the gain",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.text(0.02, -0.03, "DFlash drafter, same data/training per rank. Gains saturate "
             "by r16 here (base ~7-9%, moderate headroom) — unlike the weak-base languages "
             "(3-5%), where r64 kept paying. Rank need scales with the size of the deficit.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    out = RESULTS / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "rank_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {out}/rank_ladder.png")


if __name__ == "__main__":
    main()
