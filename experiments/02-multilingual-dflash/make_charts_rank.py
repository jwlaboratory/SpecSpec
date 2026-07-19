#!/usr/bin/env python3
"""Rank-scaling chart: does a deeper LoRA help the DFlash drafter more?

    python experiments/02-multilingual-dflash/make_charts_rank.py

Reads results/{lang}_{own,combined}[_r64].jsonl + base and writes
results/charts/rank_scaling.png — base vs rank-16 vs rank-64 own-LoRA per language.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LANGS = ["polish", "korean", "italian", "japanese", "german"]
SERIES = [("base", "base", "#898781"),
          ("own", "own-LoRA r16 (α32)", "#f0a175"),
          ("own_r64", "own-LoRA r64 (α128)", "#eb6834")]

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
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    xs = range(len(LANGS))
    w = 0.8 / len(SERIES)
    vals = {}
    for lang in LANGS:
        for key, _, _ in SERIES:
            f = RESULTS / f"{lang}_{key}.jsonl"
            vals[(lang, key)] = pooled(f) if f.exists() else 0.0
    top = max(vals.values())
    for i, (key, label, color) in enumerate(SERIES):
        offs = [x + (i - 1) * w for x in xs]
        hs = [vals[(lang, key)] for lang in LANGS]
        ax.bar(offs, hs, width=w * 0.86, color=color, zorder=3, label=label)
        for x, h in zip(offs, hs):
            ax.text(x, h + top * 0.02, f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    ax.set_ylim(0, top * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("acceptance rate (%)", color=INK2, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Deeper rank helps: DFlash own-LoRA acceptance, rank 16 vs 64",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.text(0.02, -0.03, "Same data (800 target-generated prompts/language), same "
             "training; only rank/alpha differ. r64 > r16 on 5/5 languages; the "
             "biggest extra gains land where base is weakest (Polish/Korean).",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    out = RESULTS / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "rank_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] -> {out}/rank_scaling.png")


if __name__ == "__main__":
    main()
