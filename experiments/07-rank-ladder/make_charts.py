#!/usr/bin/env python3
"""Rank-ladder charts — adapter capacity across both DFlash domain sets.

    python experiments/07-rank-ladder/make_charts.py

Rank-variant jsonls (r4/r64) live HERE in results/; the base and r16 runs they
ladder against stay with their source experiments (../02-multilingual-dflash and
../03-weird-domains — the rank runs were executed by those pipelines with a
--rank flag). Writes results/charts/:
    rank_scaling.png  5 languages: base vs own-r16 vs own-r64
    rank_ladder.png   3 weird domains: base vs own-r4 vs own-r16 vs own-r64
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHARTS = RESULTS / "charts"
ML_RESULTS = HERE.parent / "02-multilingual-dflash" / "results"
WD_RESULTS = HERE.parent / "03-weird-domains" / "results"

LANGS = ["polish", "korean", "italian", "japanese", "german"]
DOMAINS = ["translation", "roleplay", "poetry"]

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


def bar_ladder(ax, groups, series, vals):
    xs = range(len(groups))
    w = 0.8 / len(series)
    for i, (key, label, color) in enumerate(series):
        ax.bar([x + i * w for x in xs], [vals[(g, key)] for g in groups],
               width=w * 0.92, label=label, color=color, zorder=3)
    ax.set_xticks([x + 0.4 - w / 2 for x in xs])
    ax.set_xticklabels(groups, color=INK)
    ax.set_ylabel("pooled acceptance rate (%)", color=INK2)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, ncol=len(series), loc="upper left", fontsize=9)


def chart(groups, series, resolve, title, out):
    vals = {}
    for g in groups:
        for key, _, _ in series:
            f = resolve(g, key)
            vals[(g, key)] = pooled(f) if f.exists() else 0.0
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    bar_ladder(ax, groups, series, vals)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    # multilingual: r16/base live in exp 02, r64 here
    def ml(lang, key):
        if key.endswith("_r64"):
            return RESULTS / f"{lang}_{key}.jsonl"
        return ML_RESULTS / f"{lang}_{key}.jsonl"

    chart(LANGS,
          [("base", "base", "#898781"),
           ("own", "own-LoRA r16 (α32)", "#f0a175"),
           ("own_r64", "own-LoRA r64 (α128)", "#eb6834")],
          ml,
          "DFlash multilingual — rank scaling (base vs r16 vs r64 own-LoRA)",
          CHARTS / "rank_scaling.png")

    # weird domains: r16/base live in exp 03, r4/r64 here
    def wd(domain, key):
        if key.endswith(("_r4", "_r64")):
            return RESULTS / f"dflash_{domain}_{key}.jsonl"
        return WD_RESULTS / f"dflash_{domain}_{key}.jsonl"

    chart(DOMAINS,
          [("base", "base", "#898781"),
           ("own_r4", "own r4 (α8)", "#f5c9ae"),
           ("own", "own r16 (α32)", "#f0a175"),
           ("own_r64", "own r64 (α128)", "#eb6834")],
          wd,
          "DFlash weird domains — full rank ladder (base vs r4 vs r16 vs r64)",
          CHARTS / "rank_ladder.png")

    print(f"[ok] -> {CHARTS}/rank_scaling.png, rank_ladder.png")


if __name__ == "__main__":
    main()
