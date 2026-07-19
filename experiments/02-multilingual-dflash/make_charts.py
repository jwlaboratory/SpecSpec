#!/usr/bin/env python3
"""Charts for the multilingual LoRA specialization experiment.

    python finetuning/multilingual/make_charts.py

Reads results/<lang>_<variant>.jsonl (variant = base|own|combined) and writes:

    results/charts/matrix.png       5 languages x 3 variants — acceptance + mean len
    results/charts/delta.png        own & combined LoRA gain over base, per language

base = untouched pretrained drafter (gray) · own = LoRA trained on that language
only (orange) · combined = one LoRA trained on all five languages (blue).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LANGS = ["polish", "korean", "italian", "japanese", "german"]
VARIANTS = ["base", "own", "combined"]
LABEL = {"base": "base", "own": "own-language LoRA", "combined": "combined LoRA"}
COLOR = {"base": "#898781", "own": "#eb6834", "combined": "#2a78d6"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def load():
    rows = {}
    for lang in LANGS:
        for v in VARIANTS:
            f = RESULTS / f"{lang}_{v}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            if not recs:
                continue
            n = len(recs)
            tp = sum(r["proposed_draft_tokens"] for r in recs)
            rows[(lang, v)] = {
                "accept": 100 * sum(r["accepted_draft_tokens"] for r in recs) / tp if tp else 0.0,
                "mean_len": sum(r["mean_accept_length"] for r in recs) / n,
                "speedup": sum(r["speedup"] for r in recs) / n,
                "n": n,
            }
    return rows


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _grouped(ax, rows, key, fmt, title):
    xs = range(len(LANGS))
    w = 0.8 / len(VARIANTS)
    top = max(d[key] for d in rows.values())
    for i, v in enumerate(VARIANTS):
        offs = [x + (i - 1) * w for x in xs]
        hs = [rows.get((lang, v), {}).get(key, 0.0) for lang in LANGS]
        ax.bar(offs, hs, width=w * 0.86, color=COLOR[v], zorder=3,
               label=LABEL[v])
        for x, h in zip(offs, hs):
            if h:
                ax.text(x, h + top * 0.015, fmt.format(h), ha="center", va="bottom",
                        fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    ax.set_ylim(0, top * 1.18)
    _style(ax)
    ax.set_title(title, color=INK2, fontsize=11, loc="left", pad=10)


def matrix_fig(rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2))
    _grouped(axes[0], rows, "accept", "{:.1f}", "Acceptance rate  (%)")
    _grouped(axes[1], rows, "mean_len", "{:.2f}", "Mean accept length  (tokens / target pass)")
    leg = axes[0].legend(frameon=False, fontsize=10.5, loc="upper right", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    n = next(iter(rows.values()))["n"]
    fig.suptitle("Multilingual specialization  ·  base vs own-language LoRA vs combined LoRA",
                 fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.015,
             f"Qwen3-8B target · z-lab DFlash drafter · temperature 0 (lossless) · per-language "
             f"held-out test (n={n}). own = rank-16 LoRA on that language's 800 prompts; "
             f"combined = one LoRA on all five (4000).",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def delta_fig(rows, out):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    xs = range(len(LANGS))
    w = 0.36
    top, bot = 0.0, 0.0
    for i, v in enumerate(("own", "combined")):
        offs = [x + (i - 0.5) * w for x in xs]
        ds = [rows.get((lang, v), {}).get("accept", 0.0)
              - rows.get((lang, "base"), {}).get("accept", 0.0) for lang in LANGS]
        top = max(top, max(ds)); bot = min(bot, min(ds))
        ax.bar(offs, ds, width=w * 0.88, color=COLOR[v], zorder=3, label=LABEL[v])
        for x, d in zip(offs, ds):
            ax.text(x, d + (0.08 if d >= 0 else -0.08),
                    f"{'+' if d >= 0 else ''}{d:.1f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    span = (top - bot) or 1.0
    ax.set_ylim(bot - span * 0.25, top + span * 0.25)
    _style(ax)
    ax.set_ylabel("Δ acceptance rate vs base  (pp)", color=INK2, fontsize=10)
    ax.set_title("LoRA gain over the base drafter, per language",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load()
    if not rows:
        raise SystemExit(f"no result jsonls under {RESULTS}")
    cdir = RESULTS / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    matrix_fig(rows, cdir / "matrix.png")
    delta_fig(rows, cdir / "delta.png")
    print(f"[ok] -> {cdir}/matrix.png, delta.png")


if __name__ == "__main__":
    main()
