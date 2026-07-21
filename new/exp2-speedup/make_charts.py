#!/usr/bin/env python3
"""Charts for exp2-speedup: base vs own LoRA vs combined LoRA across 25 languages.

Reads results/summary.json, writes results/charts/*.png:
    speedup_dots.png      analytic speedup per language, 3 variants (dot plot)
    acceptance_dots.png   acceptance rate per language, 3 variants (dot plot)
    mean_length_dots.png  mean accept length per language, 3 variants (dot plot)
    own_gain_bars.png     own gain over base per language (bar plot)
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# palette (dataviz default, validated): base = neutral reference, own/combined = series
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_BASE = "#898781"
C_OWN = "#2a78d6"
C_COMB = "#d97706"


def style_ax(ax, xgrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load():
    s = json.load(open(HERE / "results" / "summary.json"))
    rows = []
    for lang, row in s["per_lang"].items():
        if all(v in row for v in ("base", "own", "combined")):
            rows.append((lang, row))
    return rows


def dot_plot(rows, key, scale, xlabel, title, subtitle, fname, fmt):
    rows = sorted(rows, key=lambda r: r[1]["base"][key])
    langs = [r[0] for r in rows]
    b = [r[1]["base"][key] * scale for r in rows]
    o = [r[1]["own"][key] * scale for r in rows]
    c = [r[1]["combined"][key] * scale for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8, 9), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    for i in y:  # connector: base -> furthest variant
        lo, hi = min(b[i], o[i], c[i]), max(b[i], o[i], c[i])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(b, list(y), s=34, color=C_BASE, zorder=3, label="base")
    ax.scatter(o, list(y), s=44, color=C_OWN, zorder=4, label="own")
    ax.scatter(c, list(y), s=44, color=C_COMB, zorder=4, label="combined")
    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.xaxis.set_major_formatter(fmt)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def own_gain_bars(rows):
    rows = sorted(rows, key=lambda r: r[1]["own"]["speedup_analytic"]
                  - r[1]["base"]["speedup_analytic"])
    langs = [r[0] for r in rows]
    gain = [(r[1]["own"]["speedup_analytic"] - r[1]["base"]["speedup_analytic"])
            for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8, 9), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.barh(y, gain, height=0.6, color=C_OWN, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel("speedup gain (×)", color=INK2, fontsize=9)
    ax.set_title("Language-specific LoRA outperforms shared base",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.005, "own variant gain over base, measured as analytic speedup",
            transform=ax.transAxes, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_gain_bars.png", facecolor=SURFACE)
    plt.close(fig)


def speedup_grouped_bars(rows):
    rows = sorted(rows, key=lambda r: r[1]["own"]["speedup_analytic"], reverse=True)
    langs = [r[0] for r in rows]
    b = [r[1]["base"]["speedup_analytic"] for r in rows]
    o = [r[1]["own"]["speedup_analytic"] for r in rows]
    c = [r[1]["combined"]["speedup_analytic"] for r in rows]
    y = range(len(rows))
    h = 0.25

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax, xgrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh([i - h for i in y], b, height=h, color=C_BASE, label="base", zorder=3)
    ax.barh([i for i in y], o, height=h, color=C_OWN, label="own", zorder=3)
    ax.barh([i + h for i in y], c, height=h, color=C_COMB, label="combined", zorder=3)

    # Add value labels on bars
    for i, (bi, oi, ci) in enumerate(zip(b, o, c)):
        ax.text(bi + 0.03, i - h, f"{bi:.2f}×", va="center", fontsize=7.5, color=INK2)
        ax.text(oi + 0.03, i, f"{oi:.2f}×", va="center", fontsize=7.5, color=INK2)
        ax.text(ci + 0.03, i + h, f"{ci:.2f}×", va="center", fontsize=7.5, color=INK2)

    # Find max bar length to position improvement column
    max_val = max(max(b), max(o), max(c))
    improvement_x = max_val + 0.3

    # Add improvement percentages
    ax.text(improvement_x, len(rows) + 0.3, "Improvement vs base",
            fontsize=8.5, fontweight="bold", color=INK2, ha="left")

    for i, (bi, oi, ci) in enumerate(zip(b, o, c)):
        own_pct = ((oi - bi) / bi * 100) if bi > 0 else 0
        comb_pct = ((ci - bi) / bi * 100) if bi > 0 else 0
        pct_text = f"own: +{own_pct:.1f}%  |  combined: {comb_pct:+.1f}%"
        ax.text(improvement_x, i, pct_text, fontsize=7, color=INK2, ha="left", va="center")

    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel("analytic speedup (×)", color=INK2, fontsize=9)
    ax.set_title("Speedup comparison: base vs own vs combined",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.005, "analytic wall-clock speedup per language with improvement percentages vs base",
            transform=ax.transAxes, color=INK2, fontsize=9)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "speedup_comparison.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    from matplotlib.ticker import FuncFormatter
    rows = load()
    dot_plot(rows, "speedup_analytic", 1, "analytic speedup (×)",
             "Analytic speedup by language: base vs own vs combined",
             "25 languages, 25 prompts each, language-specific LoRA experts",
             "speedup_dots.png", FuncFormatter(lambda v, _: f"{v:.2f}×"))
    dot_plot(rows, "acceptance_rate", 100, "acceptance rate (%)",
             "Acceptance rate by language: base vs own vs combined",
             "25 languages, 25 prompts each, baseline acceptance across variants",
             "acceptance_dots.png", FuncFormatter(lambda v, _: f"{v:.1f}%"))
    dot_plot(rows, "mean_accept_length", 1, "mean accept length (tokens)",
             "Mean accept length by language: base vs own vs combined",
             "25 languages, 25 prompts each, average accepted sequence length",
             "mean_length_dots.png", FuncFormatter(lambda v, _: f"{v:.2f}"))
    own_gain_bars(rows)
    speedup_grouped_bars(rows)
    print("charts ->", OUT)


if __name__ == "__main__":
    main()
