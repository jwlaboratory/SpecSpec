#!/usr/bin/env python3
"""Charts for exp1-language: base vs own-LoRA vs combined-LoRA across 40 languages.

Reads results/summary.json, writes results/charts/*.png:
    acceptance_dots.png   acceptance rate per language, 3 variants (dot plot)
    delta_bars.png        own/combined gain over base per language (paired bars)
    gain_vs_base.png      gain vs base weakness (scatter)
    transfer_vs_data.png  combined-minus-own vs training-set size (scatter)
    speedup_dots.png      analytic wall-clock speedup per language, 3 variants
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
C_COMB = "#1baf7a"

# train prompts fetched per language (pipeline fetch yields, run ap-4YP8…)
TRAIN_N = {
    "English": 1000, "Russian": 1000, "Chinese": 1000, "French": 1000,
    "Vietnamese": 1000, "Yoruba": 1000, "Arabic": 1000, "Indonesian": 1000,
    "Spanish": 1000, "Portuguese": 1000, "German": 1000, "Persian": 1000,
    "Tagalog": 1000, "Turkish": 1000, "Korean": 1000, "Italian": 1000,
    "Maori": 420, "Sotho": 659, "Polish": 1000, "Latin": 1000,
    "Japanese": 1000, "Serbian": 295, "Ukrainian": 1000, "Malay": 1000,
    "Dutch": 1000, "Esperanto": 1000, "Romanian": 1000, "Hungarian": 1000,
    "Swedish": 1000, "Somali": 542, "Estonian": 430, "Tswana": 942,
    "Bulgarian": 722, "Finnish": 739, "Catalan": 986, "Bokmal": 712,
    "Hebrew": 959, "Welsh": 791, "Hindi": 549, "Nynorsk": 670,
}


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
    for lang, row in s.items():
        if isinstance(row, dict) and all(v in row for v in ("base", "own", "combined")):
            rows.append((lang, row))
    return rows


def dot_plot(rows, key, scale, xlabel, title, subtitle, fname, fmt):
    rows = sorted(rows, key=lambda r: r[1]["base"][key])
    langs = [r[0] for r in rows]
    b = [r[1]["base"][key] * scale for r in rows]
    o = [r[1]["own"][key] * scale for r in rows]
    c = [r[1]["combined"][key] * scale for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8, 10.5), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    for i in y:  # connector: base -> furthest variant
        lo, hi = min(b[i], o[i], c[i]), max(b[i], o[i], c[i])
        ax.plot([lo, hi], [i, i], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(b, list(y), s=34, color=C_BASE, zorder=3, label="base")
    ax.scatter(o, list(y), s=44, color=C_OWN, zorder=4, label="own LoRA")
    ax.scatter(c, list(y), s=44, color=C_COMB, zorder=4, label="combined LoRA")
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


def delta_bars(rows):
    rows = sorted(rows, key=lambda r: r[1]["combined"]["acceptance_rate"]
                  - r[1]["base"]["acceptance_rate"])
    langs = [r[0] for r in rows]
    do = [(r[1]["own"]["acceptance_rate"] - r[1]["base"]["acceptance_rate"]) * 100
          for r in rows]
    dc = [(r[1]["combined"]["acceptance_rate"] - r[1]["base"]["acceptance_rate"]) * 100
          for r in rows]
    y = range(len(rows))
    h = 0.36

    fig, ax = plt.subplots(figsize=(8, 10.5), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.barh([i + h / 2 + 0.02 for i in y], do, height=h, color=C_OWN,
            label="own LoRA − base", zorder=3)
    ax.barh([i - h / 2 - 0.02 for i in y], dc, height=h, color=C_COMB,
            label="combined LoRA − base", zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel("acceptance-rate gain over base (percentage points)",
                  color=INK2, fontsize=9)
    ax.set_title("Both adapters beat base almost everywhere",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.005, "gain in pooled acceptance rate vs base drafter, "
            "100 test prompts per language", transform=ax.transAxes,
            color=INK2, fontsize=9)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "delta_bars.png", facecolor=SURFACE)
    plt.close(fig)


def gain_vs_base(rows):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    bx = [r[1]["base"]["acceptance_rate"] * 100 for r in rows]
    do = [(r[1]["own"]["acceptance_rate"] - r[1]["base"]["acceptance_rate"]) * 100
          for r in rows]
    dc = [(r[1]["combined"]["acceptance_rate"] - r[1]["base"]["acceptance_rate"]) * 100
          for r in rows]
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.scatter(bx, do, s=44, color=C_OWN, label="own LoRA", zorder=3)
    ax.scatter(bx, dc, s=44, color=C_COMB, label="combined LoRA", zorder=3)
    ax.set_xlabel("base acceptance rate (%)", color=INK2, fontsize=9)
    ax.set_ylabel("gain over base (pp)", color=INK2, fontsize=9)
    ax.set_title("Gains concentrate where the base drafter is weakest",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.01, "each dot = one language; left = weak base coverage",
            transform=ax.transAxes, color=INK2, fontsize=9)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "gain_vs_base.png", facecolor=SURFACE)
    plt.close(fig)


def transfer_vs_data(rows):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    xs, ys, langs = [], [], []
    for lang, row in rows:
        xs.append(TRAIN_N.get(lang, 1000))
        ys.append((row["combined"]["acceptance_rate"]
                   - row["own"]["acceptance_rate"]) * 100)
        langs.append(lang)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.scatter(xs, ys, s=44, color=C_OWN, zorder=3)
    for lang, x, yv in zip(langs, xs, ys):
        if x < 800 or abs(yv) > 0.9:  # annotate the short-data + outlier langs
            ax.annotate(lang, (x, yv), textcoords="offset points",
                        xytext=(5, 4), fontsize=7.5, color=INK2)
    ax.text(0.02, 0.96, "combined wins", transform=ax.transAxes,
            color=INK2, fontsize=9, style="italic", va="top")
    ax.text(0.02, 0.05, "own wins", transform=ax.transAxes,
            color=INK2, fontsize=9, style="italic")
    ax.set_xlabel("training conversations fetched", color=INK2, fontsize=9)
    ax.set_ylabel("combined − own acceptance (pp)", color=INK2, fontsize=9)
    ax.set_title("Cross-lingual transfer rescues data-starved languages",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.01, "below ~750 training conversations the shared adapter "
            "beats the specialist", transform=ax.transAxes, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_vs_data.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    from matplotlib.ticker import FuncFormatter
    rows = load()
    dot_plot(rows, "acceptance_rate", 100, "pooled acceptance rate (%)",
             "Acceptance rate by language: base vs own vs combined",
             "40 WildChat languages, 100 held-out test prompts each, r16 LoRAs",
             "acceptance_dots.png", FuncFormatter(lambda v, _: f"{v:.0f}%"))
    dot_plot(rows, "speedup_analytic", 1, "analytic wall-clock speedup (×, L/(1+0.44))",
             "Wall-clock speedup over vanilla decode",
             "speedup ≈ L/(1+c), c = 0.44 fitted in exp-08; follows acceptance",
             "speedup_dots.png", FuncFormatter(lambda v, _: f"{v:.1f}×"))
    delta_bars(rows)
    gain_vs_base(rows)
    transfer_vs_data(rows)
    print("charts ->", OUT)


if __name__ == "__main__":
    main()
