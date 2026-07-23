#!/usr/bin/env python3
"""Charts for exp1-language: base vs own-LoRA vs combined-LoRA.

Only the clean 26-language subset (>=1000 train prompts) is charted.

Reads results/summary.json, writes results/charts/*.png:
    base_own_combined_26_mintrain1000.png
                          acceptance rate per language, base vs own vs combined
    base_acceptance_26_mintrain1000.png
                          base DFlash acceptance on the clean subset
    speedup_26_mintrain1000.png
                          analytic speedup on the clean subset
    own_lora_gain_26_mintrain1000.png
                          own-language LoRA gains on the clean subset
    gain_vs_base_26_mintrain1000.png
                          gain vs base weakness (scatter), clean subset
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


def dot_plot(rows, key, scale, xlabel, title, subtitle, fname, fmt, aliases=()):
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
    for alias in aliases:
        fig.savefig(OUT / alias, facecolor=SURFACE)
    plt.close(fig)


def base_acceptance_chart(rows, fname, title, subtitle):
    rows = sorted(rows, key=lambda r: r[1]["base"]["acceptance_rate"])
    langs = [r[0] for r in rows]
    vals = [r[1]["base"]["acceptance_rate"] * 100 for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8, max(7.0, len(rows) * 0.34)), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.barh(list(y), vals, height=0.68, color=C_BASE, zorder=3)
    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("pooled base acceptance rate (%)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.006, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)
    for yy, v in zip(y, vals):
        ax.text(v + 0.08, yy, f"{v:.2f}%", va="center", ha="left",
                fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def own_gain_chart(rows, fname, title, subtitle):
    rows = sorted(rows, key=lambda r: r[1]["base"]["acceptance_rate"])
    langs = [r[0] for r in rows]
    bases = [r[1]["base"]["acceptance_rate"] * 100 for r in rows]
    gains = [(r[1]["own"]["acceptance_rate"] - r[1]["base"]["acceptance_rate"]) * 100
             for r in rows]
    rels = [g / b * 100 if b else 0.0 for g, b in zip(gains, bases)]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, max(7.0, len(rows) * 0.34)), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.barh(list(y), gains, height=0.68, color=C_OWN, zorder=3)
    ax.set_yticks(list(y), langs)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("own-language LoRA gain over base (percentage points)",
                  color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    ax.text(0, 1.006, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)
    for yy, v, rel in zip(y, gains, rels):
        ax.text(v + 0.04, yy, f"+{v:.2f}pp ({rel:+.0f}%)", va="center", ha="left",
                fontsize=7.5, color=INK2)
    ax.set_xlim(right=max(gains) + 1.1)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def speedup_labeled_chart(rows, fname, title, subtitle):
    rows = sorted(rows, key=lambda r: r[1]["base"]["speedup_analytic"])
    langs = [r[0] for r in rows]
    base = [r[1]["base"]["speedup_analytic"] for r in rows]
    own = [r[1]["own"]["speedup_analytic"] for r in rows]
    comb = [r[1]["combined"]["speedup_analytic"] for r in rows]
    y = list(range(len(rows)))
    h = 0.23

    fig, ax = plt.subplots(figsize=(11.2, max(8.2, len(rows) * 0.42)), dpi=160)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.barh([i + h for i in y], base, height=h, color=C_BASE, label="base",
            zorder=3)
    ax.barh(y, own, height=h, color=C_OWN, label="own LoRA", zorder=3)
    ax.barh([i - h for i in y], comb, height=h, color=C_COMB,
            label="combined LoRA", zorder=3)

    xmax = max(max(base), max(own), max(comb))
    label_x = xmax + 0.11
    ax.set_xlim(0.95, xmax + 0.46)
    ax.set_yticks(y, langs)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("analytic speedup (x, L/(1+0.44))", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=18,
                 weight="bold")
    ax.text(0, 1.006, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)

    for yy, b, o, c in zip(y, base, own, comb):
        ax.text(b + 0.008, yy + h, f"{b:.2f}x", va="center", ha="left",
                fontsize=6.6, color=INK2)
        ax.text(o + 0.008, yy, f"{o:.2f}x", va="center", ha="left",
                fontsize=6.6, color=INK2)
        ax.text(c + 0.008, yy - h, f"{c:.2f}x", va="center", ha="left",
                fontsize=6.6, color=INK2)
        own_pct = (o / b - 1.0) * 100 if b else 0.0
        comb_pct = (c / b - 1.0) * 100 if b else 0.0
        ax.text(label_x, yy, f"own {own_pct:+.1f}%  |  comb {comb_pct:+.1f}%",
                va="center", ha="left", fontsize=6.6, color=INK2)

    ax.text(label_x, len(rows) - 0.25, "gain vs base", va="bottom", ha="left",
            fontsize=7.5, color=INK2, weight="bold")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.075),
                    ncol=3, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
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


def gain_vs_base(rows, fname="gain_vs_base.png", subtitle=None):
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
    ax.text(0, 1.01, subtitle or "each dot = one language; left = weak base coverage",
            transform=ax.transAxes, color=INK2, fontsize=9)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURFACE)
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


def val_loss(lang="Hindi", rank=16):
    """Convergence curve for one language from results/train_logs/{lang}.json
    (produced by pipeline.py::curve). Val loss (left axis) + val accept-rate
    (right axis) vs training step, initial/final points included."""
    sfx = f"_r{rank}" if rank != 16 else ""
    path = HERE / "results" / "train_logs" / f"{lang}{sfx}.json"
    if not path.exists():
        print(f"[val_loss] missing {path} — run: modal run pipeline.py::curve "
              f"--lang {lang}")
        return
    log = json.load(open(path))
    pts = []
    if log.get("val_initial") and log["val_initial"][0] is not None:
        pts.append({"step": 0, "loss": log["val_initial"][0],
                    "acc": log["val_initial"][1]})
    pts += [p for p in log.get("val", []) if p.get("loss") is not None]
    if log.get("val_final") and log["val_final"][0] is not None:
        # place final at its true step; fall back just past the last mid point
        last = pts[-1]["step"] if pts else 0
        fstep = log.get("steps") or (last + 1)
        if fstep > last:
            pts.append({"step": fstep, "loss": log["val_final"][0],
                        "acc": log["val_final"][1]})
    if len(pts) < 2:
        print(f"[val_loss] only {len(pts)} point(s) in {path.name}; "
              f"re-run curve with a smaller --val-every")
        return
    steps = [p["step"] for p in pts]
    loss = [p["loss"] for p in pts]
    acc = [p["acc"] * 100 for p in pts]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot(steps, loss, color=C_OWN, linewidth=2, marker="o", markersize=4,
            label="val loss")
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("validation loss", color=C_OWN, fontsize=9)
    ax.tick_params(axis="y", colors=C_OWN)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(steps, acc, color=C_COMB, linewidth=1.6, marker="s", markersize=3,
             linestyle="--", label="val accept rate")
    ax2.set_ylabel("val accept rate (%)", color=C_COMB, fontsize=9)
    ax2.tick_params(axis="y", colors=C_COMB, labelsize=8.5)

    ax.set_title(f"{lang}: r{rank} LoRA converges on held-out validation",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right",
              frameon=False, fontsize=8.5)
    fig.tight_layout()
    dest = OUT / f"val_loss_{lang}{sfx}.png"
    fig.savefig(dest, facecolor=SURFACE)
    plt.close(fig)
    print("val curve ->", dest)


def main():
    from matplotlib.ticker import FuncFormatter
    rows = load()
    clean = [r for r in rows if TRAIN_N.get(r[0], 0) >= 1000]
    dot_plot(clean, "acceptance_rate", 100, "pooled acceptance rate (%)",
             "Acceptance rate by language: base vs own vs combined",
             "26 WildChat languages with 1,000 train prompts, 100 held-out test prompts each",
             "base_own_combined_26_mintrain1000.png",
             FuncFormatter(lambda v, _: f"{v:.0f}%"))
    base_acceptance_chart(
        clean,
        "base_acceptance_26_mintrain1000.png",
        "Base DFlash acceptance (Qwen3-8B and DFlash 1B)",
        "",
    )
    own_gain_chart(
        clean,
        "own_lora_gain_26_mintrain1000.png",
        "LoRA gains across languages",
        "",
    )
    speedup_labeled_chart(
        clean,
        "speedup_26_mintrain1000.png",
        "Analytic speedup by language: base vs own vs combined",
        "26 WildChat languages with 1,000 train prompts; labels show speedup and % gain vs base",
    )
    gain_vs_base(
        clean,
        fname="gain_vs_base_26_mintrain1000.png",
        subtitle="each dot = one 1k-train language; left = weak base coverage",
    )
    val_loss("Swedish")  # convergence curve; no-op if train_logs/Swedish.json absent
    print("charts ->", OUT)


if __name__ == "__main__":
    main()
