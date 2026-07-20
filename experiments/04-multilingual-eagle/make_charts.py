#!/usr/bin/env python3
"""Charts for the EAGLE3 multilingual LoRA experiment.

    python experiments/04-multilingual-eagle/make_charts.py

Writes results/charts/:
    matrix.png          5 languages x 3 variants (acceptance + mean accept len)
    delta.png           own & combined LoRA gain over base, per language
    vs_dflash.png       cross-speculator: LoRA acceptance gain, DFlash vs EAGLE3
                        (reads ../02-multilingual-dflash/results for the DFlash side)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DFLASH_RESULTS = HERE.parent / "02-multilingual-dflash" / "results"
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


def load(results_dir):
    rows = {}
    for lang in LANGS:
        for v in VARIANTS:
            f = Path(results_dir) / f"{lang}_{v}.jsonl"
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


def _rel(d, base):
    """Relative change vs base, e.g. '(+2.1%)'. Empty when there is no base."""
    if not base:
        return ""
    r = 100.0 * d / base
    return f"({r:+.0f}%)" if abs(r) >= 9.95 else f"({r:+.1f}%)"


def _label2(ax, x, y, main, rel, up=True, fs=9.0):
    va = "bottom" if up else "top"
    s = 1 if up else -1
    ax.annotate(main, (x, y), xytext=(0, s * 2), textcoords="offset points",
                ha="center", va=va, fontsize=fs, color=INK, fontweight="bold")
    if rel:
        ax.annotate(rel, (x, y), xytext=(0, s * (fs + 5)), textcoords="offset points",
                    ha="center", va=va, fontsize=fs - 1.8, color=MUTED)


def _grouped(ax, rows, key, fmt, title):
    xs = range(len(LANGS))
    w = 0.8 / len(VARIANTS)
    top = max(d[key] for d in rows.values())
    for i, v in enumerate(VARIANTS):
        offs = [x + (i - 1) * w for x in xs]
        hs = [rows.get((lang, v), {}).get(key, 0.0) for lang in LANGS]
        ax.bar(offs, hs, width=w * 0.86, color=COLOR[v], zorder=3, label=LABEL[v])
        for x, h, lang in zip(offs, hs, LANGS):
            if h:
                base = rows.get((lang, "base"), {}).get(key, 0.0)
                rel = "" if v == "base" else _rel(h - base, base)
                _label2(ax, x, h, fmt.format(h), rel, fs=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    ax.set_ylim(0, top * 1.26)
    _style(ax)
    ax.set_title(title, color=INK2, fontsize=11, loc="left", pad=10)


def matrix_fig(rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2))
    _grouped(axes[0], rows, "accept", "{:.1f}", "Acceptance rate  (%)")
    _grouped(axes[1], rows, "mean_len", "{:.2f}",
             "Mean accept length  (tokens / target pass)")
    leg = axes[0].legend(frameon=False, fontsize=10.5, loc="upper right", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    n = next(iter(rows.values()))["n"]
    fig.suptitle("EAGLE3 multilingual specialization  ·  base vs own-language LoRA vs combined LoRA",
                 fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.015,
             f"Qwen3-8B target · RedHatAI EAGLE3 head (1 layer, 3 spec tokens/step) · vLLM · "
             f"temperature 0 · per-language held-out test (n={n}). Same training data as the "
             f"DFlash experiment.", fontsize=8.5, color=MUTED, ha="left")
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
        bases = [rows.get((lang, "base"), {}).get("accept", 0.0) for lang in LANGS]
        ds = [rows.get((lang, v), {}).get("accept", 0.0) - b
              for lang, b in zip(LANGS, bases)]
        top = max(top, max(ds)); bot = min(bot, min(ds))
        ax.bar(offs, ds, width=w * 0.88, color=COLOR[v], zorder=3, label=LABEL[v])
        for x, d, b in zip(offs, ds, bases):
            _label2(ax, x, d, f"{d:+.1f}", _rel(d, b), up=d >= 0)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    span = (top - bot) or 1.0
    ax.set_ylim(bot - span * 0.35, top + span * 0.35)
    _style(ax)
    ax.set_ylabel("Δ acceptance rate vs base  (pp)", color=INK2, fontsize=10)
    ax.set_title("EAGLE3: LoRA gain over the base head, per language",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def vs_dflash_fig(eagle_rows, dflash_rows, out):
    """Own-LoRA acceptance gain over base, per language, DFlash vs EAGLE3."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    xs = range(len(LANGS))
    w = 0.36
    series = [("DFlash (block diffusion)", dflash_rows, "#4a3aa7"),
              ("EAGLE3 (autoregressive)", eagle_rows, "#1baf7a")]
    top, bot = 0.0, 0.0
    for i, (name, rows, color) in enumerate(series):
        offs = [x + (i - 0.5) * w for x in xs]
        bases = [rows.get((lang, "base"), {}).get("accept", 0.0) for lang in LANGS]
        ds = [rows.get((lang, "own"), {}).get("accept", 0.0) - b
              for lang, b in zip(LANGS, bases)]
        top = max(top, max(ds)); bot = min(bot, min(ds))
        ax.bar(offs, ds, width=w * 0.88, color=color, zorder=3, label=name)
        for x, d, b in zip(offs, ds, bases):
            _label2(ax, x, d, f"{d:+.1f}", _rel(d, b), up=d >= 0)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.capitalize() for l in LANGS])
    span = (top - bot) or 1.0
    ax.set_ylim(min(bot - span * 0.35, -0.2), top + span * 0.38)
    _style(ax)
    ax.set_ylabel("own-LoRA Δ acceptance vs base  (pp)", color=INK2, fontsize=10)
    ax.set_title("Does specialization transfer across speculators?  own-LoRA gain, DFlash vs EAGLE3",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.012, -0.02, "Same training data (target-generated answers), same rank-16 LoRA on "
             "q/k/v/o, same test prompts — only the speculator architecture differs.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load(RESULTS)
    if not rows:
        raise SystemExit(f"no result jsonls under {RESULTS}")
    cdir = RESULTS / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    matrix_fig(rows, cdir / "matrix.png")
    delta_fig(rows, cdir / "delta.png")
    made = ["matrix.png", "delta.png"]
    dflash = load(DFLASH_RESULTS)
    if dflash:
        vs_dflash_fig(rows, dflash, cdir / "vs_dflash.png")
        made.append("vs_dflash.png")
    print(f"[ok] -> {cdir}/" + ", ".join(made))


if __name__ == "__main__":
    main()
