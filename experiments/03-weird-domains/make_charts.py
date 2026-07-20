#!/usr/bin/env python3
"""Charts for the weird-domains experiment — both speculators side by side.

    python experiments/03-weird-domains/make_charts.py

Reads results/{dflash,eagle}_{domain}_{variant}.jsonl and writes results/charts/:
    matrix.png     2 rows (DFlash, EAGLE3) x acceptance-rate bars per domain/variant
    delta.png      own & combined gain over base, per domain, per speculator
    delta_len.png  same, but raw Δ mean accept length (tokens / target pass)
    len_abs.png    absolute mean accept length — base vs own vs combined
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOMAINS = ["translation", "roleplay", "poetry"]
VARIANTS = ["base", "own", "combined"]
SPECS = [("dflash", "DFlash (block diffusion, 15 drafts/step)"),
         ("eagle", "EAGLE3 (autoregressive, 3 drafts/step)")]
LABEL = {"base": "base", "own": "own-domain LoRA", "combined": "combined LoRA"}
COLOR = {"base": "#898781", "own": "#eb6834", "combined": "#2a78d6"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def load(spec):
    rows = {}
    for d in DOMAINS:
        for v in VARIANTS:
            f = RESULTS / f"{spec}_{d}_{v}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            if not recs:
                continue
            tp = sum(r["proposed_draft_tokens"] for r in recs)
            rows[(d, v)] = {
                "accept": 100 * sum(r["accepted_draft_tokens"] for r in recs) / tp if tp else 0.0,
                "mean_len": sum(r["mean_accept_length"] for r in recs) / len(recs),
                "n": len(recs),
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


def _label2(ax, x, y, main, rel, up=True, fs=8.5):
    va = "bottom" if up else "top"
    s = 1 if up else -1
    ax.annotate(main, (x, y), xytext=(0, s * 2), textcoords="offset points",
                ha="center", va=va, fontsize=fs, color=INK, fontweight="bold")
    if rel:
        ax.annotate(rel, (x, y), xytext=(0, s * (fs + 5)), textcoords="offset points",
                    ha="center", va=va, fontsize=fs - 1.8, color=MUTED)


def matrix_fig(out):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2))
    for ax, (spec, title) in zip(axes, SPECS):
        rows = load(spec)
        if not rows:
            continue
        xs = range(len(DOMAINS))
        w = 0.8 / len(VARIANTS)
        top = max(d["accept"] for d in rows.values())
        for i, v in enumerate(VARIANTS):
            offs = [x + (i - 1) * w for x in xs]
            hs = [rows.get((d, v), {}).get("accept", 0.0) for d in DOMAINS]
            ax.bar(offs, hs, width=w * 0.86, color=COLOR[v], zorder=3, label=LABEL[v])
            for x, h, d in zip(offs, hs, DOMAINS):
                if h:
                    base = rows.get((d, "base"), {}).get("accept", 0.0)
                    rel = "" if v == "base" else _rel(h - base, base)
                    _label2(ax, x, h, f"{h:.1f}", rel)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([d.capitalize() for d in DOMAINS])
        ax.set_ylim(0, top * 1.28)
        _style(ax)
        ax.set_title(f"{title}  ·  acceptance rate (%)", color=INK2, fontsize=11,
                     loc="left", pad=10)
    leg = axes[0].legend(frameon=False, fontsize=10.5, loc="upper left", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("Weird domains  ·  base vs own-domain LoRA vs combined — two speculators",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.015,
             "Qwen3-8B target · temperature 0 · n=100/domain · same target-generated training "
             "data for both speculators. Acceptance rates are NOT comparable across speculators "
             "(different drafts/step); compare within each panel.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def delta_fig(out, key="accept", fmt="{:+.1f}",
              ylabel="Δ accept vs base (pp)",
              suptitle="LoRA gain over base — specialization helps DFlash, not the stronger EAGLE3"):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
    for ax, (spec, title) in zip(axes, SPECS):
        rows = load(spec)
        if not rows:
            continue
        xs = range(len(DOMAINS))
        w = 0.36
        top, bot = 0.0, 0.0
        for i, v in enumerate(("own", "combined")):
            offs = [x + (i - 0.5) * w for x in xs]
            bases = [rows.get((d, "base"), {}).get(key, 0.0) for d in DOMAINS]
            ds = [rows.get((d, v), {}).get(key, 0.0) - b
                  for d, b in zip(DOMAINS, bases)]
            top = max(top, max(ds)); bot = min(bot, min(ds))
            ax.bar(offs, ds, width=w * 0.88, color=COLOR[v], zorder=3, label=LABEL[v])
            for x, dd, b in zip(offs, ds, bases):
                _label2(ax, x, dd, fmt.format(dd), _rel(dd, b), up=dd >= 0)
        ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([d.capitalize() for d in DOMAINS])
        span = (top - bot) or 1.0
        ax.set_ylim(bot - span * 0.38, top + span * 0.38)
        _style(ax)
        ax.set_title(title, color=INK2, fontsize=11, loc="left", pad=10)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    leg = axes[0].legend(frameon=False, fontsize=10, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle(suptitle, fontsize=13, color=INK, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def len_abs_fig(out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
    for ax, (spec, title) in zip(axes, SPECS):
        rows = load(spec)
        if not rows:
            continue
        xs = range(len(DOMAINS))
        w = 0.8 / len(VARIANTS)
        top = max(d["mean_len"] for d in rows.values())
        for i, v in enumerate(VARIANTS):
            offs = [x + (i - 1) * w for x in xs]
            hs = [rows.get((d, v), {}).get("mean_len", 0.0) for d in DOMAINS]
            ax.bar(offs, hs, width=w * 0.86, color=COLOR[v], zorder=3, label=LABEL[v])
            for x, h, d in zip(offs, hs, DOMAINS):
                if h:
                    base = rows.get((d, "base"), {}).get("mean_len", 0.0)
                    rel = "" if v == "base" else _rel(h - base, base)
                    _label2(ax, x, h, f"{h:.2f}", rel)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([d.capitalize() for d in DOMAINS])
        ax.set_ylim(0, top * 1.28)
        _style(ax)
        ax.set_title(f"{title}  ·  mean accept length (tokens / target pass)",
                     color=INK2, fontsize=11, loc="left", pad=10)
    leg = axes[0].legend(frameon=False, fontsize=10, loc="upper left", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("Mean accept length  ·  base vs own-domain LoRA vs combined — two speculators",
                 fontsize=13, color=INK, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cdir = RESULTS / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    matrix_fig(cdir / "matrix.png")
    delta_fig(cdir / "delta.png")
    delta_fig(cdir / "delta_len.png", key="mean_len", fmt="{:+.3f}",
              ylabel="Δ mean accept length vs base (tokens)",
              suptitle="LoRA gain in mean accept length over base, per domain, per speculator")
    len_abs_fig(cdir / "len_abs.png")
    print(f"[ok] -> {cdir}/matrix.png, delta.png, delta_len.png, len_abs.png")


if __name__ == "__main__":
    main()
