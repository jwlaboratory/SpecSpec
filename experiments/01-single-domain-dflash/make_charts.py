#!/usr/bin/env python3
"""Comparison charts for the finetuning experiments — base vs full fine-tune vs LoRA.

    python experiments/01-single-domain-dflash/make_charts.py                 # every domain under results/
    python experiments/01-single-domain-dflash/make_charts.py ood_indian_legal code_sql

Reads experiments/01-single-domain-dflash/results/<domain>/{base_dflash,full_finetune,lora}.jsonl (the raw
per-prompt benchmark records) and writes PNGs to experiments/01-single-domain-dflash/results/<domain>/charts/:

    summary.png       mean accept length · acceptance rate · speedup  (3 panels)
    distribution.png  per-prompt mean-accept-length distribution per variant

and, when more than one domain has results:

    experiments/01-single-domain-dflash/results/charts/cross_domain.png   mean accept length, domain × variant

Design: base is the neutral reference (gray); the two trained variants use the
CVD-safe blue/orange pair; every bar is directly labelled so identity never rests
on colour alone. CPU only — run after pulling results back from the volume.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = Path(__file__).resolve().parent / "results"
VARIANTS = ["base_dflash", "full_finetune", "lora"]
LABEL = {"base_dflash": "base", "full_finetune": "full FT", "lora": "LoRA"}
COLOR = {"base_dflash": "#898781", "full_finetune": "#2a78d6", "lora": "#eb6834"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.facecolor": SURF,
})


def load(domain):
    d = RESULTS / domain
    out = {}
    for v in VARIANTS:
        f = d / f"{v}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if recs:
            out[v] = recs
    return out


def metrics(recs):
    n = len(recs)
    tot_acc = sum(r["accepted_draft_tokens"] for r in recs)
    tot_prop = sum(r["proposed_draft_tokens"] for r in recs)
    return {
        "accept_rate": 100 * tot_acc / tot_prop if tot_prop else 0.0,
        "mean_len": sum(r["mean_accept_length"] for r in recs) / n,
        "speedup": sum(r["speedup"] for r in recs) / n,
        "lens": [r["mean_accept_length"] for r in recs],
        "n": n,
    }


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _rel(d, base):
    """Relative change vs base, e.g. '(+2.1%)'. Empty when there is no base."""
    if not base:
        return ""
    r = 100.0 * d / base
    return f"({r:+.0f}%)" if abs(r) >= 9.95 else f"({r:+.1f}%)"


def _label2(ax, x, y, main, rel, fs=9.0):
    ax.annotate(main, (x, y), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=fs, color=INK, fontweight="bold")
    if rel:
        ax.annotate(rel, (x, y), xytext=(0, fs + 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fs - 1.8, color=MUTED)


def _rounded_bars(ax, xs, heights, colors, width, top):
    """Flat bars with softly rounded top corners, 2px surface gap via width<step."""
    pad = top * 0.006
    for x, h, c in zip(xs, heights, colors):
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, 0), width, max(h - pad, top * 0.002),
            boxstyle="round,pad=0,rounding_size=%.4f" % (width * 0.10),
            mutation_aspect=top / (width * 8 + 1e-9),
            linewidth=0, facecolor=c, zorder=3, clip_on=False))


def _panel(ax, title, present, vals, fmt):
    xs = list(range(len(present)))
    top = max(vals[v] for v in present) or 1.0
    _rounded_bars(ax, xs, [vals[v] for v in present],
                  [COLOR[v] for v in present], 0.60, top)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[v] for v in present])
    ax.set_ylim(0, top * 1.30)
    ax.set_xlim(-0.6, len(present) - 0.4)
    _style(ax)
    base = vals.get("base_dflash", 0.0)
    for x, v in zip(xs, present):
        rel = "" if v == "base_dflash" else _rel(vals[v] - base, base)
        _label2(ax, x, vals[v], fmt.format(vals[v]), rel, fs=11)
    ax.set_title(title, color=INK2, fontsize=10.5, pad=10, loc="left")


def summary_fig(domain, data, out):
    present = [v for v in VARIANTS if v in data]
    M = {v: metrics(data[v]) for v in present}
    n = M[present[0]]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    _panel(axes[0], "Mean accept length  (tokens / target pass)",
           present, {v: M[v]["mean_len"] for v in present}, "{:.2f}")
    _panel(axes[1], "Acceptance rate",
           present, {v: M[v]["accept_rate"] for v in present}, "{:.1f}%")
    _panel(axes[2], "Speedup vs target-only",
           present, {v: M[v]["speedup"] for v in present}, "{:.2f}×")
    fig.suptitle(f"{domain}  ·  DFlash drafter: base vs full fine-tune vs LoRA",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.02, -0.03,
             f"Qwen3-8B target · z-lab DFlash drafter · temperature 0 (lossless) · "
             f"held-out test split (n={n}). Higher is better on all three.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def distribution_fig(domain, data, out):
    present = [v for v in VARIANTS if v in data]
    M = {v: metrics(data[v]) for v in present}
    allv = [x for v in present for x in M[v]["lens"]]
    lo, hi = min(allv), max(allv)
    bins = [lo + (hi - lo) * i / 26 for i in range(27)]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    handles = []
    for v in present:
        mu = M[v]["mean_len"]
        ax.hist(M[v]["lens"], bins=bins, color=COLOR[v], alpha=0.14, zorder=2)
        ax.hist(M[v]["lens"], bins=bins, histtype="step", color=COLOR[v],
                linewidth=2.2, zorder=3, label=f"{LABEL[v]}  (mean {mu:.2f})")
        ax.axvline(mu, color=COLOR[v], lw=1.3, ls=(0, (4, 2)), zorder=4)
    _style(ax)
    ax.set_xlabel("per-prompt mean accept length (tokens / target pass)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("prompts", color=INK2, fontsize=10)
    ax.set_title(f"{domain}  ·  where each variant's per-prompt acceptance lands",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper right",
                    handlelength=1.4, borderaxespad=0.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.012, -0.02, "dashed line = variant mean; a rightward-shifted "
             "distribution means the drafter tracks the target on more prompts.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cross_domain_fig(domains_data, out):
    doms = list(domains_data)
    present = [v for v in VARIANTS if all(v in domains_data[d] for d in doms)]
    xs = range(len(doms))
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(2.4 + 2.2 * len(doms), 4.4))
    bases = {d: metrics(domains_data[d]["base_dflash"])["mean_len"]
             if "base_dflash" in domains_data[d] else 0.0 for d in doms}
    for i, v in enumerate(present):
        heights = [metrics(domains_data[d][v])["mean_len"] for d in doms]
        offs = [x + (i - (len(present) - 1) / 2) * w for x in xs]
        ax.bar(offs, heights, width=w * 0.86, color=COLOR[v], label=LABEL[v], zorder=3)
        for x, h, d in zip(offs, heights, doms):
            rel = "" if v == "base_dflash" else _rel(h - bases[d], bases[d])
            _label2(ax, x, h, f"{h:.2f}", rel, fs=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(doms)
    top = max(metrics(domains_data[d][v])["mean_len"] for d in doms for v in present)
    ax.set_ylim(0, top * 1.26)
    _style(ax)
    ax.set_ylabel("mean accept length  (tokens / target pass)", color=INK2, fontsize=10)
    ax.set_title("Mean accept length by domain  ·  base vs full fine-tune vs LoRA",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=len(present))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    domains = args or sorted(p.name for p in RESULTS.iterdir()
                             if p.is_dir() and p.name != "charts")
    have = {}
    for domain in domains:
        data = load(domain)
        if not data:
            print(f"[skip] {domain}: no result jsonls")
            continue
        cdir = RESULTS / domain / "charts"
        cdir.mkdir(parents=True, exist_ok=True)
        summary_fig(domain, data, cdir / "summary.png")
        distribution_fig(domain, data, cdir / "distribution.png")
        print(f"[ok] {domain} -> {cdir}/summary.png, distribution.png")
        have[domain] = data
    if len(have) > 1:
        (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
        cross_domain_fig(have, RESULTS / "charts" / "cross_domain.png")
        print(f"[ok] cross-domain -> {RESULTS}/charts/cross_domain.png")


if __name__ == "__main__":
    main()
