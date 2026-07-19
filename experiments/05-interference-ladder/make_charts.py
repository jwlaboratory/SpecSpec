#!/usr/bin/env python3
"""Charts for the interference-at-scale experiment.

    python experiments/05-interference-ladder/make_charts.py

Reads results/<domain>_<variant>.jsonl (variant = base|own|comb10|comb20|comb40)
and writes:

    results/charts/matrix.png   10 core domains x 5 variants — acceptance rate
    results/charts/delta.png    gain over base per domain (own vs comb10/20/40)
    results/charts/ladder.png   the money chart: combN − own gap vs N, per domain
                                + mean with a paired per-prompt bootstrap 95% CI

base = untouched pretrained drafter (gray) · own = specialist LoRA (orange) ·
combN = one LoRA on N domains, core always included (blues, darker = larger N).
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CORE = ["code_python", "code_sql", "lang_polish", "lang_korean", "lang_german",
        "ood_legal", "ood_medical", "task_math_reasoning", "task_summarization",
        "task_roleplay_chat"]
VARIANTS = ["base", "own", "comb10", "comb20", "comb40"]
COMBOS = ["comb10", "comb20", "comb40"]
NS = {"comb10": 10, "comb20": 20, "comb40": 40}
LABEL = {"base": "base", "own": "own LoRA", "comb10": "combined (10 dom)",
         "comb20": "combined (20 dom)", "comb40": "combined (40 dom)"}
COLOR = {"base": "#898781", "own": "#eb6834", "comb10": "#8fbbe8",
         "comb20": "#2a78d6", "comb40": "#123c74"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def load():
    """rows[(domain, variant)] = summary; recs[(domain, variant)] = per-prompt list."""
    rows, recs = {}, {}
    for dom in CORE:
        for v in VARIANTS:
            f = RESULTS / f"{dom}_{v}.jsonl"
            if not f.exists():
                continue
            rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            if not rs:
                continue
            n = len(rs)
            tp = sum(r["proposed_draft_tokens"] for r in rs)
            rows[(dom, v)] = {
                "accept": 100 * sum(r["accepted_draft_tokens"] for r in rs) / tp if tp else 0.0,
                "mean_len": sum(r["mean_accept_length"] for r in rs) / n,
                "speedup": sum(r["speedup"] for r in rs) / n,
                "n": n,
            }
            recs[(dom, v)] = rs
    return rows, recs


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _short(dom):
    return dom.replace("code_", "").replace("lang_", "").replace("ood_", "") \
              .replace("task_", "").replace("_", " ")


def matrix_fig(rows, out):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    xs = range(len(CORE))
    w = 0.8 / len(VARIANTS)
    top = max(d["accept"] for d in rows.values())
    for i, v in enumerate(VARIANTS):
        offs = [x + (i - 2) * w for x in xs]
        hs = [rows.get((dom, v), {}).get("accept", 0.0) for dom in CORE]
        ax.bar(offs, hs, width=w * 0.86, color=COLOR[v], zorder=3, label=LABEL[v])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([_short(d) for d in CORE], fontsize=9)
    ax.set_ylim(0, top * 1.18)
    _style(ax)
    ax.set_ylabel("acceptance rate  (%)", color=INK2, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=5)
    for t in leg.get_texts():
        t.set_color(INK2)
    n = next(iter(rows.values()))["n"]
    fig.suptitle("Interference at scale  ·  base vs own LoRA vs combined LoRA (10/20/40 domains)",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.text(0.02, -0.02,
             f"Qwen3-8B target · z-lab DFlash drafter · temperature 0 (lossless) · per-domain "
             f"held-out test (n={n}). own = rank-16 LoRA on that domain's 800 prompts; "
             f"combN = one rank-16 LoRA on N domains (800 each, core 10 always included).",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def delta_fig(rows, out):
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    xs = range(len(CORE))
    order = ["own"] + COMBOS
    w = 0.8 / len(order)
    top, bot = 0.0, 0.0
    for i, v in enumerate(order):
        offs = [x + (i - 1.5) * w for x in xs]
        ds = [rows.get((dom, v), {}).get("accept", 0.0)
              - rows.get((dom, "base"), {}).get("accept", 0.0) for dom in CORE]
        top = max(top, max(ds)); bot = min(bot, min(ds))
        ax.bar(offs, ds, width=w * 0.88, color=COLOR[v], zorder=3, label=LABEL[v])
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([_short(d) for d in CORE], fontsize=9)
    span = (top - bot) or 1.0
    ax.set_ylim(bot - span * 0.2, top + span * 0.25)
    _style(ax)
    ax.set_ylabel("Δ acceptance vs base  (pp)", color=INK2, fontsize=10)
    ax.set_title("LoRA gain over the base drafter, per core domain",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=4)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pooled(rs, idx):
    a = sum(rs[i]["accepted_draft_tokens"] for i in idx)
    p = sum(rs[i]["proposed_draft_tokens"] for i in idx)
    return 100 * a / p if p else 0.0


def ladder_fig(rows, recs, out, n_boot=2000, seed=0):
    """combN − own acceptance gap vs N. Thin lines = core domains; bold = mean.
    CI: paired bootstrap over prompts, pooled over the core domains."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ns = [NS[c] for c in COMBOS]

    per_dom = {}
    for dom in CORE:
        if (dom, "own") not in rows:
            continue
        gaps = [rows[(dom, c)]["accept"] - rows[(dom, "own")]["accept"]
                for c in COMBOS if (dom, c) in rows]
        if len(gaps) == len(COMBOS):
            per_dom[dom] = gaps
            ax.plot(ns, gaps, color=MUTED, lw=1.0, alpha=0.45, zorder=2)
            ax.annotate(_short(dom), (ns[-1], gaps[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=7.5, color=MUTED,
                        va="center")

    mean_gap = [sum(g[i] for g in per_dom.values()) / len(per_dom)
                for i in range(len(COMBOS))]

    rng = random.Random(seed)
    lo, hi = [], []
    for ci, c in enumerate(COMBOS):
        boots = []
        doms = list(per_dom)
        for _ in range(n_boot):
            vals = []
            for dom in doms:
                ro, rc = recs[(dom, "own")], recs[(dom, c)]
                m = min(len(ro), len(rc))
                idx = [rng.randrange(m) for _ in range(m)]
                vals.append(_pooled(rc, idx) - _pooled(ro, idx))
            boots.append(sum(vals) / len(vals))
        boots.sort()
        lo.append(boots[int(0.025 * n_boot)])
        hi.append(boots[int(0.975 * n_boot)])

    ax.fill_between(ns, lo, hi, color=COLOR["comb20"], alpha=0.15, zorder=2,
                    linewidth=0)
    ax.plot(ns, mean_gap, color=COLOR["comb20"], lw=2.8, zorder=4, marker="o",
            markersize=6, label="mean over core domains (95% CI)")
    for x, y in zip(ns, mean_gap):
        ax.annotate(f"{y:+.2f}", (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK, fontweight="bold")

    ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"N = {n}" for n in ns])
    _style(ax)
    ax.set_xlabel("domains in the combined adapter", color=INK2, fontsize=10)
    ax.set_ylabel("combined − own acceptance  (pp)", color=INK2, fontsize=10)
    ax.set_title("Interference ladder — does the combined LoRA fall behind specialists?",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.02, -0.02,
             "0 = zero interference. Thin gray lines: individual core domains. "
             "CI: paired per-prompt bootstrap (2000 resamples), pooled over core domains.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, recs = load()
    if not rows:
        raise SystemExit(f"no result jsonls under {RESULTS}")
    cdir = RESULTS / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    matrix_fig(rows, cdir / "matrix.png")
    delta_fig(rows, cdir / "delta.png")
    ladder_fig(rows, recs, cdir / "ladder.png")
    print(f"[ok] -> {cdir}/matrix.png, delta.png, ladder.png")


if __name__ == "__main__":
    main()
