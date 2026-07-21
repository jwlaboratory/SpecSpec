#!/usr/bin/env python3
"""LoRA-attributable net wall-clock gain, across EVERY experiment.

    python experiments/08-wallclock/make_charts.py

Writes results/charts/lora_gain.png.

Bars: the timing-free estimator validated by this experiment — since
speedup ≈ L/(1+c) and c is a per-speculator constant, the LoRA's wall-clock
gain over base spec decode is exactly

    gain = L_variant / L_base − 1        (L = pooled mean accept length)

independent of framework, engine overhead, and container timing noise (also
what a MERGED adapter serves at — relevant for exp 06, whose measured numbers
carry unmerged-LoRA overhead). Dots: the measured pooled spec-tok/s ratio
variant/base where both spec runs exist (paired in-container for 01/02/05/06;
cross-container for the vLLM EAGLE cells — vLLM timing is stable, HF is not).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / "results" / "charts"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"own": "#eb6834", "combined": "#2a78d6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def L_and_toks(path, kind):
    recs = rows(path)
    steps = sum(r["forward_steps"] for r in recs)
    secs = sum(r["spec_seconds"] for r in recs)
    gen = sum(r["num_generated_tokens"] for r in recs)
    if kind == "eagle":
        L = 1.0 + sum(r["accepted_draft_tokens"] for r in recs) / steps if steps else 0.0
    else:
        L = gen / steps if steps else 0.0
    return L, (gen / secs if secs else 0.0)


# (section title, kind, [(domain-label, base-path, {series: path})])
def sections():
    out = []

    e01 = EXP / "01-single-domain-dflash" / "results"
    out.append(("exp01 DFlash\nsingle-domain", "dflash", [
        (d, e01 / d / "base_dflash.jsonl", {"own": e01 / d / "lora.jsonl"})
        for d in ("code_sql", "ood_indian_legal")
    ]))

    e02 = EXP / "02-multilingual-dflash" / "results"
    out.append(("exp02 DFlash multilingual", "dflash", [
        (l, e02 / f"{l}_base.jsonl",
         {"own": e02 / f"{l}_own.jsonl", "combined": e02 / f"{l}_combined.jsonl"})
        for l in ("polish", "korean", "italian", "japanese", "german")
    ]))

    e03 = EXP / "03-weird-domains" / "results"
    out.append(("exp03 DFlash\nweird", "dflash", [
        (d, e03 / f"dflash_{d}_base.jsonl",
         {"own": e03 / f"dflash_{d}_own.jsonl",
          "combined": e03 / f"dflash_{d}_combined.jsonl"})
        for d in ("translation", "roleplay", "poetry")
    ]))

    e05 = EXP / "05-interference-ladder" / "results"
    doms = ["code_python", "code_sql", "lang_polish", "lang_korean", "lang_german",
            "ood_legal", "ood_medical", "task_math_reasoning",
            "task_summarization", "task_roleplay_chat"]
    out.append(("exp05 DFlash interference ladder (combined = comb40)", "dflash", [
        (d.replace("task_", "").replace("lang_", "").replace("ood_", ""),
         e05 / f"{d}_base.jsonl",
         {"own": e05 / f"{d}_own.jsonl", "combined": e05 / f"{d}_comb40.jsonl"})
        for d in doms
    ]))

    e06 = EXP / "06-independent-drafter" / "results"
    out.append(("exp06 independent 0.6B", "dflash", [
        (d.replace("task_", "").replace("lang_", "").replace("ood_", ""),
         e06 / f"{d}_base.jsonl",
         {"own": e06 / f"{d}_own.jsonl", "combined": e06 / f"{d}_combined.jsonl"})
        for d in ("code_sql", "lang_polish", "lang_korean", "ood_legal",
                  "task_math_reasoning")
    ]))

    out.append(("exp03 EAGLE3\nweird", "eagle", [
        (d, e03 / f"eagle_{d}_base.jsonl",
         {"own": e03 / f"eagle_{d}_own.jsonl",
          "combined": e03 / f"eagle_{d}_combined.jsonl"})
        for d in ("translation", "roleplay", "poetry")
    ]))

    e04 = EXP / "04-multilingual-eagle" / "results"
    out.append(("exp04 EAGLE3 multilingual (v3)", "eagle", [
        (l, e04 / f"{l}_base.jsonl",
         {"own": e04 / f"{l}_own.jsonl", "combined": e04 / f"{l}_combined.jsonl"})
        for l in ("polish", "korean", "italian", "japanese", "german")
    ]))
    return out


def main():
    secs = sections()
    row1 = [s for s in secs if s[0].startswith(("exp01", "exp02", "exp03 DFlash", "exp05"))]
    row2 = [s for s in secs if s not in row1]

    fig = plt.figure(figsize=(16.5, 8.6))
    grids = fig.add_gridspec(2, 1, hspace=0.52)
    for gi, group in enumerate((row1, row2)):
        widths = [max(len(cells), 2) for _, _, cells in group]
        sub = grids[gi].subgridspec(1, len(group), width_ratios=widths, wspace=0.14)
        for si, (title, kind, cells) in enumerate(group):
            ax = fig.add_subplot(sub[si])
            labels, gains, meas = [], {"own": [], "combined": []}, {"own": [], "combined": []}
            for label, base_p, var_paths in cells:
                if not base_p.exists():
                    continue
                Lb, tb = L_and_toks(base_p, kind)
                labels.append(label)
                for series in ("own", "combined"):
                    p = var_paths.get(series)
                    if p is not None and p.exists() and Lb > 0:
                        Lv, tv = L_and_toks(p, kind)
                        gains[series].append(100 * (Lv / Lb - 1))
                        meas[series].append(100 * (tv / tb - 1) if tb else None)
                    else:
                        gains[series].append(None)
                        meas[series].append(None)
            nser = sum(1 for s in ("own", "combined") if any(v is not None for v in gains[s]))
            w = 0.8 / max(nser, 1)
            xs = range(len(labels))
            i = 0
            for series in ("own", "combined"):
                if not any(v is not None for v in gains[series]):
                    continue
                pos = [x + i * w for x in xs]
                ax.bar(pos, [g if g is not None else 0 for g in gains[series]],
                       width=w * 0.9, color=COLOR[series], label=f"{series} (ΔL)",
                       zorder=3)
                mp = [(p_, m) for p_, m in zip(pos, meas[series]) if m is not None]
                if mp:
                    ax.scatter([p_ for p_, _ in mp], [m for _, m in mp],
                               marker="D", s=14, color=INK, zorder=4,
                               label=f"{series} (measured)" if i == 0 else None)
                i += 1
            ax.axhline(0, color=INK2, lw=1, zorder=2)
            ax.set_xticks([x + 0.4 - w / 2 for x in xs])
            ax.set_xticklabels(labels, rotation=38, ha="right", color=INK, fontsize=8)
            ax.set_title(title, color=INK, fontsize=9.5, loc="left")
            ax.grid(axis="y", color=GRID, zorder=0)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(AXIS)
            ax.tick_params(colors=INK2, labelsize=8)
            if si == 0:
                ax.set_ylabel("net wall-clock gain\nfrom LoRA vs base spec (%)",
                              color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["own"]),
               plt.Rectangle((0, 0), 1, 1, color=COLOR["combined"]),
               plt.Line2D([], [], marker="D", ls="", color=INK, markersize=4)]
    fig.legend(handles, ["own-LoRA (ΔL, timing-free)", "combined-LoRA (ΔL)",
                         "measured tok/s ratio"],
               loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Net wall-clock speedup from LoRA specialization — every experiment\n"
                 "bars: L_variant/L_base − 1 (exact, c cancels; = merged-adapter serving) · "
                 "dots: measured pooled spec-tok/s ratio",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lora_gain.png", dpi=170)
    print(f"[ok] -> {OUT}/lora_gain.png")


if __name__ == "__main__":
    main()
