#!/usr/bin/env python3
"""Collate net wall-clock speedups: spec-decode tok/s ÷ vanilla target-only tok/s.

    python experiments/08-wallclock/aggregate.py

Reads
  results/vanilla/{vllm,hf}_<domain>.jsonl   (pipeline.py baselines, downloaded
                                              from the volume)
  ../04-multilingual-eagle/results/<lang>_<variant>.jsonl        [vLLM, EAGLE]
  ../03-weird-domains/results/eagle_<domain>_<variant>.jsonl     [vLLM, EAGLE]
  ../03-weird-domains/results/dflash_<domain>_<variant>.jsonl    [HF, DFlash]
  ../02-multilingual-dflash/results/<lang>_<variant>.jsonl       [HF, DFlash —
                                              carries its own baseline_seconds]
and writes results/comparison.csv, results/report.md, results/charts/speedup.png.

Ratios only ever divide numbers measured in the SAME framework (vLLM/vLLM,
HF/HF) with the same prompts, greedy decoding, and max_new_tokens=256.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VAN = HERE / "results" / "vanilla"
OUT = HERE / "results"
E02 = HERE.parent / "02-multilingual-dflash" / "results"
E03 = HERE.parent / "03-weird-domains" / "results"
E04 = HERE.parent / "04-multilingual-eagle" / "results"

LANGS = ["polish", "korean", "italian", "japanese", "german"]
WEIRD = ["translation", "roleplay", "poetry"]
VARIANTS = ["base", "own", "combined"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"base": "#898781", "own": "#eb6834", "combined": "#2a78d6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pooled_tok_s(recs, tok_key="num_generated_tokens", sec_key="seconds"):
    tt = sum(r[tok_key] for r in recs)
    ts = sum(r[sec_key] for r in recs)
    return tt / ts if ts else 0.0


def spec_pooled(path):
    return pooled_tok_s(rows(path), sec_key="spec_seconds")


def pooled_L(path, kind):
    """Mean accept length pooled over the run: tokens committed per target
    verify pass. eagle jsonls count drafting rounds in forward_steps and add a
    bonus token per round; dflash commits num_generated/forward_steps."""
    recs = rows(path)
    steps = sum(r["forward_steps"] for r in recs)
    if not steps:
        return 0.0
    if kind == "eagle":
        return 1.0 + sum(r["accepted_draft_tokens"] for r in recs) / steps
    return sum(r["num_generated_tokens"] for r in recs) / steps


def main():
    # (section, domain, vanilla_tok_s, {variant: (tok_s, speedup, L)})
    table = []

    # --- EAGLE3 via vLLM: multilingual (exp 04) + weird (exp 03) ---
    for domain, spec_path, van_key in (
        [(l, E04 / f"{l}_{{v}}.jsonl", f"lang_{l}") for l in LANGS]
        + [(d, E03 / f"eagle_{d}_{{v}}.jsonl", d) for d in WEIRD]
    ):
        van_file = VAN / f"vllm_{van_key}.jsonl"
        if not van_file.exists():
            continue
        v_tok_s = pooled_tok_s(rows(van_file))
        cells = {}
        for var in VARIANTS:
            p = Path(str(spec_path).format(v=var))
            if p.exists():
                s = spec_pooled(p)
                cells[var] = (s, s / v_tok_s if v_tok_s else 0.0,
                              pooled_L(p, "eagle"))
        sec = "EAGLE3 · multilingual (vLLM)" if van_key.startswith("lang_") \
            else "EAGLE3 · weird domains (vLLM)"
        table.append((sec, domain, v_tok_s, cells))

    # --- DFlash via HF: weird (exp 03, baseline from this experiment).
    # NOTE: spec numbers and this baseline ran in DIFFERENT containers days
    # apart — HF batch-1 timing varies several percent to tens of percent
    # across hosts, so these ratios carry real noise (see report). ---
    for d in WEIRD:
        van_file = VAN / f"hf_{d}.jsonl"
        if not van_file.exists():
            continue
        v_tok_s = pooled_tok_s(rows(van_file))
        cells = {}
        for var in VARIANTS:
            p = E03 / f"dflash_{d}_{var}.jsonl"
            if p.exists():
                s = spec_pooled(p)
                cells[var] = (s, s / v_tok_s if v_tok_s else 0.0,
                              pooled_L(p, "dflash"))
        table.append(("DFlash · weird domains (HF)", d, v_tok_s, cells))

    # --- DFlash via HF: multilingual (exp 02 timed baseline and spec decode
    # for the SAME prompts in the SAME container: use each variant file's own
    # baseline — the only fully noise-paired HF cells) ---
    for l in LANGS:
        if not (E02 / f"{l}_base.jsonl").exists():
            continue
        cells = {}
        v_disp = 0.0
        for var in VARIANTS:
            p = E02 / f"{l}_{var}.jsonl"
            if p.exists():
                recs = rows(p)
                v = pooled_tok_s(recs, sec_key="baseline_seconds")
                s = pooled_tok_s(recs, sec_key="spec_seconds")
                cells[var] = (s, s / v if v else 0.0, pooled_L(p, "dflash"))
                if var == "base":
                    v_disp = v
        table.append(("DFlash · multilingual (HF, paired in-container baselines)",
                      l, v_disp, cells))

    # ---- csv + report ----
    csv = ["section,domain,vanilla_tok_s," + ",".join(
        f"{v}_tok_s,{v}_speedup,{v}_mean_accept_len" for v in VARIANTS)]
    md = ["# Net wall-clock speedup — spec decode vs target-only decoding\n",
          "Same prompts, greedy, 256 max new tokens, batch 1, H200. Speedup ="
          " pooled spec tok/s ÷ pooled vanilla tok/s, always within one"
          " framework (vLLM/vLLM or HF/HF). L = pooled mean accept length"
          " (tokens committed per target pass).\n"]
    cur = None
    for sec, dom, v, cells in table:
        if sec != cur:
            md += [f"\n## {sec}\n",
                   "| domain | vanilla tok/s | base | own | combined |",
                   "|---|--:|--:|--:|--:|"]
            cur = sec
        def cell(var):
            if var not in cells:
                return "—"
            s, sp, L = cells[var]
            return f"{sp:.2f}× (L={L:.2f})"
        md.append(f"| {dom} | {v:.0f} | {cell('base')} | {cell('own')} | {cell('combined')} |")
        csv.append(f"{sec},{dom},{v:.2f}," + ",".join(
            f"{cells[var][0]:.2f},{cells[var][1]:.4f},{cells[var][2]:.4f}"
            if var in cells else ",,"
            for var in VARIANTS))

    # ---- analytic model: speedup ≈ L / (1 + c), c = per-section overhead ----
    md += ["\n## Analytic model — speedup ≈ L / (1 + c)\n",
           "c is the speculator's per-step overhead in units of one target"
           " forward, fitted as median(L/speedup − 1) over a section's cells."
           " If c is really a per-speculator constant, the model predicts every"
           " cell's wall-clock from its (already-measured) acceptance alone.\n",
           "| section | fitted c | median |err| | max |err| | cells |",
           "|---|--:|--:|--:|--:|"]
    for sec in dict.fromkeys(s for s, *_ in table):
        pts = [(L, sp) for s_, d, v, cells in table if s_ == sec
               for (tok, sp, L) in cells.values() if sp > 0 and L > 0]
        if len(pts) < 2:
            continue
        cs = sorted(L / sp - 1 for L, sp in pts)
        c = cs[len(cs) // 2]
        errs = sorted(abs(L / (1 + c) - sp) / sp for L, sp in pts)
        md.append(f"| {sec} | {c:.3f} | {errs[len(errs)//2]*100:.1f}% "
                  f"| {errs[-1]*100:.1f}% | {len(pts)} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.csv").write_text("\n".join(csv) + "\n")
    (OUT / "report.md").write_text("\n".join(md) + "\n")

    # ---- chart: grouped speedup bars per domain, one panel per section ----
    secs = []
    for sec, dom, v, cells in table:
        if sec not in secs:
            secs.append(sec)
    fig, axes = plt.subplots(1, len(secs), figsize=(4.3 * len(secs), 4.4),
                             sharey=False)
    if len(secs) == 1:
        axes = [axes]
    for ax, sec in zip(axes, secs):
        entries = [(d, c) for s, d, v, c in table if s == sec]
        xs = range(len(entries))
        w = 0.8 / len(VARIANTS)
        for i, var in enumerate(VARIANTS):
            ys = [c[var][1] if var in c else 0.0 for _, c in entries]
            ax.bar([x + i * w for x in xs], ys, width=w * 0.92, color=COLOR[var],
                   label=var, zorder=3)
        ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=2)
        ax.set_xticks([x + 0.4 - w / 2 for x in xs])
        ax.set_xticklabels([d for d, _ in entries], rotation=20, ha="right",
                           color=INK, fontsize=9)
        ax.set_title(sec, color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AXIS)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("wall-clock speedup vs target-only (×)", color=INK2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    (OUT / "charts").mkdir(exist_ok=True)
    fig.savefig(OUT / "charts" / "speedup.png", dpi=170)
    print(f"[ok] -> {OUT}/report.md, comparison.csv, charts/speedup.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
