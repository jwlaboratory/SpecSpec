#!/usr/bin/env python3
"""Charts for the batched-LoRA serving experiment.

    python experiments/09-batched-lora-serving/make_charts.py

Reads results/raw/{hf_track,vllm_*}.jsonl and writes results/charts/:
    throughput.png   decode tok/s vs batch size, per serving mode (HF | vLLM panels)
    overhead.png     % overhead vs the SAME-container base, per batch size
plus results/report.md with the numbers.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

STYLE = {  # variant -> (label, color, paired-base variant)
    "A_base":            ("A · base",                 "#898781", None),
    "B_unmerged":        ("B · unmerged wrappers",    "#c23b21", "A_base"),
    "C_merged":          ("C · merged",               "#eb6834", "A_base"),
    "D_punica_1_nolora": ("engine w/ LoRA, idle",     "#b9b7ae", None),
    "D_punica_1":        ("D · punica ×1 adapter",    "#2a78d6", "D_punica_1_nolora"),
    "E_punica_50_nolora": ("engine w/ LoRA, idle",    "#b9b7ae", None),
    "E_punica_50":       ("E · punica ×50 adapters",  "#134a86", "E_punica_50_nolora"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def load():
    data = defaultdict(dict)  # (track, variant) -> {bs: mean tok/s}
    for f in RAW.glob("*.jsonl"):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["track"], r["variant"])
            data[key].setdefault(r["batch_size"], []).append(r["tok_s"])
    return {k: {bs: sum(v) / len(v) for bs, v in d.items()}
            for k, d in data.items()}


def style_ax(ax):
    ax.grid(color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.set_xscale("log", base=2)


def main():
    data = load()
    tracks = [("hf", "HF eager (exp 06's serving stack)"),
              ("vllm", "vLLM (punica multi-LoRA kernels)")]

    # ---- throughput ----
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, (track, title) in zip(axes, tracks):
        for (t, var), series in sorted(data.items()):
            if t != track:
                continue
            label, color, _ = STYLE.get(var, (var, MUTED, None))
            xs = sorted(series)
            ax.plot(xs, [series[b] for b in xs], marker="o", ms=4, lw=1.8,
                    color=color, label=label, zorder=3)
        style_ax(ax)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("batch size", color=INK2)
        ax.set_xticks(sorted({b for s in data.values() for b in s}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("decode throughput (tok/s, total)", color=INK2)
    fig.suptitle("Qwen3-0.6B decode throughput by LoRA serving mode",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "charts" / "throughput.png", dpi=170)
    plt.close(fig)

    # ---- overhead vs paired base ----
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for (t, var), series in sorted(data.items()):
        label, color, base_var = STYLE.get(var, (var, MUTED, None))
        if base_var is None:
            continue
        base = data.get((t, base_var))
        if not base:
            continue
        xs = sorted(set(series) & set(base))
        ys = [100 * (base[b] / series[b] - 1) for b in xs]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color,
                label=f"{label}  (vs paired base)", zorder=3)
    ax.axhline(0, color=INK2, lw=1, zorder=2)
    style_ax(ax)
    ax.set_xticks(sorted({b for s in data.values() for b in s}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("batch size", color=INK2)
    ax.set_ylabel("serving overhead vs base (%)", color=INK2)
    ax.set_title("LoRA serving overhead by mode and batch size "
                 "(same-container ratios)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "overhead.png", dpi=170)
    plt.close(fig)

    # ---- report ----
    md = ["# Batched-LoRA serving — throughput and overhead by batch size\n",
          "Qwen3-0.6B, greedy, 128 forced new tokens/seq, zero-delta r16 q/k/v/o"
          " adapters (identical outputs across modes), H200. Mean of "
          "3 timed batches after warmup. Overheads computed against the"
          " same-container base only.\n"]
    for track, title in tracks:
        md += [f"\n## {title}\n"]
        variants = [v for (t, v) in data if t == track]
        bss = sorted({b for (t, v), s in data.items() if t == track for b in s})
        md += ["| variant | " + " | ".join(f"bs={b}" for b in bss) + " |",
               "|---|" + "--:|" * len(bss)]
        for var in sorted(variants):
            label, _, base_var = STYLE.get(var, (var, MUTED, None))
            s = data[(track, var)]
            cells = []
            for b in bss:
                v = s.get(b)
                if v is None:
                    cells.append("—")
                    continue
                txt = f"{v:.0f}"
                if base_var and (track, base_var) in data:
                    bb = data[(track, base_var)].get(b)
                    if bb:
                        txt += f" ({100*(bb/v-1):+.1f}%)"
                cells.append(txt)
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print(f"[ok] -> {OUT}/charts/throughput.png, overhead.png, report.md")


if __name__ == "__main__":
    main()
