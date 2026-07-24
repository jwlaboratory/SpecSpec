#!/usr/bin/env python3
"""Charts for exp13: net spec-decode speedup vs batch size.

Reads results/summary.json, writes results/charts/*.png:
    speedup_vs_batch.png   net speedup (dflash tok/s / target tok/s) vs batch,
                           with breakeven line + acceptance overlay
    throughput_vs_batch.png  target-only vs dflash tok/s vs batch
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_TARGET = "#898781"
C_DFLASH = "#2a78d6"
C_ACC = "#1baf7a"
C_WARN = "#d1495b"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_axisbelow(True)


def load():
    d = json.load(open(HERE / "results" / "summary.json"))
    by_b = {r["batch_size"]: r for r in d["rows"]}
    bs = sorted(by_b)
    def pick(mode, b):
        return by_b[b].get(mode)   # nested {tok_s, acceptance_rate, speedup_vs_target}
    return bs, pick


def speedup_vs_batch():
    bs, pick = load()
    xs = list(range(len(bs)))
    spd = [pick("dflash", b)["speedup_vs_target"] for b in bs]
    acc = [pick("dflash", b)["acceptance_rate"] * 100 for b in bs]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(1.0, color=C_WARN, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(len(bs) - 1, 1.02, "break-even (1.0×)", color=C_WARN, fontsize=8,
            ha="right", va="bottom")
    # shade the net-loss region
    ax.axhspan(0, 1.0, color=C_WARN, alpha=0.05, zorder=0)
    ax.plot(xs, spd, color=C_DFLASH, linewidth=2.2, marker="o", markersize=6,
            zorder=4, label="net speedup (spec / target)")
    for x, s in zip(xs, spd):
        ax.annotate(f"{s:.2f}×", (x, s), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8,
                    color=(C_WARN if s < 1 else INK2), fontweight="bold")
    ax.set_xticks(xs, [str(b) for b in bs])
    ax.set_ylim(0, max(spd) + 0.35)
    ax.set_xlabel("batch size (concurrent requests)", color=INK2, fontsize=9)
    ax.set_ylabel("net wall-clock speedup vs target-only", color=INK2, fontsize=9)
    ax.set_title("Spec-decode net speedup collapses with batch size",
                 color=INK, fontsize=13, loc="left", pad=22, weight="bold")
    ax.text(0, 1.045,
            "DFlash / Qwen3-8B, vLLM continuous batching, H200 — acceptance "
            "stays flat (~13%)",
            transform=ax.transAxes, color=INK2, fontsize=8.5)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(xs, acc, color=C_ACC, linewidth=1.4, marker="s", markersize=3.5,
             linestyle=":", zorder=3, label="acceptance rate")
    ax2.set_ylabel("acceptance rate (%)", color=C_ACC, fontsize=9)
    ax2.tick_params(axis="y", colors=C_ACC, labelsize=8.5)
    ax2.set_ylim(0, max(acc) * 2.2)

    lines = ax.get_lines()[1:] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "speedup_vs_batch.png", facecolor=SURFACE)
    plt.close(fig)
    print("->", OUT / "speedup_vs_batch.png")


def throughput_vs_batch():
    bs, pick = load()
    xs = list(range(len(bs)))
    tgt = [pick("target_only", b)["tok_s"] for b in bs]
    dfl = [pick("dflash", b)["tok_s"] for b in bs]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot(xs, tgt, color=C_TARGET, linewidth=2.2, marker="o", markersize=6,
            label="target-only (no speculation)")
    ax.plot(xs, dfl, color=C_DFLASH, linewidth=2.2, marker="o", markersize=6,
            label="DFlash spec-decode")
    # mark the crossover
    for i in range(1, len(bs)):
        if dfl[i] < tgt[i] and dfl[i - 1] >= tgt[i - 1]:
            ax.axvspan(i - 0.5, len(bs) - 0.5, color=C_WARN, alpha=0.05, zorder=0)
            ax.text((i - 0.5 + len(bs) - 1) / 2, max(tgt) * 0.5,
                    "spec-decode\nis slower here", color=C_WARN, fontsize=8.5,
                    ha="center", va="center")
            break
    ax.set_xticks(xs, [str(b) for b in bs])
    ax.set_xlabel("batch size (concurrent requests)", color=INK2, fontsize=9)
    ax.set_ylabel("decode throughput (tokens/s)", color=INK2, fontsize=9)
    ax.set_title("Throughput: target-only scales, spec-decode saturates",
                 color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "throughput_vs_batch.png", facecolor=SURFACE)
    plt.close(fig)
    print("->", OUT / "throughput_vs_batch.png")


def modes_vs_batch():
    """base / merged-combined / merged-own net speedup vs batch (modes_summary.json)."""
    d = json.load(open(HERE / "results" / "modes_summary.json"))
    lang = d["lang"]
    own_key = [m for m in d["modes"] if m.startswith("merged_own")][0]
    rows = sorted(d["rows"], key=lambda r: r["batch_size"])
    bs = [r["batch_size"] for r in rows]
    xs = list(range(len(bs)))
    base = [r["base"]["speedup_vs_target"] for r in rows]
    comb = [r["merged_combined"]["speedup_vs_target"] for r in rows]
    own = [r[own_key]["speedup_vs_target"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(1.0, color=C_WARN, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhspan(0, 1.0, color=C_WARN, alpha=0.05, zorder=0)
    ax.text(len(bs) - 1, 1.02, "break-even (1.0×)",
            color=C_WARN, fontsize=8, ha="right", va="bottom")
    for ys, c, lab, mk in [(own, C_DFLASH, "merged own (Swedish specialist)", "o"),
                           (comb, C_ACC, "merged combined", "s"),
                           (base, C_TARGET, "base DFlash (no adapter)", "^")]:
        ax.plot(xs, ys, color=c, linewidth=2.1, marker=mk, markersize=5.5,
                label=lab, zorder=4)
    ax.set_xticks(xs, [str(b) for b in bs])
    ax.set_ylim(0, max(own) + 0.3)
    ax.set_xlabel("batch size (concurrent requests)", color=INK2, fontsize=9)
    ax.set_ylabel("net wall-clock speedup vs target-only", color=INK2, fontsize=9)
    ax.set_title(f"Serving speedup vs batch size ({lang})",
                 color=INK, fontsize=13, loc="left", pad=20, weight="bold")
    ax.text(0, 1.04, "vLLM continuous batching, H200", transform=ax.transAxes,
            color=INK2, fontsize=8.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "modes_vs_batch.png", facecolor=SURFACE)
    plt.close(fig)
    print("->", OUT / "modes_vs_batch.png")


def mixed_vs_batch():
    """merged-combined vs base net speedup on a 16-language mixed stream, vs batch."""
    d = json.load(open(HERE / "results" / "mixed_summary.json"))
    rows = sorted(d["rows"], key=lambda r: r["batch_size"])
    bs = [r["batch_size"] for r in rows]
    xs = list(range(len(bs)))
    base = [r["base"]["speedup_vs_target"] for r in rows]
    comb = [r["merged_combined"]["speedup_vs_target"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(1.0, color=C_WARN, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhspan(0, 1.0, color=C_WARN, alpha=0.05, zorder=0)
    ax.text(len(bs) - 1, 1.02, "break-even (1.0×)", color=C_WARN, fontsize=8,
            ha="right", va="bottom")
    for ys, c, lab, mk in [(comb, C_ACC, "merged combined (one adapter, all langs)", "s"),
                           (base, C_TARGET, "base DFlash (no adapter)", "^")]:
        ax.plot(xs, ys, color=c, linewidth=2.1, marker=mk, markersize=5.5,
                label=lab, zorder=4)
    ax.set_xticks(xs, [str(b) for b in bs])
    ax.set_ylim(0, max(comb) + 0.3)
    ax.set_xlabel("batch size (concurrent requests)", color=INK2, fontsize=9)
    ax.set_ylabel("net wall-clock speedup vs target-only", color=INK2, fontsize=9)
    ax.set_title("Serving speedup vs batch size (mixed 16-language stream)",
                 color=INK, fontsize=13, loc="left", pad=20, weight="bold")
    ax.text(0, 1.04, "vLLM continuous batching, H200", transform=ax.transAxes,
            color=INK2, fontsize=8.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "mixed_vs_batch.png", facecolor=SURFACE)
    plt.close(fig)
    print("->", OUT / "mixed_vs_batch.png")


def bs1_workload_bars():
    """Batch-size-1 net speedup vs target-only, grouped by workload.

    Bars: base model (1.00x reference), then the Swedish workload
    (base DFlash vs Swedish specialist) and a random/mixed workload
    (base DFlash vs merged combined adapter). The specialist/merged bars
    are annotated with their within-group speedup over base DFlash.
    """
    # (label, value x, value-string, color, within-group gain vs group base)
    bars = [
        ("base model",         1.000, "1.00×",  C_TARGET, None),
        ("base DFlash",        1.500, "1.50×",  C_TARGET, None),
        ("specialized DFlash", 1.730, "1.73×",  C_DFLASH, 0.153),
        ("base DFlash",        1.418, "1.418×", C_TARGET, None),
        ("merged adapter",     1.495, "1.495×", C_ACC,    0.054),
    ]
    # x positions with a gap between workload groups
    xpos = [0.0, 1.3, 2.15, 3.45, 4.30]
    width = 0.72

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    fig.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(1.0, color=C_WARN, linewidth=1.2, linestyle="--", zorder=2)

    for (lab, v, vstr, c, gain), x in zip(bars, xpos):
        ax.bar(x, v, width, color=c, zorder=3)
        # x value just above the bar
        ax.annotate(vstr, (x, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom",
                    fontsize=9, color=INK2)
        # within-group speedup headline on the treatment bars
        if gain is not None:
            ax.annotate(f"+{gain*100:.1f}%", (x, v), textcoords="offset points",
                        xytext=(0, 20), ha="center", va="bottom",
                        fontsize=12.5, fontweight="bold", color=c)
        ax.annotate(lab, (x, 0), textcoords="offset points",
                    xytext=(0, -7), ha="center", va="top",
                    fontsize=8.2, color=INK2)

    # group brackets / labels under the workload pairs
    groups = [("Swedish workload", (xpos[1] + xpos[2]) / 2),
              ("Random workload", (xpos[3] + xpos[4]) / 2)]
    for name, xc in groups:
        ax.annotate(name, (xc, 0), textcoords="offset points",
                    xytext=(0, -24), ha="center", va="top",
                    fontsize=9, color=INK, fontweight="bold")

    ax.set_xticks([])
    ax.set_xlim(-0.7, xpos[-1] + 0.7)
    ax.set_ylim(0, max(v for _, v, *_ in bars) + 0.28)
    ax.set_ylabel("net wall-clock speedup vs target-only", color=INK2, fontsize=9)
    ax.set_title("Inference speeds with DFlash and specialized DFlash",
                 color=INK, fontsize=13, loc="left", pad=22, weight="bold")
    ax.text(0, 1.045, "DFlash / Qwen3-8B, vLLM, batch size 1",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(OUT / "bs1_workload_bars.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print("->", OUT / "bs1_workload_bars.png")


if __name__ == "__main__":
    speedup_vs_batch()
    throughput_vs_batch()
    modes_vs_batch()
    mixed_vs_batch()
    bs1_workload_bars()
