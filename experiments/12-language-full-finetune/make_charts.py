#!/usr/bin/env python3
"""Charts for the language full fine-tune comparison.

Reads results/summary.json and writes:

    results/charts/full_finetune_vs_lora.png
    results/charts/full_finetune_gain.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUMMARY = RESULTS / "summary.json"
CHARTS = RESULTS / "charts"

VARIANTS = ["base", "own", "full"]
LABELS = {
    "base": "base DFlash",
    "own": "own-language LoRA",
    "full": "full fine-tune",
}
COLORS = {
    "base": "#898781",
    "own": "#2a78d6",
    "full": "#e07800",
}

INK = "#111111"
INK2 = "#52514e"
MUTED = "#85837d"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.facecolor": SURF,
})


def load_summary() -> tuple[list[str], dict]:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY}")
    data = json.loads(SUMMARY.read_text())
    return data["langs"], data["rows"]


def pct(rows: dict, lang: str, variant: str) -> float:
    return 100.0 * rows[lang][variant]["acceptance_rate"]


def style_axis(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def acceptance_chart(langs: list[str], rows: dict, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    y = list(range(len(langs)))
    h = 0.22
    offsets = {"base": -h, "own": 0.0, "full": h}

    xmax = 0.0
    for variant in VARIANTS:
        vals = [pct(rows, lang, variant) for lang in langs]
        xmax = max(xmax, max(vals))
        bars = ax.barh(
            [i + offsets[variant] for i in y],
            vals,
            height=h * 0.86,
            color=COLORS[variant],
            label=LABELS[variant],
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                value + 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK2,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(langs, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax + 1.0)
    ax.set_xlabel("acceptance rate (%)", color=INK2, fontsize=10)
    style_axis(ax)

    ax.set_title(
        "Base vs LoRA vs full fine tune",
        color=INK,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    fig.savefig(CHARTS / f"full_finetune_vs_lora{suffix}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def gain_chart(langs: list[str], rows: dict, suffix: str = "", subtitle: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = list(range(len(langs)))
    w = 0.34

    for variant, offset in [("own", -w / 2), ("full", w / 2)]:
        vals = [pct(rows, lang, variant) - pct(rows, lang, "base") for lang in langs]
        bars = ax.bar(
            [i + offset for i in x],
            vals,
            width=w * 0.88,
            color=COLORS[variant],
            label=LABELS[variant],
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.04,
                f"+{value:.2f}pp",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK2,
            )

    ax.axhline(0, color=AXIS, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, color=MUTED)
    ax.set_ylabel("acceptance gain vs base (percentage points)", color=INK2, fontsize=10)
    ax.set_ylim(0, 2.55)
    style_axis(ax)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.grid(axis="x", visible=False)

    ax.set_title(
        "Full fine-tuning does not recover the LoRA gain",
        color=INK,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax.text(
        0,
        1.01,
        subtitle or "Same five weak language lanes, same held-out benchmark; higher is better",
        transform=ax.transAxes,
        color=INK2,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    fig.savefig(CHARTS / f"full_finetune_gain{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    langs, rows = load_summary()
    acceptance_chart(langs, rows)
    gain_chart(langs, rows)
    clean = [lang for lang in langs if lang != "Hebrew"]
    acceptance_chart(clean, rows, suffix="_mintrain1000")
    gain_chart(
        clean,
        rows,
        suffix="_mintrain1000",
        subtitle="Four weak language lanes with 1,000 train prompts; higher is better",
    )
    print(f"charts -> {CHARTS}")


if __name__ == "__main__":
    main()
