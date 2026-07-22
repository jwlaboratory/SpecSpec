#!/usr/bin/env python3
"""Charts for exp 10.

Run after pulling/aggregating results:

    python3 experiments/10-english-subdomains/make_charts.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TABLE_DIR = RESULTS / "english_subdomains"
if not (TABLE_DIR / "english_subdomains_comparison.csv").exists():
    TABLE_DIR = RESULTS
CSV = TABLE_DIR / "english_subdomains_comparison.csv"
CHARTS = TABLE_DIR / "charts"

DOMAINS = [
    "code_python",
    "code_sql",
    "ood_legal",
    "ood_medical",
    "ood_financial",
    "task_math_reasoning",
    "task_summarization",
]
VARIANTS = ["base", "own", "combined", "combined_equal"]
LABELS = {
    "base": "base",
    "own": "own",
    "combined": "combined",
    "combined_equal": "equal-budget combined",
}
COLORS = {
    "base": "#6b7280",
    "own": "#2563eb",
    "combined": "#059669",
    "combined_equal": "#d97706",
}


def load_rows():
    rows = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            rows[(row["domain"], row["variant"])] = {
                "accept": float(row["acceptance_rate_pooled"]),
                "mean_len": float(row["mean_accept_length"]),
                "speedup": float(row["speedup"]),
            }
    return rows


def pretty(domain: str) -> str:
    return domain.replace("task_", "").replace("ood_", "").replace("code_", "").replace("_", "\n")


def acceptance_bars(rows):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = list(range(len(DOMAINS)))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for variant, off in zip(VARIANTS, offsets):
        vals = [rows.get((d, variant), {}).get("accept", 0) * 100 for d in DOMAINS]
        ax.bar([i + off for i in x], vals, width=width, label=LABELS[variant],
               color=COLORS[variant])
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("English subdomains: base vs own vs combined LoRA")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in DOMAINS])
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "acceptance_bars.png", dpi=180)
    plt.close(fig)


def delta_bars(rows):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = list(range(len(DOMAINS)))
    width = 0.25
    variants = ["own", "combined", "combined_equal"]
    offsets = [-width, 0, width]
    for variant, off in zip(variants, offsets):
        vals = []
        for d in DOMAINS:
            b = rows.get((d, "base"), {}).get("accept", 0)
            v = rows.get((d, variant), {}).get("accept", 0)
            vals.append((v - b) * 100)
        ax.bar([i + off for i in x], vals, width=width, label=LABELS[variant],
               color=COLORS[variant])
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("Acceptance gain vs base (pp)")
    ax.set_title("Where does specialization help inside English?")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in DOMAINS])
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "delta_bars.png", dpi=180)
    plt.close(fig)


def gain_vs_base(rows):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for variant in ["own", "combined", "combined_equal"]:
        xs, ys = [], []
        for d in DOMAINS:
            b = rows.get((d, "base"), {}).get("accept")
            v = rows.get((d, variant), {}).get("accept")
            if b is None or v is None:
                continue
            xs.append(b * 100)
            ys.append((v - b) * 100)
        ax.scatter(xs, ys, s=80, label=LABELS[variant], color=COLORS[variant], alpha=0.9)
    for d in DOMAINS:
        b = rows.get((d, "base"), {}).get("accept")
        o = rows.get((d, "own"), {}).get("accept")
        if b is not None and o is not None:
            ax.annotate(pretty(d).replace("\n", " "), (b * 100, (o - b) * 100),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Base acceptance (%)")
    ax.set_ylabel("Acceptance gain vs base (pp)")
    ax.set_title("Does headroom predict LoRA gain inside English?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "gain_vs_base.png", dpi=180)
    plt.close(fig)


def main():
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")
    CHARTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    acceptance_bars(rows)
    delta_bars(rows)
    gain_vs_base(rows)
    print(f"Wrote charts to {CHARTS}")


if __name__ == "__main__":
    main()
