#!/usr/bin/env python3
"""
Compare speculators across domains — grouped bar chart from multiple runs.

    python compare_charts.py ../results/dflash_synthetic_by_category.csv \
                             ../results/eagle3_synthetic_by_category.csv

Reads several `<run>_by_category.csv` files (from aggregate.py or benchmark_vllm.py)
and overlays their per-domain acceptance rate as grouped bars — one bar per
speculator per domain — so you can see, across the whole domain distribution,
where each speculator tracks the target well and where it falls off.

Writes <results>/charts/compare_acceptance.png (+ a mean-per-speculator summary).
CPU only; run locally after pulling the result CSVs.
"""
import csv
import sys
from collections import OrderedDict
from pathlib import Path

# distinct, print-safe colors per run (speculator)
_RUN_COLORS = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1", "#76B7B2"]
_PREFIX = {"lang_": "languages", "code_": "coding", "task_": "tasks", "ood_": "ood"}


def group_of(domain):
    for pref, g in _PREFIX.items():
        if domain.startswith(pref):
            return g
    return "ood"


def load_csv(path):
    """Return (run_label, {domain: {col: value}})."""
    p = Path(path)
    label = p.stem.replace("_by_category", "")
    data = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            vals = {}
            for col in ("acceptance_rate_pooled", "mean_accept_length"):
                try:
                    vals[col] = float(row[col])
                except (TypeError, ValueError, KeyError):
                    vals[col] = 0.0
            data[row["category"]] = vals
    return label, data


# metric column -> (chart title, x-axis label, filename suffix, is_percent)
# mean_accept_length is framed as SPEEDUP: it's tokens committed per 8B target
# forward pass, i.e. the factor by which the speculator cuts target passes vs
# target-only greedy (which commits 1 token/pass = 1.0x). Same target for both
# speculators, so it's directly comparable; batch/hardware independent.
_METRICS = OrderedDict([
    ("acceptance_rate_pooled",
     ("Speculator acceptance rate by domain", "Pooled acceptance rate", "acceptance", True)),
    ("mean_accept_length",
     ("Speculator speedup by domain (8B target passes saved)",
      "Speedup ×  (tokens per target pass; target-only = 1.0×)", "speedup", False)),
])


def _source_tag(labels):
    """If all run labels (<method>_<source>) share one source, return it, else 'all'."""
    sources = {lab.split("_", 1)[1] for lab in labels if "_" in lab}
    return sources.pop() if len(sources) == 1 else "all"


def _plot_metric(plt, runs, all_domains, col, out_dir, tag):
    title, xlabel, suffix, is_pct = _METRICS[col]
    n_runs = len(runs)
    y = range(len(all_domains))
    bar_h = 0.8 / n_runs
    fig_h = max(4.0, 0.30 * len(all_domains))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    for ri, (label, data) in enumerate(runs.items()):
        offs = [(i + (ri - (n_runs - 1) / 2) * bar_h) for i in y]
        vals = [data.get(c, {}).get(col, 0.0) for c in all_domains]
        ax.barh(offs, vals, height=bar_h, label=label,
                color=_RUN_COLORS[ri % len(_RUN_COLORS)])
    if col == "mean_accept_length":  # target-only baseline reference
        ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.7, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(all_domains, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.set_size_inches(10, max(4.0, 0.30 * len(all_domains)))
    fig.tight_layout()
    out = out_dir / f"compare_{tag}_{suffix}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {out}")


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python compare_charts.py <runA_by_category.csv> "
                         "<runB_by_category.csv> [more...]")
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not installed — pip install matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = OrderedDict()
    for path in sys.argv[1:]:
        label, data = load_csv(path)
        runs[label] = data
    if len(runs) < 2:
        raise SystemExit("need at least two distinct runs to compare")

    # domains present in ANY run, ordered by group then by first run's acceptance
    all_domains = sorted(
        set().union(*[set(d) for d in runs.values()]),
        key=lambda c: (list(_PREFIX.values()).index(group_of(c)),
                       -next((d.get(c, {}).get("acceptance_rate_pooled", 0)
                              for d in runs.values()), 0)),
    )

    out_dir = Path(sys.argv[1]).resolve().parent / "charts"
    out_dir.mkdir(exist_ok=True)
    tag = _source_tag(runs.keys())
    for col in _METRICS:
        _plot_metric(plt, runs, all_domains, col, out_dir, tag)

    print("\nPer speculator over shared domains (acceptance / mean accept length):")
    shared = set.intersection(*[set(d) for d in runs.values()])
    for label, data in runs.items():
        acc = [data[c]["acceptance_rate_pooled"] for c in shared]
        mln = [data[c]["mean_accept_length"] for c in shared]
        a = sum(acc) / len(acc) if acc else 0.0
        m = sum(mln) / len(mln) if mln else 0.0
        print(f"  {label:24s} {a*100:5.1f}%   {m:5.2f} tok/pass  "
              f"(over {len(shared)} shared domains)")


if __name__ == "__main__":
    main()
