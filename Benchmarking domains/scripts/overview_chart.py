#!/usr/bin/env python3
"""
All-sources overview — one chart, every data source, both speculators.

    python overview_chart.py ../results/*_by_category.csv

Reads every `<method>_<source>_by_category.csv` run, computes each run's overall
score (mean over its domains) for acceptance rate and mean accept length, and plots
grouped bars: x-axis = data source (synthetic / wild / downloaded), one bar per
speculator (dflash / eagle3). Answers "how do the two speculators score across the
different prompt sources?" at a glance.

Writes <results>/charts/overview_acceptance.png and overview_mean_len.png.
CPU only.
"""
import csv
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

METHOD_COLORS = {"dflash": "#4E79A7", "eagle3": "#E15759"}
SOURCE_ORDER = ["synthetic", "wild", "downloaded"]

# metric -> (title, y-label, output stem, is_percent)
_METRICS = OrderedDict([
    ("acceptance_rate_pooled",
     ("Acceptance rate by source & speculator", "Mean pooled acceptance rate",
      "overview_acceptance", True)),
    ("mean_accept_length",
     ("Mean accept length by source & speculator", "Mean tokens per target pass",
      "overview_mean_len", False)),
])


def run_means(path):
    """(method, source, {metric: mean over domains})."""
    stem = Path(path).stem.replace("_by_category", "")
    method, _, source = stem.partition("_")
    sums = defaultdict(float)
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            n += 1
            for col in _METRICS:
                try:
                    sums[col] += float(row[col])
                except (TypeError, ValueError, KeyError):
                    pass
    means = {col: (sums[col] / n if n else 0.0) for col in _METRICS}
    return method, source, means


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python overview_chart.py <run>_by_category.csv [...]")
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not installed — pip install matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # scores[metric][source][method] = value
    scores = {col: defaultdict(dict) for col in _METRICS}
    methods, sources = [], []
    for path in sys.argv[1:]:
        method, source, means = run_means(path)
        if method not in methods:
            methods.append(method)
        if source not in sources:
            sources.append(source)
        for col in _METRICS:
            scores[col][source][method] = means[col]

    methods = [m for m in ("dflash", "eagle3") if m in methods] + \
              [m for m in methods if m not in ("dflash", "eagle3")]
    sources = [s for s in SOURCE_ORDER if s in sources] + \
              [s for s in sources if s not in SOURCE_ORDER]

    out_dir = Path(sys.argv[1]).resolve().parent / "charts"
    out_dir.mkdir(exist_ok=True)

    for col, (title, ylabel, stem, is_pct) in _METRICS.items():
        n_m = len(methods)
        x = range(len(sources))
        bar_w = 0.8 / max(1, n_m)
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(sources) + 2), 4.5))
        for mi, method in enumerate(methods):
            offs = [i + (mi - (n_m - 1) / 2) * bar_w for i in x]
            vals = [scores[col][s].get(method, 0.0) for s in sources]
            ax.bar(offs, vals, width=bar_w, label=method,
                   color=METHOD_COLORS.get(method, f"C{mi}"))
            for xo, v in zip(offs, vals):
                ax.text(xo, v, f"{v*100:.0f}%" if is_pct else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sources)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = out_dir / f"{stem}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[wrote] {out}")

    print("\nOverall means (acceptance / mean accept length):")
    for s in sources:
        for m in methods:
            a = scores["acceptance_rate_pooled"][s].get(m)
            ln = scores["mean_accept_length"][s].get(m)
            if a is not None:
                print(f"  {s:11s} {m:7s}  {a*100:5.1f}%   {ln:5.2f} tok/pass")


if __name__ == "__main__":
    main()
