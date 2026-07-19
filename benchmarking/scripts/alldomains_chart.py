#!/usr/bin/env python3
"""
Master chart — every source/domain on one x-axis, both metrics, both speculators.

    python alldomains_chart.py ../results/*_by_category.csv

x-axis: one tick per (source, domain) — downloaded/*, then synthetic/*, then wild/*.
Two stacked panels sharing that axis:
    top    = mean acceptance rate
    bottom = mean length accepted
Each panel overlays DFlash vs EAGLE3. Source bands are shaded/labelled.

Writes <results>/charts/alldomains.png. CPU only.
"""
import csv
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

METHOD_COLORS = {"dflash": "#4E79A7", "eagle3": "#E15759"}
SOURCE_ORDER = ["downloaded", "synthetic", "wild"]
_METRICS = [
    ("acceptance_rate_pooled", "Mean acceptance rate", True),
    ("mean_accept_length", "Mean length accepted (tokens/pass)", False),
]


def load(path):
    stem = Path(path).stem.replace("_by_category", "")
    method, _, source = stem.partition("_")
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["category"]] = {m: float(r.get(m, 0) or 0) for m, _, _ in _METRICS}
    return method, source, rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python alldomains_chart.py <run>_by_category.csv [...]")
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not installed — pip install matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # data[source][domain][method] = {metric: value}
    data = defaultdict(lambda: defaultdict(dict))
    methods = []
    for path in sys.argv[1:]:
        method, source, rows = load(path)
        if method not in methods:
            methods.append(method)
        for domain, vals in rows.items():
            data[source][domain][method] = vals
    methods = [m for m in ("dflash", "eagle3") if m in methods] + \
              [m for m in methods if m not in ("dflash", "eagle3")]
    sources = [s for s in SOURCE_ORDER if s in data] + \
              [s for s in data if s not in SOURCE_ORDER]

    # x order: source by source, domains sorted within each
    x_keys, x_labels, bands = [], [], []  # bands: (start, end, source)
    for s in sources:
        start = len(x_keys)
        for d in sorted(data[s]):
            x_keys.append((s, d))
            x_labels.append(f"{s}/{d}")
        bands.append((start, len(x_keys), s))
    n = len(x_keys)
    xs = range(n)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.22 * n), 9), sharex=True)
    for ax, (col, ylabel, is_pct) in zip(axes, _METRICS):
        # source bands
        for i, (a, b, s) in enumerate(bands):
            if i % 2:
                ax.axvspan(a - 0.5, b - 0.5, color="gray", alpha=0.07, zorder=0)
        for method in methods:
            ys = [data[s][d].get(method, {}).get(col, 0.0) for (s, d) in x_keys]
            ax.plot(list(xs), ys, "-o", ms=3, lw=1, color=METHOD_COLORS.get(method),
                    label=method)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        if is_pct:
            ax.set_ylim(bottom=0)
        # source labels at top of the upper panel
        if col == _METRICS[0][0]:
            for a, b, s in bands:
                ax.text((a + b - 1) / 2, ax.get_ylim()[1], s, ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    axes[0].set_title("DFlash vs EAGLE3 — acceptance rate & mean length accepted, "
                      "every source/domain", fontsize=13, fontweight="bold")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(x_labels, rotation=90, fontsize=5)
    axes[1].set_xlim(-0.5, n - 0.5)
    fig.tight_layout()

    out_dir = Path(sys.argv[1]).resolve().parent / "charts"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "alldomains.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[wrote] {out}  ({n} source/domain columns)")


if __name__ == "__main__":
    main()
