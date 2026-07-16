#!/usr/bin/env python3
"""
Turn an aggregated benchmark CSV into charts for the results folder.

    python make_charts.py results/dflash_bench_by_category.csv

Reads the per-domain CSV that aggregate.py writes and produces PNGs under
<results>/charts/:

    acceptance_by_domain.png   per-domain pooled acceptance rate (ranked)
    mean_len_by_domain.png     per-domain mean accept length
    speedup_by_domain.png      per-domain speedup vs target-only (if baseline ran)
    group_summary.png          mean acceptance rate per domain group
    acceptance_hist.png        distribution of per-domain acceptance rates

Bars are coloured by domain group (languages / coding / tasks / ood) so the
in-distribution vs out-of-distribution split is visible at a glance. CPU only;
no GPU or model needed — run it locally after pulling results back.
"""
import csv
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

# Group colours (colour-blind-safe, distinct in print). One per domain family.
GROUP_COLORS = OrderedDict([
    ("languages", "#4E79A7"),  # blue
    ("coding",    "#59A14F"),  # green
    ("tasks",     "#E15759"),  # red
    ("ood",       "#F28E2B"),  # orange
])
_PREFIX = {"lang_": "languages", "code_": "coding", "task_": "tasks", "ood_": "ood"}


def group_of(domain: str) -> str:
    for pref, g in _PREFIX.items():
        if domain.startswith(pref):
            return g
    return "ood"


def load_rows(csv_path: Path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _f(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def barh(ax, labels, values, colors, title, xlabel, pct=False):
    y = range(len(labels))
    ax.barh(list(y), values, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v*100:.0f}%" if pct else f" {v:.2f}",
                va="center", fontsize=6)


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]


def ranked_chart(rows, value_key, title, xlabel, out, pct=False):
    import matplotlib.pyplot as plt
    data = [(r["category"], _f(r, value_key)) for r in rows]
    data = [(c, v) for c, v in data if v is not None]
    if not data:
        return None
    data.sort(key=lambda t: t[1], reverse=True)
    labels = [c for c, _ in data]
    values = [v for _, v in data]
    colors = [GROUP_COLORS[group_of(c)] for c in labels]
    h = max(3.0, 0.22 * len(labels))
    fig, ax = plt.subplots(figsize=(9, h))
    barh(ax, labels, values, colors, title, xlabel, pct=pct)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def group_summary(rows, out):
    import matplotlib.pyplot as plt
    by_group = defaultdict(list)
    for r in rows:
        v = _f(r, "acceptance_rate_pooled")
        if v is not None:
            by_group[group_of(r["category"])].append(v)
    groups = [g for g in GROUP_COLORS if g in by_group]
    if not groups:
        return None
    means = [sum(by_group[g]) / len(by_group[g]) for g in groups]
    colors = [GROUP_COLORS[g] for g in groups]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(groups, means, color=colors)
    ax.set_ylabel("Mean acceptance rate")
    ax.set_title("Acceptance rate by domain group", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m*100:.0f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def acceptance_hist(rows, out):
    import matplotlib.pyplot as plt
    vals = [_f(r, "acceptance_rate_pooled") for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([v * 100 for v in vals], bins=20, color="#4E79A7", edgecolor="white")
    ax.set_xlabel("Per-domain acceptance rate (%)")
    ax.set_ylabel("Number of domains")
    ax.set_title("Distribution of per-domain acceptance", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python make_charts.py results/<run>_by_category.csv")
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not installed — pip install matplotlib")
    matplotlib.use("Agg")  # headless; works on GPU boxes with no display

    csv_path = Path(sys.argv[1]).resolve()
    if not csv_path.exists():
        raise SystemExit(f"not found: {csv_path}")
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    charts_dir = csv_path.parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    stem = csv_path.stem.replace("_by_category", "")

    written = []
    written.append(ranked_chart(
        rows, "acceptance_rate_pooled",
        "Acceptance rate by domain (drafter vs target)",
        "Pooled acceptance rate", charts_dir / f"{stem}_acceptance_by_domain.png", pct=True))
    written.append(ranked_chart(
        rows, "mean_accept_length", "Mean accept length by domain",
        "Tokens committed per target pass", charts_dir / f"{stem}_mean_len_by_domain.png"))
    if any(_f(r, "speedup") for r in rows):
        written.append(ranked_chart(
            rows, "speedup", "Speedup by domain (spec vs target-only greedy)",
            "Speedup (x)", charts_dir / f"{stem}_speedup_by_domain.png"))
    written.append(group_summary(rows, charts_dir / f"{stem}_group_summary.png"))
    written.append(acceptance_hist(rows, charts_dir / f"{stem}_acceptance_hist.png"))

    for w in written:
        if w:
            print(f"[wrote] {w}")


if __name__ == "__main__":
    main()
