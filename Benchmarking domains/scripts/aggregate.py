"""
Aggregate a benchmark results JSONL into per-domain stats + a markdown report.

    python aggregate.py results/dflash_bench.jsonl

Writes next to the input:
    <name>_by_category.csv     one row per domain with all key metrics
    <name>_report.md           human-readable ranked report

The whole point: see WHERE the tiny drafter tracks the target well (high accept
rate / big speedup) and where it falls apart (low accept rate), across domains.
"""
import csv
import json
import statistics as st
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else 0.0


def agg_category(rows):
    has_base = any("speedup" in r for r in rows)
    out = {
        "n": len(rows),
        "acceptance_rate": mean([r["acceptance_rate"] for r in rows]),
        "mean_accept_length": mean([r["mean_accept_length"] for r in rows]),
        "spec_tok_s": mean([r["spec_tok_s"] for r in rows]),
        "gen_tokens": mean([r["num_generated_tokens"] for r in rows]),
        "forward_steps": mean([r["forward_steps"] for r in rows]),
        # aggregate acceptance over the whole domain (token-weighted, not per-prompt mean)
        "accepted_draft_tokens": sum(r["accepted_draft_tokens"] for r in rows),
        "proposed_draft_tokens": sum(r["proposed_draft_tokens"] for r in rows),
    }
    tot_prop = out["proposed_draft_tokens"]
    out["acceptance_rate_pooled"] = (out["accepted_draft_tokens"] / tot_prop) if tot_prop else 0.0
    if has_base:
        out["speedup"] = mean([r.get("speedup") for r in rows])
        out["baseline_tok_s"] = mean([r.get("baseline_tok_s") for r in rows])
        out["agreement_frac"] = mean([r.get("agreement_frac") for r in rows if "agreement_frac" in r])
        exact = [r.get("exact_match") for r in rows if "exact_match" in r]
        out["exact_match_rate"] = (sum(bool(m) for m in exact) / len(exact)) if exact else None
        # The real bug signal: divergences with a LARGE logit gap (not bf16 near-ties).
        out["n_suspicious"] = sum(1 for r in rows if r.get("suspicious"))
        gaps = [r.get("divergence_gap") for r in rows if r.get("divergence_gap") is not None]
        out["mean_divergence_gap"] = mean(gaps) if gaps else None
        out["max_divergence_gap"] = max(gaps) if gaps else None
    return out, has_base


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python aggregate.py results/<run>.jsonl")
    path = sys.argv[1]
    rows = load(path)
    if not rows:
        raise SystemExit(f"No rows in {path}")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    cats = {}
    has_base = False
    for c, rs in by_cat.items():
        cats[c], hb = agg_category(rs)
        has_base = has_base or hb

    base = path.rsplit(".", 1)[0]
    csv_path = base + "_by_category.csv"
    md_path = base + "_report.md"

    # ---- CSV ---- #
    cols = ["category", "n", "acceptance_rate_pooled", "acceptance_rate",
            "mean_accept_length", "forward_steps", "gen_tokens", "spec_tok_s"]
    if has_base:
        cols += ["baseline_tok_s", "speedup", "agreement_frac", "exact_match_rate",
                 "n_suspicious", "mean_divergence_gap", "max_divergence_gap"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in sorted(cats, key=lambda x: -cats[x]["acceptance_rate_pooled"]):
            d = cats[c]
            w.writerow([c] + [round(d.get(k), 4) if isinstance(d.get(k), float) else d.get(k)
                              for k in cols[1:]])

    # ---- overall ---- #
    tot_acc = sum(d["accepted_draft_tokens"] for d in cats.values())
    tot_prop = sum(d["proposed_draft_tokens"] for d in cats.values())
    overall_accept = tot_acc / tot_prop if tot_prop else 0.0
    overall = {
        "acceptance_rate_pooled": overall_accept,
        "mean_accept_length": mean([r["mean_accept_length"] for r in rows]),
        "spec_tok_s": mean([r["spec_tok_s"] for r in rows]),
    }
    if has_base:
        overall["speedup"] = mean([r.get("speedup") for r in rows if "speedup" in r])
        overall["agreement_frac"] = mean([r.get("agreement_frac") for r in rows if "agreement_frac" in r])
        ex = [r.get("exact_match") for r in rows if "exact_match" in r]
        overall["exact_match_rate"] = sum(bool(m) for m in ex) / len(ex) if ex else None
        overall["n_suspicious"] = sum(1 for r in rows if r.get("suspicious"))
        gaps = [r.get("divergence_gap") for r in rows if r.get("divergence_gap") is not None]
        overall["max_divergence_gap"] = max(gaps) if gaps else None

    # ---- markdown ---- #
    ranked = sorted(cats, key=lambda x: -cats[x]["acceptance_rate_pooled"])
    lines = []
    lines.append("# DFlash Speculative Decoding — Domain Benchmark\n")
    lines.append(f"Drafter `z-lab/Qwen3-8B-DFlash-b16` + target `Qwen/Qwen3-8B`  ·  "
                 f"{len(rows)} prompts across {len(cats)} domains.\n")
    lines.append("## Overall\n")
    lines.append(f"- **Acceptance rate (pooled):** {overall['acceptance_rate_pooled']*100:.1f}%  "
                 f"(fraction of the 15 draft tokens/step the target accepts)")
    lines.append(f"- **Mean accept length:** {overall['mean_accept_length']:.2f} tokens per target pass "
                 f"(max = block size 16)")
    lines.append(f"- **Spec throughput:** {overall['spec_tok_s']:.0f} tok/s")
    if has_base:
        lines.append(f"- **Mean speedup vs target-only greedy:** {overall['speedup']:.2f}×")
        if overall.get("agreement_frac") is not None:
            lines.append(f"- **Greedy agreement:** {overall['agreement_frac']*100:.1f}% "
                         f"(mean fraction of tokens matching sequential-greedy target before first divergence)")
            lines.append(f"- **Exact-match rate:** {overall['exact_match_rate']*100:.1f}% "
                         f"(bit-identical to HF greedy — expected to be <100% in bf16; see below)")
            ns = overall["n_suspicious"]
            mg = overall.get("max_divergence_gap")
            flag = ("✅ no inference bugs — every divergence is a bf16 near-tie"
                    if ns == 0 else f"⚠️ {ns} SUSPICIOUS large-gap divergences — investigate!")
            lines.append(f"- **Suspicious divergences:** {ns} "
                         f"(large-gap, i.e. not a rounding tie; max gap seen {mg:.2f} logits)  {flag}")
    lines.append("")
    if has_base:
        lines.append("> **Why exact-match < 100% is fine.** DFlash greedy is lossless in exact "
                     "arithmetic. In bf16 the target's logits are quantized (~0.25 per step near "
                     "magnitude 32), so when the top-2 tokens are within a rounding tie the "
                     "parallel block-verification path and sequential greedy pick differently — a "
                     "benign divergence at a coin-flip position (e.g. `,` vs `.`). A *real* bug "
                     "would flip tokens with a **large** logit gap; that's what **Suspicious "
                     "divergences** counts, and it should be 0.\n")

    def fmt_table(names):
        head = ["Domain", "Accept %", "Mean len", "Steps", "Gen tok", "Spec tok/s"]
        if has_base:
            head += ["Base tok/s", "Speedup", "Agree %", "Suspicious"]
        rowsl = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] * len(head)) + "|"]
        for c in names:
            d = cats[c]
            cells = [c, f"{d['acceptance_rate_pooled']*100:.1f}", f"{d['mean_accept_length']:.2f}",
                     f"{d['forward_steps']:.0f}", f"{d['gen_tokens']:.0f}", f"{d['spec_tok_s']:.0f}"]
            if has_base:
                ns = d.get("n_suspicious", 0)
                sus = "0 ✅" if ns == 0 else f"⚠️{ns}"
                cells += [f"{d.get('baseline_tok_s', 0):.0f}", f"{d.get('speedup', 0):.2f}×",
                          f"{d.get('agreement_frac', 0)*100:.0f}", sus]
            rowsl.append("| " + " | ".join(cells) + " |")
        return "\n".join(rowsl)

    lines.append("## All domains (ranked by acceptance rate)\n")
    lines.append(fmt_table(ranked))
    lines.append("")

    lines.append("## Best-tracked domains (drafter matches target well)\n")
    lines.append(fmt_table(ranked[:5]))
    lines.append("")
    lines.append("## Worst-tracked domains (drafter struggles)\n")
    lines.append(fmt_table(ranked[-5:]))
    lines.append("")

    if has_base and overall.get("n_suspicious", 0) > 0:
        lines.append("## ⚠️ Suspicious (large-gap) divergences\n")
        lines.append("These diverged from sequential-greedy at a position where the target's "
                     "top-2 logits were **far apart** (> rounding tie) — i.e. potentially a real "
                     "inference bug, not fp noise. Investigate these prompts:\n")
        for c in ranked:
            ns = cats[c].get("n_suspicious", 0)
            if ns:
                lines.append(f"- `{c}`: {ns}/{cats[c]['n']} prompts "
                             f"(max gap {cats[c].get('max_divergence_gap', 0):.2f} logits)")
        lines.append("")

    lines.append("## Metric definitions\n")
    lines.append("- **Accept %** — pooled acceptance rate = accepted draft tokens ÷ proposed draft "
                 "tokens (15 proposed per target pass, block size 16).")
    lines.append("- **Mean len** — average tokens committed per target forward pass "
                 "(accepted drafts + 1 bonus token). Higher = fewer target passes = faster.")
    lines.append("- **Speedup** — spec throughput ÷ target-only greedy throughput (same hardware).")
    lines.append("- **Agree %** — mean fraction of tokens matching the sequential-greedy target "
                 "before the first divergence.")
    lines.append("- **Suspicious** — divergences with a LARGE top-2 logit gap (not a bf16 rounding "
                 "tie). This is the actual inference-bug signal; it should be 0. Ordinary "
                 "near-tie divergences are expected and benign in bf16.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[wrote] {csv_path}")
    print(f"[wrote] {md_path}\n")
    # console summary
    print(f"Overall: accept {overall['acceptance_rate_pooled']*100:.1f}%  "
          f"mean_len {overall['mean_accept_length']:.2f}  ", end="")
    if has_base:
        print(f"speedup {overall['speedup']:.2f}x  agree {overall.get('agreement_frac', 0)*100:.0f}%  "
              f"suspicious {overall.get('n_suspicious')}")
    else:
        print()
    print("\nTop domains by acceptance:")
    for c in ranked[:5]:
        print(f"  {c:26s} {cats[c]['acceptance_rate_pooled']*100:5.1f}%  "
              f"len {cats[c]['mean_accept_length']:.2f}")
    print("Bottom domains by acceptance:")
    for c in ranked[-5:]:
        print(f"  {c:26s} {cats[c]['acceptance_rate_pooled']*100:5.1f}%  "
              f"len {cats[c]['mean_accept_length']:.2f}")


if __name__ == "__main__":
    main()
