#!/usr/bin/env python3
"""
Unified vLLM speculative-decoding benchmark — one runner for BOTH speculators.

Runs the SAME domain test prompts through vLLM with a chosen speculator against
the shared target `Qwen/Qwen3-8B`, batched, and records per-domain acceptance:

  --method dflash   z-lab/Qwen3-8B-DFlash-b16          (15 speculative tokens)
  --method eagle3   RedHatAI/Qwen3-8B-speculator.eagle3 (3 speculative tokens)

We only care about the batch-independent, generalizable numbers — **acceptance
rate** and **mean accept length** per domain — not speedup, so everything runs
batched (vLLM continuous batching) for speed. Writes the same
`<run>_by_category.csv` shape aggregate.py produces, so make_charts.py works
unchanged.

Needs a CUDA GPU and vLLM (nightly for DFlash support). Run via scripts/modal_run_vllm.py
on Modal, or directly on a GPU box.

Usage:
  python benchmark_vllm.py --method eagle3 --source synthetic --split test
  python benchmark_vllm.py --method dflash --datagen-dir ../data/wild --run-name dflash_wild
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

# Must be set BEFORE vllm is imported (maps to `--attention-backend flash_attn`).
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = _ROOT / "results"
DEFAULT_DATA_ROOT = _ROOT / "data"

TARGET_MODEL = "Qwen/Qwen3-8B"

# speculative_config dicts go straight to vllm.LLM(..., speculative_config=...).
SPECULATORS = {
    "eagle3": {
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
    },
    "dflash": {
        "method": "dflash",
        "model": "z-lab/Qwen3-8B-DFlash-b16",
        "num_speculative_tokens": 15,
    },
}


# --------------------------------------------------------------------------- #
# Datasets                                                                      #
# --------------------------------------------------------------------------- #
def load_datagen_prompts(datagen_dir, split):
    """Read data/<source>/<domain>/<split>.jsonl -> {domain: [prompt, ...]}."""
    base = Path(datagen_dir)
    if not base.exists():
        raise SystemExit(f"Data dir not found: {base}")
    data = OrderedDict()
    for domain_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        f = domain_dir / f"{split}.jsonl"
        if not f.exists() or f.stat().st_size == 0:
            continue
        prompts = []
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line)["prompt"])
        if prompts:
            data[domain_dir.name] = prompts
    if not data:
        raise SystemExit(f"No <domain>/{split}.jsonl under {base}")
    return data


# --------------------------------------------------------------------------- #
# vLLM engine                                                                   #
# --------------------------------------------------------------------------- #
def build_llm(method, max_model_len, gpu_mem_util, max_num_batched_tokens,
              max_num_seqs, enforce_eager):
    from vllm import LLM
    kw = dict(
        model=TARGET_MODEL,
        trust_remote_code=True,
        speculative_config=SPECULATORS[method],
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_batched_tokens=max_num_batched_tokens,
        # REQUIRED: LLM defaults this to True, but get_metrics() asserts log_stats.
        disable_log_stats=False,
    )
    if max_num_seqs:
        kw["max_num_seqs"] = max_num_seqs
    if enforce_eager:
        kw["enforce_eager"] = True
    return LLM(**kw)


def render_prompts(prompts):
    """Qwen3 chat template with thinking DISABLED (per the DFlash card)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]


# --------------------------------------------------------------------------- #
# Spec-decode acceptance metrics  (filled from research agent B)                #
# --------------------------------------------------------------------------- #
# vLLM V1 spec-decode counters (cumulative; base names have NO _total suffix in
# get_metrics()). Standard path; block-diffusion may instead use vllm:diffusion_*.
_SPEC_NAMES = (
    "vllm:spec_decode_num_drafts",           # == number of target forward passes
    "vllm:spec_decode_num_draft_tokens",     # proposed draft tokens
    "vllm:spec_decode_num_accepted_tokens",  # accepted draft tokens (excl. bonus)
)
_DIFF_NAMES = (
    "vllm:diffusion_num_committed_tokens",
    "vllm:diffusion_num_canvas_positions",
    "vllm:diffusion_num_denoising_steps",
)


def snapshot_spec_metrics(llm):
    """Cumulative snapshot of all spec-decode / diffusion scalar counters,
    summed across engine label sets (data/tensor parallel)."""
    from vllm.v1.metrics.reader import Counter
    wanted = set(_SPEC_NAMES) | set(_DIFF_NAMES)
    totals = {n: 0.0 for n in wanted}
    for m in llm.get_metrics():
        if isinstance(m, Counter) and m.name in wanted:
            totals[m.name] += m.value
    return totals


def _debug_nonzero(before, after):
    diff = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    return {k: v for k, v in diff.items() if v}


def domain_stats_from_snapshots(before, after, outputs):
    """Per-domain acceptance from cumulative-counter diffs.

    Standard spec-decode path (eagle3, and dflash if routed there):
      acceptance_rate    = accepted / proposed
      mean_accept_length = 1 + accepted / drafts   (drafts = target forward passes)
    Block-diffusion fallback (if dflash emits vllm:diffusion_* instead):
      acceptance_rate    ~ committed / canvas
      mean_accept_length ~ committed / steps
    """
    diff = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    gen = sum(len(o.outputs[0].token_ids) for o in outputs)

    drafts = diff.get("vllm:spec_decode_num_drafts", 0)
    proposed = diff.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = diff.get("vllm:spec_decode_num_accepted_tokens", 0)
    if drafts > 0 and proposed > 0:
        return {"accepted": accepted, "proposed": proposed,
                "forward_steps": drafts, "gen_tokens": gen,
                "mean_accept_length": 1 + accepted / drafts}

    # block-diffusion fallback
    committed = diff.get("vllm:diffusion_num_committed_tokens", 0)
    canvas = diff.get("vllm:diffusion_num_canvas_positions", 0)
    steps = diff.get("vllm:diffusion_num_denoising_steps", 0)
    if steps > 0 and canvas > 0:
        return {"accepted": committed, "proposed": canvas,
                "forward_steps": steps, "gen_tokens": gen,
                "mean_accept_length": committed / steps}

    # no counters moved — return zeros (diagnostic printed by caller)
    return {"accepted": 0, "proposed": 0, "forward_steps": 0, "gen_tokens": gen,
            "mean_accept_length": 0.0}


# --------------------------------------------------------------------------- #
# Output (same CSV shape aggregate.py writes, so make_charts.py works)          #
# --------------------------------------------------------------------------- #
_CSV_COLS = ["category", "n", "acceptance_rate_pooled", "acceptance_rate",
             "mean_accept_length", "forward_steps", "gen_tokens", "spec_tok_s"]


def write_outputs(out_dir, run_name, rows, method):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_name}_by_category.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLS)
        for r in sorted(rows, key=lambda x: -x["acceptance_rate_pooled"]):
            w.writerow([r.get(c) for c in _CSV_COLS])

    ranked = sorted(rows, key=lambda x: -x["acceptance_rate_pooled"])
    pooled = (sum(r["accepted"] for r in rows) / sum(r["proposed"] for r in rows)
              if sum(r["proposed"] for r in rows) else 0.0)
    lines = [f"# Speculator domain benchmark — {method}\n",
             f"Speculator `{SPECULATORS[method]['model']}` + target `{TARGET_MODEL}` · "
             f"{len(rows)} domains · {SPECULATORS[method]['num_speculative_tokens']} "
             f"speculative tokens/step.\n",
             f"- **Acceptance rate (pooled):** {pooled*100:.1f}%",
             f"- **Mean accept length:** "
             f"{sum(r['mean_accept_length'] for r in rows)/len(rows):.2f} tokens/pass\n",
             "## Domains ranked by acceptance rate\n",
             "| Domain | Accept % | Mean len | Gen tok | n |",
             "|---|---|---|---|---|"]
    for r in ranked:
        lines.append(f"| {r['category']} | {r['acceptance_rate_pooled']*100:.1f} | "
                     f"{r['mean_accept_length']:.2f} | {r['gen_tokens']:.0f} | {r['n']} |")
    (out_dir / f"{run_name}_report.md").write_text("\n".join(lines) + "\n")
    return csv_path


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(SPECULATORS), required=True)
    p.add_argument("--source", default="synthetic",
                   help="data/<source> subtree (synthetic|wild|downloaded)")
    p.add_argument("--datagen-dir", default=None,
                   help="override: explicit path to a data/<source> dir")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--categories", nargs="*", default=None,
                   help="subset of domain keys (default: all in the split)")
    p.add_argument("--limit", type=int, default=None, help="cap prompts/domain")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--run-name", default=None, help="default: <method>_<source>")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-num-batched-tokens", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datagen_dir = args.datagen_dir or str(DEFAULT_DATA_ROOT / args.source)
    run_name = args.run_name or f"{args.method}_{args.source}"
    out_dir = Path(args.out_dir)

    all_prompts = load_datagen_prompts(datagen_dir, args.split)
    cats = args.categories or list(all_prompts)
    cats = [c for c in cats if c in all_prompts]
    print(f"[plan] {run_name}: {len(cats)} domains, split={args.split}, "
          f"method={args.method}")

    llm = build_llm(args.method, args.max_model_len, args.gpu_mem_util,
                    args.max_num_batched_tokens, args.max_num_seqs, args.enforce_eager)
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

    rows = []
    t0 = time.perf_counter()
    for i, cat in enumerate(cats):
        prompts = all_prompts[cat]
        if args.limit:
            prompts = prompts[:args.limit]
        rendered = render_prompts(prompts)
        before = snapshot_spec_metrics(llm)
        gen_t0 = time.perf_counter()
        outputs = llm.generate(rendered, sp)
        gen_dt = time.perf_counter() - gen_t0
        after = snapshot_spec_metrics(llm)
        if i == 0:  # one-time: reveal which counters this method actually moves
            print(f"[diag] spec/diffusion counters that moved on domain 1: "
                  f"{_debug_nonzero(before, after)}")
        st = domain_stats_from_snapshots(before, after, outputs)
        prop = st["proposed"]
        st.update({
            "category": cat, "n": len(prompts),
            "acceptance_rate_pooled": st["accepted"] / prop if prop else 0.0,
            "acceptance_rate": st["accepted"] / prop if prop else 0.0,
            "spec_tok_s": st["gen_tokens"] / gen_dt if gen_dt else 0.0,
        })
        rows.append(st)
        if not prop:
            print(f"[warn] {cat}: no spec-decode counters moved — metrics may need "
                  f"a formula tweak (see [diag] above).")
        print(f"[{cat}] accept={st['acceptance_rate_pooled']*100:.1f}% "
              f"mean_len={st['mean_accept_length']:.2f} n={len(prompts)}")

    csv_path = write_outputs(out_dir, run_name, rows, args.method)
    print(f"\n[done] {len(rows)} domains in {time.perf_counter()-t0:.0f}s -> {csv_path}")
    print(f"[next] python make_charts.py {csv_path}")


if __name__ == "__main__":
    main()
