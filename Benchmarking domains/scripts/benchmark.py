"""
DFlash speculative-decoding domain benchmark.

Loads the DFlash drafter (z-lab/Qwen3-8B-DFlash-b16) + target (Qwen/Qwen3-8B),
runs speculative decoding across many domains, and records for every prompt:

  - num_generated_tokens          tokens produced by spec decoding
  - forward_steps                 target verification passes
  - committed_per_step (list)     tokens committed each step (1..block_size)
  - accepted_draft_tokens         sum(committed-1)  == accepted drafts total
  - proposed_draft_tokens         forward_steps * (block_size-1)
  - acceptance_rate               accepted / proposed   (fraction of drafts kept)
  - mean_accept_length            generated / steps     (tokens per target pass)
  - spec_seconds / spec_tok_s     wall-clock throughput of spec decoding
  - baseline_seconds/base_tok_s   target-only greedy generate (for speedup)
  - speedup                       spec_tok_s / base_tok_s
  - lossless_match                spec output == target greedy output (temp 0)?

At temperature 0 DFlash is *lossless*: its output must equal the target's greedy
output. `lossless_match=False` therefore flags a bug in the inference code, which
is exactly the correctness signal we want -- independent of whether the target's
answer is any good.

Results stream to results/<run_name>.jsonl (one line per prompt). Run aggregate.py
afterwards to turn that into per-domain tables + a markdown report.

Requires a CUDA GPU with >= ~20 GB VRAM (bf16 8B target + 1B drafter + KV cache).

Usage examples
--------------
  # quick smoke test: 2 categories, 5 prompts each, short outputs
  python benchmark.py --limit 5 --categories lang_english code_python --max-new-tokens 256

  # the real thing: all 28 domains, 100 prompts each
  python benchmark.py --limit 100 --max-new-tokens 512

  # skip the (expensive) baseline pass -- lose speedup + correctness, keep accept stats
  python benchmark.py --limit 100 --no-baseline
"""
import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch

from prompts import build_prompts, CATEGORY_GROUPS
from spec_patch import make_instrumented_spec_generate

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"

# Paths are resolved relative to this file so the script works from any CWD,
# including after it was moved into scripts/.  scripts/ -> parent is the
# "Benchmarking domains" root, which holds DataGen/ and results/.
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = _ROOT / "results"
DEFAULT_DATAGEN_DIR = _ROOT / "DataGen" / "data"

# DataGen domain-key prefixes -> group name, for --categories languages/coding/...
_DATAGEN_GROUP_PREFIX = OrderedDict([
    ("lang_", "languages"),
    ("code_", "coding"),
    ("task_", "tasks"),
    ("ood_", "ood"),
])


def datagen_groups(all_cats):
    """Build a {group: [domain,...]} map from DataGen domain keys by prefix."""
    groups = OrderedDict((g, []) for g in _DATAGEN_GROUP_PREFIX.values())
    for c in all_cats:
        for pref, g in _DATAGEN_GROUP_PREFIX.items():
            if c.startswith(pref):
                groups[g].append(c)
                break
    return OrderedDict((g, v) for g, v in groups.items() if v)


def load_datagen_prompts(datagen_dir, split):
    """Read DataGen/data/<domain>/<split>.jsonl for every domain.

    Returns an OrderedDict {domain: [prompt, ...]}.  This is the intended
    source for the benchmark: run the drafter+target over the held-out test
    split of each generated domain.
    """
    base = Path(datagen_dir)
    if not base.exists():
        raise SystemExit(
            f"DataGen dir not found: {base}\n"
            f"Generate datasets first:  cd ../DataGen && python generate.py --group all")
    data = OrderedDict()
    for domain_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        f = domain_dir / f"{split}.jsonl"
        if not f.exists():
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
        raise SystemExit(f"No <domain>/{split}.jsonl files under {base}")
    return data


def parse_args():
    p = argparse.ArgumentParser(description="DFlash domain benchmark")
    p.add_argument("--draft-model", default=DRAFT_MODEL)
    p.add_argument("--target-model", default=TARGET_MODEL)
    p.add_argument("--prompt-source", choices=["datagen", "legacy"], default="datagen",
                   help="datagen = DataGen/data/<domain>/<split>.jsonl (default); "
                        "legacy = the deterministic prompts.py generator")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="which DataGen split to benchmark (default: test)")
    p.add_argument("--datagen-dir", default=str(DEFAULT_DATAGEN_DIR),
                   help="path to DataGen/data (default: resolved next to this script)")
    p.add_argument("--limit", type=int, default=100,
                   help="prompts per category (max = split size)")
    p.add_argument("--categories", nargs="*", default=None,
                   help="subset of domain names, or a group: languages/coding/tasks/ood/all")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 => greedy/lossless (enables correctness check)")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip target-only baseline (no speedup / no correctness check)")
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="attention implementation for both models")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-name", default="dflash_bench")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--warmup", type=int, default=2, help="warmup generations before timing")
    p.add_argument("--resume", action="store_true",
                   help="skip (category,idx) pairs already present in the output file")
    return p.parse_args()


def resolve_categories(all_cats, requested, groups):
    if not requested or requested == ["all"]:
        return list(all_cats)
    out = []
    for r in requested:
        if r in groups:
            out.extend(groups[r])
        elif r in all_cats:
            out.append(r)
        else:
            raise SystemExit(f"Unknown category/group: {r!r}. "
                             f"Groups: {list(groups)}. "
                             f"Categories: {list(all_cats)}")
    # de-dup, keep order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def load_models(args):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    print(f"[load] tokenizer  {args.target_model}")
    tok = AutoTokenizer.from_pretrained(args.target_model)
    print(f"[load] drafter    {args.draft_model}")
    draft = AutoModel.from_pretrained(
        args.draft_model, trust_remote_code=True,
        dtype="auto", attn_implementation=args.attn,
    ).to(args.device).eval()
    print(f"[load] target     {args.target_model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype="auto", attn_implementation=args.attn,
    ).to(args.device).eval()
    draft.spec_generate = make_instrumented_spec_generate(draft)
    return tok, draft, target


def build_input_ids(tok, prompt, device):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok([text], return_tensors="pt").input_ids.to(device)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def first_stop(ids_list, stop_ids):
    """Trim a python list of token ids at the first stop token (inclusive)."""
    for i, t in enumerate(ids_list):
        if t in stop_ids:
            return ids_list[:i + 1]
    return ids_list


@torch.inference_mode()
def run_spec(draft, target, input_ids, max_new_tokens, temperature, stop_ids, device):
    sync(device)
    t0 = time.perf_counter()
    out_ids, committed = draft.spec_generate(
        target=target, input_ids=input_ids,
        max_new_tokens=max_new_tokens, temperature=temperature,
        stop_token_ids=stop_ids,
    )
    sync(device)
    dt = time.perf_counter() - t0
    gen = out_ids[0, input_ids.shape[1]:].tolist()
    return gen, committed, dt


# bf16 logits are quantized: near a magnitude of ~32-64 the smallest representable
# step ("ULP") is 0.25. When the target's top-2 tokens are within a few ULPs, the
# parallel block-verification path and sequential greedy legitimately round the tie
# differently -> a benign divergence, NOT a bug. Only a divergence with a LARGE logit
# gap indicates the inference code actually picked the wrong token.
NEAR_TIE_MAX_GAP = 0.5


@torch.inference_mode()
def divergence_logit_gap(target, input_ids, prefix_tokens, device):
    """Teacher-force the target over (prompt + prefix_tokens) and return the top-2
    logit gap at the next position -- i.e. how close the tie was at a divergence."""
    if prefix_tokens:
        pref = torch.tensor([prefix_tokens], device=device, dtype=input_ids.dtype)
        seq = torch.cat([input_ids, pref], dim=1)
    else:
        seq = input_ids
    logits = target(seq, use_cache=False).logits[0, -1, :].float()
    top2 = torch.topk(logits, 2).values.tolist()
    return abs(top2[0] - top2[1])


@torch.inference_mode()
def run_baseline(target, tok, input_ids, max_new_tokens, stop_ids, device):
    sync(device)
    t0 = time.perf_counter()
    out = target.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=list(stop_ids),
        use_cache=True,
    )
    sync(device)
    dt = time.perf_counter() - t0
    gen = out[0, input_ids.shape[1]:].tolist()
    return gen, dt


def summarize_committed(committed, block_size):
    steps = len(committed)
    generated = sum(committed)                      # total committed tokens
    accepted_drafts = sum(c - 1 for c in committed)  # bonus token excluded
    proposed = steps * (block_size - 1)
    return {
        "forward_steps": steps,
        "num_generated_tokens": generated,
        "accepted_draft_tokens": accepted_drafts,
        "proposed_draft_tokens": proposed,
        "acceptance_rate": (accepted_drafts / proposed) if proposed else 0.0,
        "mean_accept_length": (generated / steps) if steps else 0.0,
        "max_committed": max(committed) if committed else 0,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit(
            "No CUDA GPU detected. This benchmark needs a GPU with ~20GB+ VRAM.\n"
            "See README.md for GPU options (Colab L4/A100, RunPod, Vast.ai, Lambda)."
        )
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_name}.jsonl")

    done = set()
    mode = "w"
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["category"], r["prompt_idx"]))
                except Exception:
                    pass
        mode = "a"
        print(f"[resume] {len(done)} results already present, appending")

    if args.prompt_source == "datagen":
        all_prompts = load_datagen_prompts(args.datagen_dir, args.split)
        groups = datagen_groups(all_prompts.keys())
        print(f"[data] DataGen split '{args.split}' from {args.datagen_dir} "
              f"({len(all_prompts)} domains)")
    else:
        all_prompts = build_prompts(100)
        groups = CATEGORY_GROUPS
    cats = resolve_categories(all_prompts, args.categories, groups)
    limit = args.limit
    print(f"[plan] {len(cats)} categories x up to {limit} prompts")
    print(f"[plan] baseline={'off' if args.no_baseline else 'on'} "
          f"max_new_tokens={args.max_new_tokens} temp={args.temperature} attn={args.attn}")

    tok, draft, target = load_models(args)
    block_size = draft.block_size
    # NOTE: keep this a list -- the model's spec_generate does torch.tensor(stop_ids),
    # which raises "Could not infer dtype of set" if given a set.
    stop_ids = [tok.eos_token_id]
    print(f"[info] block_size={block_size}  proposed drafts/step={block_size - 1}  "
          f"eos_id={tok.eos_token_id}")

    # Warmup (kernels, autotune) -- excluded from timing.
    if args.warmup:
        warm = build_input_ids(tok, "Hello, please introduce yourself briefly.", args.device)
        for _ in range(args.warmup):
            run_spec(draft, target, warm, 64, args.temperature, stop_ids, args.device)
            if not args.no_baseline:
                run_baseline(target, tok, warm, 64, stop_ids, args.device)
        print(f"[warmup] done ({args.warmup} iters)")

    n_done = 0
    t_start = time.perf_counter()
    with open(out_path, mode) as fout:
        for cat in cats:
            prompts = all_prompts[cat][:limit]
            for idx, prompt in enumerate(prompts):
                if (cat, idx) in done:
                    continue
                input_ids = build_input_ids(tok, prompt, args.device)
                n_in = input_ids.shape[1]

                gen, committed, spec_dt = run_spec(
                    draft, target, input_ids, args.max_new_tokens,
                    args.temperature, stop_ids, args.device)
                stats = summarize_committed(committed, block_size)
                spec_gen = first_stop(gen, stop_ids)
                spec_tok_s = len(spec_gen) / spec_dt if spec_dt > 0 else 0.0

                rec = {
                    "category": cat, "prompt_idx": idx,
                    "num_input_tokens": n_in,
                    "spec_seconds": spec_dt, "spec_tok_s": spec_tok_s,
                    **stats,
                    "committed_per_step": committed,
                }

                if not args.no_baseline:
                    base_gen, base_dt = run_baseline(
                        target, tok, input_ids, args.max_new_tokens, stop_ids, args.device)
                    base_gen = first_stop(base_gen, stop_ids)
                    base_tok_s = len(base_gen) / base_dt if base_dt > 0 else 0.0
                    # Greedy agreement: spec vs sequential-greedy target, token by token.
                    m = min(len(spec_gen), len(base_gen))
                    exact = (args.temperature == 0.0 and spec_gen == base_gen)
                    divergence = next((i for i in range(m) if spec_gen[i] != base_gen[i]), m)
                    agreement_frac = (divergence / m) if m else 1.0
                    # If they diverge WITHIN the overlap, measure how close the tie was.
                    gap, near_tie, suspicious = None, None, False
                    if args.temperature == 0.0 and divergence < m:
                        gap = divergence_logit_gap(
                            target, input_ids, base_gen[:divergence], args.device)
                        near_tie = gap <= NEAR_TIE_MAX_GAP
                        suspicious = not near_tie   # large-gap divergence == real bug signal
                    rec.update({
                        "baseline_seconds": base_dt, "baseline_tok_s": base_tok_s,
                        "baseline_gen_tokens": len(base_gen),
                        "speedup": (spec_tok_s / base_tok_s) if base_tok_s > 0 else 0.0,
                        "exact_match": bool(exact),
                        "first_divergence": divergence,
                        "agreement_frac": agreement_frac,
                        "divergence_gap": gap,
                        "near_tie": near_tie,
                        "suspicious": bool(suspicious),
                    })

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1

                if n_done % 10 == 0 or idx == 0:
                    el = time.perf_counter() - t_start
                    extra = ""
                    if not args.no_baseline:
                        extra = (f" speedup={rec['speedup']:.2f}x "
                                 f"agree={rec['agreement_frac']*100:.0f}% "
                                 f"{'SUSPICIOUS' if rec['suspicious'] else 'ok'}")
                    print(f"[{n_done}] {cat}#{idx} "
                          f"accept={stats['acceptance_rate']*100:.0f}% "
                          f"mean_len={stats['mean_accept_length']:.2f} "
                          f"spec={spec_tok_s:.0f}tok/s{extra}  ({el:.0f}s elapsed)")

    total = time.perf_counter() - t_start
    print(f"\n[done] {n_done} prompts in {total:.0f}s -> {out_path}")
    print(f"[next] python aggregate.py {out_path}")


if __name__ == "__main__":
    main()
