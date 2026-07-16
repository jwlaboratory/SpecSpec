#!/usr/bin/env python3
"""
WildDataGen — sort real WildChat prompts into the DataGen domains.

Streams the WildChat dataset, extracts each conversation's first user prompt,
routes it into one of the 51 DataGen domains (see router.py), and writes the same
train/val/test layout DataGen produces:

    data/<domain>/{train,val,test}.jsonl     rows: {"prompt": ..., "domain": ...}

This is the **real-human-prompt control set**: run benchmark.py against it exactly
like the synthetic set to check whether the drafter's per-domain behaviour on
Claude-generated prompts holds up on genuine user prompts.

    cd ../scripts
    python benchmark.py --datagen-dir ../WildDataGen/data --split test \
        --run-name dflash_wild --categories all

WildChat is gated on Hugging Face — accept the terms on the dataset page and
`huggingface-cli login` (or set HF_TOKEN) before running.

Examples
--------
    python sort.py --dry-run                          # show the plan, no download
    python sort.py --group languages --n 200          # just language domains
    python sort.py --classifier claude --group all    # Claude-labelled, full coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATAGEN = HERE.parent / "DataGen"
# Shared top-level data/ folder (sibling of DataGen/WildDataGen); wild prompts go
# in the wild/ subtree, alongside data/synthetic/ from DataGen.
DATA_DIR = HERE.parent / "data" / "wild"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DATAGEN))
from domains import REGISTRY, GROUPS  # noqa: E402
from generate import Accumulator, split_prompts, write_split, SEED  # noqa: E402
import router  # noqa: E402

DEFAULT_DATASET = "allenai/WildChat-1M"


# --------------------------------------------------------------------------- #
# Extract the first user prompt from a WildChat row                            #
# --------------------------------------------------------------------------- #
def first_user_prompt(row):
    """Return (prompt_text, detected_language) or (None, None)."""
    conv = row.get("conversation") or []
    for turn in conv:
        if turn.get("role") == "user":
            content = (turn.get("content") or "").strip()
            lang = turn.get("language") or row.get("language")
            return content, lang
    return None, None


def select_domains(args):
    if args.domains:
        unknown = [d for d in args.domains if d not in REGISTRY]
        if unknown:
            raise SystemExit(f"Unknown domain(s): {unknown}")
        return list(args.domains)
    if args.group and args.group != "all":
        return [k for k, s in REGISTRY.items() if s["group"] == args.group]
    return list(REGISTRY.keys())


def write_domain(key, acc, splits):
    n_train, n_val, n_test = splits
    target = n_train + n_val + n_test
    have = len(acc)
    if have == 0:
        return {"domain": key, "collected": 0, "train": 0, "val": 0, "test": 0}
    if have < target:  # scale splits down proportionally when real data is scarce
        n_train = round(n_train * have / target)
        n_val = round(n_val * have / target)
        n_test = have - n_train - n_val
    train, val, test = split_prompts(acc.items, n_train, n_val, n_test)
    dd = DATA_DIR / key
    write_split(dd / "train.jsonl", train, key)
    write_split(dd / "val.jsonl", val, key)
    write_split(dd / "test.jsonl", test, key)
    return {"domain": key, "collected": have,
            "train": len(train), "val": len(val), "test": len(test)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id")
    p.add_argument("--hf-split", default="train", help="HF split to stream")
    p.add_argument("--domains", nargs="+", help="explicit domain keys")
    p.add_argument("--group", choices=GROUPS + ["all"], default="all")
    p.add_argument("--n", type=int, default=None,
                   help="total prompts/domain (8:1:1 train/val/test)")
    p.add_argument("--splits", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                   default=[800, 100, 100])
    p.add_argument("--classifier", choices=["heuristic", "claude"], default="heuristic")
    p.add_argument("--model", default="claude-sonnet-5",
                   help="model for --classifier claude")
    p.add_argument("--claude-batch", type=int, default=40,
                   help="prompts per Claude classify call")
    p.add_argument("--max-scan", type=int, default=300_000,
                   help="max WildChat rows to scan before stopping")
    p.add_argument("--min-len", type=int, default=12, help="min prompt chars")
    p.add_argument("--max-len", type=int, default=4000, help="max prompt chars")
    p.add_argument("--keep-toxic", action="store_true", help="don't skip toxic rows")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="no download; print plan")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.n is not None:
        v = round(args.n * 0.1)
        args.splits = [args.n - 2 * v, v, v]
    return args


def main():
    args = parse_args()
    if args.list:
        for k, s in REGISTRY.items():
            print(f"  {k:28s} [{s['group']:9s}] {s['description']}")
        return

    keys = select_domains(args)
    target = sum(args.splits)
    accs = OrderedDict((k, Accumulator()) for k in keys)
    wanted = set(keys)
    print(f"WildDataGen: sorting WildChat -> {len(keys)} domain(s), "
          f"target {target}/domain, classifier={args.classifier}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    if args.dry_run:
        print(f"  would stream {args.dataset} (split={args.hf_split}), scan up to "
              f"{args.max_scan} rows, route via {args.classifier}, write to {DATA_DIR}")
        return

    from datasets import load_dataset
    print(f"[stream] {args.dataset} (accept terms + HF login if gated)")
    ds = load_dataset(args.dataset, split=args.hf_split, streaming=True)

    client = None
    buf_prompts, buf_langs, buf_domhint = [], [], []
    if args.classifier == "claude":
        import anthropic
        client = anthropic.Anthropic()

    def route(prompt, lang):
        return router.classify_heuristic(prompt, lang)

    def full(k):
        return len(accs[k]) >= target

    def all_full():
        return all(full(k) for k in keys)

    def flush_claude():
        if not buf_prompts:
            return
        labels = router.classify_claude_batch(client, buf_prompts, args.model)
        for prompt, dom in zip(buf_prompts, labels):
            if dom in wanted and not full(dom):
                accs[dom].add(prompt)
        buf_prompts.clear(); buf_langs.clear()

    scanned = kept = 0
    for row in ds:
        if scanned >= args.max_scan or all_full():
            break
        scanned += 1
        if not args.keep_toxic and row.get("toxic"):
            continue
        prompt, lang = first_user_prompt(row)
        if not prompt or not (args.min_len <= len(prompt) <= args.max_len):
            continue

        if args.classifier == "claude":
            buf_prompts.append(prompt); buf_langs.append(lang)
            if len(buf_prompts) >= args.claude_batch:
                flush_claude()
                kept = sum(len(a) for a in accs.values())
        else:
            dom = route(prompt, lang)
            if dom in wanted and not full(dom):
                if accs[dom].add(prompt):
                    kept += 1

        if scanned % 5000 == 0:
            filled = sum(1 for k in keys if full(k))
            print(f"  scanned {scanned}  kept {sum(len(a) for a in accs.values())}  "
                  f"domains full {filled}/{len(keys)}")

    if args.classifier == "claude":
        flush_claude()

    results = [write_domain(k, accs[k], args.splits) for k in keys]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": args.dataset, "classifier": args.classifier,
        "splits": {"train": args.splits[0], "val": args.splits[1], "test": args.splits[2]},
        "seed": SEED, "scanned": scanned, "domains": results,
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    got = [r for r in results if r["collected"] > 0]
    print(f"\n[done] scanned {scanned} rows; filled {len(got)}/{len(keys)} domains.")
    print("Per-domain collected:")
    for r in sorted(results, key=lambda x: -x["collected"]):
        bar = "" if r["collected"] >= target else "  (short)"
        print(f"  {r['domain']:28s} {r['collected']:5d}{bar}")
    print(f"\nManifest: {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
