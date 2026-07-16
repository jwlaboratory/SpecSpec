#!/usr/bin/env python3
"""
Dedicated real-dataset sources for the specialised domains.

WildChat (see sort.py) is general chat — it covers the language and general-task
domains well but comes up thin on specialised domains (medical, legal, financial,
SQL/data-analysis, ...). For those, purpose-built public datasets give far better
real-prompt coverage, and no classification is needed: the whole dataset *is* the
domain, so we just extract each row's user-facing prompt.

Output goes to the same shared control tree the WildChat sort writes to:

    ../data/downloaded/<domain>/{train,val,test}.jsonl

so benchmark.py treats it identically:

    cd ../scripts
    python benchmark.py --datagen-dir ../data/downloaded --split test --run-name dflash_dl

Each SOURCE maps one domain to (HF dataset, config, split, extract). `extract(row)`
returns a self-contained prompt string (or None to skip). All are real human /
curated text — this is a control for the synthetic DataGen set.

Add more by dropping an entry in SOURCES; the loader handles the rest.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATAGEN = HERE.parent / "DataGen"
DATA_DIR = HERE.parent / "data" / "downloaded"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DATAGEN))
from generate import Accumulator, split_prompts, write_split, SEED  # noqa: E402


def _clean(s):
    return (s or "").strip()


# --------------------------------------------------------------------------- #
# Extraction helpers — each returns a self-contained prompt string or None.     #
# --------------------------------------------------------------------------- #
def _medquad(row):
    q = _clean(row.get("Question"))
    return q or None


def _finance_alpaca(row):
    instr = _clean(row.get("instruction"))
    inp = _clean(row.get("input"))
    if not instr:
        return None
    return f"{instr}\n\n{inp}" if inp else instr


def _legal_contract(row):
    contract = _clean(row.get("contract"))
    q = _clean(row.get("question"))
    if not q:
        return None
    if contract:
        return f"Given the following contract clause:\n\n{contract}\n\n{q}"
    return q


def _sql_context(row):
    ctx = _clean(row.get("context"))
    q = _clean(row.get("question"))
    if not q:
        return None
    if ctx:
        return f"Given the schema:\n{ctx}\n\nWrite a SQL query to answer: {q}"
    return f"Write a SQL query to answer: {q}"


# domain -> source spec.  (dataset, config, split, extract fn)
SOURCES = OrderedDict([
    ("ood_medical",   ("keivalya/MedQuAD-MedicalQnADataset", None, "train", _medquad)),
    ("ood_financial", ("gbharti/finance-alpaca",             None, "train", _finance_alpaca)),
    ("ood_legal",     ("nguha/legalbench", "consumer_contracts_qa", "test", _legal_contract)),
    ("code_sql",      ("b-mc2/sql-create-context",           None, "train", _sql_context)),
])


def collect_domain(key, spec, splits, max_scan, min_len, max_len):
    dataset, config, split, extract = spec
    target = sum(splits)
    from datasets import load_dataset
    print(f"[src ] {key}  <- {dataset}" + (f":{config}" if config else ""))
    ds = (load_dataset(dataset, config, split=split, streaming=True) if config
          else load_dataset(dataset, split=split, streaming=True))
    acc = Accumulator()
    scanned = 0
    for row in ds:
        if len(acc) >= target or scanned >= max_scan:
            break
        scanned += 1
        try:
            prompt = extract(row)
        except Exception:
            prompt = None
        if prompt and (min_len <= len(prompt) <= max_len):
            acc.add(prompt)

    have = len(acc)
    n_train, n_val, n_test = splits
    if have < target:
        n_train = round(n_train * have / target)
        n_val = round(n_val * have / target)
        n_test = have - n_train - n_val
    train, val, test = split_prompts(acc.items, n_train, n_val, n_test)
    dd = DATA_DIR / key
    write_split(dd / "train.jsonl", train, key)
    write_split(dd / "val.jsonl", val, key)
    write_split(dd / "test.jsonl", test, key)
    print(f"    collected {have} (scanned {scanned}) -> "
          f"train={len(train)} val={len(val)} test={len(test)}")
    return {"domain": key, "dataset": dataset, "collected": have,
            "train": len(train), "val": len(val), "test": len(test)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", nargs="+", help="subset of SOURCE-backed domains")
    p.add_argument("--n", type=int, default=None, help="total/domain (8:1:1)")
    p.add_argument("--splits", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                   default=[800, 100, 100])
    p.add_argument("--max-scan", type=int, default=200_000)
    p.add_argument("--min-len", type=int, default=12)
    p.add_argument("--max-len", type=int, default=4000)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.n is not None:
        v = round(args.n * 0.1)
        args.splits = [args.n - 2 * v, v, v]
    return args


def main():
    args = parse_args()
    if args.list:
        for k, (ds, cfg, split, _) in SOURCES.items():
            print(f"  {k:20s} <- {ds}{':' + cfg if cfg else ''}  [{split}]")
        return

    keys = args.domains or list(SOURCES)
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        raise SystemExit(f"No dedicated source for: {unknown}\n"
                         f"Available: {list(SOURCES)}")

    print(f"Dedicated sources -> {len(keys)} domain(s), target {sum(args.splits)}/domain\n")
    results = [collect_domain(k, SOURCES[k], args.splits,
                              args.max_scan, args.min_len, args.max_len) for k in keys]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    man_path = DATA_DIR / "sources_manifest.json"
    man_path.write_text(json.dumps(
        {"splits": {"train": args.splits[0], "val": args.splits[1], "test": args.splits[2]},
         "seed": SEED, "domains": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {len(results)} domains from dedicated datasets.")
    print(f"Manifest: {man_path}")


if __name__ == "__main__":
    main()
