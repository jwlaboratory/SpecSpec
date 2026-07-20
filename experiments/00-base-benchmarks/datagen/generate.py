#!/usr/bin/env python3
"""
DataGen — build train/val/test prompt datasets for every domain using Claude.

For each domain in domains.REGISTRY this:
  1. seeds the accumulator with the deterministic prompts from ../prompts.py
     (where a matching legacy category exists) so we reuse existing work,
  2. tops up to the target count by asking Claude for batches of diverse,
     domain-specific *user prompts* (structured JSON output, deduped),
  3. deterministically shuffles and splits into train / val / test,
  4. writes data/<domain>/{train,val,test}.jsonl  ({"prompt","domain"} per line),
  5. records everything in data/manifest.json.

Each domain is independent and resumable: a domain whose splits already exist and
match the requested sizes is skipped unless --overwrite is passed.

Auth: uses the Anthropic SDK, which reads ANTHROPIC_API_KEY (or an `ant auth`
profile) from the environment. Nothing is hard-coded.

Examples
--------
    # Dry run — show the plan, no API calls, no writes
    python generate.py --dry-run

    # One domain, small, to sanity-check wiring + cost
    python generate.py --domains lang_hindi --n 50 --splits 40 5 5

    # A whole group
    python generate.py --group coding

    # Everything (the full 1000/domain build), resumable
    python generate.py --group all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent               # ".../experiments/00-base-benchmarks"
REPO_ROOT = BENCH_DIR.parent.parent
# Datasets live in the shared top-level data/ folder (not inside DataGen/), with
# synthetic (Claude-generated) and wild (WildChat) kept in separate subtrees so
# the benchmark can compare them per domain.
DATA_DIR = REPO_ROOT / "data" / "synthetic"
SEED = 1234

sys.path.insert(0, str(HERE))
from domains import REGISTRY, GROUPS  # noqa: E402

# Structured-output schema: Claude must return {"prompts": [str, ...]}.
_PROMPTS_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompts"],
    "additionalProperties": False,
}

# Rotating "angle" nudges injected per batch to spread coverage and reduce
# near-duplicates across batches (temperature isn't available on Opus 4.8).
_ANGLES = [
    "everyday, casual phrasing",
    "detailed, expert-level requests",
    "very short, terse prompts",
    "long, multi-part prompts",
    "beginner-level questions",
    "unusual or niche sub-topics",
    "prompts that embed a concrete example or snippet",
    "comparison / trade-off style prompts",
    "step-by-step or how-to phrasing",
    "opinion / recommendation requests",
]


# --------------------------------------------------------------------------- #
# Normalisation & dedup                                                         #
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Key for dedup: collapse whitespace, strip, casefold."""
    return re.sub(r"\s+", " ", text).strip().casefold()


class Accumulator:
    """Ordered, deduped collection of prompt strings."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.items: list[str] = []

    def add(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        key = _norm(text)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.items.append(text)
        return True

    def extend(self, texts) -> int:
        return sum(1 for t in texts if self.add(t))

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- #
# Seeding from the legacy deterministic generator                               #
# --------------------------------------------------------------------------- #
_LEGACY_CACHE = None


def _load_legacy():
    """Load prompts.py:build_prompts(100) once; return {category: [prompt]}.

    Looks for prompts.py in ../scripts (post-reorg) and .. (legacy layout).
    """
    global _LEGACY_CACHE
    if _LEGACY_CACHE is not None:
        return _LEGACY_CACHE
    try:
        for cand in (BENCH_DIR / "scripts", BENCH_DIR):
            if (cand / "prompts.py").exists():
                sys.path.insert(0, str(cand))
                break
        import prompts as legacy  # noqa: E402
        _LEGACY_CACHE = legacy.build_prompts(100)
    except Exception as e:  # pragma: no cover - seeding is best-effort
        print(f"  ! could not seed from legacy prompts.py: {e}")
        _LEGACY_CACHE = {}
    return _LEGACY_CACHE


def seed_domain(acc: Accumulator, legacy_key: str | None) -> int:
    if not legacy_key:
        return 0
    legacy = _load_legacy()
    return acc.extend(legacy.get(legacy_key, []))


# --------------------------------------------------------------------------- #
# Claude generation                                                             #
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You produce datasets of USER PROMPTS for benchmarking a language model. "
    "You return prompts that a user would send to an AI assistant — never the "
    "answers. Prompts must be diverse, self-contained, and free of numbering, "
    "bullets, or surrounding commentary. Return only the structured JSON."
)


def _build_client():
    import anthropic
    return anthropic.Anthropic()


def generate_batch(client, spec: dict, batch_size: int, angle: str,
                   avoid: list[str], model: str, max_retries: int = 4) -> list[str]:
    """Ask Claude for one batch of prompts; returns a list of strings."""
    avoid_block = ""
    if avoid:
        sample = "\n".join(f"- {p}" for p in avoid[:12])
        avoid_block = (
            "\n\nHere are prompts already collected — produce DIFFERENT ones, "
            f"do not repeat these:\n{sample}"
        )
    user = (
        f"{spec['instruction']}\n\n"
        f"Produce exactly {batch_size} prompts. Lean toward: {angle}."
        f"{avoid_block}"
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=8000,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _PROMPTS_SCHEMA,
                    }
                },
            )
            if resp.stop_reason == "refusal":
                print("    ! batch refused by safety classifier; skipping")
                return []
            text = next((b.text for b in resp.content if b.type == "text"), "")
            data = json.loads(text)
            prompts = data.get("prompts", [])
            return [p for p in prompts if isinstance(p, str)]
        except Exception as e:  # noqa: BLE001 - retry on transient/API errors
            last_err = e
            wait = min(2 ** attempt, 20)
            print(f"    ! batch error ({type(e).__name__}): {e} — retry in {wait}s")
            time.sleep(wait)
    print(f"    ! batch failed after {max_retries} attempts: {last_err}")
    return []


# --------------------------------------------------------------------------- #
# Splitting & writing                                                           #
# --------------------------------------------------------------------------- #
def split_prompts(items: list[str], n_train: int, n_val: int, n_test: int):
    rng = random.Random(SEED)
    items = list(items)
    rng.shuffle(items)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def write_split(path: Path, prompts: list[str], domain: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p, "domain": domain}, ensure_ascii=False) + "\n")


def splits_present(domain_dir: Path, n_train: int, n_val: int, n_test: int) -> bool:
    def count(name: str) -> int:
        p = domain_dir / name
        if not p.exists():
            return -1
        with p.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    return (count("train.jsonl") == n_train
            and count("val.jsonl") == n_val
            and count("test.jsonl") == n_test)


# --------------------------------------------------------------------------- #
# Per-domain driver                                                             #
# --------------------------------------------------------------------------- #
def build_domain(client, key: str, spec: dict, args) -> dict:
    n_train, n_val, n_test = args.splits
    target = n_train + n_val + n_test
    domain_dir = DATA_DIR / key

    if not args.overwrite and splits_present(domain_dir, n_train, n_val, n_test):
        print(f"[skip] {key}: splits already present ({target} prompts)")
        return {"domain": key, "status": "skipped", "total": target,
                "train": n_train, "val": n_val, "test": n_test}

    print(f"[gen ] {key}  ({spec['description']}) -> target {target}")
    acc = Accumulator()
    seeded = seed_domain(acc, spec.get("legacy_key"))
    if seeded:
        print(f"    seeded {seeded} from legacy prompts.py")

    if args.dry_run:
        print(f"    (dry-run) would generate {max(0, target - len(acc))} more via {args.model}")
        return {"domain": key, "status": "dry-run", "seeded": seeded,
                "target": target}

    empty_streak = 0
    batch_no = 0
    while len(acc) < target:
        need = target - len(acc)
        bs = min(args.batch_size, max(need + 5, 10))  # slight overshoot for dedup
        angle = _ANGLES[batch_no % len(_ANGLES)]
        avoid = acc.items[-12:]
        got = generate_batch(client, spec, bs, angle, avoid, args.model)
        added = acc.extend(got)
        batch_no += 1
        print(f"    batch {batch_no}: +{added} new (have {len(acc)}/{target})")
        if added == 0:
            empty_streak += 1
            if empty_streak >= 3:
                print(f"    ! 3 empty batches in a row — stopping at {len(acc)}")
                break
        else:
            empty_streak = 0

    have = len(acc)
    # Trim requested split sizes down proportionally if we came up short.
    if have < target:
        print(f"    ! only collected {have}/{target}; scaling splits down")
        n_train = round(n_train * have / target)
        n_val = round(n_val * have / target)
        n_test = have - n_train - n_val

    train, val, test = split_prompts(acc.items, n_train, n_val, n_test)
    write_split(domain_dir / "train.jsonl", train, key)
    write_split(domain_dir / "val.jsonl", val, key)
    write_split(domain_dir / "test.jsonl", test, key)
    print(f"    wrote train={len(train)} val={len(val)} test={len(test)}")

    return {"domain": key, "status": "generated", "seeded": seeded,
            "collected": have, "train": len(train), "val": len(val),
            "test": len(test)}


# --------------------------------------------------------------------------- #
# Selection & CLI                                                               #
# --------------------------------------------------------------------------- #
def select_domains(args) -> list[str]:
    if args.domains:
        unknown = [d for d in args.domains if d not in REGISTRY]
        if unknown:
            raise SystemExit(f"Unknown domain(s): {unknown}\n"
                             f"Run --list to see valid keys.")
        return list(args.domains)
    if args.group and args.group != "all":
        return [k for k, s in REGISTRY.items() if s["group"] == args.group]
    return list(REGISTRY.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", nargs="+", help="explicit domain keys")
    p.add_argument("--group", choices=GROUPS + ["all"], default="all",
                   help="generate a whole group (default: all)")
    p.add_argument("--n", type=int, default=None,
                   help="total prompts/domain; shorthand that sets --splits "
                        "to a 8:1:1 train/val/test ratio of N")
    p.add_argument("--splits", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                   default=[800, 100, 100],
                   help="prompts per split (default: 800 100 100)")
    p.add_argument("--model", default="claude-opus-4-8",
                   help="Claude model (default: claude-opus-4-8; "
                        "use claude-sonnet-5 for a cheaper bulk run)")
    p.add_argument("--batch-size", type=int, default=40,
                   help="prompts requested per API call (default: 40)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="number of domains to build in parallel (default: 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate domains even if splits already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan; no API calls, no writes")
    p.add_argument("--list", action="store_true", help="list domains and exit")
    args = p.parse_args()
    if args.n is not None:
        n_val = round(args.n * 0.1)
        n_test = round(args.n * 0.1)
        args.splits = [args.n - n_val - n_test, n_val, n_test]
    return args


def main():
    args = parse_args()

    if args.list:
        for key, spec in REGISTRY.items():
            print(f"  {key:28s} [{spec['group']:9s}] {spec['description']}")
        print(f"\n{len(REGISTRY)} domains.")
        return

    keys = select_domains(args)
    n_train, n_val, n_test = args.splits
    total_each = n_train + n_val + n_test
    print(f"DataGen: {len(keys)} domain(s) x {total_each} prompts "
          f"(train={n_train} val={n_val} test={n_test})  model={args.model}"
          f"{'  [DRY RUN]' if args.dry_run else ''}\n")

    client = None if args.dry_run else _build_client()
    _load_legacy()  # warm the shared seed cache before any threads start

    results = []
    if args.concurrency > 1 and not args.dry_run:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"(building up to {args.concurrency} domains in parallel)\n")
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(build_domain, client, k, REGISTRY[k], args): k
                    for k in keys}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for key in keys:
            results.append(build_domain(client, key, REGISTRY[key], args))

    if not args.dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "splits": {"train": n_train, "val": n_val, "test": n_test},
            "model": args.model,
            "seed": SEED,
            "domains": results,
        }
        (DATA_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote manifest: {DATA_DIR / 'manifest.json'}")

    gen = sum(1 for r in results if r["status"] == "generated")
    skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone. generated={gen} skipped={skip} total={len(results)}")


if __name__ == "__main__":
    main()
