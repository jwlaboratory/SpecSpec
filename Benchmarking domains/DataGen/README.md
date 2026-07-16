# DataGen — multi-domain prompt datasets for the DFlash drafter

Generates **train / val / test** prompt datasets across many domains using Claude,
so we can measure where a tiny 1B block-diffusion **drafter** tracks the 8B
**target** and where it stops generalising (in-distribution vs out-of-distribution).

Each row is a single user prompt:

```json
{"prompt": "Explain compound interest to a 12-year-old.", "domain": "task_question_answering"}
```

Prompts only — the benchmark (`../benchmark.py`) generates the target's answers at
run time, so the drafter is scored on real target output, not pre-baked responses.

## Domains (51)

| Group | Count | Examples |
|---|---|---|
| `languages` | 16 | English, Spanish, French, German, Hindi, Chinese, Japanese, Russian, Arabic, Portuguese, **Korean, Italian, Turkish, Vietnamese, Polish, Swahili** |
| `coding` | 15 | Python, JavaScript, SQL, C++, Ruby, Rust, Go, Bash, **Java, TypeScript, C, Kotlin, Swift, Haskell, R** |
| `tasks` | 11 | summarisation, question answering, JSON extraction, translation, creative writing, email, roleplay, math, logic, tabular, **data generation** |
| `ood` | 9 | **medical, financial, legal, customer support, chemistry/LaTeX, regex, shell one-liners, formal poetry, ASCII art** |

Lower-resource languages, non-mainstream code langs, and the specialised `ood`
group are deliberately far from the drafter's likely training mix — that's the
generalisation signal we're after.

`python domains.py` prints the full registry.

## Setup

```bash
pip install anthropic            # already added to ../requirements.txt
export ANTHROPIC_API_KEY=...     # or use an `ant auth login` profile
```

## Generate

```bash
# See the plan (no API calls, no writes)
python generate.py --dry-run

# One domain, tiny, to sanity-check wiring and cost
python generate.py --domains lang_hindi --n 20

# A whole group
python generate.py --group coding

# The full build — 1000 prompts/domain (800 train / 100 val / 100 test), resumable
python generate.py --group all
```

Useful flags:

- `--splits TRAIN VAL TEST` — split sizes (default `800 100 100`).
- `--n N` — shorthand: total N split 8:1:1 into train/val/test.
- `--domains k1 k2 ...` — explicit domain keys (overrides `--group`).
- `--model` — default `claude-opus-4-8`; pass `--model claude-sonnet-5` for a
  cheaper bulk run.
- `--batch-size` — prompts requested per API call (default 40).
- `--overwrite` — regenerate domains even if their splits already exist.
- `--list` — list domain keys and exit.

## How it works

1. **Seed from existing work.** For domains that already existed in the old
   `../prompts.py` generator, the deterministic prompts are loaded first and
   deduped in — we build on what was there rather than throwing it away.
2. **Top up via Claude.** Batches of diverse, domain-specific *user prompts* are
   requested using structured JSON output, rotating a "coverage angle" per batch
   and passing recent prompts to avoid to reduce near-duplicates. Everything is
   deduped (whitespace/case-normalised).
3. **Split** deterministically (fixed seed) into `train` / `val` / `test`.
4. **Write** `../data/synthetic/<domain>/{train,val,test}.jsonl` and a
   `../data/synthetic/manifest.json` summary (per-domain counts, model, seed).

Generation is **per-domain and resumable**: a domain whose splits already exist
at the requested sizes is skipped, so you can stop and restart, or build one
group at a time. If a domain comes up short, its split sizes scale down
proportionally rather than failing.

## Output layout

Datasets are written to the **shared top-level `data/` folder** (a sibling of
`DataGen/`), under the `synthetic/` subtree — `data/downloaded/` is the WildChat control
set from `../WildDataGen`.

```
../data/synthetic/
  manifest.json
  lang_english/   train.jsonl  val.jsonl  test.jsonl
  code_python/    train.jsonl  val.jsonl  test.jsonl
  ood_medical/    train.jsonl  val.jsonl  test.jsonl
  ...
```

## Notes

- **Cost.** A full 51-domain × 1000-prompt build is a lot of prompts. Start with
  a `--group` or a few `--domains` and check the manifest before committing to
  `--group all`. `--model claude-sonnet-5` cuts cost substantially for bulk runs.
- **Determinism.** Split assignment is seeded (`SEED = 1234`), so the same
  collected prompts always land in the same split. The Claude *generation* itself
  is not deterministic — regenerate a domain with `--overwrite` to refresh it.
- `../prompts.py` (the original deterministic generator) still exists and is used
  purely as a seed source here; the benchmark can keep using either.
