# WildDataGen — real-prompt control set from WildChat

Sorts genuine human prompts from **WildChat** into the same 51 domains as
`DataGen/`, producing the identical split layout in the shared top-level data
folder at `../data/wild/<domain>/{train,val,test}.jsonl` (alongside
`../data/synthetic/` from DataGen).

It exists to answer the "are the synthetic prompts representative?" question: the
Claude-generated `DataGen` prompts are clean and low-perplexity, which can inflate
speculative-decoding acceptance and hide the drafter's worst cases. WildDataGen is
the **real-human-prompt control** — run the same benchmark on it and compare
per-domain acceptance against the synthetic set.

The prompts here are 100% real human text. Only the *routing label* (which domain a
prompt belongs to) is assigned by a classifier — so this does **not** reintroduce
synthetic-text bias.

## Setup

WildChat is gated on Hugging Face:

1. Accept the terms on the dataset page (e.g. `allenai/WildChat-1M`).
2. `huggingface-cli login` (or set `HF_TOKEN`).

`datasets` and (for `--classifier claude`) `anthropic` come from `../requirements.txt`.

## Sort

```bash
python sort.py --dry-run                       # show the plan, no download
python sort.py --group all                     # heuristic routing, all domains
python sort.py --classifier claude --group all # Claude-labelled (better coverage)
```

It **streams** WildChat (no full download), takes each conversation's first user
turn, filters (length, skips toxic), routes to a domain, and stops once every
domain hits its target or `--max-scan` rows are seen.

Flags: `--splits TRAIN VAL TEST` (default `800 100 100`), `--n N` (8:1:1 shorthand),
`--domains`/`--group`, `--classifier {heuristic,claude}`, `--model`, `--max-scan`,
`--min-len`/`--max-len`, `--keep-toxic`, `--dataset`, `--overwrite`, `--dry-run`.

## Dedicated sources for specialised domains (`sources.py`)

WildChat is general chat, so its **specialised** buckets (medical, legal, financial,
SQL/data-analysis) come up thin. `sources.py` fills those from **purpose-built public
datasets** instead — no classification needed, the whole dataset *is* the domain:

| Domain | Dataset |
|---|---|
| `ood_medical` | `keivalya/MedQuAD-MedicalQnADataset` (real medical Q&A) |
| `ood_financial` | `gbharti/finance-alpaca` |
| `ood_legal` | `nguha/legalbench` (consumer-contract Q&A) |
| `code_sql` | `b-mc2/sql-create-context` (text-to-SQL with schema) |

```bash
python sources.py --list                 # show the registry
python sources.py                         # fill all dedicated domains into ../data/wild
python sources.py --domains ood_medical  # just one
```

Writes to the same `../data/wild/<domain>/` tree as the WildChat sort — **run it
after `sort.py`** so the dedicated (higher-quality) data overwrites WildChat's thin
versions for those domains. Add more domains by dropping an entry in `SOURCES`
(e.g. `bitext/...` for customer support, `openai_humaneval` for code, `HuggingFaceH4/no_robots`
for general human prompts).

## Two classifiers

| | Heuristic (default) | Claude (`--classifier claude`) |
|---|---|---|
| Cost | free, deterministic | API calls (batched) |
| Languages | WildChat's own `language` label — reliable | same, plus better topical routing |
| Code / task / OOD | keyword + code-fence rules; **high precision, lower recall** (clear matches land in-domain, the rest fall to the language bucket) | balanced coverage across the full taxonomy |

Real prompts are unevenly distributed — WildChat is English-heavy, so rare
languages (Swahili, Vietnamese) and niche code langs (Haskell, Kotlin, R) will
come up short. Those domains get smaller control sets (splits scale down
proportionally); the manifest records per-domain counts.

## Benchmark it (same as DataGen)

`benchmark.py` is source-agnostic — just point it at the `data/wild` dir:

```bash
cd ../scripts
python benchmark.py --datagen-dir ../data/wild --split test \
    --run-name dflash_wild --categories all
python aggregate.py ../results/dflash_wild.jsonl
python make_charts.py ../results/dflash_wild_by_category.csv
```

Then compare `dflash_wild` (real) vs `dflash_bench` (synthetic) per domain: if they
track, the synthetic set is validated; where synthetic acceptance is much higher,
you've quantified the inflation.

## Files

| File | Purpose |
|---|---|
| `sort.py` | stream WildChat → route → split → write |
| `router.py` | domain classifiers (heuristic + Claude), taxonomy from `../DataGen/domains` |
| `../data/wild/` | the sorted real prompts (generated, in the shared data folder) |
