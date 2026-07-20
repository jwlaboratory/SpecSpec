# 00 — base-speculator domain benchmarks

*(Experiment 0 of the repo: characterizes the BASE speculators across all
domains — every fine-tuning experiment builds on the headroom map produced
here. Formerly the top-level `benchmarking/` folder.)*

Measure where a tiny **1B block-diffusion drafter** (`z-lab/Qwen3-8B-DFlash-b16`)
tracks its **8B target** (`Qwen/Qwen3-8B`) well and where it doesn't, across many
domains — natural languages, programming languages, general tasks, and deliberately
out-of-distribution / specialised domains.

**The question:** does a tiny speculator generalise across all the domains of the
main model? We don't care whether the *target's* answer is good — only (a) that
inference is correct and (b) how well the drafter's proposals match the target,
measured by acceptance rate, mean accept length, and speedup.

> **Current goal (not training yet):** ① build clean per-domain datasets, ② run the
> target + drafter over each domain's held-out test set and see which domains the
> drafter tracks best/worst and what the distribution looks like. Training better
> speculators (e.g. a LoRA on the diffusion drafter) is a later, separate step.

## Layout

```
benchmarking/   (datasets live in ../data/)
├── data/             # the datasets (shared, top-level — the actual data)
│   ├── synthetic/<domain>/    Claude-generated prompts        (from datagen)
│   ├── wild/<domain>/         sorted real WildChat prompts    (Wilddatagen/sort.py)
│   └── downloaded/<domain>/   straight from HF datasets       (Wilddatagen/sources.py)
│                              …each with train.jsonl · val.jsonl · test.jsonl
├── datagen/          # Claude generator                 ->  ../data/synthetic
│   ├── generate.py       Claude-based prompt-dataset generator
│   ├── domains.py        the 51-domain registry
│   └── README.md
├── Wilddatagen/      # real-prompt collectors
│   ├── sort.py           sort WildChat into domains     ->  ../data/wild
│   ├── sources.py        pull purpose-built HF datasets ->  ../data/downloaded
│   ├── router.py         domain classifiers
│   └── README.md
├── scripts/          # the vLLM speculator benchmark
│   ├── benchmark_vllm.py   unified vLLM runner: both speculators, batched, per-domain
│   ├── modal_run_vllm.py   run it on Modal (per method × source; ::all for everything)
│   ├── make_charts.py      CSV -> per-run charts (PNG)
│   ├── compare_charts.py   overlay speculators per domain (accept rate + mean len)
│   ├── overview_chart.py   all sources × both speculators (grouped bars)
│   └── prompts.py          legacy deterministic prompts (datagen seed source)
├── results/          # the results data + charts
│   ├── <method>_<source>_by_category.csv   one row per domain
│   ├── <method>_<source>_report.md         ranked human-readable report
│   └── charts/                             per-domain, compare, and overview PNGs
├── README.md
└── requirements.txt
```

## 1 · Generate the datasets  (no GPU)

See `datagen/README.md` for the full story. Quick version:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or an `ant auth login` profile
cd datagen
python generate.py --group all --model claude-sonnet-5 --concurrency 4
```

Produces `data/synthetic/<domain>/{train,val,test}.jsonl` (default **800 / 100 / 100**
per domain) across **51 domains** (16 natural languages, 15 programming languages,
11 general tasks, 9 specialised/OOD domains). Rows are prompt-only:
`{"prompt": ..., "domain": ...}`. Resumable per domain.

## 2 · Run the benchmark  (needs a CUDA GPU)

Two **speculators** are benchmarked against the same target `Qwen/Qwen3-8B`:

| Speculator | method | model |
|---|---|---|
| DFlash | `dflash` | `z-lab/Qwen3-8B-DFlash-b16` (block-diffusion, 15 spec tokens) |
| EAGLE3 | `eagle3` | `RedHatAI/Qwen3-8B-speculator.eagle3` (3 spec tokens) |

Both run through **one vLLM setup** — the *same* domain test prompts, batched, both
speculators, one stack — recording per-domain **acceptance rate** and **mean accept
length** (the batch-independent, generalizable numbers; we don't chase speedup).

**Primary path — unified vLLM on Modal:**

```bash
cd scripts
# one (method, source):
modal run modal_run_vllm.py::main --method eagle3 --source synthetic
modal run modal_run_vllm.py::main --method dflash --source wild
# smoke test first:
modal run modal_run_vllm.py::main --method eagle3 --source downloaded --limit 5
# everything — both speculators × all three sources, in parallel:
modal run modal_run_vllm.py::all
```

Each run writes `results/<method>_<source>_by_category.csv` + `_report.md`, and
`make_charts.py` renders the per-domain charts. Then overlay the speculators:

```bash
# per-domain overlay of the two speculators (acceptance rate + mean accept length):
python compare_charts.py ../results/dflash_synthetic_by_category.csv \
                         ../results/eagle3_synthetic_by_category.csv
# all sources × both speculators in one view:
python overview_chart.py ../results/*_by_category.csv
```

vLLM **nightly** is required for DFlash's `method: "dflash"` support; the Modal image
installs it. `benchmark_vllm.py` reads `data/<source>/<domain>/test.jsonl` batched and
reads acceptance from vLLM's spec-decode counters (`get_metrics()`).

**Real-prompt controls.** Beyond `data/synthetic`, `Wilddatagen/` produces two
real-prompt sets in the same layout: `data/wild` (genuine WildChat prompts, sorted
into the domains) and `data/downloaded` (straight from purpose-built HF datasets —
medical, legal, financial, SQL). Benchmark each source and compare per-domain
acceptance vs synthetic — this checks whether the clean, low-perplexity synthetic
prompts inflate acceptance or hide a speculator's failure modes. See
`Wilddatagen/README.md`.

## 3 · Read the results

Everything lands in `results/`:

- `<method>_<source>_report.md` — domains ranked by acceptance rate.
- `<method>_<source>_by_category.csv` — one row per domain.
- `charts/` — per-run (acceptance/mean-length/group/distribution), `compare_*`
  (DFlash vs EAGLE3 per domain), and `overview_*` (all sources × both speculators).

## Metrics

Per domain, for each speculator:

| Metric | Meaning |
|---|---|
| **acceptance rate** | accepted draft tokens ÷ proposed draft tokens. The batch-independent, generalizable "does the speculator predict the target's next token" number. **The fair cross-speculator metric** — it normalises for the fact that DFlash proposes 15 tokens/step and EAGLE3 proposes 3. |
| **mean accept length** | tokens committed per target forward pass (accepted + 1 bonus). Higher = fewer target passes. Not directly comparable across speculators (DFlash's ceiling is ~16, EAGLE3's ~4) — read it as a within-speculator speed proxy. |
| **forward steps** | number of target verification passes (`num_drafts`). |

Speculative decoding is lossless at temperature 0, so the emitted text equals the
target's greedy output regardless of speculator — we don't measure output quality,
only how well each speculator's proposals match the target.

**Reading it.** High acceptance (usually English, code, math) → the speculator
generalises to that domain. Low acceptance (often low-resource scripts / specialised
OOD domains) → it mispredicts the target frequently. Comparing sources
(synthetic vs wild vs downloaded) shows whether clean synthetic prompts overstate the
numbers vs real ones.

## GPU notes

The benchmark runs on Modal (`scripts/modal_run_vllm.py`) — an `A100-40GB` per
(method, source), which comfortably holds the Qwen3-8B target + 1B speculator + KV
cache in bf16. A full test-split run (100 prompts × ~49 domains) is ~10–15 min per
source, batched. To run on your own GPU box instead, install vLLM (`pip install -U
vllm --extra-index-url https://wheels.vllm.ai/nightly`) and call `benchmark_vllm.py`
directly.
