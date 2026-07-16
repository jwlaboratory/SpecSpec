# DFlash Speculative Decoding — Domain Benchmark

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
Benchmarking domains/
├── data/             # the datasets (shared, top-level — the actual data)
│   ├── synthetic/<domain>/    Claude-generated prompts        (from DataGen)
│   ├── wild/<domain>/         sorted real WildChat prompts    (WildDataGen/sort.py)
│   └── downloaded/<domain>/   straight from HF datasets       (WildDataGen/sources.py)
│                              …each with train.jsonl · val.jsonl · test.jsonl
├── DataGen/          # Claude generator                 ->  ../data/synthetic
│   ├── generate.py       Claude-based prompt-dataset generator
│   ├── domains.py        the 51-domain registry
│   └── README.md
├── WildDataGen/      # real-prompt collectors
│   ├── sort.py           sort WildChat into domains     ->  ../data/wild
│   ├── sources.py        pull purpose-built HF datasets ->  ../data/downloaded
│   ├── router.py         domain classifiers
│   └── README.md
├── scripts/          # everything that runs the benchmark
│   ├── benchmark_vllm.py   PRIMARY: unified vLLM runner, both speculators, batched
│   ├── modal_run_vllm.py   run the vLLM benchmark on Modal (per method × source)
│   ├── compare_charts.py   overlay speculators per domain (grouped bars)
│   ├── make_charts.py      CSV -> per-run charts (PNG)
│   ├── benchmark.py        REFERENCE: transformers DFlash runner (batch=1, lossless check)
│   ├── spec_patch.py       instrumented DFlash spec_generate (per-step accepts)
│   ├── aggregate.py        JSONL -> per-domain CSV + report (transformers path)
│   ├── modal_run.py        Modal runner for the transformers path (sharded)
│   ├── prompts.py          legacy deterministic generator (kept as a seed source)
│   ├── run.sh              local one-command runner (transformers path)
│   └── colab_dflash_benchmark.ipynb
├── results/          # the report, charts, and results data
│   ├── <run>.jsonl               one JSON line per prompt (raw metrics)
│   ├── <run>_by_category.csv     one row per domain
│   ├── <run>_report.md           ranked human-readable report
│   └── charts/                   per-domain + distribution PNGs (and screenshots)
├── README.md
└── requirements.txt
```

## 1 · Generate the datasets  (no GPU)

See `DataGen/README.md` for the full story. Quick version:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or an `ant auth login` profile
cd DataGen
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
python compare_charts.py ../results/dflash_synthetic_by_category.csv \
                         ../results/eagle3_synthetic_by_category.csv
# -> results/charts/compare_acceptance.png  (grouped bars, per domain)
```

vLLM **nightly** is required (DFlash's `method: "dflash"` support); the Modal image
installs it. `benchmark_vllm.py` reads `data/<source>/<domain>/test.jsonl` batched
and reads acceptance from vLLM's spec-decode counters.

**Reference path — transformers (DFlash only, batch=1).** `benchmark.py` +
`spec_patch.py` run DFlash through HuggingFace transformers one prompt at a time.
Slower and DFlash-only, but it also does the temp-0 **lossless correctness check**
(`suspicious` divergences) that the vLLM path doesn't expose. Keep it for validation:

```bash
cd scripts
python benchmark.py --run-name dflash_ref --split test --categories all   # needs a GPU
# or ./run.sh ; or  modal run modal_run.py::full --run-name dflash_ref
```

**Real-prompt controls.** `WildDataGen/` produces two real-prompt sets to compare
against the synthetic one: `data/wild` (genuine WildChat prompts, sorted into the
domains) and `data/downloaded` (straight from purpose-built HF datasets — medical,
legal, financial, SQL). Run the same benchmark against each (`--datagen-dir
../data/wild` or `../data/downloaded`) and compare per-domain acceptance vs
synthetic — this checks whether the clean, low-perplexity synthetic prompts inflate
acceptance or hide the drafter's failure modes. See `WildDataGen/README.md`.

## 3 · Read the results

Everything lands in `results/`:

- `<run>_report.md` — overall numbers, all-domains table ranked by acceptance,
  best/worst-tracked domains, and a ⚠️ section for any correctness mismatches.
- `<run>_by_category.csv` — one row per domain.
- `charts/` — acceptance-by-domain, mean-length, speedup, group summary, and the
  acceptance distribution (coloured by domain group). Drop screenshots here too.

## Metrics

| Metric | Meaning |
|---|---|
| **acceptance rate (%)** | accepted draft tokens ÷ proposed draft tokens (15/step). The headline "does the drafter agree with the target" number. |
| **mean accept length** | tokens committed per target forward pass (accepted + 1 bonus), 1..16. Drives speed. |
| **forward steps** | number of target verification passes. |
| **speedup** | spec throughput ÷ target-only greedy throughput, same GPU. |
| **agreement_frac** | fraction of tokens matching sequential-greedy target before first divergence. |
| **suspicious** | a divergence where the target's top-2 logits were far apart (not a bf16 tie). The real inference-bug signal — should be **0**. |

**On correctness.** At temperature 0 DFlash greedy is lossless *in exact arithmetic*,
but bit-exact equality with HF sequential greedy is **not** expected in bf16 and that's
fine. The correctness gate is not "100% exact match" — it's **zero `suspicious`
(large-logit-gap) divergences**. A genuine bug (wrong positions, cache mishandling,
off-by-one) corrupts tokens with a large logit gap, which `suspicious` flags.

**Reading it.** High acceptance % + high mean length (usually English, code, math) →
the drafter generalises there; big speedups. Low acceptance % (often low-resource
scripts / specialised OOD domains) → the drafter mispredicts the target frequently;
small or no speedup — the "where does a tiny speculator fail to generalise" signal we
want.

## GPU notes

Needs a CUDA GPU with **~20 GB+ VRAM** (Qwen3-8B target ≈16 GB + DFlash drafter ≈2 GB
+ KV cache, all bf16). The free Colab T4 (16 GB) OOMs. Cheapest realistic paths: a
24 GB GPU on RunPod/Vast (~$0.30/hr), Colab Pro (L4/A100), or Modal (`scripts/modal_run.py`).
A full test-split run (100 prompts × 51 domains) is a few GPU-hours.
