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
├── DataGen/          # the datasets + the code that generates them
│   ├── generate.py       Claude-based prompt-dataset generator
│   ├── domains.py        the 51-domain registry
│   ├── data/<domain>/    train.jsonl · val.jsonl · test.jsonl   (the actual data)
│   └── README.md         dataset docs
├── scripts/          # everything that runs the benchmark
│   ├── benchmark.py      loads both models, runs spec + baseline per prompt
│   ├── aggregate.py      JSONL -> per-domain CSV + markdown report
│   ├── make_charts.py    CSV -> charts (PNG)
│   ├── spec_patch.py     instrumented DFlash spec_generate (per-step accepts)
│   ├── modal_run.py      run it on Modal (GPU in the cloud), sharded
│   ├── prompts.py        legacy deterministic generator (kept as a seed source)
│   ├── run.sh            local one-command runner
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

Produces `DataGen/data/<domain>/{train,val,test}.jsonl` (default **800 / 100 / 100**
per domain) across **51 domains** (16 natural languages, 15 programming languages,
11 general tasks, 9 specialised/OOD domains). Rows are prompt-only:
`{"prompt": ..., "domain": ...}`. Resumable per domain.

## 2 · Run the benchmark  (needs a CUDA GPU, ~20 GB VRAM)

The benchmark reads each domain's **held-out `test` split** and runs the drafter +
target over it. It does **not** run on Mac/Apple Silicon (custom CUDA kernels).

**On Modal (recommended — GPU in the cloud, sharded):**

```bash
cd scripts
modal run modal_run.py::full --run-name dflash_bench          # all domains, test split, sharded
# smoke test first:
modal run modal_run.py --categories "lang_english code_python" --limit 5 --run-name smoke
```

**On a local/rented GPU box (RunPod, Vast, Lambda, Colab):**

```bash
cd scripts
./run.sh                        # RUN_NAME/LIMIT/SPLIT/CATEGORIES/EXTRA_ARGS via env
# or directly:
python benchmark.py --run-name dflash_bench --split test --categories all
python aggregate.py ../results/dflash_bench.jsonl
python make_charts.py ../results/dflash_bench_by_category.csv
```

`benchmark.py` reads from `DataGen/data` by default (`--prompt-source datagen
--split test`); `--categories` accepts domain keys or a group
(`languages`/`coding`/`tasks`/`ood`/`all`). Pass `--prompt-source legacy` to use the
old deterministic `prompts.py` instead.

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
