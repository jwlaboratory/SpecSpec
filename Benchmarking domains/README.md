# DFlash Speculative Decoding — Domain Benchmark

Benchmark the [`z-lab/Qwen3-8B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)
**drafter** against its **target** `Qwen/Qwen3-8B` across many domains (10 natural
languages, 8 programming languages, 10 general tasks), and record where the tiny
1B block-diffusion drafter tracks the 8B target well and where it doesn't.

The question we're answering: **does a tiny speculator generalize across all the
domains of the main model?** We don't care whether the *target's* answer is good —
we only care (a) that inference is correct (no bugs) and (b) how well the drafter's
proposals match the target, measured by acceptance rate, mean accept length, and
speedup.

## What DFlash is (so the metrics make sense)

- The drafter is a lightweight (**~1B**, bf16) block-diffusion model. Each decode
  step it proposes a **block of 16** tokens (position 0 is already known ⇒ **15
  draft tokens proposed per step**).
- The **target** (`Qwen/Qwen3-8B`) then verifies the whole block in one forward
  pass and accepts the longest matching prefix, plus 1 free "bonus" token.
- At **temperature 0 it is lossless**: the emitted text is *exactly* the target's
  greedy output. We exploit this as a built-in correctness check.

## Metrics collected (per prompt, aggregated per domain)

| Metric | Meaning |
|---|---|
| **acceptance rate (%)** | accepted draft tokens ÷ proposed draft tokens (15/step). The headline "does the drafter agree with the target" number. |
| **mean accept length** | tokens committed per target forward pass (accepted + 1 bonus), 1..16. This is what drives speed. |
| **forward steps** | number of target verification passes. |
| **speedup** | spec throughput ÷ target-only greedy throughput, same GPU. |
| **spec / baseline tok/s** | raw decode throughput for each path. |
| **agreement_frac** | fraction of tokens matching the sequential-greedy target before the first divergence. |
| **suspicious** | a divergence at a position where the target's top-2 logits were **far apart** (not a bf16 rounding tie). **This is the real inference-bug signal — it should be 0.** |

### On correctness (important — measured, not assumed)

DFlash greedy is lossless *in exact arithmetic*, but **bit-exact equality with HF's
sequential greedy is not expected in bf16**, and that's fine. We verified this
directly (`modal_diagnose.py`): our own sequential-greedy loop is bit-identical to
`transformers.generate`, and every spec-vs-greedy divergence occurred at a position
where the target's top-2 logits were **exactly one bf16 ULP apart (~0.25)** on trivial
near-ties like `,` vs `.`. The parallel 16-token block verification and the
one-token-at-a-time path simply round the tie differently. So the correctness gate is
**not** "100% exact match" (that would false-alarm on every long generation); it's
**zero `suspicious` (large-logit-gap) divergences**. A genuine bug — wrong positions,
cache mishandling, off-by-one — corrupts tokens with a large logit gap, which this flags.

## Do I need a GPU? (yes)

**Yes — a CUDA (NVIDIA) GPU is required.** It will not run on a Mac / Apple Silicon:
the custom DFlash kernels and the whole spec loop assume CUDA, and you need enough
VRAM to hold both models in bf16 at once:

- Qwen3-8B target ≈ **16 GB**, DFlash drafter ≈ **2 GB**, + KV cache ⇒ **~20 GB minimum**.

### Cheap / free GPU options

| Option | GPU | Cost | Notes |
|---|---|---|---|
| **Google Colab (free)** | T4 16 GB | free | ❌ **Too small** — the 8B target alone is ~16 GB in bf16. Will OOM. |
| **Google Colab Pro** | L4 24 GB / A100 40 GB | ~$10/mo | ✅ Recommended easiest path. Use `colab_dflash_benchmark.ipynb`. Pick L4 or A100 in *Runtime → Change runtime type*. |
| **RunPod** | RTX 4090 / L4 / A10 (24 GB) | ~$0.20–0.40/hr | ✅ Cheapest per-hour. Spin up a "PyTorch" pod, `git`/upload this folder, run. |
| **Vast.ai** | 3090/4090 24 GB | ~$0.15–0.35/hr | ✅ Cheapest overall, marketplace pricing. |
| **Lambda / Modal / Together** | A10/L4/A100 | ~$0.5–1.5/hr | ✅ Clean, reliable. |
| **Kaggle Notebooks** | T4 x2 / P100 16 GB | free (30 hr/wk) | ⚠️ 16 GB is tight; single-GPU won't fit bf16. Skip unless you shard. |

**Bottom line:** the free Colab T4 won't fit this. The cheapest realistic paths are
a **24 GB GPU on RunPod/Vast (~$0.30/hr)** or **Colab Pro with an L4/A100**. A full
100-prompt × 28-domain run is a few GPU-hours, so well under a few dollars.

## Install (on the GPU box)

```bash
pip install -r requirements.txt      # torch==2.9.1 (CUDA), transformers==4.57.3, accelerate
```

## Run

```bash
# Quick smoke test (2 domains, 5 prompts each) — verifies wiring + correctness check
python benchmark.py --run-name smoke --limit 5 --max-new-tokens 256 \
    --categories lang_english code_python
python aggregate.py results/smoke.jsonl

# Full run: all 28 domains x 100 prompts (the task as specified)
./run.sh                                   # or:
python benchmark.py --run-name dflash_bench --limit 100 --max-new-tokens 512 --categories all
python aggregate.py results/dflash_bench.jsonl
```

Useful flags / env for `run.sh`: `LIMIT`, `MAX_NEW_TOKENS`, `CATEGORIES`
(`all` | `languages` | `coding` | `tasks` | explicit names), `EXTRA_ARGS`
(e.g. `--no-baseline`, `--resume`).

`benchmark.py` options:
- `--limit N` prompts per category (1..100)
- `--categories ...` names or a group (`languages` / `coding` / `tasks` / `all`)
- `--max-new-tokens` generation length cap (default 512)
- `--no-baseline` skip the target-only pass — **faster but loses speedup AND the
  lossless correctness check**. Keep baseline on for at least one run.
- `--resume` continue a run, skipping prompts already written
- `--attn {sdpa,eager,flash_attention_2}` (default `sdpa`, portable)

## Outputs

- `results/<run>.jsonl` — one JSON line per prompt (all raw metrics + `committed_per_step`).
- `results/<run>_by_category.csv` — one row per domain, ranked by acceptance.
- `results/<run>_report.md` — human-readable report: overall numbers, all-domains
  table, best/worst-tracked domains, and a ⚠️ section listing any correctness mismatches.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Deterministic generator: 28 domains × 100 prompts (fixed seed). |
| `spec_patch.py` | Instrumented copy of DFlash's `spec_generate` that also returns per-step acceptance counts (upstream discards them). |
| `benchmark.py` | Loads both models, runs spec + baseline, streams per-prompt metrics. |
| `aggregate.py` | Rolls JSONL up into per-domain CSV + markdown report. |
| `run.sh` | One-command install-check + benchmark + aggregate. |
| `colab_dflash_benchmark.ipynb` | Colab notebook (L4/A100 runtime). |

## Reading the results

- **High acceptance % + high mean length** (usually English, code, math) → the
  drafter generalizes to that domain; big speedups.
- **Low acceptance %** (often low-resource scripts / languages the drafter saw
  little of) → the drafter mispredicts the target frequently; small or no speedup.
  This is the "where does a tiny speculator fail to generalize" signal you want.
- **`lossless_match` < 100%** → **stop and inspect**: at temp 0 this means the spec
  path diverged from the target's greedy output, i.e. an inference/harness bug, not
  a domain-quality issue. The report calls these out explicitly.

## Notes

- Uses the HuggingFace **transformers** path (`spec_generate`) because it exposes
  per-step acceptance cleanly. The model card also documents vLLM and SGLang server
  backends (faster, but harder to extract per-step acceptance and they need
  nightly/PR builds). Throughput numbers here are apples-to-apples (spec vs
  baseline on the *same* transformers stack), which is what the speedup metric needs.
- `enable_thinking=False` in the chat template — this DFlash drafter is trained for
  Qwen3 thinking-mode-disabled, per the model card.
