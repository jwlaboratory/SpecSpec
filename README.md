# SpecSpec — specializing speculative-decoding drafters

**Question:** can a small, cheap LoRA adapter make a speculative-decoding drafter
track its target model better on a chosen domain — and is one combined adapter as
good as per-domain specialists?

**Answer (short):** yes — on both speculators. LoRA specialization works reliably
on the weak speculator (DFlash: positive on every domain tested), gains scale
with how weak the base is, and a single combined adapter matches specialists at
3–5 domains and nearly matches them out to 40. EAGLE3 initially looked immune —
that turned out to be a train/serve alignment bug (a missing one-token shift in
the TTT forward); once fixed, EAGLE gains on 5/5 multilingual domains
(+0.6..+2.1pp, combined ≥ own) and sits flat, not negative, on its strongest
domains. Full fine-tuning the 1B drafter ≈ base everywhere — the rank-16 adapter
(~2M params) is the better tool. Exp 06 had predicted the EAGLE result from the
other direction: an independent Qwen3-0.6B drafter (vanilla spec decode, no
feature conditioning, trivially aligned plain-CE channel) specializes on 5/5
domains — the strongest base drafter tested, the biggest per-proposal gains in
the repo — though its drafting latency makes vanilla speculation itself a
wall-clock loss at batch-1.

Read **[REPORT.md](REPORT.md)** for the full story;
**[EXPERIMENTS.md](EXPERIMENTS.md)** is the run-by-run ledger (every experiment,
config, verdict, and where its results live).

## Stack

| component | choice |
|---|---|
| target | `Qwen/Qwen3-8B` (frozen everywhere) |
| speculator A | **DFlash** `z-lab/Qwen3-8B-DFlash-b16` — 1B block-diffusion drafter |
| speculator B | **EAGLE3** `RedHatAI/Qwen3-8B-speculator.eagle3` — 1-layer autoregressive head |
| speculator C | **independent** `Qwen/Qwen3-0.6B` — vanilla two-model spec decode, no feature conditioning |
| adaptation | unmerged LoRA on q/k/v/o (r16 α=32 unless stated); self-distillation on the target's own generations |
| compute | Modal H200s (volume `code-sql-pipeline`) |

## Map

```
REPORT.md                 the full report — read this first
EXPERIMENTS.md            chronological ledger of every run
lib/                      shared modules every experiment vendors into Modal:
                            lora.py (LoRALinear), online_dflash.py (SpecForge
                            DFlash loss), spec_patch.py (instrumented spec_generate),
                            vanilla_spec.py (two-model Leviathan-style spec decode)
data/                     canonical datasets, train/val/test.jsonl per domain
  synthetic/<domain>/       Claude-generated prompts (51 domains)
  wild/<domain>/            real WildChat prompts sorted into the same domains
  downloaded/<domain>/      straight from HF datasets (incl. exp-01's code_sql,
                            ood_indian_legal)
experiments/              every experiment, one folder each; each has
                          README + pipeline/scripts + results/
  00-base-benchmarks/       measures the *base* speculators across all domains
                            (datagen/ + wilddatagen/ build data/, scripts/ bench it)
  01-single-domain-dflash/  LoRA vs full fine-tune on code_sql + indian legal
  02-multilingual-dflash/   5 language LoRAs + combined (r16)
  03-weird-domains/         translation/roleplay/poetry, DFlash + EAGLE (r16)
  04-multilingual-eagle/    EAGLE3 replication (v1+v2 archived invalid, v3 current)
  05-interference-ladder/   one combined LoRA vs specialists at 10/20/40 domains
  06-independent-drafter/   vanilla spec decode (Qwen3-0.6B drafter), CE-only LoRAs
  07-rank-ladder/           adapter capacity: r4 vs r16 vs r64 across 02+03 domains
  08-wallclock/             net wall-clock speedup: spec decode vs plain decoding
router/                   MLP on target hidden states that picks the adapter (100% test acc)
serving/                  multi-adapter serving: N LoRAs over one resident backbone
third_party/SpecForge/    gitignored clone — git clone https://github.com/sgl-project/SpecForge.git third_party/SpecForge
MODELS.md                 where every trained checkpoint lives (local + Modal volume)
```

Trained adapters (`*.pt`) are gitignored; see [MODELS.md](MODELS.md). Results
(jsonl, csv, reports, charts) are tracked in each experiment's `results/`.
Invalid runs are archived in place (e.g. `results-v1-*-bug/`), never deleted —
the ledger explains each one.
