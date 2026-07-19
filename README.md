# SpecSpec — specializing speculative-decoding drafters

**Question:** can a small, cheap LoRA adapter make a speculative-decoding drafter
track its target model better on a chosen domain — and is one combined adapter as
good as per-domain specialists?

**Answer (short):** yes. LoRA specialization works reliably on the weak speculator
(DFlash: positive on every domain tested), gains scale with how weak the base is,
a single combined adapter matches specialists at 3–5 domains and nearly matches
them out to 40, and none of it helps the already-strong speculator (EAGLE3) except
on its weakest domain. Full fine-tuning the 1B drafter ≈ base everywhere — the
rank-16 adapter (~2M params) is the better tool.

Read **[REPORT.md](REPORT.md)** for the full story;
**[EXPERIMENTS.md](EXPERIMENTS.md)** is the run-by-run ledger (every experiment,
config, verdict, and where its results live).

## Stack

| component | choice |
|---|---|
| target | `Qwen/Qwen3-8B` (frozen everywhere) |
| speculator A | **DFlash** `z-lab/Qwen3-8B-DFlash-b16` — 1B block-diffusion drafter |
| speculator B | **EAGLE3** `RedHatAI/Qwen3-8B-speculator.eagle3` — 1-layer autoregressive head |
| adaptation | unmerged LoRA on q/k/v/o (r16 α=32 unless stated); self-distillation on the target's own generations |
| compute | Modal H200s (volume `code-sql-pipeline`) |

## Map

```
REPORT.md                 the full report — read this first
EXPERIMENTS.md            chronological ledger of every run
lib/                      shared modules every experiment vendors into Modal:
                            lora.py (LoRALinear), online_dflash.py (SpecForge
                            DFlash loss), spec_patch.py (instrumented spec_generate)
data/                     canonical datasets, train/val/test.jsonl per domain
  synthetic/<domain>/       Claude-generated prompts (51 domains)
  wild/<domain>/            real WildChat prompts sorted into the same domains
  downloaded/<domain>/      straight from HF datasets (incl. exp-01's code_sql,
                            ood_indian_legal)
benchmarking/             measures the *base* speculators across all domains
                          (datagen/ + wilddatagen/ build data/, scripts/ bench it)
experiments/              the finetuning experiments, one folder each, numbered
                          like the ledger; each has README + pipeline + results/
  01-single-domain-dflash/  LoRA vs full fine-tune on code_sql + indian legal
  02-multilingual-dflash/   5 language LoRAs + combined; r64 rank scaling
  03-weird-domains/         translation/roleplay/poetry, DFlash + EAGLE, r4/r16/r64 ladder
  04-multilingual-eagle/    EAGLE3 replication (v1 archived invalid, v2 current)
  05-interference-ladder/   one combined LoRA vs specialists at 10/20/40 domains
router/                   MLP on target hidden states that picks the adapter (100% test acc)
serving/                  multi-adapter serving: N LoRAs over one resident backbone
third_party/SpecForge/    gitignored clone — git clone https://github.com/sgl-project/SpecForge.git third_party/SpecForge
MODELS.md                 where every trained checkpoint lives (local + Modal volume)
```

Trained adapters (`*.pt`) are gitignored; see [MODELS.md](MODELS.md). Results
(jsonl, csv, reports, charts) are tracked in each experiment's `results/`.
Invalid runs are archived in place (e.g. `results-v1-*-bug/`), never deleted —
the ledger explains each one.
