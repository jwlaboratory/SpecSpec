# Truncation Fix & Rerun — Status

_Working doc. Last updated: 2026-07-23._

## Goal
The blog's training pipelines truncated sequences at `max_seq_len=512`, keeping
the first 512 tokens. Since responses sit at the *end*, prompt-heavy records lost
their whole response and contributed **zero training signal**. Fix it (no
truncation, train + test), re-run the affected experiments, get new numbers,
update figures, assess blog validity.

## The bug (quantified)
Truncation keeps `input_ids[:512]`; records with `prompt_len` ≳ 480 have no
anchorable response left and are dropped by `run_batch`'s `keep` mask. Severity
tracks tokenizer bloat (non-Latin scripts blow past 512):

| language | script | records dropped | response tokens lost |
|---|---|---:|---:|
| English / Chinese / Russian / Arabic | Latin/CJK | 5–9% | 2–7% |
| German / Spanish | Latin | 9–10% | 9–14% |
| Yoruba | Latin+diacritics | 20% | 16% |
| Hebrew | Hebrew | 30% | 17% |
| Hindi | Devanagari | **48%** | **53%** |

Systemic: every training pipeline (`new/exp1-language`, exps 01,02,03,05,06,09,10,12)
did `input_ids[:512]`. **Test/bench path never truncated** (full prompts,
pre-filtered to fit 2048 − max_new_tokens).

Key reassurance: benchmark ran full-length prompts, so **no reported gain is
inflated** — truncation only limited what training saw, biasing *against* the weak
non-Latin languages. True gains are ≥ reported.

## Fix applied
- `max_seq_len` default `512 → 0` (no truncation; capture already caps at 2048).
- Token-budgeted batching (`batch_tokens`) so full-2048 sequences don't OOM while
  short-sequence languages still batch wide.
- Training moved to **H200** (`GPU_TRAIN`).
- Files: `new/exp1-language/pipeline.py`, `experiments/12-language-full-finetune/pipeline.py`.

## Status

| item | status |
|---|---|
| Fix `new/exp1-language/pipeline.py` | ✅ done |
| Validate on Hindi (H200) | ✅ val loss 1.00→0.62, no OOM |
| Retrain 40 own LoRAs (no-trunc, H200) | ✅ done, saved |
| **Retrain combined LoRA** (no-trunc, H200) | ✅ done (3 ep, step 8700), saved |
| Re-bench all 40 × {base,own,combined} | ✅ done (batch-1 HF harness) |
| Aggregate → new `summary.json` | ✅ done, pulled local |
| Regenerate exp1 charts | ✅ done (15:12) |
| Training-curve chart (clean, headline lang) | ✅ Swedish dense curve (val 4.78→3.60) rendered + in blog |
| Fix + rerun exp12 (LoRA vs full-FT) | ✅ done: own +1.54pp vs full-FT +0.19pp (was +1.37/+0.10) |
| Blog edits (both) + figures repointed | ✅ done (exp1 tables, full-FT tables, methodology notes) |

## Remaining
- **Task #6 — batch-size net-speedup serving experiment** (NEW, not truncation).
  Needs design + go-ahead. Not started.
- **Commit** — everything is uncommitted in the working tree.

## NEW NUMBERS (no-truncation vs truncated), 40 langs
Base acceptance **identical** old vs new (deterministic greedy bench — sanity ✓).
Only own/combined adapters changed.

- **Mean own-LoRA gain: +0.634pp → +0.704pp** (higher, as predicted)
- **Mean combined-LoRA gain: +0.843pp** (combined still ≥ own — transfer story intact)
- Biggest increases on prompt-heavy / previously-truncated langs:
  - Swedish +0.88 → **+2.09pp**
  - Hindi +0.69 → **+1.08pp** (the 48%-dropped language)
  - Indonesian +1.34→+1.50, Ukrainian +1.48→+1.58, Russian +0.30→+0.41,
    Estonian +0.38→+0.48, French +0.13→+0.26, Maori −0.01→+0.08
- Headroom law intact: English −0.09, Welsh −0.07, Somali −0.02 (no headroom)
- **No blog claim invalidated; weak-language gains are LARGER → headroom law cleaner.**
- Caveat: `speedup_measured` is batch-1 (see task #6); acceptance is batch-invariant.

## Notes / decisions
- Earlier full run was killed mid-bench by a client heartbeat drop — **all training
  survived** (LoRAs committed to volume); resumed with `launch --skip-data --skip-train`.
- Out of scope (appendix + note, per user): exps 02-multilingual, 07-rank-ladder,
  05-interference, 03-weird-domains, 10-english-subdomains.

## Open / requested experiments (not truncation-related)
- **Net speedup vs batch size** (serving): sweep batch sizes for net speedup incl.
  adapter swap cost and full merged-adapter reload cost. Motivation: spec-decode
  speedup collapses as batch grows (memory-bound → compute-bound); merge vs
  hot-swap tradeoff shifts with batch. Extends exps 09/11 (currently batch-1).
  → needs design (engine, batch sizes, modes); batched spec-decode harness may
  need work. NOT started.
