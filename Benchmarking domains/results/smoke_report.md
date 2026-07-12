# DFlash Speculative Decoding — Domain Benchmark

Drafter `z-lab/Qwen3-8B-DFlash-b16` + target `Qwen/Qwen3-8B`  ·  6 prompts across 2 domains.

## Overall

- **Acceptance rate (pooled):** 21.8%  (fraction of the 15 draft tokens/step the target accepts)
- **Mean accept length:** 4.89 tokens per target pass (max = block size 16)
- **Spec throughput:** 97 tok/s
- **Mean speedup vs target-only greedy:** 3.49×
- **Lossless match rate:** 16.7%  (5 mismatches)  ⚠️ MISMATCHES — inference bug!

## All domains (ranked by acceptance rate)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Lossless |
|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 144 | 28 | 5.18× | ⚠️0% |
| lang_english | 10.5 | 2.47 | 48 | 123 | 50 | 28 | 1.80× | ⚠️33% |

## Best-tracked domains (drafter matches target well)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Lossless |
|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 144 | 28 | 5.18× | ⚠️0% |
| lang_english | 10.5 | 2.47 | 48 | 123 | 50 | 28 | 1.80× | ⚠️33% |

## Worst-tracked domains (drafter struggles)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Lossless |
|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 144 | 28 | 5.18× | ⚠️0% |
| lang_english | 10.5 | 2.47 | 48 | 123 | 50 | 28 | 1.80× | ⚠️33% |

## ⚠️ Correctness mismatches

At temperature 0 DFlash is lossless, so any mismatch means the spec output diverged from the target's greedy output — a bug in the inference/harness code, not a model-quality issue. Domains with mismatches:

- `code_python`: 3/3 prompts diverged
- `lang_english`: 2/3 prompts diverged

## Metric definitions

- **Accept %** — pooled acceptance rate = accepted draft tokens ÷ proposed draft tokens (15 proposed per target pass, block size 16).
- **Mean len** — average tokens committed per target forward pass (accepted drafts + 1 bonus token). Higher = fewer target passes = faster.
- **Speedup** — spec throughput ÷ target-only greedy throughput (same hardware).
- **Lossless** — % of prompts where spec output == target greedy output. Should be 100%; anything less is an inference bug.
