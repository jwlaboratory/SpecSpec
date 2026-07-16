# DFlash Speculative Decoding — Domain Benchmark

Drafter `z-lab/Qwen3-8B-DFlash-b16` + target `Qwen/Qwen3-8B`  ·  9 prompts across 3 domains.

## Overall

- **Acceptance rate (pooled):** 26.2%  (fraction of the 15 draft tokens/step the target accepts)
- **Mean accept length:** 5.58 tokens per target pass (max = block size 16)
- **Spec throughput:** 91 tok/s
- **Mean speedup vs target-only greedy:** 4.00×
- **Greedy agreement:** 63.0% (mean fraction of tokens matching sequential-greedy target before first divergence)
- **Exact-match rate:** 33.3% (bit-identical to HF greedy — expected to be <100% in bf16; see below)
- **Suspicious divergences:** 0 (large-gap, i.e. not a rounding tie; max gap seen 0.25 logits)  ✅ no inference bugs — every divergence is a bf16 near-tie

> **Why exact-match < 100% is fine.** DFlash greedy is lossless in exact arithmetic. In bf16 the target's logits are quantized (~0.25 per step near magnitude 32), so when the top-2 tokens are within a rounding tie the parallel block-verification path and sequential greedy pick differently — a benign divergence at a coin-flip position (e.g. `,` vs `.`). A *real* bug would flip tokens with a **large** logit gap; that's what **Suspicious divergences** counts, and it should be 0.

## All domains (ranked by acceptance rate)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Agree % | Suspicious |
|---|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 119 | 23 | 5.23× | 40 | 0 ✅ |
| task_math_reasoning | 37.2 | 6.94 | 31 | 206 | 114 | 23 | 4.99× | 69 | 0 ✅ |
| lang_english | 10.5 | 2.47 | 48 | 123 | 41 | 23 | 1.79× | 81 | 0 ✅ |

## Best-tracked domains (drafter matches target well)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Agree % | Suspicious |
|---|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 119 | 23 | 5.23× | 40 | 0 ✅ |
| task_math_reasoning | 37.2 | 6.94 | 31 | 206 | 114 | 23 | 4.99× | 69 | 0 ✅ |
| lang_english | 10.5 | 2.47 | 48 | 123 | 41 | 23 | 1.79× | 81 | 0 ✅ |

## Worst-tracked domains (drafter struggles)

| Domain | Accept % | Mean len | Steps | Gen tok | Spec tok/s | Base tok/s | Speedup | Agree % | Suspicious |
|---|---|---|---|---|---|---|---|---|---|
| code_python | 40.0 | 7.32 | 29 | 205 | 119 | 23 | 5.23× | 40 | 0 ✅ |
| task_math_reasoning | 37.2 | 6.94 | 31 | 206 | 114 | 23 | 4.99× | 69 | 0 ✅ |
| lang_english | 10.5 | 2.47 | 48 | 123 | 41 | 23 | 1.79× | 81 | 0 ✅ |

## Metric definitions

- **Accept %** — pooled acceptance rate = accepted draft tokens ÷ proposed draft tokens (15 proposed per target pass, block size 16).
- **Mean len** — average tokens committed per target forward pass (accepted drafts + 1 bonus token). Higher = fewer target passes = faster.
- **Speedup** — spec throughput ÷ target-only greedy throughput (same hardware).
- **Agree %** — mean fraction of tokens matching the sequential-greedy target before the first divergence.
- **Suspicious** — divergences with a LARGE top-2 logit gap (not a bf16 rounding tie). This is the actual inference-bug signal; it should be 0. Ordinary near-tie divergences are expected and benign in bf16.
