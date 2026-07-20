# Independent drafter — vanilla speculative decoding, base vs own vs combined

Held-out test (n=100/domain) · target **Qwen/Qwen3-8B** · drafter **Qwen/Qwen3-0.6B** (k=4 proposals/step, greedy verification, lossless) · rank-16 LoRA on q/k/v/o, plain-CE self-distillation.

own = LoRA trained on that domain only (800 ex); combined = one LoRA on all five domains (4000 ex).

| domain | base | own | combined | base len | own len | comb len | base spd | own spd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| code_sql | 50.0% | 55.5% (+5.5pp) | 54.7% (+4.7pp) | 3.04 | 3.29 | 3.26 | 0.72× | 0.64× |
| lang_polish | 35.5% | 39.7% (+4.1pp) | 38.2% (+2.7pp) | 2.44 | 2.61 | 2.56 | 0.58× | 0.50× |
| lang_korean | 37.3% | 40.9% (+3.6pp) | 38.2% (+0.9pp) | 2.51 | 2.65 | 2.54 | 0.60× | 0.51× |
| ood_legal | 41.8% | 47.0% (+5.2pp) | 42.4% (+0.6pp) | 2.70 | 2.92 | 2.72 | 0.65× | 0.56× |
| task_math_reasoning | 77.3% | 79.9% (+2.6pp) | 78.1% (+0.7pp) | 4.12 | 4.24 | 4.15 | 0.97× | 0.80× |

exact-match vs HF `generate`: code_sql 31% · lang_polish 12% · lang_korean 16% · ood_legal 10% · task_math_reasoning 42% — same rates as the DFlash benches on these domains (chunked verification computes logits over k+1-token chunks; bf16 numerics flip occasional argmax ties vs step-by-step `generate`). The match *vector* is byte-identical across base/own/combined on every domain: adapters change speed, never output.
