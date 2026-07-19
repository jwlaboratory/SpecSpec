# Interference at scale — base vs own vs comb10/comb20/comb40

Core-domain held-out test (n=100 each) · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · rank-16 LoRA on q/k/v/o.

own = LoRA trained on that domain only (800 ex); combN = one LoRA trained on N domains together (800 ex/domain), the core 10 always included.

| domain | base | own | comb10 | comb20 | comb40 |
|---|--:|--:|--:|--:|--:|
| code_python | 20.9% | 21.5% (+0.6pp) | 21.1% (+0.1pp) | 21.3% (+0.4pp) | 21.6% (+0.7pp) |
| code_sql | 18.0% | 18.8% (+0.7pp) | 18.4% (+0.4pp) | 18.4% (+0.4pp) | 18.5% (+0.5pp) |
| lang_polish | 3.1% | 4.4% (+1.3pp) | 4.5% (+1.4pp) | 4.3% (+1.1pp) | 4.1% (+1.0pp) |
| lang_korean | 3.5% | 5.1% (+1.6pp) | 5.1% (+1.6pp) | 5.0% (+1.4pp) | 4.8% (+1.2pp) |
| lang_german | 6.8% | 7.2% (+0.4pp) | 7.0% (+0.2pp) | 7.0% (+0.2pp) | 7.0% (+0.2pp) |
| ood_legal | 11.6% | 12.2% (+0.6pp) | 12.0% (+0.4pp) | 11.9% (+0.3pp) | 11.9% (+0.3pp) |
| ood_medical | 13.2% | 13.6% (+0.4pp) | 13.6% (+0.4pp) | 13.5% (+0.3pp) | 13.5% (+0.3pp) |
| task_math_reasoning | 37.6% | 39.5% (+2.0pp) | 38.8% (+1.2pp) | 38.6% (+1.0pp) | 38.5% (+0.9pp) |
| task_summarization | 9.6% | 9.8% (+0.2pp) | 9.7% (+0.1pp) | 9.8% (+0.2pp) | 9.8% (+0.2pp) |
| task_roleplay_chat | 8.1% | 8.5% (+0.4pp) | 8.4% (+0.2pp) | 8.4% (+0.2pp) | 8.3% (+0.2pp) |

**combN − own gap (pp, mean over core domains with results):** comb10: -0.21 · comb20: -0.27 · comb40: -0.27
(deltas vs base in parentheses; gap ≈ 0 ⇒ no interference at that N)
