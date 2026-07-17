# Multilingual LoRA specialization — base vs own-language LoRA vs combined LoRA

Per-language held-out test (n=100 each) · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · rank-16 LoRA on q/k/v/o.

own = LoRA trained on that language only (800 ex); combined = one LoRA trained on all five languages (4000 ex).

| language | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| polish | 3.1% | 4.4% (+1.3pp) | 4.5% (+1.3pp) | 1.48 | 1.68 (+0.20) | 1.69 (+0.21) |
| korean | 3.5% | 5.1% (+1.6pp) | 5.1% (+1.6pp) | 1.54 | 1.78 (+0.24) | 1.79 (+0.25) |
| italian | 8.1% | 8.5% (+0.4pp) | 8.4% (+0.4pp) | 2.25 | 2.32 (+0.07) | 2.31 (+0.06) |
| japanese | 5.0% | 5.9% (+0.9pp) | 5.9% (+0.9pp) | 1.76 | 1.89 (+0.14) | 1.90 (+0.14) |
| german | 6.8% | 7.2% (+0.4pp) | 7.1% (+0.3pp) | 2.04 | 2.10 (+0.06) | 2.09 (+0.05) |

**own-LoRA beats base on 5/5 languages; combined LoRA beats base on 5/5.** (deltas vs base in parentheses)
