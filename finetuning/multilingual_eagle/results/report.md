# EAGLE3 multilingual LoRA specialization — base vs own vs combined

Per-language held-out test · target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (1-layer EAGLE3 head, 3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA on the head's q/k/v/o (canonical speculators TTT loss), merged for serving.

| language | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| polish | 9.1% | 7.0% (-2.1pp) | 7.0% (-2.1pp) | 1.28 | 1.21 (-0.06) | 1.21 (-0.07) |
| korean | 7.9% | 7.1% (-0.8pp) | 6.5% (-1.4pp) | 1.24 | 1.22 (-0.02) | 1.20 (-0.04) |
| italian | 13.9% | 11.9% (-2.0pp) | 11.2% (-2.7pp) | 1.43 | 1.36 (-0.06) | 1.34 (-0.08) |
| japanese | 4.9% | 5.2% (+0.3pp) | 4.9% (+0.0pp) | 1.15 | 1.16 (+0.01) | 1.15 (+0.00) |
| german | 12.2% | 8.9% (-3.3pp) | 8.9% (-3.3pp) | 1.38 | 1.28 (-0.10) | 1.28 (-0.10) |

**own-LoRA beats base on 1/5 languages; combined beats base on 1/5.**
