# EAGLE3 multilingual LoRA specialization — base vs own vs combined

Per-language held-out test · target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (1-layer EAGLE3 head, 3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA on the head's q/k/v/o (canonical speculators TTT loss), merged for serving.

| language | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| polish | 9.1% | 10.1% (+1.0pp) | 10.3% (+1.2pp) | 1.28 | 1.31 (+0.03) | 1.31 (+0.04) |
| korean | 7.9% | 8.8% (+0.9pp) | 8.9% (+1.1pp) | 1.24 | 1.27 (+0.03) | 1.27 (+0.03) |
| italian | 13.9% | 14.7% (+0.9pp) | 15.2% (+1.4pp) | 1.43 | 1.45 (+0.03) | 1.47 (+0.04) |
| japanese | 4.9% | 6.9% (+2.1pp) | 7.1% (+2.3pp) | 1.15 | 1.21 (+0.06) | 1.22 (+0.07) |
| german | 12.2% | 12.8% (+0.6pp) | 13.0% (+0.8pp) | 1.38 | 1.39 (+0.02) | 1.40 (+0.02) |

**own-LoRA beats base on 5/5 languages; combined beats base on 5/5.**
