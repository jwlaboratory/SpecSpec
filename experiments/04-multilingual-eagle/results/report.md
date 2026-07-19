# EAGLE3 multilingual LoRA specialization — base vs own vs combined

Per-language held-out test · target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (1-layer EAGLE3 head, 3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA on the head's q/k/v/o (canonical speculators TTT loss), merged for serving.

| language | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| polish | 9.1% | 6.0% (-3.1pp) | 5.9% (-3.2pp) | 1.28 | 1.19 (-0.09) | 1.18 (-0.10) |
| korean | 7.9% | 7.2% (-0.7pp) | 6.4% (-1.5pp) | 1.24 | 1.22 (-0.02) | 1.20 (-0.04) |
| italian | 13.9% | 10.0% (-3.9pp) | 9.8% (-4.0pp) | 1.43 | 1.31 (-0.12) | 1.30 (-0.12) |
| japanese | 4.9% | 6.4% (+1.6pp) | 6.5% (+1.6pp) | 1.15 | 1.19 (+0.05) | 1.20 (+0.05) |
| german | 12.2% | 8.3% (-3.9pp) | 8.5% (-3.7pp) | 1.38 | 1.26 (-0.12) | 1.26 (-0.11) |

**own-LoRA beats base on 1/5 languages; combined beats base on 1/5.**
