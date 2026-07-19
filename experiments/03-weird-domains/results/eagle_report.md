# Weird-domains LoRA specialization (EAGLE3) — base vs own vs combined

Domains: translation · roleplay · poetry. Target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA (canonical speculators TTT loss), merged.

| domain | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| translation | 20.6% | 19.2% (-1.3pp) | 17.8% (-2.8pp) | 1.54 | 1.49 (-0.05) | 1.44 (-0.10) |
| roleplay | 36.0% | 33.5% (-2.5pp) | 32.7% (-3.3pp) | 2.09 | 2.02 (-0.07) | 2.00 (-0.10) |
| poetry | 33.5% | 29.4% (-4.0pp) | 27.9% (-5.6pp) | 2.01 | 1.90 (-0.11) | 1.86 (-0.15) |

**own-LoRA beats base on 0/3 domains; combined beats base on 0/3.**
