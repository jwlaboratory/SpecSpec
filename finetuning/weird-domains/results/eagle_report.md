# Weird-domains LoRA specialization (EAGLE3) — base vs own vs combined

Domains: translation · roleplay · poetry. Target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA (canonical speculators TTT loss), merged.

| domain | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| translation | 20.6% | 18.7% (-1.9pp) | 19.3% (-1.2pp) | 1.54 | 1.50 (-0.05) | 1.51 (-0.03) |
| roleplay | 36.0% | 33.9% (-2.0pp) | 33.5% (-2.5pp) | 2.09 | 2.03 (-0.07) | 2.02 (-0.08) |
| poetry | 33.5% | 28.9% (-4.5pp) | 31.0% (-2.5pp) | 2.01 | 1.88 (-0.13) | 1.94 (-0.07) |

**own-LoRA beats base on 0/3 domains; combined beats base on 0/3.**
