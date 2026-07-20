# Weird-domains LoRA specialization (EAGLE3) — base vs own vs combined

Domains: translation · roleplay · poetry. Target **Qwen/Qwen3-8B** · speculator **RedHatAI/Qwen3-8B-speculator.eagle3** (3 spec tokens/step) · vLLM · temperature 0 · rank-16 LoRA (canonical speculators TTT loss), merged.

| domain | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| translation | 20.6% | 20.3% (-0.3pp) | 20.4% (-0.2pp) | 1.54 | 1.50 (-0.04) | 1.53 (-0.02) |
| roleplay | 36.0% | 35.2% (-0.7pp) | 35.1% (-0.9pp) | 2.09 | 2.07 (-0.03) | 2.06 (-0.03) |
| poetry | 33.5% | 33.0% (-0.5pp) | 32.9% (-0.6pp) | 2.01 | 2.00 (-0.01) | 2.00 (-0.01) |

**own-LoRA beats base on 0/3 domains; combined beats base on 0/3.**
