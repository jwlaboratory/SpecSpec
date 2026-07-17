# Weird-domains LoRA specialization (DFlash) — base vs own vs combined

Domains: translation · roleplay · poetry (heterogeneous task types — a real interference test for the combined adapter). Target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 · rank-16 LoRA.

| domain | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| translation | 8.7% | 9.5% (+0.8pp) | 9.4% (+0.6pp) | 2.12 | 2.20 (+0.08) | 2.15 (+0.03) |
| roleplay | 8.1% | 8.5% (+0.4pp) | 8.5% (+0.3pp) | 2.24 | 2.30 (+0.06) | 2.28 (+0.05) |
| poetry | 7.0% | 7.6% (+0.7pp) | 7.6% (+0.6pp) | 2.18 | 2.33 (+0.14) | 2.29 (+0.11) |

**own-LoRA beats base on 3/3 domains; combined beats base on 3/3.** own-vs-combined gaps indicate cross-task interference inside one adapter.
