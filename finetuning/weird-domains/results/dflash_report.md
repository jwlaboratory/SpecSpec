# Weird-domains LoRA specialization (DFlash) — base vs own vs combined

Domains: translation · roleplay · poetry (heterogeneous task types — a real interference test for the combined adapter). Target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 · rank-16 LoRA.

| domain | base accept | own accept | combined accept | base len | own len | combined len |
|---|--:|--:|--:|--:|--:|--:|
| translation | 2.0% | 9.5% (+7.5pp) | 9.4% (+7.5pp) | 1.30 | 2.20 (+0.90) | 2.19 (+0.89) |
| poetry | 7.0% | 7.6% (+0.6pp) | 7.6% (+0.6pp) | 2.18 | 2.30 (+0.11) | 2.28 (+0.10) |

**own-LoRA beats base on 2/3 domains; combined beats base on 2/3.** own-vs-combined gaps indicate cross-task interference inside one adapter.
