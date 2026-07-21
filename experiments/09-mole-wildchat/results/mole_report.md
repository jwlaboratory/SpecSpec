# MoLE on WildChat — base vs own vs comb_r16 vs comb_r64 vs mole

Wild test splits (n ≤ 100/domain) · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0.

mole = K=8 experts r=8 (rank-64-equivalent params) + latent gate on the pooled 20480-dim context feature, trained on the FULL unlabeled wild pool. own = per-domain r16 specialist (800 wild ex). comb_r16/comb_r64 = one LoRA on the full pool.

pred_speedup = mean_accept_length / (1 + c), c = 0.44 (exp-08 fitted DFlash-HF); acceptance deltas in parentheses vs base.

| domain | base | own | comb_r16 | comb_r64 | mole |
|---|--:|--:|--:|--:|--:|
| lang_chinese | 7.3% | 7.7% (+0.4pp) | 7.6% (+0.3pp) | 7.5% (+0.3pp) | 7.7% (+0.4pp) |
| lang_english | 12.1% | 12.2% (+0.1pp) | 12.0% (-0.0pp) | 11.8% (-0.3pp) | 12.2% (+0.1pp) |
| lang_french | 10.7% | 10.9% (+0.2pp) | 10.8% (+0.1pp) | 10.5% (-0.2pp) | 10.9% (+0.2pp) |
| lang_german | 7.4% | 7.8% (+0.4pp) | 7.7% (+0.2pp) | 7.5% (+0.1pp) | 7.8% (+0.4pp) |
| lang_japanese | 8.6% | — | 8.9% (+0.2pp) | 8.7% (+0.1pp) | 8.9% (+0.3pp) |
| lang_polish | 4.4% | — | 5.2% (+0.8pp) | 5.4% (+1.0pp) | 5.2% (+0.8pp) |
| lang_russian | 7.5% | 7.8% (+0.3pp) | 7.8% (+0.3pp) | 7.7% (+0.3pp) | 7.9% (+0.4pp) |
| lang_spanish | 11.4% | 11.6% (+0.1pp) | 11.4% (-0.0pp) | 11.1% (-0.3pp) | 11.6% (+0.1pp) |
| code_javascript | 15.8% | — | 16.0% (+0.2pp) | 15.5% (-0.3pp) | 16.2% (+0.4pp) |
| code_python | 17.0% | — | 17.4% (+0.4pp) | 16.7% (-0.3pp) | 17.4% (+0.3pp) |
| ood_financial | 13.7% | — | 13.7% (-0.1pp) | 13.5% (-0.2pp) | 13.9% (+0.1pp) |
| task_creative_writing | 10.1% | 10.4% (+0.3pp) | 10.3% (+0.2pp) | 10.1% (+0.1pp) | 10.4% (+0.4pp) |
| task_question_answering | 11.5% | 11.5% (+0.0pp) | 11.5% (+0.0pp) | 11.2% (-0.3pp) | 11.7% (+0.2pp) |
| task_roleplay_chat | 10.2% | 10.4% (+0.2pp) | 10.3% (+0.2pp) | 10.1% (-0.0pp) | 10.4% (+0.3pp) |
| task_summarization | 12.4% | 12.3% (-0.1pp) | 12.3% (-0.1pp) | 11.8% (-0.5pp) | 12.5% (+0.1pp) |
| task_translation | 8.9% | — | 8.8% (-0.1pp) | 8.8% (-0.1pp) | 9.0% (+0.2pp) |

**mean vs base:** own: +0.21pp / +0.039x · comb_r16: +0.17pp / +0.015x · comb_r64: -0.05pp / -0.015x · mole: +0.30pp / +0.030x

**head-to-head predicted speedup:** own − comb_r16: +0.0178x mean (range -0.000..+0.059) · mole − comb_r16: +0.0146x mean (range -0.007..+0.060) · mole − comb_r64: +0.0450x mean (range +0.002..+0.100)


## Latent gate vs (held-out) domain labels

| domain | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | top | H(sample) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| lang_chinese | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_english | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_french | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_german | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_japanese | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_polish | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_russian | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| lang_spanish | 0.14 | 0.13 | 0.14 | 0.13 | 0.11 | 0.10 | 0.12 | 0.13 | e0 | 2.07 |
| code_javascript | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| code_python | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| ood_financial | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| task_creative_writing | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| task_question_answering | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| task_roleplay_chat | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| task_summarization | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |
| task_translation | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.10 | 0.12 | 0.12 | e0 | 2.07 |

marginal usage: [0.14, 0.132, 0.137, 0.136, 0.111, 0.102, 0.118, 0.125] · H(marginal) = 2.07 nats (uniform = 2.08) — specialization = low H(sample), high H(marginal).
