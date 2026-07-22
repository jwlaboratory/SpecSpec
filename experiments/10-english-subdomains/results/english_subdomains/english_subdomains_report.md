# English subdomain LoRA specialization — base vs own vs combined

Seven English-domain synthetic prompt datasets · held-out test n=100/domain · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · rank-16 LoRA on q/k/v/o.

`combined_equal` uses 114 train examples per domain (798 total) as a matched-budget control.

| domain | base | own | combined | combined_equal | base len | own len | combined len | equal len |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| code_python | 20.9% | 21.3% (+0.3pp) | 21.4% (+0.4pp) | 21.1% (+0.2pp) | 4.48 | 4.56 (+0.07) | 4.56 (+0.07) | 4.53 (+0.04) |
| code_sql | 18.0% | 18.7% (+0.7pp) | 18.0% (-0.0pp) | 18.0% (+0.0pp) | 3.90 | 4.01 (+0.11) | 3.90 (+0.00) | 3.91 (+0.01) |
| ood_legal | 11.6% | 12.2% (+0.6pp) | 11.6% (+0.0pp) | 11.7% (+0.1pp) | 2.78 | 2.86 (+0.08) | 2.78 (+0.00) | 2.80 (+0.02) |
| ood_medical | 13.2% | 13.7% (+0.5pp) | 13.2% (-0.0pp) | 13.3% (+0.1pp) | 3.03 | 3.12 (+0.09) | 3.03 (-0.00) | 3.05 (+0.01) |
| ood_financial | 14.8% | 15.5% (+0.7pp) | 14.8% (+0.0pp) | 15.2% (+0.4pp) | 3.59 | 3.69 (+0.10) | 3.59 (-0.00) | 3.66 (+0.07) |
| task_math_reasoning | 37.6% | 39.4% (+1.8pp) | 37.7% (+0.1pp) | 37.5% (-0.0pp) | 6.98 | 7.30 (+0.31) | 6.99 (+0.01) | 6.98 (-0.00) |
| task_summarization | 9.6% | 9.8% (+0.2pp) | 9.7% (+0.0pp) | 9.6% (-0.0pp) | 2.54 | 2.60 (+0.06) | 2.54 (+0.00) | 2.57 (+0.03) |

**Wins vs base:** own 7/7 · combined 5/7 · combined_equal 5/7.
**Mean retained specialist gain:** combined 20% · combined_equal 18%.

Interpretation target: if weak base domains get the largest own gains and combined/combined_equal retain most of them, the language result is really a broader English coverage result.
