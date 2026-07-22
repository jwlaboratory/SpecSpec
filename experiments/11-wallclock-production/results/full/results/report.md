# Production wall-clock LoRA serving — full

650 prompts · 26 languages · 624 language switches · H200 · greedy decode

| mode | tok/s | speedup vs target | relative vs base DFlash | accept | L | swaps | setup merge s |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 46.78 | 1.000 | 1.000 | 0.0000 | 0.000 | 0 | 0.000 |
| base | 66.33 | 1.418 | 1.000 | 0.0646 | 1.969 | 0 | 0.000 |
| merged_combined | 70.23 | 1.501 | 1.059 | 0.0717 | 2.076 | 0 | 0.073 |
| merged_own | 70.55 | 1.508 | 1.064 | 0.0733 | 2.099 | 0 | 0.125 |
| hotswap_own | 54.73 | 1.170 | 0.825 | 0.0732 | 2.098 | 625 | 0.000 |
