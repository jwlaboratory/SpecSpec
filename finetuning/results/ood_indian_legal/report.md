# ood_indian_legal — base vs full fine-tune vs LoRA (DFlash drafter)

Domain: **ood_indian_legal** · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · held-out test split.

Metric that matters: **mean accept length** (tokens committed per target pass) and **acceptance rate** — higher = the drafter tracks the target better on this domain.

| variant | n | accept rate | mean accept len | fwd steps | spec tok/s | speedup | exact-match |
|---|--:|--:|--:|--:|--:|--:|--:|
| base (pretrained) | 150 | 10.9% | 2.67 | 96.1 | 74.0 | 1.94x | 18% |
| full fine-tune | 150 | 11.1% (+0.2pp) | 2.70 (+0.03) | 95.0 | 91.7 | 1.96x (+0.01) | 18% |
| LoRA | 150 | 13.5% (+2.5pp) | 3.09 (+0.42) | 84.1 | 104.2 | 2.11x (+0.17) | 18% |

_exact-match = spec output equals the target's own greedy output (temperature 0 ⇒ DFlash is lossless; the adapter changes speed, not correctness). Deltas in parentheses are vs. the base drafter._
