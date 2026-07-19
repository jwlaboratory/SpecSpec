# code_sql — base vs full fine-tune vs LoRA (DFlash drafter)

Domain: **code_sql** · target **Qwen/Qwen3-8B** · drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · held-out test split.

Metric that matters: **mean accept length** (tokens committed per target pass) and **acceptance rate** — higher = the drafter tracks the target better on this domain, so fewer target passes and higher speedup.

| variant | n | accept rate | mean accept len | fwd steps | spec tok/s | speedup | exact-match |
|---|--:|--:|--:|--:|--:|--:|--:|
| base (pretrained) | 100 | 25.0% | 5.07 | 39.9 | 87.2 | 3.70x | 38% |
| full fine-tune | 100 | 25.1% (+0.2pp) | 5.12 (+0.04) | 39.7 | 96.0 | 3.68x (-0.02) | 38% |
| LoRA | 100 | 28.3% (+3.4pp) | 6.03 (+0.96) | 36.1 | 102.6 | 4.11x (+0.42) | 38% |

_exact-match = spec output equals the target's own greedy output (temperature 0 ⇒ DFlash is lossless; the adapter changes speed, not correctness). Deltas in parentheses are vs. the base drafter._
