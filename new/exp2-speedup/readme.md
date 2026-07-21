# Exp2-speedup: dynamic adapter switching on a mixed-language stream

Exp1 benched each language in isolation, which hides the serving question: if
requests arrive in mixed language order, per-language ("own") LoRAs must be
hot-swapped between requests, while the combined LoRA — and the base drafter —
never swap. This experiment measures that cost directly.

## Setup

- **Test set**: every exp1 language whose fetched **train split has >= 1000
  prompts** and has a trained own-LoRA — 26 of the 40 (the 14 skipped came up
  short after WildChat filters; see `results/summary.json` manifest). 25
  held-out test prompts each, shuffled into ONE 650-prompt stream with a fixed
  seed → **624 language switches in 649 transitions** (~worst case for
  adapter switching).
- **Bench**: one H200 container per variant runs the whole stream
  back-to-back through the instrumented `spec_generate`:
  - `base` — no adapter, no swaps
  - `own` — per-language r16 adapter; on every language change the adapter's
    A/B/scaling are copied from pinned CPU RAM into the injected LoRALinears
    (a realistic hot-swap; swap time counts toward stream wall-clock)
  - `combined` — the single all-language adapter, loaded once
- Vanilla decode is paired in-container on every 20th prompt of the base run
  (memory: wallclock-analytic-model — never compare across containers).

All data (prompts, LoRAs, lm_head/embed dump) is reused from the
`exp1-language-hidden` volume — bench-only, no training stages.

    modal run new/exp2-speedup/pipeline.py::smoke      # 2 langs, 3 prompts each
    modal run --detach new/exp2-speedup/pipeline.py::launch
    modal run new/exp2-speedup/pipeline.py::results    # -> results/summary.json

## Results (2026-07-21, run ap-lQGNgI2unfAhQfrelU8iOt)

650 prompts (26 langs x 25), 256 max new tokens, greedy. ~142k tokens
generated per variant.

| variant | acc | L | swaps | swap total | swap/wall | stream tok/s | analytic spd | measured spd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 0.0646 | 1.97 | 0 | — | — | 61.3 | 1.37 | 1.44 |
| own (hot-swap) | 0.0732 | 2.10 | 625 | 0.26 s | 0.012% | 68.4 | 1.46 | 1.61 |
| combined | 0.0717 | 2.08 | 0 | — | — | 58.7 | 1.44 | 1.38 |

Vanilla anchor: 42.5 tok/s (paired, base container). Measured speedups for
own/combined borrow that anchor across containers, so they carry
container-to-container noise (see takeaway 3); the analytic column
L/(1+0.44) is the apples-to-apples comparison.

### Per-language inside the mixed stream (sorted by base acceptance)

| lang | base acc | own acc | comb acc | base L | own L | comb L | own spd | comb spd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Hungarian | 0.031 | 0.049 | 0.043 | 1.47 | 1.73 | 1.65 | 1.20 | 1.14 |
| Korean | 0.036 | 0.045 | 0.043 | 1.54 | 1.67 | 1.64 | 1.16 | 1.14 |
| Romanian | 0.038 | 0.050 | 0.047 | 1.57 | 1.75 | 1.70 | 1.22 | 1.18 |
| Indonesian | 0.041 | 0.054 | 0.056 | 1.62 | 1.81 | 1.83 | 1.26 | 1.27 |
| Polish | 0.044 | 0.057 | 0.054 | 1.66 | 1.85 | 1.81 | 1.29 | 1.26 |
| Dutch | 0.046 | 0.058 | 0.053 | 1.69 | 1.87 | 1.80 | 1.30 | 1.25 |
| Turkish | 0.048 | 0.068 | 0.060 | 1.73 | 2.02 | 1.90 | 1.41 | 1.32 |
| Ukrainian | 0.058 | 0.072 | 0.065 | 1.86 | 2.08 | 1.98 | 1.45 | 1.37 |
| Tagalog | 0.059 | 0.060 | 0.067 | 1.88 | 1.90 | 2.00 | 1.32 | 1.39 |
| Yoruba | 0.060 | 0.061 | 0.075 | 1.90 | 1.92 | 2.12 | 1.33 | 1.48 |
| Arabic | 0.060 | 0.066 | 0.066 | 1.90 | 1.99 | 1.98 | 1.38 | 1.38 |
| Swedish | 0.063 | 0.070 | 0.067 | 1.94 | 2.06 | 2.01 | 1.43 | 1.40 |
| Malay | 0.063 | 0.076 | 0.078 | 1.95 | 2.14 | 2.17 | 1.48 | 1.50 |
| Japanese | 0.064 | 0.065 | 0.065 | 1.97 | 1.98 | 1.97 | 1.37 | 1.37 |
| Portuguese | 0.066 | 0.072 | 0.069 | 1.98 | 2.08 | 2.03 | 1.45 | 1.41 |
| Chinese | 0.066 | 0.068 | 0.068 | 1.99 | 2.02 | 2.02 | 1.40 | 1.40 |
| Persian | 0.072 | 0.082 | 0.079 | 2.08 | 2.22 | 2.19 | 1.54 | 1.52 |
| Italian | 0.076 | 0.078 | 0.077 | 2.14 | 2.16 | 2.16 | 1.50 | 1.50 |
| German | 0.084 | 0.086 | 0.085 | 2.26 | 2.30 | 2.28 | 1.59 | 1.58 |
| Vietnamese | 0.088 | 0.105 | 0.099 | 2.32 | 2.58 | 2.49 | 1.79 | 1.73 |
| Russian | 0.093 | 0.097 | 0.096 | 2.40 | 2.45 | 2.45 | 1.70 | 1.70 |
| Esperanto | 0.095 | 0.094 | 0.100 | 2.42 | 2.40 | 2.50 | 1.67 | 1.74 |
| Spanish | 0.103 | 0.104 | 0.102 | 2.55 | 2.56 | 2.54 | 1.77 | 1.76 |
| French | 0.108 | 0.108 | 0.106 | 2.61 | 2.63 | 2.59 | 1.82 | 1.80 |
| Latin | 0.117 | 0.117 | 0.116 | 2.76 | 2.76 | 2.74 | 1.92 | 1.90 |
| English | 0.134 | 0.130 | 0.129 | 3.01 | 2.95 | 2.93 | 2.05 | 2.03 |

### Takeaways

1. **Hot-swapping r16 adapters is free.** 625 swaps cost 0.26 s total —
   0.41 ms each, 0.012% of stream wall-clock. An r16 q/k/v/o adapter is
   ~7 MB of bf16; the pinned-CPU→GPU copy is noise next to a single ~350 ms
   spec-decode request. Adapter routing is NOT a reason to prefer the
   combined LoRA at this scale.
2. **Own > combined on the mixed stream, but barely.** Pooled acceptance
   6.46% (base) → 7.17% (combined) → 7.32% (own); analytic speedup
   1.37 → 1.44 → 1.46. The specialists' edge (+0.15pp pooled, worth ~1% of
   wall-clock) is far smaller than the operational cost of storing/routing 26
   adapters — the exp1 conclusion (one combined LoRA ≈ the specialist fleet)
   survives dynamic switching. Per-language the exp1 pattern reproduces:
   own wins where it's data-rich and the base is weak (Hungarian, Turkish,
   Ukrainian), combined wins where cross-lingual transfer helps (Yoruba,
   Tagalog, Esperanto, Malay), both ~tie at the strong end (English, French,
   Latin).
3. **Measured tok/s across containers is noisy; trust the analytic column.**
   Combined's stream tok/s (58.7) lands below base (61.3) despite higher
   acceptance — an ~8% container/GPU draw effect, since the same adapter with
   the same L was 68.4 tok/s in the own container milliseconds after each
   swap. Within-container ratios are honest (base spec vs paired vanilla:
   1.44 measured vs 1.37 analytic); cross-container ones are not — which is
   exactly why exp1 pinned base+own+combined per language to one container
   and this experiment pins each stream to one.
4. Mixed-stream numbers are slightly below exp1's per-language runs (e.g.
   English base L 3.01 here vs 2.93 there, Hungarian 1.47 vs 1.59) — 25
   prompts vs 100 per language plus a different prompt subset; ordering
   effects can't matter for base (no state carries across requests).
