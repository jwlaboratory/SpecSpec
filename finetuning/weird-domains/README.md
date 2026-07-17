# weird-domains — translation · roleplay · poetry, on BOTH speculators

Three deliberately **heterogeneous task types** (unlike the 5-language experiment
where every task was "same job, different language") — so the combined-vs-own
comparison is a genuine interference test. Trained on **both** speculators from
the SAME target-generated answers (shared prep at `/work/prep/weird`):

- `pipeline_dflash.py` — DFlash drafter (z-lab, block diffusion), validated stack
- `pipeline_eagle.py` — EAGLE3 head (RedHatAI, autoregressive), speculators TTT loss

4 rank-16 LoRAs per speculator (one per domain + combined), then a 3×3 matrix per
speculator: base vs own-domain vs combined, n=100/domain, temperature 0.

## Results

**DFlash** — specialization works, no interference:

| domain | base | own | combined |
|---|--:|--:|--:|
| translation | 8.7% | **9.5% (+0.8pp)** | 9.4% |
| roleplay | 8.1% | **8.5% (+0.4pp)** | 8.5% |
| poetry | 7.0% | **7.6% (+0.7pp)** | 7.6% |

**EAGLE3 (v2, reversed-features bug fixed)** — still no gains: its base is
2.5-4x stronger, so there is little headroom (see `../multilingual_eagle/README.md`
for the full analysis; v1's invalid run is archived in
`results-v1-eagle-reversed-features-bug/`):

| domain | base | own | combined |
|---|--:|--:|--:|
| translation | 20.6% | 19.2% (−1.3pp) | 17.8% |
| roleplay | 36.0% | 33.5% (−2.5pp) | 32.7% |
| poetry | 33.5% | 29.4% (−4.0pp) | 27.9% |

→ `results/charts/matrix.png`, `delta.png`

## Takeaways

1. **Interference didn't materialize even here.** Combined ≈ own on DFlash
   (gaps ≤0.2pp) across genuinely different task types — a rank-16 adapter has
   room for translation+roleplay+poetry simultaneously. Two experiments, zero
   interference: one combined adapter is the operationally simpler choice.
2. **EAGLE3's base is 2.5–4× stronger on these domains** (e.g. roleplay 36.0% vs
   8.1% acceptance — noting acceptance rates aren't directly comparable across
   speculators: EAGLE proposes 3 drafts/step vs DFlash's 15; mean accept length
   is closer, ~2.0 vs ~2.2). Specialization helps the weak speculator, not the
   strong one.

## Reproduce

```bash
modal run --detach finetuning/weird-domains/pipeline_dflash.py::launch          # prep + train + bench
modal run --detach finetuning/weird-domains/pipeline_eagle.py::launch --aux "1:std"
python3 finetuning/weird-domains/make_charts.py
```
