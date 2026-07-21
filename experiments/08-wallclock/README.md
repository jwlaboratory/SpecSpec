# 08 — net wall-clock speedup (and the analytic model that predicts it)

Every bench in the repo records spec-decode tok/s, but only exps 01/02/06 ever
measured the **target-only baseline**, so "does the acceptance gain survive as
wall-clock gain?" was unanswered for the EAGLE (vLLM) and weird-domains DFlash
(HF) matrices. This experiment fills that hole with two vanilla-decoding runs on
the exact prompts/settings of the spec benches — one per framework, because a
tok/s ratio across engines is meaningless (vLLM decodes the 8B target at ~195
tok/s batch-1; HF at ~40-50):

- `pipeline.py::vanilla_vllm` — vLLM engine, no `speculative_config`, 8 domains
- `pipeline.py::vanilla_hf` — HF bf16/sdpa greedy `generate`, 3 weird domains
- `aggregate.py` — collates speedups offline from the repo's committed jsonls
  (also folds in exp 02, whose jsonls carry their own in-container baselines)

## Results (pooled speedup vs target-only; L = mean accept length)

| section | speedups (base → best LoRA) | verdict |
|---|---|---|
| EAGLE3 multilingual (vLLM) | 0.95–1.13× → 1.02–1.22× | **own > base on 5/5** — the v3 acceptance gains survive to wall-clock; japanese crosses break-even (0.95→1.02×) |
| EAGLE3 weird (vLLM) | 1.31–1.68× | LoRA ≈ base (flat acceptance ⇒ flat wall-clock) |
| DFlash weird (HF) | 1.46–1.71× | LoRA ≈ base within cross-container noise (see below) |
| DFlash multilingual (HF, paired baselines) | 1.07–1.61× → 1.15–1.30× | own > base 4/5 |

Full per-domain table: `results/report.md` · chart: `results/charts/speedup.png`

**The cross-experiment money chart — `results/charts/lora_gain.png`
(`make_charts.py`)**: the LoRA-attributable net wall-clock gain for every
experiment in the repo. Because speedup ≈ L/(1+c) and c is per-speculator, the
gain over base spec decode is exactly `L_variant/L_base − 1` — timing-free,
framework-independent, and equal to what a *merged* adapter serves at. Bars are
that estimator; diamonds are measured pooled spec-tok/s ratios, which hug the
bars on vLLM (stable engine timing) and scatter wildly on the HF cells
(cross-container noise) — visual proof of lesson 2 below. Exp 06's dots sit
far below its bars for a different reason: those runs benched *unmerged*
adapters, whose wrapper overhead a merge removes.

## The analytic model — you don't need more baseline runs

Speculative decoding at batch 1 obeys **speedup ≈ L / (1 + c)** where L is the
mean accept length (measured in every bench already) and c is the speculator's
per-step overhead in target-forward units. Fitting c as a per-section median:

| section | fitted c | median abs err | cells |
|---|--:|--:|--:|
| EAGLE3 multilingual (vLLM) | 0.18 | **2.0%** | 15 |
| EAGLE3 weird (vLLM) | 0.24 | **1.3%** | 9 |
| DFlash multilingual (HF, paired) | 0.44 | **0.3%** | 15 |
| DFlash weird (HF, unpaired) | 0.36 | 6.0% (max 18%) | 9 |

Three lessons:

1. **c is a per-speculator/per-engine constant** to within a few percent —
   EAGLE's 3 sequential 1-layer drafts cost ~0.2 target-forwards per step under
   vLLM; DFlash's single 1B forward ~0.44 under HF. Given c, any cell's
   wall-clock follows from its acceptance alone: future experiments can skip
   baseline runs entirely and report `L/(1+c)`.
2. **Pair your baselines.** The one poorly-predicted section (DFlash weird) is
   the one whose baseline ran in a *different container days later* — HF batch-1
   timing drifts tens of percent across hosts. Exp 02 timed vanilla and spec in
   the same container per prompt and lands at 0.3% median error. `vanilla_vllm`
   is stable because vLLM's engine timing barely drifts (191–199 tok/s across
   all 8 domains).
3. **Acceptance is the whole game at these L values.** dL/(1+c) ≈ 0.8× per
   accepted token for EAGLE — the v3 LoRA's +0.03..+0.06 L buys 2–6% wall-clock,
   real but small; the reason EAGLE beats DFlash end-to-end is c (0.2 vs 0.44),
   and the reason exp 06's independent drafter loses everywhere is c ≈ 3.

## Reproduce

```bash
modal run --detach experiments/08-wallclock/pipeline.py::launch
# then pull results/wallclock/*.jsonl from the volume into results/vanilla/ and:
python3 experiments/08-wallclock/aggregate.py
```
