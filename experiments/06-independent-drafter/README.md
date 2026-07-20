# independent drafter — specializing the *original* speculation method

EAGLE3 and DFlash condition the drafter on the **target's hidden states**, so
fine-tuning them drags in a training-time feature pipeline that must match
serving byte-for-byte — that contract produced most of this repo's bugs and
(we suspected, later confirmed as the missing shift_batch alignment — see
`../04-multilingual-eagle/README.md`) the residual gap behind EAGLE's v2
flat-to-negative LoRA results.
This experiment tries the other original speculation method (Leviathan/Chen
2023): a **completely separate small LM** drafts k tokens autoregressively and
the target verifies them in one forward pass. The drafter's only alignment
channel is the token stream itself — exactly what plain CE self-distillation
trains — so there is *no train/serve feature gap to get wrong*. Hypothesis:
specialization lands more easily here.

## Setup

| component | choice |
|---|---|
| target | `Qwen/Qwen3-8B` (frozen, as everywhere) |
| drafter | `Qwen/Qwen3-0.6B` — off-the-shelf independent model, same tokenizer/vocab |
| spec decode | vanilla two-model loop, k=4 proposals/step, greedy verification (`lib/vanilla_spec.py`) — lossless at temperature 0 |
| adaptation | rank-16 LoRA (α=32) on q/k/v/o of the drafter — the repo recipe |
| training signal | plain CE on the target's own generations (self-distillation); prep data reused from exp 05's vLLM generations on the volume |
| domains | code_sql · lang_polish · lang_korean · ood_legal · task_math_reasoning — all five have DFlash specialist numbers in `../05-interference-ladder` for direct comparison, spanning weak→strong base acceptance |
| protocol | own = 1 LoRA/domain (800 ex) + combined on all 5 (4000 ex), 3 epochs; bench 5×3 matrix (base/own/combined), held-out test n=100 |

## Results (n=100/domain, temperature 0)

| domain | base | own | combined | base len | own len | comb len |
|---|--:|--:|--:|--:|--:|--:|
| code_sql | 50.0% | 55.5% (+5.5pp) | 54.7% (+4.7pp) | 3.04 | 3.29 | 3.26 |
| lang_polish | 35.5% | 39.7% (+4.1pp) | 38.2% (+2.7pp) | 2.44 | 2.61 | 2.56 |
| lang_korean | 37.3% | 40.9% (+3.6pp) | 38.2% (+0.9pp) | 2.51 | 2.65 | 2.54 |
| ood_legal | 41.8% | 47.0% (+5.2pp) | 42.4% (+0.6pp) | 2.70 | 2.92 | 2.72 |
| task_math_reasoning | 77.3% | 79.9% (+2.6pp) | 78.1% (+0.7pp) | 4.12 | 4.24 | 4.15 |

**own-LoRA beats base on 5/5 domains** — +2.6..+5.5pp per-proposal, the largest
pp gains of any speculator in the repo on these domains (DFlash own on the same
five: +0.6..+2.0pp). → `results/charts/matrix.png`, `delta.png`, `vs_dflash.png`

## Takeaways

1. **The alignment hypothesis holds — and was later proven.** Plain CE
   self-distillation — no hidden states, no feature pipeline, nothing to get
   wrong — specialized the drafter on every domain, first try, zero alignment
   bugs. Contrast EAGLE at the time of this experiment (four training-side
   bugs, gains on 1/8 domains). Three days later the actual EAGLE
   misalignment (a missing one-token shift in the TTT forward) was found and
   fixed, flipping the multilingual EAGLE run to 5/5 gains — this experiment's
   inference from the other direction was correct.
2. **It breaks the pure-headroom reading of the repo's gain law.** This is by
   far the *strongest* base drafter tested (35–77% per-proposal vs EAGLE's
   12–36% and DFlash's 3–25%), yet it gains on 5/5 — including +2.6pp on top of
   a 77% base. EAGLE's v2 flat results weren't just "strong base ⇒ no
   headroom"; with a clean training channel, even a strong drafter keeps
   improving. The headroom gradient survives only in relative terms: ~10–12%
   relative gain on the four 35–50%-base domains vs +3% on the 77%-base math
   domain.
3. **Interference shows up at N=5 here** (combined < own on 5/5, −0.8..−4.6pp),
   where DFlash showed *zero* interference at 3–5 domains. Exactly what exp 05's
   law predicts: the tax concentrates where specialist gains are biggest, and
   this drafter's specialist gains are the biggest in the repo. Bigger shifts
   compete harder for the same rank-16 subspace.
4. **But vanilla spec decode itself doesn't pay at bs=1 in this harness** —
   speedup 0.58–0.97× (base) / 0.50–0.80× (own) vs target-only. Two stacked
   systems effects, both orthogonal to the science: (a) at batch-1 eager
   decoding, latency is layer-count-bound — a Qwen3-0.6B forward (28 layers,
   ~20 ms) costs almost as much as the 8B target's (36 layers, ~21–27 ms/tok),
   so k=4 sequential draft forwards + 1 verify ≈ 5 target-tokens of wall-clock
   for 2.4–4.2 committed tokens; (b) the unmerged LoRA wrappers add ~9 ms per
   draft forward, which is why *own* benches slower than base despite strictly
   better acceptance — merging ΔW (`lib/lora.py::delta_weight`) removes that
   cost at serve time. This is precisely the niche EAGLE (1-layer head) and
   DFlash (one forward per 16-token block) were invented to fill: they cut
   drafting latency, and that — not acceptance — is where the independent
   drafter loses. A shallower independent drafter, batched serving, or CUDA
   graphs would move the systems verdict; the acceptance/alignment result
   stands either way.
5. **Losslessness note:** exact-match vs HF `generate` is 10–42%, the same
   rates the DFlash benches show on these domains — chunked verification flips
   occasional bf16 argmax ties vs step-by-step decoding. The per-prompt match
   vector is byte-identical across base/own/combined on every domain: adapters
   change speed, never output.

## Files

```
pipeline.py       Modal pipeline: (prep reuse) → train 6 LoRAs (parallel)
                  → bench 5×3 matrix (parallel) → aggregate. Detached-safe.
make_charts.py    matrix.png + delta.png + vs_dflash.png (gain comparison
                  against exp-05's DFlash specialists on the same domains)
models/           6 trained adapters (*.pt, gitignored)
results/          15 per-run jsonls · independent_report.md · comparison csv · charts/
```

## Reproduce

```bash
modal run experiments/06-independent-drafter/pipeline.py::smoke      # validate paths
modal run --detach experiments/06-independent-drafter/pipeline.py::launch
modal volume get code-sql-pipeline results/independent experiments/06-independent-drafter/results/
modal volume get code-sql-pipeline models/independent experiments/06-independent-drafter/models/
python3 experiments/06-independent-drafter/make_charts.py
```
