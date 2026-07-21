# 09 — MoLE on WildChat: latent LoRA experts vs one combined LoRA

**Question.** Every human-partitioned comparison so far found a single combined
adapter ≈ per-domain specialists (exp 02/03/05/07). Maybe the problem is the
partition, not specialization: our domain labels may not be the clusters that
matter to the drafter. Here the model carves its own clusters — K LoRA experts
on the DFlash drafter, mixed by a gate that lives in the drafter's own latent
space, trained end-to-end on the full unlabeled wild (WildChat) pool. Does
emergent specialization beat (a) base DFlash and (b) one monolithic combined
LoRA — and, per the follow-up question, is own-vs-combined worth anything in
wall-clock terms?

**Architecture** (`lib/mole.py`):

- K=8 experts, rank 8, α 16, on q/k/v/o of `z-lab/Qwen3-8B-DFlash-b16` —
  total expert params exactly = one rank-64 LoRA.
- One `LatentGate` (LayerNorm → 20480→64→8, softmax; ~1.3M params) on the
  mean-pooled prompt `extract_context_feature` — the same 20480-dim feature the
  drafter conditions on and `router/` classifies (there: 100% test accuracy on
  labels, so the latent clearly separates domains; here the gate is free to
  carve whatever partition helps the loss).
- Mixture is per-sequence: `h = W0x + s·Σ_k g_k B_k A_k x`. Per-token FLOPs =
  one rank-64 LoRA; serving cost = one tiny MLP at prefill (like the router).
- Init: B=0 (every expert a no-op), gate output zero-init (uniform mixture).
  The gate's gradient is exactly zero until experts differentiate — expected
  cold start, verified in the CPU sanity test.
- Anti-collapse: switch-style importance loss `K·Σ(mean_b g)²` (1.0 at uniform),
  coefficient 0.01.

**Training pool.** All 47 wild domains pooled, ~15.3k prompts, no labels
anywhere in training. Self-distillation prep identical to exp-05 (vLLM greedy,
256 tokens). 3 epochs, lr 1e-3, bs 12, same recipe as every prior LoRA.

**Matrix** — wild test splits, 16 eval domains (test n ≥ 37), 5 variants:

| variant | what | controls for |
|---|---|---|
| base | no adapter | floor |
| own | r16 specialist on that domain's wild train (10 domains with 800 ex) | is human partitioning worth it on real data? |
| comb_r16 | one r16 on the full pool | active-capacity control, matches all prior combineds |
| comb_r64 | one r64 on the full pool | total-param control for MoLE |
| mole | 8×r8 experts + latent gate | the hypothesis |

**Readouts.**

1. Acceptance per domain (primary), and **predicted wall-clock speedup
   = L/(1+c)** with the exp-08 fitted DFlash c ≈ 0.44 — no per-prompt HF
   baselines (see the wall-clock memory/exp-08: pair in-container or use the
   model). Head-to-heads own−comb, mole−comb are reported directly in ×.
2. Emergent specialization: per-prompt gate vectors are logged at bench time;
   the aggregate crosses them with held-out domain labels
   (`gate_by_domain.csv`) and reports H(sample) vs H(marginal) — real
   specialization = low per-sample entropy, high marginal entropy. Uniform
   collapse (mole ≡ one rank-64 LoRA) or single-expert collapse (mole ≡
   comb_r64 with extra steps) are both legible failure modes.

**Result (2026-07-20, ledger #19).** The latent gate **collapsed to uniform**:
per-prompt gate std ≤ 0.004 per expert, H(sample) = 2.07 ≈ ln 8, and the mean
mixture is byte-identical across all 16 eval domains — no emergent routing in
3 epochs (the B=0 cold start + per-sequence soft mixing lets experts learn
redundant averaged solutions before the gate ever gets signal). Two findings
survive anyway:

1. **Uniform ensemble > monolithic adapter.** mole (+0.30pp mean vs base,
   ≥ base on 16/16 domains) beats its exact param-match comb_r64, which is net
   NEGATIVE (−0.05pp): mole − comb_r64 = **+0.045× predicted speedup**. The
   win is an ensemble/regularization effect of 8 independently-initialized r8
   adapters, not specialization.
2. **Own vs combined, now on real WildChat:** own − comb_r16 = **+0.018×
   predicted speedup** (~2%, range 0..+0.06×) — the combined-≈-specialists
   result holds on real data; a per-domain adapter fleet buys ~2% wall-clock.

All wild-data gains are an order of magnitude smaller than synthetic
(≤+0.4pp except lang_polish +0.8/+1.0pp) — real WildChat traffic is
heterogeneous inside any label. To actually test latent specialization the
gate needs asymmetric pressure from step 0: noisy top-k (hard) routing so
experts see different data, a per-sample entropy bonus, warm-starting the gate
from `router/` (which separates these domains at 100%), or per-block gating.

**Run.**

```
modal run experiments/09-mole-wildchat/pipeline.py::smoke        # cheap validation
modal run --detach experiments/09-mole-wildchat/pipeline.py::launch
modal run experiments/09-mole-wildchat/pipeline.py::agg_only     # re-aggregate
```

Results land in `results/` (volume `code-sql-pipeline` under
`/work/{prep,models,results}/mole_wild`).
