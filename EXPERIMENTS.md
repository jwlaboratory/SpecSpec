# Experiment log — every run, preserved

Chronological record of every finetuning experiment. Each entry names its exact
config and where its results live; superseded runs are archived, never deleted.
(The pre-ledger baseline characterization of both speculators across all domains
is experiment 00 — `experiments/00-base-benchmarks/`.)

| # | date | experiment | config | verdict | results |
|---|---|---|---|---|---|
| 1 | 2026-07-16 | **code_sql** DFlash | r16 LoRA + full FT, 800 train, n=100 | LoRA +3.4pp; full FT ≈ base | `experiments/01-single-domain-dflash/results/code_sql/` |
| 2 | 2026-07-16 | **ood_indian_legal** DFlash | r16 LoRA + full FT, 8000 train, n=150 | LoRA +2.5pp; full FT ≈ base | `experiments/01-single-domain-dflash/results/ood_indian_legal/` |
| 3 | 2026-07-16 | **multilingual** DFlash | 6× r16 LoRA (5 langs + combined), n=100/lang | own > base 5/5 (+0.4..+1.6pp); combined ≈ own | `experiments/02-multilingual-dflash/results/` |
| 4 | 2026-07-16 | **router** | MLP 20480→512→6 on target hidden states | 100% test accuracy | `router/results/` |
| 5 | 2026-07-16 | **weird-domains** DFlash | 4× r16 LoRA (translation/roleplay/poetry + combined), n=100 | own > base 3/3 (+0.4..+0.8pp); combined ≈ own | `experiments/03-weird-domains/results/dflash_*` |
| 6 | 2026-07-17 | **multilingual EAGLE v1** | 6× r16 LoRA on RedHat EAGLE3 head, TTT loss | ❌ INVALID — `order` variable shadowing trained every step on REVERSED aux features [33,18,2]; own "beat" base 1/5 | `experiments/04-multilingual-eagle/results-v1-reversed-features-bug/` |
| 7 | 2026-07-17 | **weird-domains EAGLE v1** | 4× r16 LoRA, same stack | ❌ INVALID — same shadowing bug; 0/3 | `experiments/03-weird-domains/results-v1-eagle-reversed-features-bug/` |
| 8 | 2026-07-17 | **verification suite** | zero-merge exactness, null vLLM bench, objective test | merge byte-exact ✓, serve path exact ✓ — pointed to training-side feature error, confirmed as the shadowing bug | `experiments/04-multilingual-eagle/pipeline_eagle.py::verify` |
| 9 | 2026-07-17 | **multilingual EAGLE v2** | as v1, shadowing FIXED (`perm` rename), lr 1e-4, aux `1:std` | ❌ INVALID — TTT forward fed UNSHIFTED (input_ids, aux, targets): trained to predict x_{t+1} from (embed(x_t), aux_t) while serving predicts x_{t+2} from (embed(x_{t+1}), aux_t); plus packed docs shared attention in the target pass. own "beat" base 1/5; superseded by #16 | `experiments/04-multilingual-eagle/results-v2-unshifted-ttt-bug/` |
| 10 | 2026-07-17 | **weird-domains EAGLE v2** | as v1, fixed | ❌ INVALID — same unshifted-TTT bug; 0/3 (−1.3..−4.0pp); superseded by #17 | `experiments/03-weird-domains/results-v2-eagle-unshifted-ttt-bug/` |
| 11 | 2026-07-17 | **multilingual DFlash r64** | 6× rank-64 (α=128) LoRA; base benches reused from #3 | r64 > r16 on 5/5; korean +2.3pp total (+66% rel), polish +1.9pp; combined ≈ own | `experiments/07-rank-ladder/results/*_r64*` + `charts/rank_scaling.png` |
| 12 | 2026-07-17 | **weird-domains DFlash r64** | 4× rank-64 LoRA; base benches reused from #5 | r64 ≈ r16 (gains saturate — base 7-9% has moderate headroom, unlike weak-base languages where r64 kept paying) | `experiments/07-rank-ladder/results/dflash_*_r64*` |
| 13 | 2026-07-17 | **weird-domains DFlash r4** | 4× rank-4 (α8, ~130K params) LoRA — capacity-floor + interference-under-scarcity probe | own-r4 beats base 3/3 (+0.4..+0.5pp, ≈90% of r16's gain); combined-r4 ≈ own-r4 — no interference even at rank 4 | `experiments/07-rank-ladder/results/dflash_*_r4.*` + `charts/rank_ladder.png` |
| 14 | 2026-07-17 | **interference ladder** DFlash | 13× r16 LoRA: 10 core specialists + comb10/comb20/comb40 (core+distractors, 800 ex/domain, 3 ep), 10×5 bench n=100 | own > base 10/10 (+0.2..+2.0pp); comb > base 10/10 at every N; comb−own gap −0.21pp (N=10) → −0.27 (N=20) → −0.28 (N=40), CIs exclude 0 — interference becomes measurable at N≈10 but SATURATES (no phase boundary); tax concentrates on highest-gain domains (math −1.1pp of +2.0) | `experiments/05-interference-ladder/results/` + `charts/ladder.png` |
| 15 | 2026-07-19 | **independent drafter** (vanilla spec decode) | Qwen3-0.6B independent drafter for Qwen3-8B, k=4, greedy verify (`lib/vanilla_spec.py`); 6× r16 LoRA (5 own + combined), plain-CE self-distillation (prep reused from #14), 5×3 bench n=100 | own > base **5/5** (+2.6..+5.5pp per-proposal — largest pp gains in the repo; ~10–12% relative, +3% on 77%-base math). Clean alignment: zero training-side bugs, strong base still gains ⇒ EAGLE's flat result was alignment, not just headroom. combined < own 5/5 (−0.8..−4.6pp) — interference already at N=5, where DFlash had none. BUT wall-clock 0.5–0.97× — at bs=1 eager, drafting latency is layer-bound (0.6B/28L ≈ 8B/36L per forward) so vanilla spec decode loses regardless of acceptance | `experiments/06-independent-drafter/results/` + `charts/vs_dflash.png` |
| 16 | 2026-07-20 | **multilingual EAGLE v3** | as v2 + canonical `shift_batch` alignment (embed(x_{t+1}) ↔ aux_t ↔ verifier dist at t+1, per-doc, position_ids from 1), doc-masked per-doc target passes, aux `0:std` (vLLM serving convention), real acc metric (`full_acc_0`) | own > base **5/5** (+0.6..+2.1pp); combined > base **5/5** (+0.8..+2.3pp, ≥ own everywhere — zero interference); gains track headroom (japanese 4.9% base → +2.1pp). EAGLE DOES specialize once train matches serve | `experiments/04-multilingual-eagle/results/` |
| 17 | 2026-07-20 | **weird-domains EAGLE v3** | same fix, aux `0:std` | own 0/3 but now ≈ base (−0.3..−0.7pp vs v2's −1.3..−4.0) — strong base (21–36%), no headroom, and no longer any harm. Clean "no headroom → flat" result | `experiments/03-weird-domains/results/eagle_*` |

Notes:
- The v2 EAGLE bug (found 2026-07-20): the `speculators` package's canonical data
  pipeline (`eagle3/data.py::shift_batch`) requires the alignment (x1, g0, y1, l1) —
  input_ids/verifier-targets/loss_mask shifted left one token against the aux
  features — which our live-feature `_ttt_forward` never applied. The 4-way
  offset/order probe could not catch it: ALL conventions scored ~20 because the
  missing shift dominated. Diagnostic that worked: fixing alignment makes train-time
  step-0 top-1 accuracy (~0.25 on polish/german) match serving position-1 acceptance
  (~0.24 implied by base bench) — that agreement is the gate for any future change.
  (The probe's old "acc" was also bogus — `cond_acc_0_sum` is a raw count; use
  `full_acc_0_sum/full_acc_0_total`.) Second fix in the same pass: per-document
  attention masks + position ids in the packed target pass (docs were attending to
  earlier pack-mates, contaminating features vs serving).
- v1 EAGLE runs also had two REAL bugs fixed before the shadowing was found:
  double-normalized soft-distillation targets (HF `hidden_states[-1]` is already
  post-norm) and flex-attention pack-length mismatch (packs must be padded to the
  128-block mask size). Those fixes are in v2 as well.
- All adapters (`*.pt`) are gitignored; they live locally under each experiment's
  `models/` and on the Modal volume `code-sql-pipeline`.
