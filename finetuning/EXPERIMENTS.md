# Experiment log — every run, preserved

Chronological record of every finetuning experiment. Each entry names its exact
config and where its results live; superseded runs are archived, never deleted.

| # | date | experiment | config | verdict | results |
|---|---|---|---|---|---|
| 1 | 2026-07-16 | **code_sql** DFlash | r16 LoRA + full FT, 800 train, n=100 | LoRA +3.4pp; full FT ≈ base | `results/code_sql/` |
| 2 | 2026-07-16 | **ood_indian_legal** DFlash | r16 LoRA + full FT, 8000 train, n=150 | LoRA +2.5pp; full FT ≈ base | `results/ood_indian_legal/` |
| 3 | 2026-07-16 | **multilingual** DFlash | 6× r16 LoRA (5 langs + combined), n=100/lang | own > base 5/5 (+0.4..+1.6pp); combined ≈ own | `multilingual/results/` |
| 4 | 2026-07-16 | **router** | MLP 20480→512→6 on target hidden states | 100% test accuracy | `../router/results/` |
| 5 | 2026-07-16 | **weird-domains** DFlash | 4× r16 LoRA (translation/roleplay/poetry + combined), n=100 | own > base 3/3 (+0.4..+0.8pp); combined ≈ own | `weird-domains/results/dflash_*` |
| 6 | 2026-07-17 | **multilingual EAGLE v1** | 6× r16 LoRA on RedHat EAGLE3 head, TTT loss | ❌ INVALID — `order` variable shadowing trained every step on REVERSED aux features [33,18,2]; own "beat" base 1/5 | `multilingual_eagle/results-v1-reversed-features-bug/` |
| 7 | 2026-07-17 | **weird-domains EAGLE v1** | 4× r16 LoRA, same stack | ❌ INVALID — same shadowing bug; 0/3 | `weird-domains/results-v1-eagle-reversed-features-bug/` |
| 8 | 2026-07-17 | **verification suite** | zero-merge exactness, null vLLM bench, objective test | merge byte-exact ✓, serve path exact ✓ — pointed to training-side feature error, confirmed as the shadowing bug | `multilingual_eagle/pipeline_eagle.py::verify` |
| 9 | 2026-07-17 | **multilingual EAGLE v2** | as v1, shadowing FIXED (`perm` rename), lr 1e-4, aux `1:std` | own > base 1/5 — japanese (weakest base) +1.6pp; strong-base langs −0.7..−3.9pp. Headroom gradient; residual train/serve gap suspected (base TTT loss ~20) | `multilingual_eagle/results/` |
| 10 | 2026-07-17 | **weird-domains EAGLE v2** | as v1, fixed | 0/3 (−1.3..−4.0pp) — EAGLE base 2.5-4x stronger here, no headroom | `weird-domains/results/eagle_*` |
| 11 | 2026-07-17 | **multilingual DFlash r64** | 6× rank-64 (α=128) LoRA; base benches reused from #3 | r64 > r16 on 5/5; korean +2.3pp total (+66% rel), polish +1.9pp; combined ≈ own | `multilingual/results/*_r64*` + `charts/rank_scaling.png` |
| 12 | 2026-07-17 | **weird-domains DFlash r64** | 4× rank-64 LoRA; base benches reused from #5 | r64 ≈ r16 (gains saturate — base 7-9% has moderate headroom, unlike weak-base languages where r64 kept paying) | `weird-domains/results/dflash_*_r64*` |
| 13 | 2026-07-17 | **weird-domains DFlash r4** | 4× rank-4 (α8, ~130K params) LoRA — capacity-floor + interference-under-scarcity probe | own-r4 beats base 3/3 (+0.4..+0.5pp, ≈90% of r16's gain); combined-r4 ≈ own-r4 — no interference even at rank 4 | `weird-domains/results/dflash_*_r4.*` + `charts/rank_ladder.png` |
| 14 | 2026-07-17 | **interference ladder** DFlash | 13× r16 LoRA: 10 core specialists + comb10/comb20/comb40 (core+distractors, 800 ex/domain, 3 ep), 10×5 bench n=100 | own > base 10/10 (+0.2..+2.0pp); comb > base 10/10 at every N; comb−own gap −0.21pp (N=10) → −0.27 (N=20) → −0.28 (N=40), CIs exclude 0 — interference becomes measurable at N≈10 but SATURATES (no phase boundary); tax concentrates on highest-gain domains (math −1.1pp of +2.0) | `interference/results/` + `charts/ladder.png` |

Notes:
- v1 EAGLE runs also had two REAL bugs fixed before the shadowing was found:
  double-normalized soft-distillation targets (HF `hidden_states[-1]` is already
  post-norm) and flex-attention pack-length mismatch (packs must be padded to the
  128-block mask size). Those fixes are in v2 as well.
- All adapters (`*.pt`) are gitignored; they live locally under each experiment's
  `models/` and on the Modal volume `code-sql-pipeline`.
