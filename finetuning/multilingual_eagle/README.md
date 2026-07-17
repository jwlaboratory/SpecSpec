# multilingual_eagle — MoLA for EAGLE3 (cross-speculator replication)

The `../multilingual/` experiment (5 language LoRAs + 1 combined, base/own/combined
matrix) replicated on the **EAGLE3** speculator (`RedHatAI/Qwen3-8B-speculator.eagle3`,
1-layer autoregressive head, 3 spec tokens/step). Same training data (the DFlash
run's target-generated answers, reused from the volume), same rank-16 LoRA on
q/k/v/o, same test prompts — only the speculator differs.

Training uses the `speculators` package's canonical TTT forward (3-step unroll,
soft distillation against the verifier lm_head), aux hidden states at layers
[2, 18, 33] (`1:std` convention, settled by probe), lr 1e-4. Adapters are MERGED
into speculators-format dirs and benched through vLLM.

## v1 vs v2 — the reversed-features bug

The first run (**v1**, archived in `results-v1-reversed-features-bug/`) was
invalidated by a one-variable bug: the aux-concat convention string `order`
("std") was clobbered by the epoch shuffle list, so every training step fed the
head aux states in REVERSED order [33,18,2] while validation/probe/serving used
[2,18,33]. Found by code review after a verification suite (zero-merge
byte-exactness ✓, null vLLM bench exact ✓, objective-improved ✓) proved the
error had to be on the training side. **v2** (below) fixes it (`perm` rename).

## Result — v2, fixed (n=100/language, temperature 0)

| language | base accept | own accept | combined accept |
|---|--:|--:|--:|
| polish | 9.1% | 6.0% (−3.1pp) | 5.9% (−3.2pp) |
| korean | 7.9% | 7.2% (−0.7pp) | 6.4% (−1.5pp) |
| italian | 13.9% | 10.0% (−3.9pp) | 9.8% (−4.0pp) |
| **japanese** | **4.9%** | **6.4% (+1.6pp)** | **6.5% (+1.6pp)** |
| german | 12.2% | 8.3% (−3.9pp) | 8.5% (−3.7pp) |

→ `results/charts/matrix.png`, `delta.png`, `vs_dflash.png`

## Honest interpretation

1. **Fixing the bug barely moved the aggregate — but japanese jumped** from
   +0.3pp (v1) to **+1.6pp (+33% relative)**. The one language where the base
   head is truly weak (4.9%) now shows a real, replicated specialization gain.
2. **The headroom gradient is the story.** Gains/losses track base strength
   almost monotonically: japanese (4.9% base) +1.6, korean (7.9%) −0.7, polish
   (9.1%) −3.1, german/italian (12–14%) −3.7..−4.0. Same law as within DFlash,
   extended: where the base is strong, fine-tuning on 800 prompts has nothing to
   add — and tips slightly negative rather than flat.
3. **Why negative instead of flat?** Likely a residual train/serve gap: the
   BASE head's TTT loss under our reconstructed feature pipeline stays high
   (~20) even on the correct convention, suggesting some detail (aux capture
   point, data format of RedHat's original training, or the soft-CE objective
   vs argmax acceptance) still differs. The verification suite rules out the
   merge and serve paths (both byte-exact). Definitive next step, if pursued:
   dump ground-truth aux tensors from inside vLLM's serving path and diff.
4. **Net recommendation:** LoRA-specialize DFlash (works, and rank 64 works
   better — see `../multilingual/`); for EAGLE3, only weak-base domains
   (japanese-like) show gains under this recipe.

## Reproduce

```bash
modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch --aux "1:std"
modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch_bench   # benches only
modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch_verify  # diagnostics
python3 finetuning/multilingual_eagle/make_charts.py
```
