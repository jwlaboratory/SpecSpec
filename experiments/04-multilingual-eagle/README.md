# multilingual_eagle — MoLA for EAGLE3 (cross-speculator replication)

The `../02-multilingual-dflash/` experiment (5 language LoRAs + 1 combined, base/own/combined
matrix) replicated on the **EAGLE3** speculator (`RedHatAI/Qwen3-8B-speculator.eagle3`,
1-layer autoregressive head, 3 spec tokens/step). Same training data (the DFlash
run's target-generated answers, reused from the volume), same rank-16 LoRA on
q/k/v/o, same test prompts — only the speculator differs.

Training uses the `speculators` package's canonical TTT forward (3-step unroll,
soft distillation against the verifier lm_head) with the canonical `shift_batch`
alignment applied per packed document, aux hidden states at layers [2, 18, 33]
(`0:std` — the vLLM serving convention), lr 1e-4. Adapters are MERGED into
speculators-format dirs and benched through vLLM.

## v1 → v2 → v3 — the bug chain

- **v1** (archived `results-v1-reversed-features-bug/`): the aux-concat
  convention string `order` ("std") was clobbered by the epoch shuffle list —
  every training step fed REVERSED aux features [33,18,2] while serving used
  [2,18,33]. Found by code review after the verification suite (zero-merge
  byte-exactness ✓, null vLLM bench exact ✓) proved the error was training-side.
- **v2** (archived `results-v2-unshifted-ttt-bug/`): shadowing fixed, but the
  TTT forward still fed input_ids / aux / verifier targets UNSHIFTED. The
  canonical contract (speculators `eagle3/data.py::shift_batch`, = vLLM serving)
  pairs `embed(x_{t+1})` with `aux_t`, supervised by the verifier's distribution
  at t+1. Training one position off meant every LoRA improved its objective
  while *degrading* serving acceptance (own < base on 4/5). Tells: base TTT
  loss ~20 under every probed convention, and the probe's "acc" was a raw count.
  Same pass also fixed packed-document contamination in the target pass
  (block-diagonal attention mask + per-doc position ids).
- **v3** (current, `results/`): both alignment fixes in. Gate that it's right:
  train-time step-0 top-1 accuracy (`full_acc_0` ≈ 0.25 on polish/german val)
  matches serving position-1 acceptance (~0.24 implied by the base bench).

## Result — v3 (n=100/language, temperature 0)

| language | base accept | own accept | combined accept |
|---|--:|--:|--:|
| polish | 9.1% | **10.1% (+1.0pp)** | 10.3% (+1.2pp) |
| korean | 7.9% | **8.8% (+0.9pp)** | 8.9% (+1.1pp) |
| italian | 13.9% | **14.7% (+0.9pp)** | 15.2% (+1.4pp) |
| **japanese** | **4.9%** | **6.9% (+2.1pp)** | **7.1% (+2.3pp)** |
| german | 12.2% | **12.8% (+0.6pp)** | 13.0% (+0.8pp) |

→ `results/charts/matrix.png`, `delta.png`, `vs_dflash.png`

## Interpretation

1. **Own beats base 5/5; combined beats base 5/5 and ≥ own everywhere** — the
   DFlash multilingual result replicates on the strong speculator, with zero
   interference at N=5.
2. **The headroom law holds within EAGLE**: japanese (4.9% base) +2.1pp (+43%
   relative), the 8–14%-base languages +0.6..+1.0pp. Compare the weird-domains
   EAGLE run (21–36% base): flat (−0.3..−0.7pp ≈ noise) — no headroom, no harm.
3. **The v2 "EAGLE resists specialization" conclusion was entirely the bug.**
   Fine-tuning feature-conditioned heads is unforgiving: the training forward
   must reproduce the serving contract token-for-token, and a one-position slip
   silently inverts the result.

## Reproduce

```bash
modal run --detach experiments/04-multilingual-eagle/pipeline_eagle.py::launch --aux "0:std"
modal run --detach experiments/04-multilingual-eagle/pipeline_eagle.py::launch_one_cell # cheap 1-language end-to-end check
modal run --detach experiments/04-multilingual-eagle/pipeline_eagle.py::launch_bench   # benches only
modal run --detach experiments/04-multilingual-eagle/pipeline_eagle.py::launch_verify  # diagnostics
python3 experiments/04-multilingual-eagle/make_charts.py
```
