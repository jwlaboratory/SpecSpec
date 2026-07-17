# multilingual_eagle — MoLA for EAGLE3 (cross-speculator replication)

The `../multilingual/` experiment (5 language LoRAs + 1 combined, base/own/combined
matrix) replicated on the **EAGLE3** speculator (`RedHatAI/Qwen3-8B-speculator.eagle3`,
1-layer autoregressive head, 3 spec tokens/step) instead of DFlash. Same training
data (the DFlash run's target-generated answers, reused from the volume), same
rank-16 LoRA on q/k/v/o, same test prompts — only the speculator differs.

Training uses the `speculators` package's canonical TTT forward (3-step unroll,
soft distillation against the verifier lm_head), aux hidden states at layers
[2, 18, 33] (offset probed empirically, `1:std`), lr 1e-4. Adapters are MERGED
into speculators-format dirs and benched through vLLM (`speculative_config`),
acceptance from `vllm:spec_decode_*` counter diffs.

## Result (n=100/language, temperature 0)

| language | base accept | own accept | combined accept |
|---|--:|--:|--:|
| polish | 9.1% | 7.0% (−2.1pp) | 7.0% (−2.1pp) |
| korean | 7.9% | 7.1% (−0.8pp) | 6.5% (−1.4pp) |
| italian | 13.9% | 11.9% (−2.0pp) | 11.2% (−2.7pp) |
| japanese | 4.9% | **5.2% (+0.3pp)** | 4.9% (+0.0pp) |
| german | 12.2% | 8.9% (−3.3pp) | 8.9% (−3.3pp) |

**own-LoRA beats base on 1/5; combined on 1/5** — the opposite of DFlash (5/5).
→ `results/charts/matrix.png`, `delta.png`, **`vs_dflash.png`** (the cross-speculator chart)

## Honest interpretation

1. **Headroom gradient, continued.** Within DFlash, gains shrank as base strength
   grew; EAGLE3's base is stronger still, and the gains go ≈0-to-negative. The
   *only* positive language (japanese +0.3pp) is the one with the weakest base
   (4.9%) — the gradient holds even in sign.
2. **Residual training/serving mismatch.** The base head's TTT loss under our
   reconstructed feature pipeline stays elevated (~20), suggesting the exact
   training view (aux capture point / norm details) isn't perfectly reproduced;
   any mismatch turns "no headroom" into "small loss". An earlier bug (double-
   normalized soft targets + lr 1e-3) caused 3–4× worse degradation before being
   fixed — what remains may be a smaller cousin of the same problem. The
   definitive check (not run): dump ground-truth aux features from inside vLLM's
   serving path and diff against the HF-side extraction.
3. **Net recommendation:** specialize DFlash with LoRA (works, 11/11 domains);
   for EAGLE3, the pretrained head is already strong and this recipe does not
   improve it.

## Reproduce

```bash
modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch --aux "1:std"
modal run finetuning/multilingual_eagle/pipeline_eagle.py::agg_only
modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch_bench  # benches only
python3 finetuning/multilingual_eagle/make_charts.py
```
