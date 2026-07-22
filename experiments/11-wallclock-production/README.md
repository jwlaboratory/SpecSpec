# 11 — production wall-clock LoRA serving

This reruns `new/exp2-speedup` with the serving modes separated cleanly.

The old experiment showed the right analytic signal, but the measured stream
numbers mixed two problems:

- `combined` was loaded once into unmerged LoRA wrappers, not folded into the
  drafter weights.
- variants were timed in separate containers, so container/GPU variance could
  dominate small LoRA gains.

This experiment runs the key modes sequentially in one H200 container:

| mode | meaning |
|---|---|
| `target_only` | Qwen3-8B greedy decoding, no speculator |
| `base` | DFlash, no LoRA |
| `merged_combined` | one combined language LoRA folded into DFlash weights |
| `merged_own` | ideal per-language specialists: each language LoRA folded into weights, prompts grouped by language |
| `hotswap_own` | dynamic per-language LoRA hot-swap using unmerged wrappers on the mixed stream |

`merged_own` is an oracle for "N merged own LoRAs served independently." The
merge/unmerge setup time is measured and reported separately; the main tok/s
number excludes that setup, matching a deployment with already-merged replicas.

Run:

```bash
modal run experiments/11-wallclock-production/pipeline.py::smoke
modal run --detach experiments/11-wallclock-production/pipeline.py::launch
modal run experiments/11-wallclock-production/pipeline.py::results
modal volume get --force exp1-language-hidden exp11/full experiments/11-wallclock-production/results/full
python experiments/11-wallclock-production/make_charts.py
```

