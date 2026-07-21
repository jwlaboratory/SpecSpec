# 09 — batched-LoRA serving: what each serving mode costs, by batch size

The follow-up to exp 08's wall-clock model: if merged serving is the only
zero-overhead mode, how expensive are the alternatives — and does batching
rescue them? Five serving modes for the same rank-16 q/k/v/o recipe on
**Qwen3-0.6B** (the repo's independent drafter — the model where exp 06
measured the unmerged tax), batch sizes 1→64:

| | mode | stack |
|---|---|---|
| A | base model, no LoRA | each track's in-container baseline |
| B | single **unmerged** LoRA (naive `lib/lora.py` wrappers) | HF eager |
| C | single **merged** LoRA (ΔW folded into weights) | HF eager |
| D | **punica**-batched LoRA, 1 adapter shared by all requests | vLLM `enable_lora` |
| E | **punica**-batched LoRA, **50 distinct adapters** round-robin | vLLM `enable_lora` |

Timing hygiene (exp 08's lessons, applied): zero-delta adapters (`lora_B=0`) so
every mode does identical kernel work AND decodes byte-identical output; forced
128 tokens/seq (`ignore_eos`/`min_new_tokens`); prefix caching off; 3 timed
batches after warmup; every overhead ratio computed against a **same-container**
base — the D/E containers bench their own engine with the adapter idle. (The
idle bases across containers spanned 558–900 tok/s at bs=1 — host variance that
would have completely poisoned unpaired ratios.)

## Results (overhead vs paired base; full tables in `results/report.md`)

| mode | bs=1 | bs=8 | bs=64 | trend |
|---|--:|--:|--:|---|
| C merged | **+3%** | **−0%** | **−0%** | free at every batch size |
| B unmerged wrappers | +33% | +21% | +20% | flat ~20% tax — never amortizes |
| D punica ×1 | +58% | +37% | +22% | fixed kernel cost, slowly amortizes |
| E punica ×50 | +67% | +98% | **+311%** | grows with DISTINCT adapters live per batch |

→ `results/charts/throughput.png`, `overhead.png`

## Takeaways

1. **Merged is exactly free** — C tracks A within ±1% at every batch size. The
   "merge-on-swap" recipe (fold ΔW per request/domain switch, milliseconds)
   costs nothing at serve time.
2. **The naive unmerged tax never amortizes.** B's wrapper matmuls are ~20%
   throughput at every batch size (+33% at bs=1) — on a drafter whose LoRA buys
   +3–8% acceptance, unmerged serving is strictly net-negative. This is exp
   06's measured slowdown, now isolated and swept.
3. **Punica is not free on a small model.** Even one adapter costs +58% at
   bs=1, amortizing only to +22% at bs=64 — the SGMV kernels' fixed cost looms
   large when the base forward is a 0.6B. (On an 8B backbone the same absolute
   overhead is a few percent — model-scale matters; these numbers are the
   *drafter-side* worst case, which is exactly the repo's serving scenario.)
4. **Adapter DIVERSITY per batch is the real killer.** E costs no more than D
   at bs=1–2 (only 1–2 adapters live), then blows up as round-robin fills the
   batch with distinct adapters: +146% at bs=16, +311% at bs=64 (50 live
   segments per SGMV call). Multi-tenant per-domain adapters on a small
   drafter don't batch.
5. **Net recipe, unchanged but now fully quantified:** serve **one combined
   LoRA, merged** (combined ≥ own on EAGLE v3, ≈ own on DFlash to N≈40 with a
   −0.3pp tax) — zero overhead, no swap machinery, mixed-domain batches for
   free. Reach for punica only on a large backbone with few distinct adapters
   per batch; never serve the naive wrappers.

## Reproduce

```bash
modal run --detach experiments/09-batched-lora-serving/pipeline.py::launch
modal volume get --force code-sql-pipeline results/batched_lora/ results/raw/  # into this folder
python3 experiments/09-batched-lora-serving/make_charts.py
```
