# Python-specialized DFlash drafter via (unmerged) LoRA — plan

## What we're building
Fine-tune a **DFlash drafter** to match Qwen3-8B better **on Python queries** by
training an **unmerged LoRA** adapter on the drafter. Base never changes:

- **frozen:** Qwen3-8B target (`Qwen/Qwen3-8B`) + pretrained DFlash drafter
  (`eigen-ai-labs/qwen3-8b-dflash-demo`, or your own checkpoint)
- **trained:** only LoRA `{A,B}` on the drafter's q/k/v/o projections
- **unmerged:** run `base`, or add `ΔW = (α/r)·B·A` on top — verified in `demo.py [A]`

### Why LoRA and not NaRA
A DFlash block drafts in **one step at a fixed masking pattern** (1 anchor +
block_size-1 masks), so the noise level λ is ~constant. NaRA's noise-aware core
`C(λ)` would never move ⟹ it collapses to plain LoRA. So we use plain LoRA.

## The objective — NOT perplexity
The drafter is trained to **match the target model's own tokens**: DFlash's
exponentially-weighted block cross-entropy `w_k = exp(-(k-1)/γ)` (γ=7) with
dynamic anchor sampling. Perplexity is the wrong lens (it measures standalone
language modeling; the drafter's job is agreement with the 8B).

**Eval = acceptance length + tokens/sec speedup** on held-out Python queries,
base drafter vs. base+LoRA.

## Dataset — WildChat-1M → Python
`allenai/WildChat-1M`, filtered to Python (langtag + heuristics: "python", ```py
fences, common imports), user turns used as **prompts only**. Responses are
**regenerated with Qwen3-8B** (DFlash labels must be the target's tokens). Hold
out a Python eval split for the speedup benchmark.

## Multi-adapter: 3 domains, swap + batched routing
Three domains — **python, sql, prose** — each gets its own unmerged LoRA on the
same frozen drafter. `batched_lora.py` serves them three ways:
- `use_base()` — no adapter
- `use_adapter("sql")` — one adapter for the whole batch (**hot-swap**)
- `route(ids)` — **different adapter per sequence in ONE batch** (S-LoRA style):
  `a = einsum('bsi,bri->bsr', x, A[ids]); u = einsum('bsr,bor->bso', a, B[ids])`

## Files
- `lora.py` — single unmerged LoRA (`LoRALinear`, `inject_lora`). (DONE)
- `batched_lora.py` — multi-adapter LoRA: hot-swap + per-sequence batched routing. (DONE)
- `demo.py` — CPU proof of unmerged LoRA math + DFlash-loss training on the drafter. (DONE, passing)
- `multi_lora_demo.py` — CPU: 3 adapters, specialization matrix, swap, exact batched routing. (DONE, passing)
- `modal_lora.py` — Modal A100: same machinery on the REAL 1B drafter (smoke). (RUNNING/DONE)
- `data_prep.py` — WildChat → {python,sql,prose} prompts → regenerate w/ Qwen3-8B. (TODO)
- `benchmark.py` — spec-decoding acceptance length, per-domain, base vs +adapter. (TODO)

## Steps (Modal GPU)
1. [done] smoke — real 1B drafter + 3-adapter swap/routing on A100.
2. data_prep — filter WildChat to 3 domains, regenerate responses with Qwen3-8B.
3. train — DFlash matching loss, 3 adapters (LoRA-only), base frozen.
4. benchmark — per-domain acceptance length / speedup; show routing on a mixed batch.

## Status
- [x] `lora.py` + `demo.py` — unmerged LoRA-on-drafter math verified locally.
- [x] `batched_lora.py` + `multi_lora_demo.py` — 3-adapter swap + exact batched routing.
- [x] objective settled (DFlash matching loss; eval = acceptance/speedup, not ppl).
- [x] domains = python / sql / prose;  dataset = WildChat-1M.
- [x] Modal smoke on real 1B drafter (A100): 1.05B draft built, 3 adapters train
      via DFlash loss, batched routing bit-exact (err 0.0). Synthetic targets, so
      loss is flat — machinery validated, not yet learning real data.
- [ ] real-data: data_prep (WildChat→3 domains, regen w/ Qwen3-8B) → train 3
      adapters for real → per-domain acceptance-length benchmark + mixed-batch routing.
