# Multi-adapter serving over a shared DFlash backbone

Serve **N NaRA/LoRA adapters** on **one shared backbone** (Qwen3-8B target + the
`z-lab/Qwen3-8B-DFlash-b16` ~1B block-diffusion drafter). Five requests, each on a
different adapter, all run against the **one resident backbone** — no reload
between requests.

## Why plain LoRA + naive swap (not NaRA's C(λ), not a Punica kernel)

The DFlash drafter proposes a whole block in **one denoising step**, so the noise
level `λ = (block_size-1)/block_size ≈ 0.9375` is ~constant. NaRA's noise-aware
core `C(λ)` therefore never varies across the trajectory → **NaRA collapses to
plain LoRA**. So at load we fold the fixed `C(λ)` into `A` (`A' = C(λ)·A`) and each
adapter becomes a single static low-rank delta `ΔW = s·B·A'`.

`spec_generate` is batch-1 with ragged acceptance lengths, so there's nothing to
token-batch across adapters — a Punica gather kernel buys nothing here. The real
win is the **shared resident backbone**: load the 8B+1B once, and for each request
just point the drafter's q/k/v/o LoRA layers at that request's adapter — an O(1)
index swap. That's `MultiAdapterLoRALinear.set_active(i)`.

Note: at temperature 0 DFlash is **lossless** — the emitted text is exactly the
8B's greedy output regardless of the adapter. The adapter only changes the draft's
acceptance length, i.e. **speed**, not correctness.

## Files
| File | Role |
|---|---|
| `adapter_bank.py` | `MultiAdapterLoRALinear` (N unmerged deltas, O(1) swap), `AdapterBank` (loads `nara_state_dict`, folds `C(λ_fix)`), `make_toy_adapter` |
| `serve_nara.py` | `AdapterServer`: load backbone once, serve requests by swapping adapters via real `spec_generate` |
| `modal_serve_nara.py` | Modal app `nara-multi-adapter-serve` (A100, reuses `dflash-hf-cache`) — isolated from the benchmark app |
| `dry_run_serving.py` | CPU test of the swap machinery on the real `DFlashDraftModel` |

## Run
```bash
# CPU: verify the multi-adapter swap machinery (no downloads)
../.venv/bin/python dry_run_serving.py

# GPU on Modal: 5 requests over 5 toy adapters, real 8B+1B backbone
export PATH="$HOME/Library/Python/3.11/bin:$PATH"
modal run modal_serve_nara.py --n-toy 5 --max-new-tokens 128

# once the real adapter is trained (drop-in; nara_state_dict format):
modal run modal_serve_nara.py --adapter-glob '/adapters/*.pt' --n-toy 4
#   (put the .pt on the `nara-adapters` Modal volume, or serve locally:)
python serve_nara.py --adapters /path/to/nara_adapter.pt --n-toy 4
```

## Plugging in the real adapter
The training agent lives in `../python-lora-drafter/` (renamed from
`python-nara-drafter` when it dropped NaRA for plain LoRA). It trains **3 unmerged
LoRA adapters** — python / sql / prose — on the drafter's q/k/v/o.

`load_lora_checkpoint()` is **format-agnostic** and reads any of:
- `python-lora-drafter`'s `lora_state_dict`: flat `{layer: {"A","B","scaling"}}`
  (per-layer scaling honored) — **the expected real format**;
- PEFT: a dir (`adapter_config.json` + `adapter_model.safetensors`) or raw
  `*.lora_A`/`*.lora_B` state_dict (scaling from `lora_alpha`);
- NaRA `nara_state_dict` (`{"hypernetwork", "layers"}`), folding `C(λ_fix)`.

So the real checkpoint is a drop-in for the toy adapters — pass it via
`--adapters <path...>` / `--adapter-glob`. Note the training PLAN references the
base drafter as `eigen-ai-labs/qwen3-8b-dflash-demo`; if the trained adapter uses
that instead of `z-lab/Qwen3-8B-DFlash-b16`, set `DRAFT_MODEL` to match (same
Qwen3 layer names, so injection is unaffected).

## Verified
- **CPU dry-run**: 5 adapters swap on the real `DFlashDraftModel`, distinct outputs,
  `activate(None)`==base, stateless re-swap, shared backbone byte-identical.
- **Modal A100**: real Qwen3-8B + DFlash drafter loaded once; 5 requests each on a
  different toy adapter served through the full block-diffusion `spec_generate`;
  coherent Qwen3-8B output; clean exit.
