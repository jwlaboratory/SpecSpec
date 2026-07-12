# DFlash + NaRA

Two separate things live here:

| File | What it is |
|------|-----------|
| `run_qwen3_8b_dflash.sh`, `dry_run.py` | **DFlash pre-training** — trains the 8B→1B block-diffusion *draft model* for speculative decoding (SpecForge recipe). |
| `nara.py`, `finetune_nara.py` | **NaRA fine-tuning** — parameter-efficient (LoRA-style) fine-tuning of a diffusion LLM, noise-aware. |

Both ship with a `--dry-run` / CPU smoke test because this machine has no CUDA GPU.

Environment: `../.venv` (torch 2.11.0, transformers 5.8.1). Activate with
`source ../.venv/bin/activate` or call `../.venv/bin/python`.

---

## NaRA — Noise-Aware Low-Rank Adaptation  (arXiv:2605.29716)

Standard LoRA freezes `W₀` and learns one static update `ΔW = BA`, reused at
every noise level. A diffusion LLM's task changes drastically along the
denoising trajectory (few masked tokens = easy/local; many = hard/global), so a
static adapter is a poor fit. **NaRA makes the update depend on the current
noise level λ** (= fraction of masked tokens) by inserting a small, dynamic
`r×r` core matrix between the two static LoRA projections:

```
h = W₀x + B · C(λ) · A · x            C(λ) = I_r + η · F_φ(e_λ)
e_λ = [cos(2π k λ) ⊕ sin(2π k λ)],  k ~ N(0, σ²)   (Gaussian Fourier features)
```

- `A (r×k)`, `B (d×r)` — static, per-layer (as in LoRA).
- `C(λ) (r×r)` — dynamic, produced by **one globally-shared hypernetwork** `F_φ`
  (a tiny MLP) conditioned on λ. Computed once per step, broadcast to all layers.
  Negligible extra params/latency.
- Init: `B=0` and hypernetwork's last layer `=0` ⟹ `C(λ)=I` and `ΔW=0` at
  step 0, so training starts identical to plain LoRA and specializes from there.
- Trained with the masked-diffusion loss: sample `t~U(ε,1)`, mask each token
  w.p. `t`, predict the originals, weight by `1/t`.

### Files
- `nara.py` — the adapter: `GaussianFourierEmbedding`, `NaRAHypernetwork`,
  `NaRAController` (owns the shared hypernet, caches `C(λ)`), `NaRALinear`
  (wraps a frozen `nn.Linear`), and `inject_nara(model, ...)` which freezes the
  base model and swaps target `nn.Linear`s (default `q/k/v/o_proj`) for
  `NaRALinear`, all sharing one hypernetwork.
- `finetune_nara.py` — the fine-tuning script: `masked_diffusion_step(...)`,
  a real HF training path, and a self-contained CPU dry run.

### Run
```bash
# CPU smoke test (no downloads) — verifies the whole NaRA machinery
../.venv/bin/python finetune_nara.py --dry-run --steps 20

# Real fine-tune of a diffusion LLM (needs a GPU + a masked-diffusion base model)
../.venv/bin/python finetune_nara.py \
    --model <hf-diffusion-llm> \
    --data data.jsonl \
    --rank 32 --eta 0.1 \
    --target-modules q_proj k_proj v_proj o_proj \
    --mask-token-id <MASK_ID> \
    --output-dir ./outputs/nara
```
`--data` is a JSONL with a `text` field. Point `--model` at a masked-diffusion
LLM (e.g. LLaDA / Dream); `--mask-token-id` defaults to the tokenizer's mask
token. Only the NaRA adapter is saved (`nara_adapter.pt`), not the base weights.

The dry run asserts: base frozen; only `{A,B,φ}` trainable; `C(λ)=I` at init;
`C` drifts off identity after training; gradients reach both `{A,B}` and `φ`.

---

## DFlash draft pre-training (context / earlier work)

`dry_run.py` builds the real ~1.05B DFlash draft from
`../SpecForge/configs/qwen3-8b-dflash.json` and runs a full
forward→backward→step through the actual `OnlineDFlashModel` loss on CPU with
synthetic target tensors. The real 8-GPU run is `run_qwen3_8b_dflash.sh`
(wraps SpecForge; needs CUDA + sglang + the regenerated dataset).

```bash
../.venv/bin/python dry_run.py          # CPU smoke test of the 8B->1B draft
```
