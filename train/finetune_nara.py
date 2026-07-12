#!/usr/bin/env python3
"""Fine-tune a diffusion LLM with NaRA (Noise-Aware LoRA).

This is a *fine-tuning* script (PEFT), NOT the DFlash pre-training / distillation
trainer. It freezes a base diffusion LLM and trains only the NaRA adapters
({A, B} per layer + one shared hypernetwork φ) with the masked-diffusion
objective from arXiv:2605.29716.

Masked-diffusion loss (LLaDA-style):
    1. sample noise level  t ~ U(ε, 1)  per sequence
    2. mask each response token independently with probability t  -> r_t
    3. λ = (#masked) / (response length)          # per-sequence noise level
    4. C(λ) computed once by the shared hypernetwork, broadcast to all layers
    5. predict the original tokens at masked positions
    6. loss = mean_b (1/t_b) * (Σ_{i∈masked_b} CE_i) / L_response

Run modes:
    python finetune_nara.py --dry-run           # CPU smoke test, no downloads
    python finetune_nara.py --model <hf_dllm> --data data.jsonl   # real run
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nara import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    inject_nara,
    nara_state_dict,
    nara_trainable_parameters,
)


# --------------------------------------------------------------------------- #
# Masked-diffusion training step (model-agnostic: needs forward -> logits)
# --------------------------------------------------------------------------- #
def masked_diffusion_step(
    model: nn.Module,
    controller,
    input_ids: torch.Tensor,
    mask_token_id: int,
    prompt_len: int = 0,
    eps: float = 1e-3,
    logits_fn=None,
):
    """One forward pass returning (loss, metrics). `logits_fn(model, ids)` maps
    input ids -> logits (batch, seq, vocab); defaults to HF `.logits`."""
    device = input_ids.device
    B, L = input_ids.shape
    resp_len = L - prompt_len
    assert resp_len > 0, "sequence must be longer than prompt_len"

    # 1. per-sequence noise level t ~ U(eps, 1)
    t = torch.rand(B, device=device) * (1.0 - eps) + eps  # (B,)

    # 2. mask response tokens with prob t (prompt region is always clean)
    rand = torch.rand(B, resp_len, device=device)
    resp_mask = rand < t.unsqueeze(1)  # (B, resp_len) True = masked
    # guarantee at least one masked token per sequence (else 1/t*0 = no signal)
    no_mask = ~resp_mask.any(dim=1)
    if no_mask.any():
        resp_mask[no_mask, 0] = True

    full_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    full_mask[:, prompt_len:] = resp_mask

    noisy = input_ids.clone()
    noisy[full_mask] = mask_token_id

    # 3. per-sequence lambda = fraction of RESPONSE tokens masked
    lam = resp_mask.float().sum(dim=1) / resp_len  # (B,)

    # 4. compute C(λ) once, shared across all NaRA layers
    controller.set_noise_level(lam)

    # 5. forward + predict originals at masked positions
    if logits_fn is None:
        logits = model(input_ids=noisy).logits
    else:
        logits = logits_fn(model, noisy)

    vocab = logits.size(-1)
    ce = F.cross_entropy(
        logits.view(-1, vocab), input_ids.view(-1), reduction="none"
    ).view(B, L)

    # 6. weighted masked loss: (1/t) * sum(CE over masked) / resp_len, mean over batch
    per_seq = (ce * full_mask.float()).sum(dim=1) / resp_len
    loss = (per_seq / t).mean()

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = ((pred == input_ids) & full_mask).sum().float()
        acc = correct / full_mask.sum().clamp_min(1)
    return loss, {"acc": acc.item(), "mask_frac": full_mask.float().mean().item()}


# --------------------------------------------------------------------------- #
# Tiny bidirectional diffusion LM used only by --dry-run (no external downloads)
# --------------------------------------------------------------------------- #
class _TinyAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = (
            t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
        )
        # bidirectional attention (diffusion LLMs are NOT causal)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class _TinyBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = _TinyAttention(dim, heads)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class TinyDiffusionLM(nn.Module):
    def __init__(self, vocab, dim=128, heads=4, layers=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([_TinyBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, input_ids):
        L = input_ids.size(1)
        pos = torch.arange(L, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


def dry_run(args):
    print("=" * 72)
    print("NaRA fine-tuning DRY RUN (tiny synthetic diffusion LM, CPU)")
    print("=" * 72)
    torch.manual_seed(0)
    vocab, mask_id = 512, 511
    model = TinyDiffusionLM(vocab, dim=128, heads=4, layers=2, max_len=64)

    base_params = sum(p.numel() for p in model.parameters())
    controller, replaced = inject_nara(
        model,
        rank=args.rank,
        eta=args.eta,
        target_modules=DEFAULT_TARGET_MODULES,
    )
    print(f"base params: {base_params:,}")
    print(f"NaRA-adapted layers ({len(replaced)}): {replaced}")

    trainable = list(nara_trainable_parameters(model, controller))
    n_train = sum(p.numel() for p in trainable)
    n_frozen_with_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"trainable NaRA params: {n_train:,} ({100*n_train/base_params:.2f}% of base)")
    # base weights must be frozen; only A/B live inside NaRALinear (not model.parameters
    # of the frozen base). Confirm nothing in the frozen base leaks grad:
    assert n_frozen_with_grad == 0 or all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if ".base." in n
    ), "base weights should be frozen"

    logits_fn = lambda m, ids: m(ids)  # noqa: E731  TinyDiffusionLM returns logits directly

    # sanity: at init C(λ) == I_r  => adapter is a no-op (ΔW = 0 because B=0)
    controller.set_noise_level(torch.tensor([0.3]))
    C0 = controller.C
    I = torch.eye(args.rank)
    print(f"init ‖C(λ)-I‖ = {(C0.squeeze(0)-I).abs().max().item():.2e} (expect ~0)")

    opt = torch.optim.AdamW(trainable, lr=1e-3)
    before = trainable[0].detach().clone()
    hyper_before = next(controller.hypernetwork.net[-1].parameters()).detach().clone()

    print("-" * 72)
    losses = []
    for step in range(args.steps):
        ids = torch.randint(0, vocab - 1, (args.batch_size, 32))
        loss, m = masked_diffusion_step(
            model, controller, ids, mask_token_id=mask_id,
            prompt_len=4, logits_fn=logits_fn,
        )
        opt.zero_grad()
        loss.backward()
        # verify gradients reach A/B and the hypernetwork
        assert trainable[0].grad is not None, "no grad on NaRA A matrix"
        grad_to_hyper = sum(
            p.grad.abs().sum().item()
            for p in controller.hypernetwork.parameters()
            if p.grad is not None
        )
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(
                f"step {step:3d} | loss {loss.item():.4f} | acc {m['acc']:.3f} "
                f"| mask_frac {m['mask_frac']:.2f} | Σ|grad→φ| {grad_to_hyper:.3e}"
            )

    after = trainable[0].detach()
    hyper_after = next(controller.hypernetwork.net[-1].parameters()).detach()
    print("-" * 72)
    assert not torch.equal(before, after), "NaRA A/B did not update"
    assert not torch.equal(hyper_before, hyper_after), "hypernetwork φ did not update"
    assert all(math.isfinite(x) for x in losses), "non-finite loss encountered"

    # after training, C(λ) should have moved OFF identity (noise-awareness engaged)
    controller.set_noise_level(torch.tensor([0.3]))
    drift = (controller.C.squeeze(0) - I).abs().max().item()
    print(f"post-train ‖C(0.3)-I‖ = {drift:.3e} (should be > 0: C now noise-aware)")
    assert drift > 0, "core matrix never left identity"

    print()
    print("=" * 72)
    print("NaRA DRY RUN PASSED")
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f} over {args.steps} steps")
    print("  - base frozen, only {A,B,φ} trained")
    print("  - C(λ)=I at init, drifts off identity after training (noise-aware)")
    print("  - gradients flow to A/B AND the shared hypernetwork")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Real fine-tuning path (HF diffusion LLM, e.g. LLaDA / Dream)
# --------------------------------------------------------------------------- #
def real_run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    mask_id = args.mask_token_id
    if mask_id is None:
        mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("--mask-token-id required (tokenizer has no mask token).")

    controller, replaced = inject_nara(
        model,
        rank=args.rank,
        eta=args.eta,
        target_modules=tuple(args.target_modules),
    )
    print(f"Injected NaRA into {len(replaced)} layers.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    controller.to(device)

    ds = load_dataset("json", data_files=args.data)["train"]
    text_key = "text" if "text" in ds.column_names else ds.column_names[0]

    def collate(batch):
        texts = [b[text_key] for b in batch]
        enc = tok(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=args.max_length,
        )
        return enc["input_ids"]

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    opt = torch.optim.AdamW(
        list(nara_trainable_parameters(model, controller)), lr=args.lr
    )

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for input_ids in loader:
            input_ids = input_ids.to(device)
            loss, m = masked_diffusion_step(
                model, controller, input_ids,
                mask_token_id=mask_id, prompt_len=args.prompt_len,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(nara_trainable_parameters(model, controller)), 1.0
            )
            opt.step()
            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} acc {m['acc']:.3f}")
            step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, "nara_adapter.pt")
    torch.save(nara_state_dict(model, controller), ckpt)
    print(f"Saved NaRA adapter to {ckpt}")


def parse_args():
    p = argparse.ArgumentParser(description="NaRA fine-tuning for diffusion LLMs")
    p.add_argument("--dry-run", action="store_true", help="CPU smoke test, no downloads")
    p.add_argument("--model", type=str, default=None, help="HF diffusion LLM id/path")
    p.add_argument("--data", type=str, default=None, help="jsonl with a text field")
    p.add_argument("--output-dir", type=str, default="./outputs/nara")
    # NaRA hyperparameters (paper defaults)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    p.add_argument("--mask-token-id", type=int, default=None)
    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=20, help="dry-run steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    elif args.model and args.data:
        real_run(args)
    else:
        raise SystemExit("Use --dry-run, or provide --model and --data for a real run.")


if __name__ == "__main__":
    main()
