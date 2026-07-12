#!/usr/bin/env python3
"""DFlash Qwen3-8B -> 1B draft: local CPU dry run / smoke test.

The official training entrypoint (SpecForge/scripts/train_dflash.py) needs an
8xGPU box: it imports sglang (CUDA-only) for the target-model backend and uses
torchrun + FSDP + flex_attention. None of that runs on this Mac.

This script validates everything that CAN be checked without CUDA:

  1. Builds the REAL DFlash draft model from the exact recipe config
     (configs/qwen3-8b-dflash.json) and reports its parameter count -- this is
     the actual "1B" draft the recipe produces.
  2. Runs a real forward + backward + optimizer step through the actual
     OnlineDFlashModel training wrapper (the same anchor sampling, block-diffusion
     noise masking, attention masking, DFlash decayed loss, and accuracy code that
     the GPU trainer runs), using synthetic target hidden states / embeddings in
     place of the frozen 8B target components (which are not what gets trained).

If this prints "DRY RUN PASSED", the DFlash training code path is wired up
correctly for this 8B->1B configuration; the only missing piece for a real run
is CUDA hardware + the regenerated dataset.
"""
import argparse
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SPECFORGE = os.environ.get("SPECFORGE_ROOT", os.path.join(HERE, "..", "SpecForge"))
SPECFORGE = os.path.abspath(SPECFORGE)


def _load_dflash_modules():
    """Import the two self-contained DFlash modules WITHOUT running
    specforge/__init__.py (which pulls in triton / sglang / yunchang, none of
    which install on macOS). Both target files only depend on torch +
    transformers, so we stub the parent packages and exec the files directly."""

    def stub(name, path):
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
        sys.modules[name] = m

    sf = os.path.join(SPECFORGE, "specforge")
    stub("specforge", sf)
    stub("specforge.modeling", os.path.join(sf, "modeling"))
    stub("specforge.modeling.draft", os.path.join(sf, "modeling", "draft"))
    stub("specforge.core", os.path.join(sf, "core"))

    def load(modname, relpath):
        spec = importlib.util.spec_from_file_location(
            modname, os.path.join(SPECFORGE, relpath)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    draft = load(
        "specforge.modeling.draft.dflash", "specforge/modeling/draft/dflash.py"
    )
    core = load("specforge.core.dflash", "specforge/core/dflash.py")
    return draft.DFlashDraftModel, core.OnlineDFlashModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.path.join(SPECFORGE, "configs", "qwen3-8b-dflash.json"),
        help="DFlash draft config (the 8B->1B recipe config).",
    )
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-anchors", type=int, default=64)
    ap.add_argument("--loss-decay-gamma", type=float, default=7.0)
    ap.add_argument(
        "--vocab",
        type=int,
        default=8192,
        help="Synthetic vocab for the FROZEN target embed/head only "
        "(keeps RAM low; the trained draft body stays full real size).",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    from transformers import AutoConfig

    DFlashDraftModel, OnlineDFlashModel = _load_dflash_modules()

    print("=" * 72)
    print("STEP 1: Build the real DFlash draft model from the recipe config")
    print("=" * 72)
    print(f"config: {args.config}")
    cfg = AutoConfig.from_pretrained(args.config)
    print(
        f"  block_size={cfg.block_size}  hidden_size={cfg.hidden_size}  "
        f"num_hidden_layers={cfg.num_hidden_layers}  "
        f"num_target_layers={cfg.num_target_layers}"
    )
    draft = DFlashDraftModel(cfg).to(dtype=torch.float32)
    draft.train()
    n_params = sum(p.numel() for p in draft.parameters())
    print(f"  target_layer_ids = {draft.target_layer_ids}")
    print(f"  >>> DFlash draft model parameters: {n_params:,} (~{n_params/1e9:.2f}B)")

    print()
    print("=" * 72)
    print("STEP 2: Real forward + backward + step through OnlineDFlashModel")
    print("=" * 72)
    device = torch.device("cpu")
    B, S, H = args.batch_size, args.seq_len, cfg.hidden_size
    V = args.vocab
    mask_token_id = V - 1
    n_ctx_feats = len(draft.target_layer_ids) * H  # what draft.fc expects

    # Frozen target components (stand-ins for the 8B model's embed/head).
    embed_tokens = torch.nn.Embedding(V, H).to(device)
    lm_head = torch.nn.Linear(H, V, bias=False).to(device)
    for p in embed_tokens.parameters():
        p.requires_grad_(False)
    for p in lm_head.parameters():
        p.requires_grad_(False)

    online = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        mask_token_id=mask_token_id,
        block_size=cfg.block_size,
        attention_backend="sdpa",  # flex_attention requires CUDA
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type="dflash",
    ).to(device)

    input_ids = torch.randint(0, V - 1, (B, S), device=device)
    # target hidden states captured from the 8B model (concatenated layers).
    hidden_states = torch.randn(B, S, n_ctx_feats, device=device)
    loss_mask = torch.ones(B, S, device=device)  # all tokens count toward loss

    opt = torch.optim.SGD([p for p in draft.parameters() if p.requires_grad], lr=1e-3)

    print(f"  batch={B} seq_len={S} hidden={H} ctx_feats={n_ctx_feats} vocab={V}")
    print("  running forward...")
    loss, accuracy = online(
        input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
    )
    print(f"    loss={loss.item():.4f}  accuracy={accuracy.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    assert loss.requires_grad, "loss has no grad -- graph not built"

    print("  running backward...")
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in draft.parameters() if p.grad is not None]
    assert grads, "no gradients produced on the draft model"
    total_grad = sum(g.abs().sum().item() for g in grads)
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
    print(f"    params with grad: {len(grads)}  sum|grad|={total_grad:.2f}")

    print("  running optimizer step...")
    before = next(p for p in draft.parameters() if p.requires_grad).detach().clone()
    opt.step()
    after = next(p for p in draft.parameters() if p.requires_grad).detach()
    changed = not torch.equal(before, after)
    assert changed, "optimizer step did not update weights"
    print("    weights updated: OK")

    print()
    print("=" * 72)
    print("DRY RUN PASSED")
    print("  - real 8B->1B DFlash draft builds from the recipe config")
    print("  - forward/backward/step through the real DFlash loss all work")
    print("  Next: run on CUDA hardware via train/run_qwen3_8b_dflash.sh")
    print("=" * 72)


if __name__ == "__main__":
    main()
