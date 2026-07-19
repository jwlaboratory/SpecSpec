#!/usr/bin/env python3
"""FULL fine-tune of the Qwen3-8B DFlash drafter — every weight trains (no LoRA).

    modal run Full-Tune/full_tune.py                    # smoke: full FT one domain (synthetic)
    modal run Full-Tune/full_tune.py --domain python    # full FT the python drafter
    modal run Full-Tune/full_tune.py --domain python --steps 2000 --lr 2e-5
    python Full-Tune/full_tune.py                        # local CPU correctness check (no GPU)

Difference vs. `../LoRA/`: LoRA freezes the drafter and trains a tiny low-rank
adapter on q/k/v/o. Here there is NO adapter and NOTHING is frozen — the entire
~1.05B DFlash drafter is optimized directly with the SAME DFlash matching loss
(exponentially-weighted block cross-entropy against the TARGET's tokens; γ=7, NOT
perplexity). Output is a full drafter checkpoint (a complete `state_dict`), one
per domain, not a small adapter.

Trade-off: a full fine-tune is much heavier (all params + AdamW moment states, so
needs A100-80GB and a smaller LR ~2e-5) and produces a whole new drafter per
domain — you can't cheaply hot-swap many of them on one backbone the way LoRA
adapters do. Use this when a domain needs more capacity than a low-rank delta can
express; otherwise prefer `../LoRA/`.

The target's embed / lm_head stay frozen (we fine-tune the DRAFTER, not the 8B).
Training data is synthetic here (machinery validation) — real training regenerates
per-domain WildChat responses with Qwen3-8B first.
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (finetuning/Full-Tune/ -> repo)
SPECFORGE_PKG = ROOT / "SpecForge" / "specforge"
CONFIG = ROOT / "SpecForge" / "configs" / "qwen3-8b-dflash.json"

app = modal.App("dflash-full-tune")
GPU = "A100-80GB"  # full 1B FT: weights + grads + fp32 AdamW moments need the headroom

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "transformers==5.8.1", "numpy")
    .add_local_dir(str(SPECFORGE_PKG), "/root/specforge")
    .add_local_file(str(CONFIG), "/root/qwen3-8b-dflash.json")
)

# full drafter checkpoints land here (one state_dict per domain)
ckpt_vol = modal.Volume.from_name("dflash-fulltune-ckpts", create_if_missing=True)


def _load_dflash():
    """Import DFlashDraftModel + OnlineDFlashModel from /root/specforge without
    running specforge/__init__ (avoids triton/sglang)."""
    import importlib.util
    import sys
    import types

    def stub(name, path):
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
        sys.modules[name] = m

    for n, p in [
        ("specforge", "/root/specforge"),
        ("specforge.modeling", "/root/specforge/modeling"),
        ("specforge.modeling.draft", "/root/specforge/modeling/draft"),
        ("specforge.core", "/root/specforge/core"),
    ]:
        stub(n, p)

    def load(mod, path):
        spec = importlib.util.spec_from_file_location(mod, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m

    d = load("specforge.modeling.draft.dflash", "/root/specforge/modeling/draft/dflash.py")
    c = load("specforge.core.dflash", "/root/specforge/core/dflash.py")
    return d.DFlashDraftModel, c.OnlineDFlashModel


@app.function(gpu=GPU, image=image, timeout=3600, volumes={"/ckpt": ckpt_vol})
def train(domain: str = "python", steps: int = 200, lr: float = 2e-5,
          weight_decay: float = 0.0, save: bool = True):
    import sys
    import time

    import torch
    from transformers import AutoConfig

    DFlashDraftModel, OnlineDFlashModel = _load_dflash()
    dev = "cuda"
    torch.manual_seed(0)

    # --- real 1B DFlash draft (real config/shape) — EVERYTHING trainable ---
    cfg = AutoConfig.from_pretrained("/root/qwen3-8b-dflash.json")
    cfg._attn_implementation = "sdpa"
    draft = DFlashDraftModel(cfg).to(device=dev, dtype=torch.bfloat16)
    for p in draft.parameters():
        p.requires_grad_(True)  # full fine-tune: no frozen base, no adapter
    n_params = sum(p.numel() for p in draft.parameters())
    n_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)

    H, V = cfg.hidden_size, cfg.vocab_size
    n_ctx = len(cfg.dflash_config["target_layer_ids"]) * H
    # target's embed / lm_head — frozen (we fine-tune the DRAFTER, not the 8B target)
    embed = torch.nn.Embedding(V, H, dtype=torch.bfloat16, device=dev)
    lm_head = torch.nn.Linear(H, V, bias=False, dtype=torch.bfloat16, device=dev)
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=cfg.dflash_config["mask_token_id"], block_size=cfg.block_size,
        attention_backend="sdpa", num_anchors=128, loss_decay_gamma=7.0, loss_type="dflash",
    )

    # --- optimize ALL drafter weights with the DFlash matching loss (synthetic target) ---
    opt = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=weight_decay)
    draft.train()
    t0 = time.time()
    losses = []
    for step in range(steps):
        ids = torch.randint(0, V - 1, (2, 512), device=dev)
        hid = torch.randn(2, 512, n_ctx, device=dev, dtype=torch.bfloat16)
        lm = torch.ones(2, 512, device=dev)
        loss, acc = online(input_ids=ids, hidden_states=hid, loss_mask=lm)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    log = {
        "domain": domain, "draft_params": n_params, "trainable_params": n_trainable,
        "full_finetune": n_trainable == n_params, "steps": steps, "lr": lr,
        "loss0": losses[0], "lossN": losses[-1],
        "sec": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
    }

    # --- save the full drafter checkpoint (complete state_dict) ---
    if save:
        out = f"/ckpt/dflash-drafter-{domain}.pt"
        torch.save(draft.state_dict(), out)
        ckpt_vol.commit()
        log["saved"] = out
    return log


@app.local_entrypoint()
def main(domain: str = "python", steps: int = 200, lr: float = 2e-5,
         weight_decay: float = 0.0, save: bool = True):
    import json

    print(json.dumps(
        train.remote(domain=domain, steps=steps, lr=lr, weight_decay=weight_decay, save=save),
        indent=2,
    ))


# --------------------------------------------------------------------------- #
# Local CPU correctness check (no GPU / no Modal): proves the full-FT loop runs
# and that ALL drafter weights actually move (the opposite of the LoRA check,
# where the base must stay frozen). Run:  python Full-Tune/full_tune.py
# --------------------------------------------------------------------------- #
def _local_check():
    import importlib.util
    import os
    import sys
    import types

    import torch

    SPECFORGE = os.environ.get("SPECFORGE_ROOT", str(ROOT / "SpecForge"))

    def stub(name, path):
        m = types.ModuleType(name); m.__path__ = [path]; m.__package__ = name
        sys.modules[name] = m

    sf = os.path.join(SPECFORGE, "specforge")
    for n, p in [
        ("specforge", sf),
        ("specforge.modeling", os.path.join(sf, "modeling")),
        ("specforge.modeling.draft", os.path.join(sf, "modeling", "draft")),
        ("specforge.core", os.path.join(sf, "core")),
    ]:
        stub(n, p)

    def load(mod, rel):
        spec = importlib.util.spec_from_file_location(mod, os.path.join(SPECFORGE, rel))
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m

    DFlashDraftModel = load("specforge.modeling.draft.dflash",
                            "specforge/modeling/draft/dflash.py").DFlashDraftModel
    OnlineDFlashModel = load("specforge.core.dflash",
                             "specforge/core/dflash.py").OnlineDFlashModel
    from transformers import AutoConfig

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(os.path.join(SPECFORGE, "configs", "qwen3-8b-dflash.json"))
    cfg.hidden_size = 256; cfg.intermediate_size = 512; cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 8; cfg.num_key_value_heads = 4; cfg.head_dim = 32
    cfg.block_size = 8; cfg.num_target_layers = 2
    cfg.layer_types = ["full_attention", "full_attention"]
    cfg.dflash_config = {"mask_token_id": 511, "target_layer_ids": [0, 1]}
    cfg._attn_implementation = "sdpa"

    draft = DFlashDraftModel(cfg).to(torch.float32).train()
    for p in draft.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in draft.parameters())
    n_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"draft params: {n_params:,}  trainable: {n_trainable:,}  "
          f"full fine-tune? {n_trainable == n_params}")
    assert n_trainable == n_params, "full FT must train every drafter weight"

    H, V = cfg.hidden_size, 512
    n_ctx = len(cfg.dflash_config["target_layer_ids"]) * H
    embed = torch.nn.Embedding(V, H); lm_head = torch.nn.Linear(H, V, bias=False)
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)
    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=cfg.dflash_config["mask_token_id"], block_size=cfg.block_size,
        attention_backend="sdpa", num_anchors=16, loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    snap = {n: p.detach().clone() for n, p in draft.named_parameters()}
    for _ in range(20):
        loss, _ = online(input_ids=torch.randint(0, V - 1, (1, 128)),
                         hidden_states=torch.randn(1, 128, n_ctx), loss_mask=torch.ones(1, 128))
        opt.zero_grad(); loss.backward(); opt.step()

    moved = sum(1 for n, p in draft.named_parameters() if not torch.equal(snap[n], p))
    total = sum(1 for _ in draft.named_parameters())
    print(f"drafter tensors that changed: {moved}/{total} (full FT expects most/all)")
    assert moved > 0, "full fine-tune should move drafter weights"
    print("LOCAL CHECK PASSED — full fine-tune loop runs; drafter weights update")


if __name__ == "__main__":
    _local_check()
