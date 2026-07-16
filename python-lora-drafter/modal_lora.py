#!/usr/bin/env python3
"""Validate the multi-adapter LoRA machinery on the REAL 1B DFlash drafter, on an
A100, via Modal.

  modal run python-lora-drafter/modal_lora.py            # smoke (real drafter, GPU)

The smoke builds the real ~1.05B Qwen3-8B DFlash draft (real config/shape), injects
3 batched LoRA adapters, runs a few DFlash-matching-loss training steps, and
verifies hot-swap + exact batched per-sequence routing -- all on GPU. It uses the
sglang-free module loader (same trick as the local demo), so the image is just
torch + transformers (no sglang/triton/8B download): fast and cheap.

Full real-data training (3 domains regenerated with Qwen3-8B) is the next step and
is scaffolded in PLAN.md.
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root
SPECFORGE_PKG = ROOT / "SpecForge" / "specforge"
CONFIG = ROOT / "SpecForge" / "configs" / "qwen3-8b-dflash.json"
LOCAL = pathlib.Path(__file__).resolve().parent

app = modal.App("dflash-lora-multiadapter")
GPU = "A100-40GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "transformers==5.8.1", "numpy")
    .add_local_dir(str(SPECFORGE_PKG), "/root/specforge")
    .add_local_file(str(CONFIG), "/root/qwen3-8b-dflash.json")
    .add_local_file(str(LOCAL / "batched_lora.py"), "/root/batched_lora.py")
)


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


@app.function(gpu=GPU, image=image, timeout=1800)
def smoke():
    import sys
    import time

    import torch
    from transformers import AutoConfig

    sys.path.insert(0, "/root")
    from batched_lora import (
        BatchedLoRALinear,
        adapter_parameters,
        inject_batched_lora,
    )

    DFlashDraftModel, OnlineDFlashModel = _load_dflash()
    dev = "cuda"
    torch.manual_seed(0)
    domains = ["python", "sql", "prose"]

    # --- real 1B DFlash draft (real config/shape) ---
    cfg = AutoConfig.from_pretrained("/root/qwen3-8b-dflash.json")
    cfg._attn_implementation = "sdpa"
    draft = DFlashDraftModel(cfg).to(device=dev, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in draft.parameters())

    controller, replaced = inject_batched_lora(
        draft, domains, rank=16, alpha=32,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    # move the freshly-injected LoRA params onto the GPU in bf16 (base already is)
    draft.to(device=dev, dtype=torch.bfloat16)

    H = cfg.hidden_size
    V = cfg.vocab_size
    n_ctx = len(cfg.dflash_config["target_layer_ids"]) * H
    embed = torch.nn.Embedding(V, H, dtype=torch.bfloat16, device=dev)
    lm_head = torch.nn.Linear(H, V, bias=False, dtype=torch.bfloat16, device=dev)
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=cfg.dflash_config["mask_token_id"], block_size=cfg.block_size,
        attention_backend="sdpa", num_anchors=128, loss_decay_gamma=7.0, loss_type="dflash",
    )

    # --- train each adapter a few DFlash-loss steps (synthetic target here) ---
    log = {"draft_params": n_params, "adapted_layers": len(replaced), "train": {}}
    for d, name in enumerate(domains):
        opt = torch.optim.AdamW(list(adapter_parameters(draft, d)), lr=1e-3, weight_decay=0.0)
        controller.use_adapter(d)
        draft.train()
        t0 = time.time()
        losses = []
        for _ in range(20):
            ids = torch.randint(0, V - 1, (2, 512), device=dev)
            hid = torch.randn(2, 512, n_ctx, device=dev, dtype=torch.bfloat16)
            lm = torch.ones(2, 512, device=dev)
            loss, acc = online(input_ids=ids, hidden_states=hid, loss_mask=lm)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        log["train"][name] = {"loss0": losses[0], "lossN": losses[-1],
                              "sec": round(time.time() - t0, 1)}

    # --- hot-swap + exact batched routing, verified on a real drafter LoRA layer ---
    # (the drafter's block forward is driven correctly by OnlineDFlashModel above;
    #  here we prove the multi-adapter routing math itself is exact on GPU.)
    draft.eval()
    layer = next(m for m in draft.modules() if isinstance(m, BatchedLoRALinear))
    routes = [0, 0, 1, 1, 2, 2]
    x = torch.randn(len(routes), 8, layer.base.in_features, device=dev, dtype=torch.bfloat16)

    controller.route(routes)
    routed = layer(x)                       # one batched forward, per-seq adapter
    max_err = 0.0
    for i, d in enumerate(routes):
        controller.use_adapter(d)           # hot-swap to adapter d
        single = layer(x[i : i + 1])
        max_err = max(max_err, (routed[i : i + 1] - single).abs().max().item())
    log["routing_max_abs_err"] = max_err
    log["routing_exact"] = max_err < 1e-2   # bf16 tolerance
    log["gpu"] = torch.cuda.get_device_name(0)
    return log


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(smoke.remote(), indent=2))
