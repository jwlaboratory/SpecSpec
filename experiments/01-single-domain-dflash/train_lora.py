#!/usr/bin/env python3
"""Train an UNMERGED LoRA adapter on the real Qwen3-8B DFlash drafter (A100, Modal).

    modal run experiments/01-single-domain-dflash/train_lora.py                        # smoke: train one adapter (synthetic)
    modal run experiments/01-single-domain-dflash/train_lora.py --domain python        # train the python adapter
    modal run experiments/01-single-domain-dflash/train_lora.py --domain python --steps 2000
    python experiments/01-single-domain-dflash/train_lora.py                            # local CPU correctness check (no GPU)

What this does: builds the real ~1.05B Qwen3-8B DFlash draft (real config/shape),
injects a single LoRA on the drafter's q/k/v/o (base frozen), and trains only the
LoRA {A,B} with DFlash's exponentially-weighted block matching loss (predict the
TARGET's tokens — NOT perplexity). It saves a compact `lora_state_dict` .pt to the
`lora-adapters` volume; that flat `{layer: {"A","B","scaling"}}` format is exactly
what `../serving/adapter_bank.py` loads, so a trained adapter is a drop-in there.

One adapter per run (one per domain: python / sql / prose). Serving banks the
per-domain checkpoints and swaps between them on a shared backbone.

Training data is synthetic here (machinery validation). Real training regenerates
per-domain WildChat responses with Qwen3-8B first — see PLAN.md.
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (experiments/01-single-domain-dflash/ -> repo)
SPECFORGE_PKG = ROOT / "third_party" / "SpecForge" / "specforge"
CONFIG = ROOT / "third_party" / "SpecForge" / "configs" / "qwen3-8b-dflash.json"
LOCAL = pathlib.Path(__file__).resolve().parent

app = modal.App("dflash-lora-train")
GPU = "A100-40GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "transformers==5.8.1", "numpy")
    .add_local_dir(str(SPECFORGE_PKG), "/root/specforge")
    .add_local_file(str(CONFIG), "/root/qwen3-8b-dflash.json")
    .add_local_file(str(ROOT / "lib" / "lora.py"), "/root/lora.py")
)

# trained adapters land here; serving reads them via --adapter-glob '/adapters/*.pt'
adapters_vol = modal.Volume.from_name("lora-adapters", create_if_missing=True)


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


@app.function(gpu=GPU, image=image, timeout=3600, volumes={"/adapters": adapters_vol})
def train(domain: str = "python", steps: int = 200, rank: int = 16, alpha: int = 32,
          lr: float = 1e-3, save: bool = True):
    import sys
    import time

    import torch
    from transformers import AutoConfig

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_state_dict, lora_trainable_parameters

    DFlashDraftModel, OnlineDFlashModel = _load_dflash()
    dev = "cuda"
    torch.manual_seed(0)

    # --- real 1B DFlash draft (real config/shape), base frozen ---
    cfg = AutoConfig.from_pretrained("/root/qwen3-8b-dflash.json")
    cfg._attn_implementation = "sdpa"
    draft = DFlashDraftModel(cfg).to(device=dev, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in draft.parameters())

    replaced = inject_lora(
        draft, rank=rank, alpha=alpha,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    draft.to(device=dev, dtype=torch.bfloat16)  # move freshly-injected LoRA to GPU/bf16
    n_lora = sum(p.numel() for p in lora_trainable_parameters(draft))

    H, V = cfg.hidden_size, cfg.vocab_size
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

    # --- train ONLY the LoRA with the DFlash matching loss (synthetic target here) ---
    opt = torch.optim.AdamW(list(lora_trainable_parameters(draft)), lr=lr, weight_decay=0.0)
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
        "domain": domain, "draft_params": n_params, "lora_params": n_lora,
        "adapted_layers": len(replaced), "steps": steps,
        "loss0": losses[0], "lossN": losses[-1],
        "sec": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
    }

    # --- save the adapter (lora_state_dict: flat {layer: {"A","B","scaling"}}) ---
    if save:
        out = f"/adapters/{domain}.pt"
        torch.save(lora_state_dict(draft), out)
        adapters_vol.commit()
        log["saved"] = out
    return log


@app.local_entrypoint()
def main(domain: str = "python", steps: int = 200, rank: int = 16, alpha: int = 32,
         lr: float = 1e-3, save: bool = True):
    import json

    print(json.dumps(
        train.remote(domain=domain, steps=steps, rank=rank, alpha=alpha, lr=lr, save=save),
        indent=2,
    ))


# --------------------------------------------------------------------------- #
# Local CPU correctness check (no GPU / no Modal): proves the LoRA-on-drafter math
# before spending a GPU. Run:  python experiments/01-single-domain-dflash/train_lora.py
# --------------------------------------------------------------------------- #
def _local_check():
    import importlib.util
    import os
    import sys
    import types

    import torch

    HERE = os.path.dirname(os.path.abspath(__file__))
    SPECFORGE = os.environ.get("SPECFORGE_ROOT", str(ROOT / "third_party" / "SpecForge"))
    sys.path.insert(0, HERE)
    sys.path.insert(0, str(ROOT / "lib"))    # lora.py lives in lib/ now

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
    from lora import LoRALinear, inject_lora, lora_trainable_parameters
    from transformers import AutoConfig

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(os.path.join(SPECFORGE, "configs", "qwen3-8b-dflash.json"))
    cfg.hidden_size = 256; cfg.intermediate_size = 512; cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 8; cfg.num_key_value_heads = 4; cfg.head_dim = 32
    cfg.block_size = 8; cfg.num_target_layers = 2
    cfg.layer_types = ["full_attention", "full_attention"]
    cfg.dflash_config = {"mask_token_id": 511, "target_layer_ids": [0, 1]}
    cfg._attn_implementation = "sdpa"

    draft = DFlashDraftModel(cfg).to(torch.float32).eval()
    replaced = inject_lora(draft, rank=8, alpha=16,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    print(f"adapted layers: {len(replaced)}  e.g. {replaced[:2]}")

    # [A] unmerged == merged, [B] ΔW is rank-r
    layer = next(m for m in draft.modules() if isinstance(m, LoRALinear))
    with torch.no_grad():
        layer.B.normal_(0, 0.02)
    x = torch.randn(4, layer.base.in_features)
    err = (layer(x) - (layer.base(x) + x @ layer.delta_weight().T)).abs().max().item()
    rank = torch.linalg.matrix_rank(layer.delta_weight().detach()).item()
    print(f"[A] unmerged == merged: max|Δ|={err:.2e}  [B] rank(ΔW)={rank} (==r={layer.rank})")
    assert err < 1e-5 and rank == layer.rank

    # [C] DFlash matching loss trains ONLY LoRA; drafter base stays frozen
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
    opt = torch.optim.AdamW(list(lora_trainable_parameters(draft)), lr=2e-3)
    base_snap = {n: p.detach().clone() for n, p in draft.named_parameters() if ".base." in n}
    draft.train()
    for _ in range(20):
        loss, _ = online(input_ids=torch.randint(0, V - 1, (1, 128)),
                         hidden_states=torch.randn(1, 128, n_ctx), loss_mask=torch.ones(1, 128))
        opt.zero_grad(); loss.backward(); opt.step()
    base_changed = any(not torch.equal(base_snap[n], p)
                       for n, p in draft.named_parameters() if ".base." in n)
    print(f"[C] drafter base changed? {base_changed} (must be False)")
    assert not base_changed, "drafter base must stay frozen"
    print("LOCAL CHECK PASSED — LoRA-on-drafter math correct; base frozen")


if __name__ == "__main__":
    _local_check()
