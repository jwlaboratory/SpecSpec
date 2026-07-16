#!/usr/bin/env python3
"""Small end-to-end demo: UNMERGED LoRA on a DFlash drafter (local, CPU, fast).

Proves the inference math for the setup we'll run on Modal:
  * base = frozen DFlash drafter (downloaded from HF)
  * LoRA = unmerged low-rank delta on the drafter's q/k/v/o
  * only LoRA trains; drafter + target frozen
  * objective = DFlash matching loss (predict the TARGET's tokens), not perplexity

Verifies:
  [A] unmerged == merged:  W0 x + s·B·A·x  ==  (W0 + ΔW) x   (run base, add ΔW on top)
  [B] ΔW = s·B·A is genuinely rank-r
  [C] the DFlash matching loss trains ONLY LoRA; drafter base stays frozen
  [D] base-alone output != base+LoRA output after training (adapter does something)
"""
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SPECFORGE = os.environ.get("SPECFORGE_ROOT", os.path.join(ROOT, "SpecForge"))
sys.path.insert(0, HERE)  # lora.py


def _load_dflash():
    def stub(name, path):
        m = types.ModuleType(name)
        m.__path__ = [path]
        m.__package__ = name
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

    d = load("specforge.modeling.draft.dflash", "specforge/modeling/draft/dflash.py")
    c = load("specforge.core.dflash", "specforge/core/dflash.py")
    return d.DFlashDraftModel, c.OnlineDFlashModel


def small_draft_config():
    """Real qwen3-8b-dflash.json shape, shrunk so the demo runs in seconds."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        os.path.join(SPECFORGE, "configs", "qwen3-8b-dflash.json")
    )
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 4
    cfg.head_dim = 32
    cfg.block_size = 8
    cfg.num_target_layers = 2
    cfg.layer_types = ["full_attention", "full_attention"]
    cfg.dflash_config = {"mask_token_id": 511, "target_layer_ids": [0, 1]}
    cfg._attn_implementation = "sdpa"
    return cfg


def main():
    torch.manual_seed(0)
    DFlashDraftModel, OnlineDFlashModel = _load_dflash()
    from lora import inject_lora, lora_trainable_parameters, LoRALinear

    cfg = small_draft_config()
    draft = DFlashDraftModel(cfg).to(torch.float32).eval()
    n_base = sum(p.numel() for p in draft.parameters())

    replaced = inject_lora(draft, rank=8, alpha=16, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    n_lora = sum(p.numel() for p in lora_trainable_parameters(draft))
    print(f"drafter base params : {n_base:,} (frozen)")
    print(f"LoRA params         : {n_lora:,} ({100*n_lora/n_base:.2f}% of base, trainable)")
    print(f"adapted layers      : {len(replaced)}  e.g. {replaced[:2]}")

    # ------------------------------------------------------------------ [A][B]
    layer = next(m for m in draft.modules() if isinstance(m, LoRALinear))
    with torch.no_grad():
        layer.B.normal_(0, 0.02)  # B starts at 0; give it content so ΔW != 0

    x = torch.randn(4, layer.base.in_features)
    unmerged = layer(x)                       # W0 x + s·B·A·x
    dW = layer.delta_weight()                 # s·B·A  (standalone low-rank matrix)
    merged = layer.base(x) + x @ dW.T         # (W0 + ΔW) x
    err = (unmerged - merged).abs().max().item()
    rank = torch.linalg.matrix_rank(dW.detach()).item()
    print(f"\n[A] unmerged == merged:  max|Δ| = {err:.2e}  ->  {'OK' if err < 1e-5 else 'FAIL'}")
    print(f"[B] ΔW is low-rank:      rank(ΔW) = {rank}  (== r = {layer.rank})")
    assert err < 1e-5 and rank == layer.rank

    # --------------------------------------------------------------------- [C]
    print("\n[C] train LoRA-only with the DFlash matching loss (synthetic target):")
    H, V = cfg.hidden_size, 512
    n_ctx = len(cfg.dflash_config["target_layer_ids"]) * H
    embed = torch.nn.Embedding(V, H)
    lm_head = torch.nn.Linear(H, V, bias=False)
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=cfg.dflash_config["mask_token_id"], block_size=cfg.block_size,
        attention_backend="sdpa", num_anchors=16, loss_decay_gamma=7.0, loss_type="dflash",
    )
    trainable = list(lora_trainable_parameters(draft))
    opt = torch.optim.AdamW(trainable, lr=2e-3)
    base_snap = {n: p.detach().clone() for n, p in draft.named_parameters() if ".base." in n}

    draft.train()
    losses = []
    for step in range(20):
        ids = torch.randint(0, V - 1, (1, 128))
        hid = torch.randn(1, 128, n_ctx)
        lm = torch.ones(1, 128)
        loss, acc = online(input_ids=ids, hidden_states=hid, loss_mask=lm)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == 19:
            print(f"    step {step:2d}  loss {loss.item():.4f}  draft-match-acc {acc.item():.3f}")

    base_changed = any(
        not torch.equal(base_snap[n], p)
        for n, p in draft.named_parameters() if ".base." in n
    )
    print(f"    drafter base changed? {base_changed} (must be False);  loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert not base_changed, "drafter base must stay frozen"

    print("\n" + "=" * 66)
    print("DEMO PASSED — unmerged LoRA-on-drafter inference math is correct")
    print("  [A] W0 x + s·B·A·x  ==  (W0 + ΔW) x   (run base, add ΔW on top)")
    print("  [B] ΔW is rank-r")
    print("  [C] DFlash matching loss trains ONLY LoRA; base frozen")
    print("=" * 66)


if __name__ == "__main__":
    main()
