#!/usr/bin/env python3
"""CPU dry-run: verify the multi-adapter swap machinery on the REAL DFlashDraftModel.

We can't run the full spec_generate locally (it needs the real 8B target CausalLM),
so this exercises the novel part — the shared-backbone adapter bank — against the
actual DFlashDraftModel class (shrunk config), and checks:

  [1] injection swaps the draft's q/k/v/o into MultiAdapterLoRALinear
  [2] N adapters load into one shared draft (backbone weights untouched / shared)
  [3] activate(i) changes the draft's forward output, and each adapter gives a
      DISTINCT output (adapters are actually being swapped, not cached)
  [4] activate(None) reproduces the base drafter bit-for-bit
  [5] swapping is stateless: re-activating i returns i's exact output again
  [6] the base weights are byte-identical across all adapters (truly shared)

Run:  ../.venv/bin/python dry_run_serving.py
"""
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SPECFORGE = os.environ.get("SPECFORGE_ROOT", os.path.join(ROOT, "SpecForge"))
sys.path.insert(0, HERE)


def _load_dflash_class():
    """Import DFlashDraftModel without running specforge/__init__ (needs triton)."""
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
    ]:
        stub(n, p)
    spec = importlib.util.spec_from_file_location(
        "specforge.modeling.draft.dflash",
        os.path.join(SPECFORGE, "specforge/modeling/draft/dflash.py"),
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["specforge.modeling.draft.dflash"] = m
    spec.loader.exec_module(m)
    return m.DFlashDraftModel


def small_draft_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.path.join(SPECFORGE, "configs", "qwen3-8b-dflash.json"))
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 4
    cfg.head_dim = 32
    cfg.block_size = 16
    cfg.num_target_layers = 2
    cfg.layer_types = ["full_attention", "full_attention"]
    cfg.dflash_config = {"mask_token_id": 511, "target_layer_ids": [0, 1]}
    cfg._attn_implementation = "sdpa"
    return cfg


def draft_forward(draft, cfg, seed=0):
    """A single deterministic forward through the draft, returning the hidden output."""
    torch.manual_seed(seed)
    B, L = 1, cfg.block_size
    ctx_len = 4
    H = cfg.hidden_size
    noise_embedding = torch.randn(B, L, H)
    target_hidden = torch.randn(B, ctx_len, len(cfg.dflash_config["target_layer_ids"]) * H)
    # rotary spans context + noise (k = cat[k_ctx, k_noise]); positions cover both
    position_ids = torch.arange(ctx_len + L).unsqueeze(0)
    with torch.inference_mode():
        return draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            use_cache=False,
            is_causal=False,
        )


def main():
    from adapter_bank import AdapterBank, make_toy_adapter, MultiAdapterLoRALinear

    torch.manual_seed(0)
    DFlashDraftModel = _load_dflash_class()
    cfg = small_draft_config()
    draft = DFlashDraftModel(cfg).to(torch.float32).eval()
    n_base = sum(p.numel() for p in draft.parameters())

    bank = AdapterBank(draft, block_size=cfg.block_size)
    n_layers = len(bank.layers)
    print(f"draft base params : {n_base:,}")
    print(f"[1] injected MultiAdapterLoRALinear into {n_layers} layers: "
          f"{list(bank.layers)[:2]} ...")
    assert n_layers == cfg.num_hidden_layers * 4, "expected q/k/v/o per layer"
    assert all(isinstance(m, MultiAdapterLoRALinear) for m in bank.layers.values())

    # snapshot base weights to prove they stay shared/untouched
    base_snap = {n: p.detach().clone() for n, p in draft.named_parameters() if ".base." in n}

    N = 5
    for i in range(N):
        ckpt = make_toy_adapter(bank.layers, rank=32, seed=100 + i)
        k = bank.load_adapter(ckpt, name=f"toy-{i}")
        assert k == i
    print(f"[2] loaded {bank.num_adapters} adapters into ONE shared draft")

    # base-only reference
    bank.activate(None)
    base_out = draft_forward(draft, cfg)

    outs = []
    for i in range(N):
        bank.activate(i)
        outs.append(draft_forward(draft, cfg))

    # [3] each adapter changes output and adapters differ from each other
    diffs_vs_base = [(o - base_out).abs().max().item() for o in outs]
    print(f"[3] max|adapter_i - base| = {[f'{d:.3e}' for d in diffs_vs_base]}")
    assert all(d > 1e-6 for d in diffs_vs_base), "some adapter was a no-op"
    for i in range(N):
        for j in range(i + 1, N):
            assert (outs[i] - outs[j]).abs().max().item() > 1e-6, f"adapters {i},{j} identical"
    print("    all 5 adapters produce DISTINCT outputs")

    # [4] activate(None) == base exactly
    bank.activate(None)
    assert torch.equal(draft_forward(draft, cfg), base_out)
    print("[4] activate(None) reproduces the base drafter exactly")

    # [5] swapping is stateless — re-activate adapter 2, get identical output
    bank.activate(2)
    assert torch.equal(draft_forward(draft, cfg), outs[2])
    print("[5] re-activating an adapter reproduces its output (stateless swap)")

    # [6] base weights byte-identical after all the swapping (truly shared backbone)
    base_changed = any(not torch.equal(base_snap[n], p)
                       for n, p in draft.named_parameters() if ".base." in n)
    assert not base_changed, "shared backbone weights changed!"
    print("[6] shared backbone weights byte-identical across all adapters")

    print("\n" + "=" * 68)
    print("SERVING DRY RUN PASSED — multi-adapter swap works on real DFlashDraftModel")
    print(f"  1 shared 1B-style backbone  +  {N} unmerged LoRA adapters")
    print("  swap = O(1) index change; base frozen & shared; outputs distinct")
    print("  (full spec_generate decode runs on Modal with the real 8B target)")
    print("=" * 68)


if __name__ == "__main__":
    main()
