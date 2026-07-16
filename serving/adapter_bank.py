#!/usr/bin/env python3
"""Multi-adapter serving for NaRA-on-DFlash-drafter — the *naive adapter-swap* path.

Why naive-swap (and not a Punica batched-gather kernel):

  * The DFlash drafter proposes a whole block in ONE denoising step, so the noise
    level λ = (block_size-1)/block_size is ~constant (~0.9375). NaRA's core C(λ)
    therefore never varies across the trajectory -> NaRA collapses to plain LoRA.
    (The training agent's own PLAN.md says exactly this.) So each adapter reduces
    to a single static low-rank delta ΔW_i = s · B_i · C_i(λ_fix) · A_i, and we
    fold C_i(λ_fix) into A once at load: A'_i = C_i(λ_fix) @ A_i. Serving is then
    ordinary LoRA: h = W0 x + s · (x A'_iᵀ) B_iᵀ.

  * spec_generate is batch=1 and its acceptance lengths are ragged per request, so
    there is nothing to gather-batch across adapters at the token level. The real
    efficiency win the user asked for is the SHARED BACKBONE: load the 8B target +
    1B draft ONCE, keep them resident, and for each request just point the draft's
    LoRA layers at that request's adapter (a cheap tensor-pointer swap, no reload).

This module provides:
  * MultiAdapterLoRALinear — a frozen nn.Linear carrying an in-memory bank of
    unmerged LoRA adapters, selected by `set_active(i)` (or None = base only).
  * inject_multi_adapter(draft, ...) — swap the draft's q/k/v/o_proj for the above.
  * AdapterBank — owns the draft + its MultiAdapterLoRALinear layers; loads
    nara_state_dict checkpoints (folding C(λ_fix)) and exposes activate(i).
  * make_toy_adapter(...) — synth a shape-correct adapter so the whole path runs
    before the real `nara_adapter.pt` from the training agent exists.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# nara.py (GaussianFourierEmbedding / NaRAHypernetwork / nara_state_dict format)
# lives in ../train locally, or flat alongside this file on Modal. Try both so we
# always read the SAME checkpoint format the trainer wrote.
for _cand in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
from nara import NaRAHypernetwork  # noqa: E402

# A vanilla DFlash block is 1 anchor + (block_size-1) masks. With block_size=16
# that fixes the serve-time noise level; C(λ_fix) is folded into A at load.
DEFAULT_BLOCK_SIZE = 16
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def fixed_lambda(block_size: int = DEFAULT_BLOCK_SIZE) -> float:
    return (block_size - 1) / block_size


class MultiAdapterLoRALinear(nn.Module):
    """A frozen nn.Linear plus an in-memory bank of unmerged LoRA deltas.

    forward(x) = W0 x + (base bias) + [active adapter: scaling · (x A'ᵀ) Bᵀ]

    Adapters are appended by `add_adapter(A_eff, B, scaling)` where A_eff already
    has C(λ_fix) folded in. `set_active(i)`/`set_active(None)` picks which delta is
    applied (None -> pure base). Swapping is O(1): just an integer index.
    """

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        # parallel lists, one entry per adapter; kept as buffers so .to(device) moves them
        self._A: List[torch.Tensor] = []   # each (r, in)
        self._B: List[torch.Tensor] = []   # each (out, r)
        self._scaling: List[float] = []
        self._active: Optional[int] = None

    def add_adapter(self, A_eff: torch.Tensor, B: torch.Tensor, scaling: float) -> int:
        assert A_eff.shape[1] == self.in_features, (
            f"A in-dim {A_eff.shape[1]} != layer in {self.in_features}"
        )
        assert B.shape[0] == self.out_features, (
            f"B out-dim {B.shape[0]} != layer out {self.out_features}"
        )
        assert A_eff.shape[0] == B.shape[1], "rank mismatch between A and B"
        idx = len(self._A)
        dev = self.base.weight.device
        dt = self.base.weight.dtype
        # register as buffers so they follow the module to GPU and are not trained
        self.register_buffer(f"_A_{idx}", A_eff.to(device=dev, dtype=dt), persistent=False)
        self.register_buffer(f"_B_{idx}", B.to(device=dev, dtype=dt), persistent=False)
        self._A.append(getattr(self, f"_A_{idx}"))
        self._B.append(getattr(self, f"_B_{idx}"))
        self._scaling.append(float(scaling))
        return idx

    def set_active(self, idx: Optional[int]) -> None:
        if idx is not None and not (0 <= idx < len(self._A)):
            raise IndexError(f"adapter {idx} out of range (have {len(self._A)})")
        self._active = idx

    @property
    def num_adapters(self) -> int:
        return len(self._A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._active is None:
            return out
        A = self._A[self._active]          # (r, in)
        B = self._B[self._active]          # (out, r)
        s = self._scaling[self._active]
        lora = F.linear(F.linear(x, A), B)  # (x A'ᵀ) Bᵀ  -> (..., out)
        return out + s * lora


def inject_multi_adapter(
    draft: nn.Module, target_modules=TARGET_MODULES
) -> Dict[str, MultiAdapterLoRALinear]:
    """Replace the draft's target q/k/v/o Linear layers with MultiAdapterLoRALinear.

    Returns {layer_name: module} in a stable order for loading adapter weights.
    """
    for p in draft.parameters():
        p.requires_grad_(False)
    replaced: Dict[str, MultiAdapterLoRALinear] = {}
    for name, module in list(draft.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and (
                child_name in target_modules or any(full.endswith(t) for t in target_modules)
            ):
                wrapped = MultiAdapterLoRALinear(child)
                setattr(module, child_name, wrapped)
                replaced[full] = wrapped
    if not replaced:
        raise ValueError(
            f"No layers matched {target_modules}. named_modules e.g.: "
            f"{[n for n,_ in list(draft.named_modules())[:8]]}"
        )
    return replaced


def _fold_core(A: torch.Tensor, hypernetwork_sd: dict, rank: int, lam: float) -> torch.Tensor:
    """A'  =  C(λ_fix) @ A , where C(λ) = I + η·F_φ(e_λ) from the saved hypernet.

    If the checkpoint has no usable hypernetwork (or C==I), A' == A and serving is
    pure LoRA. Either way the result is a single static low-rank factor.
    """
    if not hypernetwork_sd:
        return A
    # rebuild the hypernetwork to evaluate C(λ_fix). fourier/hidden dims are read
    # from the saved tensor shapes so we match whatever the trainer used.
    try:
        w0 = hypernetwork_sd["net.0.weight"]        # (hidden, fourier)
        fourier_dim = w0.shape[1]
        hidden_dim = w0.shape[0]
        hyper = NaRAHypernetwork(rank=rank, fourier_dim=fourier_dim, hidden_dim=hidden_dim)
        hyper.load_state_dict(hypernetwork_sd)
        hyper.eval()
        with torch.no_grad():
            C = hyper(torch.tensor([lam], dtype=A.dtype))  # (1, r, r) or (r, r)
        C = C.reshape(rank, rank).to(A.dtype)
        return C @ A
    except Exception as e:  # noqa: BLE001 — be robust; fall back to plain LoRA
        print(f"[adapter_bank] C(λ) fold skipped ({type(e).__name__}: {e}); using plain LoRA A.")
        return A


def _load_state_file(path: str) -> dict:
    """Load a .pt (torch) or .safetensors state dict to CPU tensors."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def _peft_sd_to_layers(sd: dict) -> Dict[str, dict]:
    """Map a PEFT LoRA state_dict -> {layer_name: {"A": (r,in), "B": (out,r)}}.

    PEFT keys look like  base_model.model.<path>.lora_A.weight  (r, in)
                         base_model.model.<path>.lora_B.weight  (out, r)
    We strip the base_model.model. prefix and the .lora_{A,B}.weight suffix so the
    remaining <path> (e.g. layers.0.self_attn.q_proj) matches the bank's layer keys.
    """
    layers: Dict[str, dict] = {}
    for k, v in sd.items():
        if "lora_A" not in k and "lora_B" not in k:
            continue
        which = "A" if "lora_A" in k else "B"
        name = k.split(".lora_")[0]
        for pre in ("base_model.model.", "base_model.", "model."):
            if name.startswith(pre):
                name = name[len(pre):]
        layers.setdefault(name, {})[which] = v.float()
    # keep only complete A+B pairs
    return {n: ab for n, ab in layers.items() if "A" in ab and "B" in ab}


def load_lora_checkpoint(path: str):
    """Format-agnostic loader. Returns (ckpt, alpha) where ckpt is in the internal
    nara_state_dict shape {"hypernetwork": {}, "layers": {name: {"A":.., "B":..}}}.

    Handles: (1) NaRA nara_state_dict .pt, (2) a PEFT LoRA directory
    (adapter_config.json + adapter_model.safetensors), (3) a raw PEFT state_dict
    file with *.lora_A/lora_B keys. alpha is read from adapter_config.json when
    present (PEFT scaling = alpha/r); None -> scaling defaults to 1.
    """
    import json

    # (2) PEFT directory
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "adapter_config.json")
        alpha = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                alpha = json.load(f).get("lora_alpha")
        weights = None
        for fn in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
            if os.path.exists(os.path.join(path, fn)):
                weights = os.path.join(path, fn)
                break
        if weights is None:
            raise FileNotFoundError(f"no adapter_model.* in {path}")
        return {"hypernetwork": {}, "layers": _peft_sd_to_layers(_load_state_file(weights))}, alpha

    sd = _load_state_file(path)
    # (1) NaRA nara_state_dict: {"hypernetwork":.., "layers": {name: {A,B}}}
    if isinstance(sd, dict) and "layers" in sd and isinstance(sd["layers"], dict):
        return sd, None
    # (1b) python-lora-drafter lora_state_dict: FLAT {name: {"A","B","scaling"}}
    if isinstance(sd, dict) and sd and all(
        isinstance(v, dict) and "A" in v and "B" in v for v in sd.values()
    ):
        return {"hypernetwork": {}, "layers": sd}, None   # per-layer scaling kept in each entry
    # (3) raw PEFT state_dict
    layers = _peft_sd_to_layers(sd)
    if layers:
        # look for a sibling adapter_config.json for alpha
        alpha = None
        cfg_path = os.path.join(os.path.dirname(path), "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                alpha = json.load(f).get("lora_alpha")
        return {"hypernetwork": {}, "layers": layers}, alpha
    raise ValueError(f"Unrecognized adapter format at {path}: keys e.g. {list(sd)[:5]}")


class AdapterBank:
    """Owns the draft's MultiAdapterLoRALinear layers and the loaded adapters.

    activate(i) points every layer at adapter i; activate(None) -> base drafter.
    """

    def __init__(self, draft: nn.Module, block_size: int = DEFAULT_BLOCK_SIZE,
                 target_modules=TARGET_MODULES):
        self.draft = draft
        self.block_size = block_size
        self.lam = fixed_lambda(block_size)
        self.layers = inject_multi_adapter(draft, target_modules)
        self.names: List[str] = []        # adapter display names, index-aligned

    def load_adapter(self, ckpt: dict, name: str, alpha: Optional[float] = None) -> int:
        """Load one nara_state_dict-format checkpoint into every layer as adapter k.

        ckpt = {"hypernetwork": <sd or {}>, "layers": {layer_name: {"A":.., "B":..}}}
        Missing layers get a zero (no-op) delta so indexing stays consistent.
        """
        hyper_sd = ckpt.get("hypernetwork", {}) or {}
        layer_ckpts = ckpt["layers"]
        idx = None
        for lname, layer in self.layers.items():
            entry = layer_ckpts.get(lname)
            if entry is None:
                # some checkpoints key by the module path without a prefix; try suffix match
                entry = next((v for k, v in layer_ckpts.items() if lname.endswith(k) or k.endswith(lname)), None)
            if entry is None:
                r = 1
                A = torch.zeros(r, layer.in_features)
                B = torch.zeros(layer.out_features, r)
                scaling = 1.0
            else:
                A = entry["A"].float()
                B = entry["B"].float()
                r = A.shape[0]
                A = _fold_core(A, hyper_sd, r, self.lam)
                # prefer the checkpoint's own per-layer scaling (python-lora-drafter
                # lora_state_dict stores it); else alpha/r; else 1.0.
                if "scaling" in entry:
                    scaling = float(entry["scaling"])
                elif alpha is not None:
                    scaling = alpha / r
                else:
                    scaling = 1.0
            k = layer.add_adapter(A, B, scaling)
            idx = k if idx is None else idx
            assert k == idx, "adapters loaded out of sync across layers"
        self.names.append(name)
        return idx

    def load_adapter_path(self, path: str, name: Optional[str] = None) -> int:
        """Load one adapter from a file/dir of any supported format (NaRA or PEFT)."""
        ckpt, alpha = load_lora_checkpoint(path)
        return self.load_adapter(ckpt, name=name or os.path.basename(path.rstrip("/")), alpha=alpha)

    def activate(self, idx: Optional[int]) -> None:
        for layer in self.layers.values():
            layer.set_active(idx)

    @property
    def num_adapters(self) -> int:
        return len(self.names)


def make_toy_adapter(bank_layers: Dict[str, MultiAdapterLoRALinear], rank: int = 32,
                     seed: int = 0, strength: float = 0.02) -> dict:
    """Build a shape-correct nara_state_dict for the given layers, for dry runs.

    B is small-random (not zero) so the adapter actually changes the output, and a
    lightly-perturbed hypernetwork so C(λ_fix) != I (exercises the fold path). Its
    layout is byte-compatible with train/nara.py's nara_state_dict, so the real
    checkpoint is a drop-in replacement.
    """
    g = torch.Generator().manual_seed(seed)
    hyper = NaRAHypernetwork(rank=rank, fourier_dim=128, hidden_dim=256)
    with torch.no_grad():
        # perturb final layer so C(λ) drifts off identity (emulates a trained φ)
        hyper.net[-1].weight.normal_(0, 0.3, generator=g)
    layers = {}
    for lname, layer in bank_layers.items():
        A = torch.empty(rank, layer.in_features)
        nn.init.kaiming_uniform_(A, a=5 ** 0.5)
        B = torch.empty(layer.out_features, rank)
        B.normal_(0, strength, generator=g)
        layers[lname] = {"A": A, "B": B}
    return {"hypernetwork": hyper.state_dict(), "layers": layers}
