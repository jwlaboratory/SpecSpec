#!/usr/bin/env python3
"""Plain (unmerged) LoRA for a DFlash drafter.

We dropped NaRA on purpose: a DFlash block drafts in a single step at a fixed
masking pattern (1 anchor + block_size-1 masks), so the noise level λ is ~constant
and a noise-aware core C(λ) would never move — it collapses to ordinary LoRA.
So we use ordinary LoRA:

    h = W0 x + (α/r) · B · A · x            ΔW = (α/r) · B · A     (rank r)

Only {A, B} train; the base drafter (and the Qwen3-8B target) stay frozen. The
adapter is kept UNMERGED so you can run the base alone, or add ΔW on top:
    (W0 + ΔW) x  ==  W0 x + (α/r) B A x      (verified in demo.py [A])

Init: A ~ Kaiming, B = 0  ⟹  ΔW = 0 at start (training begins as a no-op delta).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank update h = base(x) + s·B·A·x."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)  # W0 frozen

        self.rank = rank
        alpha = rank if alpha is None else alpha
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.empty(rank, in_f))  # (r, in)
        self.B = nn.Parameter(torch.zeros(out_f, rank))  # (out, r), zero init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.drop(x), self.A), self.B)  # B (A x)
        return self.base(x) + self.scaling * update

    def delta_weight(self) -> torch.Tensor:
        """ΔW = s·B·A  (the standalone low-rank delta you add on top of W0)."""
        return self.scaling * (self.B @ self.A)


DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_modules: Tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> List[str]:
    """Freeze `model` and replace target nn.Linear layers with LoRALinear.
    Returns the list of replaced layer names. Trainable params afterwards are
    exactly the per-layer {A, B}."""
    for p in model.parameters():
        p.requires_grad_(False)

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and (
                child_name in target_modules
                or any(full.endswith(t) for t in target_modules)
            ):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                replaced.append(full)
    if not replaced:
        raise ValueError(
            f"No layers matched target_modules={target_modules}. "
            "Inspect model.named_modules() for the right suffixes."
        )
    return replaced


def lora_trainable_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m.A
            yield m.B


def lora_state_dict(model: nn.Module) -> dict:
    """Compact adapter checkpoint (no base weights) — unmerged."""
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[name] = {
                "A": m.A.detach().cpu(),
                "B": m.B.detach().cpu(),
                "scaling": m.scaling,
            }
    return sd
