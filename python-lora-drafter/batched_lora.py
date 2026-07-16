#!/usr/bin/env python3
"""Multi-adapter LoRA with hot-swap AND per-sequence batched routing (S-LoRA style).

One frozen base, N unmerged LoRA adapters living side by side on each q/k/v/o.
Three serving modes, selected on a shared controller:

  controller.use_base()          -> base only (no adapter)
  controller.use_adapter("sql")  -> one adapter for the whole batch  (hot-swap)
  controller.route(ids)          -> DIFFERENT adapter per sequence in ONE batch

Routing math (the interesting bit): for a batch x (B, S, in) with per-sequence
adapter ids (B,), we gather each sequence's own (A_i, B_i) and apply them in a
single batched matmul:

    a = einsum('bsi,bri->bsr', x, A[ids])      # per-seq down-proj
    u = einsum('bsr,bor->bso', a, B[ids])      # per-seq up-proj
    h = base(x) + scaling * u                  # ids < 0 => base only

So a batch mixing Python / SQL / prose queries each drafts under its own adapter
without splitting the batch. Every adapter stays UNMERGED (ΔW_i = s·B_i·A_i).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLoRAController(nn.Module):
    def __init__(self, adapter_names: List[str]):
        super().__init__()
        self.adapter_names = list(adapter_names)
        self.index = {n: i for i, n in enumerate(self.adapter_names)}
        self.n = len(adapter_names)
        self.mode = "base"          # "base" | "single" | "route"
        self.active: Optional[int] = None
        self.routing: Optional[torch.Tensor] = None

    def use_base(self):
        self.mode, self.active, self.routing = "base", None, None

    def use_adapter(self, which: Union[int, str]):
        self.active = self.index[which] if isinstance(which, str) else which
        self.mode, self.routing = "single", None

    def route(self, ids):
        """ids: LongTensor (batch,); entry in [0,n) picks an adapter, <0 = base."""
        if not torch.is_tensor(ids):
            ids = torch.tensor(
                [self.index[i] if isinstance(i, str) else i for i in ids],
                dtype=torch.long,
            )
        self.routing, self.mode, self.active = ids.long(), "route", None


class BatchedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, controller: MultiLoRAController,
                 rank: int = 16, alpha: Optional[float] = None):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.controller = controller
        self.rank = rank
        self.scaling = (rank if alpha is None else alpha) / rank
        in_f, out_f = base_linear.in_features, base_linear.out_features
        # one (A, B) per adapter; kept as ParameterLists so a single adapter can
        # be trained in isolation, stacked on the fly for batched routing.
        self.A = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, in_f)) for _ in range(controller.n)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_f, rank)) for _ in range(controller.n)]
        )
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        c = self.controller
        if c.mode == "base":
            return out
        if c.mode == "single":
            u = F.linear(F.linear(x, self.A[c.active]), self.B[c.active])
            return out + self.scaling * u
        # route: per-sequence adapter selection
        ids = c.routing.to(x.device)
        A_stack = torch.stack(list(self.A))          # (n, r, in)
        B_stack = torch.stack(list(self.B))          # (n, out, r)
        safe = ids.clamp_min(0)
        A_sel, B_sel = A_stack[safe], B_stack[safe]  # (B, r, in), (B, out, r)
        a = torch.einsum("bsi,bri->bsr", x, A_sel)
        u = torch.einsum("bsr,bor->bso", a, B_sel)
        u = torch.where((ids < 0).view(-1, 1, 1), torch.zeros_like(u), u)
        return out + self.scaling * u

    def delta_weight(self, adapter: int) -> torch.Tensor:
        return self.scaling * (self.B[adapter] @ self.A[adapter])


DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def inject_batched_lora(
    model: nn.Module,
    adapter_names: List[str],
    rank: int = 16,
    alpha: Optional[float] = None,
    target_modules: Tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> Tuple[MultiLoRAController, List[str]]:
    for p in model.parameters():
        p.requires_grad_(False)
    controller = MultiLoRAController(adapter_names)
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and (
                child_name in target_modules or any(full.endswith(t) for t in target_modules)
            ):
                setattr(module, child_name,
                        BatchedLoRALinear(child, controller, rank=rank, alpha=alpha))
                replaced.append(full)
    if not replaced:
        raise ValueError(f"No layers matched {target_modules}")
    return controller, replaced


def adapter_parameters(model: nn.Module, adapter: int):
    """Parameters of a SINGLE adapter across all layers (for isolated training)."""
    for m in model.modules():
        if isinstance(m, BatchedLoRALinear):
            yield m.A[adapter]
            yield m.B[adapter]


def save_adapters(model: nn.Module, controller: MultiLoRAController) -> dict:
    out = {"adapter_names": controller.adapter_names, "layers": {}}
    for name, m in model.named_modules():
        if isinstance(m, BatchedLoRALinear):
            out["layers"][name] = {
                "A": [a.detach().cpu() for a in m.A],
                "B": [b.detach().cpu() for b in m.B],
                "scaling": m.scaling,
            }
    return out
