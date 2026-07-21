#!/usr/bin/env python3
"""MoLE — mixture of latent LoRA experts for a DFlash drafter.

K independent LoRA experts per wrapped linear (q/k/v/o), mixed by a single
model-level gate that is conditioned on the SAME latent the drafter already
uses: `extract_context_feature(target_hidden)` (20480-dim for Qwen3-8B),
mean-pooled over the prompt. No domain labels anywhere — the gate + experts
are trained end-to-end through the DFlash loss, so any specialization that
emerges is carved by the latent space itself.

    h = W0 x + s · Σ_k g_k · B_k (A_k x)          g = softmax(MLP(pooled_feat))

g is per-SEQUENCE (computed once from the prompt), so the effective adapter is
a data-dependent LoRA of rank ≤ K·r and the per-token FLOP cost equals a single
rank-(K·r) LoRA. Serving needs one extra tiny MLP forward at prefill, exactly
like router/router.py.

Init: A_k ~ Kaiming (independent per expert), B_k = 0 ⟹ every expert starts as
a no-op; symmetry breaks because each B_k's gradient goes through its own A_k.
The gate's output layer is zero-init ⟹ mixing starts uniform (1/K each).

Collapse control: training should add the switch-style importance loss
`load_balance_loss(g)` (=1 at a uniform marginal, >1 otherwise) with a small
coefficient so the gate cannot dump everything on one expert.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class GateBox:
    """Shared mutable holder for the per-sequence mixture weights.

    Every MoLELinear keeps a reference to the same box; setting `box.g` once
    (B, K) routes the whole drafter. Kept outside nn.Module so it never lands
    in state_dicts."""

    def __init__(self):
        self.g: Optional[torch.Tensor] = None


class MoLELinear(nn.Module):
    """Frozen nn.Linear + K LoRA experts mixed by the gate box's weights."""

    def __init__(self, base_linear: nn.Linear, gate_box: GateBox,
                 num_experts: int = 8, rank: int = 8, alpha: Optional[float] = None):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.gate_box = gate_box
        self.num_experts = num_experts
        self.rank = rank
        alpha = rank if alpha is None else alpha
        self.scaling = alpha / rank

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.empty(num_experts, rank, in_f))   # (K, r, in)
        self.B = nn.Parameter(torch.zeros(num_experts, out_f, rank))  # (K, out, r)
        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_box.g
        if g is None:
            raise RuntimeError("MoLELinear: gate_box.g is unset — compute the "
                               "gate from the prompt feature before the forward.")
        assert x.dim() == 3 and x.shape[0] == g.shape[0], \
            f"MoLE expects (B,T,d) with B == gate batch; got {tuple(x.shape)} vs g {tuple(g.shape)}"
        # low-rank space mixing: cost == one rank-(K*r) LoRA
        xa = torch.einsum("bti,kri->btkr", x, self.A.to(x.dtype))
        xa = xa * g.to(x.dtype).view(g.shape[0], 1, g.shape[1], 1)
        update = torch.einsum("btkr,kor->bto", xa, self.B.to(x.dtype))
        return self.base(x) + self.scaling * update


class LatentGate(nn.Module):
    """pooled context feature (B, feat_dim) -> mixture weights (B, K), fp32."""

    def __init__(self, feat_dim: int = 20480, hidden: int = 64, num_experts: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)
        nn.init.zeros_(self.fc2.weight)   # start at the uniform mixture
        nn.init.zeros_(self.fc2.bias)
        self.temperature = temperature

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.gelu(self.fc1(self.norm(pooled.float()))))
        return torch.softmax(h / self.temperature, dim=-1)


def pool_prompt_feature(hid: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
    """hid: (B, S, feat_dim) context feature; prompt_mask: (B, S) 1 = prompt token.
    Returns (B, feat_dim) fp32 mean over prompt positions."""
    m = prompt_mask.to(hid.device).unsqueeze(-1).float()
    return (hid.float() * m).sum(1) / m.sum(1).clamp_min(1.0)


def load_balance_loss(g: torch.Tensor) -> torch.Tensor:
    """Switch-style importance loss on the batch marginal: K·Σ_k mean_b(g)².
    Equals 1.0 when the marginal is uniform; grows as the gate collapses."""
    K = g.shape[-1]
    imp = g.mean(dim=0)
    return K * (imp ** 2).sum()


def gate_entropy(g: torch.Tensor) -> torch.Tensor:
    """Mean per-sample entropy in nats (0 = hard routing, ln K = uniform)."""
    return -(g * (g + 1e-9).log()).sum(-1).mean()


def inject_mole(model: nn.Module, num_experts: int = 8, rank: int = 8,
                alpha: Optional[float] = None,
                target_modules: Tuple[str, ...] = DEFAULT_TARGET_MODULES,
                ) -> Tuple[List[str], GateBox]:
    """Freeze `model`, replace target linears with MoLELinear sharing one GateBox."""
    for p in model.parameters():
        p.requires_grad_(False)

    box = GateBox()
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and (
                child_name in target_modules
                or any(full.endswith(t) for t in target_modules)
            ):
                setattr(module, child_name,
                        MoLELinear(child, box, num_experts=num_experts,
                                   rank=rank, alpha=alpha))
                replaced.append(full)
    if not replaced:
        raise ValueError(f"No layers matched target_modules={target_modules}.")
    return replaced, box


def mole_trainable_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, MoLELinear):
            yield m.A
            yield m.B


def mole_expert_state_dict(model: nn.Module) -> dict:
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, MoLELinear):
            sd[name] = {"A": m.A.detach().cpu(), "B": m.B.detach().cpu(),
                        "scaling": m.scaling}
    return sd


def load_mole_expert_state(model: nn.Module, sd: dict) -> None:
    named = dict(model.named_modules())
    for name, entry in sd.items():
        m = named.get(name)
        assert isinstance(m, MoLELinear), f"missing MoLELinear at {name}"
        with torch.no_grad():
            m.A.copy_(entry["A"].to(m.A.device, m.A.dtype))
            m.B.copy_(entry["B"].to(m.B.device, m.B.dtype))
        m.scaling = float(entry["scaling"])
