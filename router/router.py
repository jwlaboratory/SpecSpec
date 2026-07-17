#!/usr/bin/env python3
"""AdapterRouter — pick which LoRA to route a request to, from hidden states.

Inference-side counterpart of pipeline_router.py. Torch-only; no Modal.

The router consumes the SAME feature the DFlash drafter already conditions on —
`extract_context_feature(hidden_states, target_layer_ids)`, i.e. the target's
prefill hidden states at layers [1,9,17,25,33] concatenated to 20480-dim — so in
a spec-decode server routing adds one tiny MLP forward and nothing else.

Usage in serving (e.g. ../serving/serve_nara.py or batched_lora routing):

    from router import AdapterRouter
    router = AdapterRouter.load("router_mlp.pt")

    # target prefill already produced `target_hidden` (B, S, 20480) + attn mask:
    names, probs = router.route_hidden(target_hidden, attention_mask)
    # -> names like ["korean", "other", ...]; map to adapter ids for
    #    MultiLoRAController.route(ids) with "other" -> -1 (base drafter).

    ids = [controller.index[n] if n != "other" else -1 for n in names]

Standalone (no spec-decode context) — route raw prompts with the target model:

    names, probs = router.route_prompts(prompts, target, tok)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AdapterRouter:
    def __init__(self, mlp: nn.Module, classes: List[str], layer_ids: List[int],
                 mu: torch.Tensor, sd: torch.Tensor, pool: str = "mean"):
        self.mlp = mlp.eval()
        self.classes = list(classes)
        self.layer_ids = list(layer_ids)
        self.mu = mu
        self.sd = sd
        self.pool = pool

    # ---- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AdapterRouter":
        ckpt = torch.load(path, map_location=device)
        mlp = nn.Sequential(
            nn.Linear(ckpt["dim"], ckpt["hidden"]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(ckpt["hidden"], len(ckpt["classes"])),
        )
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.to(device).eval()
        return cls(mlp, ckpt["classes"], ckpt["layer_ids"],
                   ckpt["mu"].to(device), ckpt["sd"].to(device), ckpt.get("pool", "mean"))

    # ---- core: route from the drafter's own conditioning feature ------------
    @torch.inference_mode()
    def route_hidden(self, target_hidden: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     ) -> Tuple[List[str], torch.Tensor]:
        """target_hidden: (B, S, 20480) — extract_context_feature output over the
        prompt. attention_mask: (B, S) 1=real token. Returns (names, probs (B, C))."""
        x = target_hidden.float()
        if attention_mask is not None:
            m = attention_mask.to(x.device).unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp_min(1)
        else:
            pooled = x.mean(1)
        pooled = (pooled - self.mu.to(pooled.device)) / self.sd.to(pooled.device)
        logits = self.mlp.to(pooled.device)(pooled)
        probs = torch.softmax(logits, dim=-1)
        names = [self.classes[i] for i in probs.argmax(-1).tolist()]
        return names, probs

    # ---- convenience: route raw prompts (runs the target's prefill) ---------
    @torch.inference_mode()
    def route_prompts(self, prompts: List[str], target, tok, max_len: int = 512,
                      ) -> Tuple[List[str], torch.Tensor]:
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for p in prompts]
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(target.device)
        hs = target(**enc, use_cache=False, output_hidden_states=True).hidden_states
        feat = torch.cat([hs[lid + 1] for lid in self.layer_ids], dim=-1)
        return self.route_hidden(feat, enc.attention_mask)

    # ---- mapping to adapter ids ("other" -> -1 == base, per batched_lora) ---
    def to_adapter_ids(self, names: List[str], adapter_index: dict) -> List[int]:
        """adapter_index: {adapter_name: id} (e.g. MultiLoRAController.index).
        Unknown/'other' -> -1, which batched_lora treats as base (no adapter)."""
        return [adapter_index.get(n, -1) if n != "other" else -1 for n in names]
