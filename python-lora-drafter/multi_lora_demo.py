#!/usr/bin/env python3
"""Three domain LoRA adapters on one frozen base: hot-swap + BATCHED routing.

Runs locally in a few seconds. It uses the exact `batched_lora.py` machinery that
scales to the real DFlash drafter on Modal (see modal_train_lora.py) -- only the
base model and the loss differ. Here the base is a tiny causal LM and each
"domain" is a distinct, learnable token rule, so specialization is crisp and the
routing correctness is provable.

Domains (deterministic next-token rule, one token band each -> maximally distinct):
  python : counts within band 0
  sql    : counts within band 1
  prose  : counts within band 2

Shows:
  1. specialization  -> 3x3 (adapter x domain) accuracy matrix, diagonal dominates
  2. hot-swap        -> same frozen base, swap adapter, behavior changes
  3. batched routing -> ONE batch, per-sequence adapter; provably == per-adapter,
                        and beats forcing a single adapter on a mixed batch
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_lora import (  # noqa: E402
    adapter_parameters,
    inject_batched_lora,
)

DOMAINS = ["python", "sql", "prose"]
V = 192
BAND = 64  # tokens per domain band; bands are [0,64) [64,128) [128,192)
SEQ = 24


# --------------------------------------------------------------------------- #
def make_batch(domain: int, bsz: int, device) -> torch.Tensor:
    """domain d: sequence counts upward within band d (wraps). Deterministic rule
    the matching adapter can learn; other domains' tokens never appear."""
    start = domain * BAND
    offs = torch.randint(0, BAND, (bsz, 1), device=device)
    steps = torch.arange(SEQ, device=device).view(1, -1)
    return start + (offs + steps) % BAND  # (bsz, SEQ)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab, dim=128, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(SEQ, dim)
        self.blocks = nn.ModuleList(
            [_Block(dim, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, ids):
        L = ids.size(1)
        x = self.embed(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        B, L, _ = x.shape
        y = self.n1(x)
        q, k, v = self.q_proj(y), self.k_proj(y), self.v_proj(y)
        q, k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal LM
        a = a.transpose(1, 2).reshape(B, L, -1)
        x = x + self.o_proj(a)
        x = x + self.mlp(self.n2(x))
        return x


def next_token_loss(logits, ids):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )


@torch.no_grad()
def accuracy(logits, ids):
    pred = logits[:, :-1].argmax(-1)
    return (pred == ids[:, 1:]).float().mean().item()


def main():
    torch.manual_seed(0)
    device = "cpu"
    base = TinyCausalLM(V).to(device)
    n_base = sum(p.numel() for p in base.parameters())

    controller, replaced = inject_batched_lora(
        base, DOMAINS, rank=16, alpha=32,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "lm_head"),
    )
    print(f"frozen base params : {n_base:,}")
    print(f"adapters           : {DOMAINS}  (each on {len(replaced)} layers)")
    per_adapter = sum(p.numel() for p in adapter_parameters(base, 0))
    print(f"params per adapter : {per_adapter:,} ({100*per_adapter/n_base:.1f}% of base)")

    # -------- train each adapter on ONLY its domain (isolated params) ---------
    print("\ntraining 3 adapters (each on its own domain)...")
    for d, name in enumerate(DOMAINS):
        opt = torch.optim.AdamW(list(adapter_parameters(base, d)), lr=3e-3, weight_decay=0.0)
        controller.use_adapter(d)
        base.train()
        for step in range(400):
            ids = make_batch(d, 64, device)
            loss = next_token_loss(base(ids), ids)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  [{name:6}] final train loss {loss.item():.3f}")

    base.eval()

    # -------- 1. specialization matrix (adapter x domain accuracy) -----------
    print("\n[1] specialization  (rows = adapter loaded, cols = eval domain):")
    print("        " + "".join(f"{d:>9}" for d in DOMAINS))
    diag_ok = True
    for a, aname in enumerate(DOMAINS):
        controller.use_adapter(a)
        row = []
        for d in range(len(DOMAINS)):
            ids = make_batch(d, 128, device)
            row.append(accuracy(base(ids), ids))
        best = max(range(len(row)), key=lambda i: row[i])
        diag_ok &= (best == a)
        print(f"  {aname:6}" + "".join(f"{x:9.2f}" for x in row) + f"   -> best: {DOMAINS[best]}")
    # base-only for reference
    controller.use_base()
    base_acc = [accuracy(base(make_batch(d, 128, device)), make_batch(d, 128, device)) for d in range(3)]
    print(f"  {'(base)':6}" + "".join(f"{x:9.2f}" for x in base_acc) + "   -> no adapter")
    assert diag_ok, "each adapter should be best on its own domain"
    print("  => every adapter is best on its own domain (specialization works)")

    # -------- 2. hot-swap ----------------------------------------------------
    print("\n[2] hot-swap on the SAME frozen base (first predicted token, python seq):")
    ids = make_batch(0, 1, device)
    for a, aname in enumerate(DOMAINS):
        controller.use_adapter(aname)
        pred = base(ids)[0, 0].argmax().item()
        print(f"  adapter={aname:6} -> predicts token {pred:3d} "
              f"({'in python band' if pred < BAND else 'wrong band'})")

    # -------- 3. batched routing (the 'in batches' part) ---------------------
    print("\n[3] batched routing: ONE batch, per-sequence adapter")
    # a mixed batch: 2 python, 2 sql, 2 prose sequences
    routes = [0, 0, 1, 1, 2, 2]
    batch = torch.cat([make_batch(d, 1, device) for d in routes], dim=0)  # (6, SEQ)

    controller.route(routes)
    routed_logits = base(batch)

    # correctness: routed batch must equal running each seq with its own adapter
    max_err = 0.0
    for i, d in enumerate(routes):
        controller.use_adapter(d)
        single = base(batch[i : i + 1])
        max_err = max(max_err, (routed_logits[i : i + 1] - single).abs().max().item())
    print(f"  routed-batch == per-adapter singles?  max|Δ| = {max_err:.2e} "
          f"-> {'OK' if max_err < 1e-4 else 'FAIL'}")
    assert max_err < 1e-4

    # per-sequence accuracy under correct routing vs forcing one adapter on all
    controller.route(routes)
    routed_acc = accuracy(base(batch), batch)
    forced = {}
    for a, aname in enumerate(DOMAINS):
        controller.use_adapter(a)
        forced[aname] = accuracy(base(batch), batch)
    print(f"  mixed-batch accuracy  | per-seq routing: {routed_acc:.2f}")
    for aname, acc in forced.items():
        print(f"                        | force '{aname}' on all: {acc:.2f}")
    assert routed_acc > max(forced.values()) + 0.1
    print("  => per-sequence routing beats forcing any single adapter on the batch")

    print("\n" + "=" * 68)
    print("MULTI-LoRA DEMO PASSED")
    print("  - 3 domain adapters on ONE frozen base, each specialized")
    print("  - hot-swap changes behavior with no base reload")
    print("  - batched per-sequence routing is exact AND beats single-adapter")
    print("  same batched_lora.py scales to the real DFlash drafter on Modal")
    print("=" * 68)


if __name__ == "__main__":
    main()
