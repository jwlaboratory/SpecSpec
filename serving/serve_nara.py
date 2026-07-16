#!/usr/bin/env python3
"""Serve N NaRA/LoRA adapters over ONE shared DFlash backbone.

Shared backbone  = frozen target Qwen/Qwen3-8B + frozen DFlash drafter
                   (z-lab/Qwen3-8B-DFlash-b16), loaded ONCE, resident in memory.
N adapters       = per-request unmerged LoRA deltas on the drafter's q/k/v/o.
Serving          = for each incoming request, point the drafter's LoRA layers at
                   that request's adapter (O(1) pointer swap — no reload) and run
                   the real DFlash block-diffusion spec_generate against the target.

This is the "naive adapter-swap" path: no Punica gather kernel. The efficiency
comes from never reloading the 8B target / 1B draft between requests.

Usage (real GPU run — see modal_serve_nara.py for the Modal wrapper):
    python serve_nara.py --adapters a.pt b.pt c.pt --n-toy 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_bank import AdapterBank, make_toy_adapter  # noqa: E402

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"


@dataclass
class Request:
    prompt: str
    adapter: int          # index into the bank (which fine-tune to use)
    max_new_tokens: int = 256


@dataclass
class Result:
    adapter: int
    adapter_name: str
    text: str
    n_tokens: int
    mean_accept_len: float
    seconds: float
    tokens_per_sec: float


class AdapterServer:
    """Holds the shared backbone + adapter bank; serves requests by swapping."""

    def __init__(self, draft, target, tokenizer, block_size: int = 16):
        self.draft = draft
        self.target = target
        self.tok = tokenizer
        self.bank = AdapterBank(draft, block_size=block_size)
        self.device = target.device

    # ---- adapter management -------------------------------------------------
    def load_adapter_paths(self, paths):
        """Load real adapters from files/dirs of any supported format (NaRA or PEFT)."""
        for path in paths:
            k = self.bank.load_adapter_path(path)
            print(f"[server] loaded adapter #{k}: {self.bank.names[k]}  (from {path})")

    def add_toy_adapters(self, n: int, rank: int = 32):
        for i in range(n):
            ckpt = make_toy_adapter(self.bank.layers, rank=rank, seed=100 + i)
            k = self.bank.load_adapter(ckpt, name=f"toy-{i}")
            print(f"[server] loaded TOY adapter #{k} (rank {rank})")

    # ---- serving ------------------------------------------------------------
    def _build_ids(self, prompt: str) -> torch.Tensor:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self.tok([text], return_tensors="pt").input_ids.to(self.device)

    def serve_one(self, req: Request) -> Result:
        self.bank.activate(req.adapter)             # <-- the swap (O(1))
        input_ids = self._build_ids(req.prompt)
        stop_ids = [self.tok.eos_token_id]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids = self.draft.spec_generate(
            target=self.target,
            input_ids=input_ids,
            max_new_tokens=req.max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=0.0,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_new = out_ids.shape[1] - input_ids.shape[1]
        text = self.tok.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        # mean accept length is exposed via the instrumented wrapper if present;
        # fall back to n_new/steps if the model recorded it.
        mal = float(getattr(self.draft, "_last_mean_accept_len", 0.0)) or 0.0
        return Result(
            adapter=req.adapter,
            adapter_name=self.bank.names[req.adapter],
            text=text,
            n_tokens=n_new,
            mean_accept_len=mal,
            seconds=dt,
            tokens_per_sec=n_new / dt if dt > 0 else 0.0,
        )

    def serve_batch(self, requests: List[Request]) -> List[Result]:
        print(f"\n[server] serving {len(requests)} requests over 1 shared backbone "
              f"({self.bank.num_adapters} adapters resident)")
        results = []
        for i, req in enumerate(requests):
            r = self.serve_one(req)
            print(f"  req {i}: adapter='{r.adapter_name}'  {r.n_tokens} tok  "
                  f"{r.seconds:.2f}s  {r.tokens_per_sec:.1f} tok/s")
            results.append(r)
        return results


def load_backbone(attn: str = "sdpa", device: str = "cuda"):
    """Load target + draft ONCE. Returns (draft, target, tokenizer)."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    print(f"[load] tokenizer {TARGET_MODEL}")
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    print(f"[load] draft     {DRAFT_MODEL}")
    draft = AutoModel.from_pretrained(
        DRAFT_MODEL, trust_remote_code=True, dtype="auto", attn_implementation=attn,
    ).to(device).eval()
    print(f"[load] target    {TARGET_MODEL}")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype="auto", attn_implementation=attn,
    ).to(device).eval()
    return draft, target, tok


# Five demo requests, each routed to a different adapter (0..4).
DEMO_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Reverse a linked list in Python.",
    "Parse a CSV file and sum the second column in Python.",
    "Implement binary search over a sorted list in Python.",
    "Write a decorator that memoizes a Python function.",
]


def make_demo_requests(n_adapters: int, max_new_tokens: int = 256) -> List[Request]:
    return [
        Request(prompt=DEMO_PROMPTS[i % len(DEMO_PROMPTS)],
                adapter=i % n_adapters, max_new_tokens=max_new_tokens)
        for i in range(5)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapters", nargs="*", default=[], help="paths to nara_adapter.pt files")
    p.add_argument("--n-toy", type=int, default=0, help="add N toy adapters (dry run)")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    draft, target, tok = load_backbone(attn=args.attn, device=args.device)
    server = AdapterServer(draft, target, tok, block_size=draft.config.block_size)

    if args.adapters:
        server.load_adapter_paths(args.adapters)
    if args.n_toy:
        server.add_toy_adapters(args.n_toy, rank=args.rank)

    if server.bank.num_adapters == 0:
        raise SystemExit("No adapters loaded. Pass --adapters or --n-toy.")

    reqs = make_demo_requests(server.bank.num_adapters, args.max_new_tokens)
    results = server.serve_batch(reqs)

    print("\n" + "=" * 72)
    for i, r in enumerate(results):
        print(f"[req {i}] adapter='{r.adapter_name}'  ({r.tokens_per_sec:.1f} tok/s)")
        print("   " + r.text.replace("\n", "\n   ")[:400])
    print("=" * 72)


if __name__ == "__main__":
    main()
