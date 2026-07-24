# Cost of the 26-language router MLP — measured

_Draft scratch doc. Numbers measured on an NVIDIA H200 (Modal), 2026-07-23.
Benchmark: `scratchpad/bench_router26.py`, loading the real trained
`router/results26/router26_mlp.pt`. Copy-paste what's useful into the blog._

## TL;DR
The router is a 2-layer MLP, `20480 → 512 → 26`, **10.5M params**. It runs
**once per request** on the mean-pooled prompt feature the DFlash drafter
already extracts — so it adds **no extra target prefill and no extra feature
traffic**, only one tiny MLP forward. Measured cost:

- **Time:** ~**48 µs** at batch 1, ~**1–5 µs/request** when batched — i.e.
  **~0.1% of a single Qwen3-8B target prefill** (~47 ms) at worst, and far less
  batched or amortized over a full generation.
- **Compute:** **21 MFLOP/request**, ~**5 parts per million** of the target
  prefill's FLOPs.
- **Memory bandwidth:** reads its **42 MB** of fp32 weights once per forward
  (21 MB in bf16), independent of batch → 42 MB/B per request.
- **Footprint:** **42 MB** VRAM (fp32) alongside the 16 GB target — **0.26%**.

The "basically negligible" claim in the blog is correct; these are the numbers
behind it.

## What the router actually is
`nn.Sequential(Linear(20480,512), GELU, Dropout, Linear(512,26))`

| quantity | value |
|---|---|
| input dim | 20480 (5 target layers × 4096, mean-pooled over the prompt) |
| hidden | 512 |
| classes | 26 |
| **parameters** | **10,499,610** (10.5M) — 99.9% in the first `Linear` |
| FLOPs / request | 20,998,144 (**≈ 21 MFLOP**) |
| weights, fp32 | 41,998,440 B (**42.0 MB**) — matches the 42 MB `.pt` |
| weights, bf16 | 20,999,220 B (**21.0 MB**) |

Key structural point: it runs **once per request**, on the pooled prompt
feature — **not per generated token**. And that feature is exactly what DFlash
already computes for the drafter (target hidden states at layers
[1,9,17,25,33]), so the router adds **zero** feature-extraction work — just the
MLP forward.

## Time — measured on H200
Serving path is fp32 (the router casts features to `float()`); bf16 shown for
reference. Warmup 50, timed 200 iters, CUDA-synced.

| batch | fp32 total | fp32 / req | bf16 total | bf16 / req |
|---:|---:|---:|---:|---:|
| 1 | 48.5 µs | 48.5 µs | 63.3 µs | 63.3 µs |
| 4 | 99.8 µs | 24.9 µs | 52.9 µs | 13.2 µs |
| 16 | 77.6 µs | 4.85 µs | 51.7 µs | 3.23 µs |
| 64 | 110.5 µs | 1.73 µs | 52.3 µs | 0.82 µs |
| 256 | 296.1 µs | 1.16 µs | 51.3 µs | 0.20 µs |

At batch 1 the ~48 µs is **kernel-launch/dispatch bound**, not compute — the
math is trivial (21 MFLOP is <10 µs of H200 flops). As batch grows the fixed
launch cost amortizes to ~1 µs/request.

### vs. the real target forward (same GPU)
Measured single-sequence **Qwen3-8B** prefill (bf16, `output_hidden_states`):

| prompt len | prefill |
|---:|---:|
| 64 | 47.3 ms |
| 256 | 46.6 ms |
| 512 | 46.5 ms |

(Short prefills are overhead/bandwidth-bound, hence roughly flat.) So the router
adds:

- **batch 1:** 48.5 µs / 46.6 ms = **0.10%** of one target prefill.
- **batched (B=64):** 1.7 µs / 46.6 ms = **0.0037%**.
- **over a full generation:** the router fires *once*; the target then runs
  hundreds of decode steps — so the router's share of end-to-end request time is
  smaller still.

## Compute
21 MFLOP/request. Qwen3-8B is ~2·8.2e9 = **16.4 GFLOP per token**, so a 256-token
prefill is ~4.2 TFLOP. Router / target = 21e6 / 4.2e12 = **5.0e-6 (~5 ppm)**. In
raw arithmetic the router is five parts per million of the prefill it rides on.

## Memory bandwidth
At batch 1 the MLP is a matrix-vector product → **memory-bound**: it must read
its weights once. That's **42 MB (fp32)** / **21 MB (bf16)** per forward,
independent of batch, so **42 MB / B per request**. Input activation is
20480×4 = 80 KB/req (fp32), output negligible.

Ideal weight-read time on H200 (~4.8 TB/s HBM3e): 42 MB / 4.8 TB/s ≈ **8.8 µs**
(fp32), 4.4 µs (bf16). Measured 48 µs at B=1 → we're launch-bound, well under the
bandwidth ceiling. Either way it's a one-time read of a 42 MB tensor per request,
against a target that streams its full 16 GB of weights every decode step.

## Footprint
42 MB resident (fp32) beside the 16 GB target = **+0.26% VRAM**. In bf16, +0.13%.

## One-line for the blog
> The router is a 10.5M-param, 2-layer MLP (`20480→512→26`, 42 MB) that runs once
> per request on features DFlash already extracts. Measured on H200 it costs
> ~48 µs at batch 1 (~1 µs/request batched) — about 0.1% of a single Qwen3-8B
> prefill, ~5 ppm of its FLOPs, and +0.26% VRAM: negligible on every axis.
