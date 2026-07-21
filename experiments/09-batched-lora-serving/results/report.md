# Batched-LoRA serving — throughput and overhead by batch size

Qwen3-0.6B, greedy, 128 forced new tokens/seq, zero-delta r16 q/k/v/o adapters (identical outputs across modes), H200. Mean of 3 timed batches after warmup. Overheads computed against the same-container base only.


## HF eager (exp 06's serving stack)

| variant | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| A · base | 51 | 64 | 127 | 254 | 505 | 1008 | 1998 |
| B · unmerged wrappers | 38 (+32.9%) | 53 (+20.3%) | 106 (+20.5%) | 211 (+20.5%) | 419 (+20.5%) | 832 (+21.1%) | 1661 (+20.4%) |
| C · merged | 49 (+3.0%) | 63 (+0.7%) | 126 (+1.1%) | 254 (-0.0%) | 508 (-0.6%) | 1017 (-0.9%) | 2006 (-0.4%) |

## vLLM (punica multi-LoRA kernels)

| variant | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| A · base | 568 | 1046 | 2039 | 4035 | 7685 | 13512 | 21719 |
| D · punica ×1 adapter | 568 (+58.5%) | 1182 (+44.4%) | 2350 (+41.9%) | 4514 (+36.9%) | 8596 (+35.8%) | 15918 (+26.8%) | 25822 (+22.3%) |
| engine w/ LoRA, idle | 900 | 1707 | 3334 | 6178 | 11677 | 20182 | 31582 |
| E · punica ×50 adapters | 334 (+67.2%) | 665 (+64.5%) | 1170 (+80.9%) | 1946 (+97.6%) | 2926 (+146.4%) | 3906 (+225.0%) | 5056 (+310.8%) |
| engine w/ LoRA, idle | 558 | 1094 | 2116 | 3845 | 7210 | 12695 | 20771 |
