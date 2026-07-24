# Serving Cost


We first compared all the different ways to serve the Specialized drafter. Merging is a process in which you take the low-rank adapter LoRa weights and you mathematically multiply them with the existing weights to create a merged single set of weights as if it was just a new fine-tuned model. The benefit of keeping them unmerged is that you can keep most of the weights the same so you have a lower memory footprint because you just need to swap out your final adapter weights. As soon as you merge, then you need to keep multiple copies that are very similar but to the computer look entirely different. 

We compared one merged model with the combined LORAs, compared with multiple N individually merged specialized LORAs, compared with N unmerged hot‑swappable LORAs. 

Because VLLM does not support hot swapping unmerged loras, we tested this on HF.

| mode | meaning |
| :---- | :---- |
| base | DFlash, no LoRA |
| merged\_combined | one combined language LoRA folded into DFlash weights |
| merged\_own | routed traffic to N already-merged language specialists |
| hotswap\_own | one drafter with unmerged per-language LoRA hot swapping |

![Comparing base, merged_combined, merged_own, and hotswap_own at batch size 1](experiments/11-wallclock-production/results/full/results/charts/speedup_modes.png)

| mode | tok/s | vs target-only | vs base DFlash | acceptance | mean accept length |
| :---- | ----: | ----: | ----: | ----: | ----: |
| target-only | 46.78 | 1.000× | — | — | — |
| base DFlash | 66.33 | 1.418× | 1.000× | 6.46% | 1.969 |
| merged combined LoRA | 70.23 | 1.501× | 1.059× | 7.17% | 2.076 |
| N merged own LoRAs | 70.55 | 1.508× | 1.064× | 7.33% | 2.099 |
| hot-swapped own LoRAs | 54.73 | 1.170× | 0.825× | 7.32% | 2.098 |

This shows us that hotswapping LoRAs without optimizing this (punica styled batch kernels perhaps) is unworkable. On the other hand, merged-combined and N merged-own show promising results. Combined LoRA gives a \+5.9% wall-clock gain over base DFlash on this mixed-language serving stream, and the one-time merge setup was only 0.073s. The N-merged-specialist path gives \+6.4% over base DFlash, only about \+0.5% relative to the merged combined LoRA.

We tried on vLLM at different batch sizes on one of the best performing LoRAs (Swedish), which performed remarkable. Note that the batch size 1 result is different from above because we use vLLM and not HF.

All numbers below are net wall-clock speedup vs no speculative decoding (target-only).

| batch size | merged own | merged combined | base DFlash | no spec decoding |
| ----: | ----: | ----: | ----: | ----: |
| 1 | 1.73× | 1.69× | 1.50× | 1.00× |
| 4 | 1.78× | 1.73× | 1.56× | 1.00× |
| 8 | 1.63× | 1.59× | 1.42× | 1.00× |
| 16 | 1.29× | 1.28× | 1.13× | 1.00× |
| 32 | 0.82× | 0.81× | 0.73× | 1.00× |
| 64 | 0.51× | 0.48× | 0.45× | 1.00× |

![Swedish serving speedup vs batch size](experiments/13-batchsize-speedup/results/charts/modes_vs_batch.png)

At a higher batch size, even naive speculative decoding doesn't help anymore because we're no longer memory bound, but rather compute bound. But the cool thing to observe is that at these lower batch sizes, the Swedish Specialist gives up to a 15.3% gain over the base D-Flash, and the combined LoRa nearly matches it, plus 12% at batch size 1. 

The benefit of merged-combined is that you only need 1 set of weights. It's essentially just the drafter with more knowledge. However, it's not specialized and may have interference (as we've somewhat shown).

The benefit of N-merged LoRAs is that you do not have any intereference and can have extreme specialization. The negative is that you now need to store more weights and this may perform poorly when constantly needing to swap in and out weights with heavy batch sizes. With larger batches (say size B), we will need to pull in potentially N different experts, instead of previously only needing to pull in 1 expert. Similar to the MOE speculation problem, this is bad because in an already memory bound system we are further hurting the memory pipe.

We then benchmarked using vLLM to see at different batch sizes with a "MIXED BAG" of different requests from different languages. This forced the model to use the combined merged LoRA and the speedups are shown below:

*(table — pending: combined vs base vs no-spec on the 16-language mixed stream across batch sizes; results uploading soon)*

*(chart — pending: mixed-bag speedup vs batch size)*

We leave a future experiment to try to update the vLLM implementation and kernels to test how much slowdown hotswapping adapters or swapping entire drafters would cause within batches.


