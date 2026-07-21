https://huggingface.co/datasets/allenai/WildChat-4.8M
this dataset split by language

## Language distribution

Counted from the `language` column across all 86 parquet shards (full dataset,
not the partial ~781k slice the HF dataset viewer stats show). Total rows:
3,199,860 conversations ("4.8M" refers to turns). 75 detected languages plus
`Nolang` (no language detected). Top 10 languages cover ~95% of the set.

| Language | Conversations | Share |
|---|---:|---:|
| English | 1,679,371 | 52.483% |
| Russian | 363,265 | 11.353% |
| Chinese | 266,258 | 8.321% |
| French | 173,167 | 5.412% |
| Vietnamese | 127,498 | 3.984% |
| Yoruba | 91,834 | 2.870% |
| Arabic | 77,487 | 2.422% |
| Indonesian | 67,037 | 2.095% |
| Spanish | 60,321 | 1.885% |
| Portuguese | 58,148 | 1.817% |
| German | 25,754 | 0.805% |
| Persian | 23,937 | 0.748% |
| Tagalog | 19,600 | 0.613% |
| Turkish | 16,889 | 0.528% |
| Korean | 16,190 | 0.506% |
| Italian | 13,655 | 0.427% |
| Maori | 11,414 | 0.357% |
| Nolang | 9,593 | 0.300% |
| Sotho | 8,473 | 0.265% |
| Polish | 7,684 | 0.240% |
| Latin | 7,628 | 0.238% |
| Japanese | 7,396 | 0.231% |
| Serbian | 6,276 | 0.196% |
| Ukrainian | 5,086 | 0.159% |
| Malay | 4,298 | 0.134% |
| Dutch | 4,061 | 0.127% |
| Esperanto | 4,060 | 0.127% |
| Romanian | 3,228 | 0.101% |
| Hungarian | 2,789 | 0.087% |
| Swedish | 2,588 | 0.081% |
| Somali | 2,357 | 0.074% |
| Estonian | 1,960 | 0.061% |
| Tswana | 1,712 | 0.054% |
| Bulgarian | 1,637 | 0.051% |
| Finnish | 1,607 | 0.050% |
| Catalan | 1,476 | 0.046% |
| Bokmal | 1,467 | 0.046% |
| Hebrew | 1,391 | 0.043% |
| Welsh | 1,389 | 0.043% |
| Hindi | 1,277 | 0.040% |
| Nynorsk | 1,210 | 0.038% |


| Swahili | 1,096 | 0.034% |
| Czech | 1,046 | 0.033% |
| Azerbaijani | 1,032 | 0.032% |
| Thai | 973 | 0.030% |
| Danish | 945 | 0.030% |
| Tsonga | 845 | 0.026% |
| Greek | 784 | 0.025% |
| Shona | 761 | 0.024% |
| Slovene | 736 | 0.023% |
| Lithuanian | 684 | 0.021% |
| Xhosa | 646 | 0.020% |
| Urdu | 602 | 0.019% |
| Albanian | 589 | 0.018% |
| Bengali | 563 | 0.018% |
| Afrikaans | 547 | 0.017% |
| Macedonian | 541 | 0.017% |
| Basque | 497 | 0.016% |
| Slovak | 492 | 0.015% |
| Croatian | 415 | 0.013% |
| Irish | 411 | 0.013% |
| Mongolian | 406 | 0.013% |
| Ganda | 395 | 0.012% |
| Zulu | 395 | 0.012% |
| Kazakh | 366 | 0.011% |
| Bosnian | 363 | 0.011% |
| Latvian | 329 | 0.010% |
| Belarusian | 325 | 0.010% |
| Tamil | 261 | 0.008% |
| Georgian | 152 | 0.005% |
| Icelandic | 119 | 0.004% |
| Marathi | 36 | 0.001% |
| Armenian | 19 | 0.001% |
| Gujarati | 14 | 0.000% |
| Telugu | 4 | 0.000% |
| Punjabi | 3 | 0.000% |

## Pipeline (`pipeline.py`)

Full experiment over the **40 languages with >= 1,100 conversations** (all of the
table above down to Nynorsk; Swahili at 1,096 misses; "Nolang" excluded):

1. **fetch** (CPU) — stream WildChat-4.8M once; 1000 train / 100 val / 100 test
   first-turn prompts per language (dedup, skip toxic/redacted; short tail
   languages split 10:1:1 proportionally).
2. **generate** (GPU) — Qwen3-8B answers train+val greedily (self-distillation).
3. **capture** (GPU) — one teacher forward per sequence; hidden states at
   `target_layer_ids [1, 9, 17, 25, 33]` frozen to shards; also dumps the
   target's `lm_head`/`embed_tokens` (~2.5 GB once).
4. **verify** (GPU) — stored hidden ≈ live teacher hidden (cosine parity) +
   real LoRA steps purely from shards, no 8B loaded.
5. **train** (GPU) — one r16 LoRA per language + one combined LoRA on all 40,
   streamed shard-by-shard (combined set ≈ 880 GB, never fits in RAM),
   teacher-free.
6. **bench** (H200) — base vs own vs combined on each language's 100 test
   prompts; spec runs instrumented, vanilla decode paired in-container on a
   15-prompt subset.
7. **aggregate** — pooled acceptance rate, mean accept length, measured
   speedup (spec/base tok/s) and analytic speedup L/(1+c), c=0.44 (exp-08 fit).

Volume `exp1-language-hidden`:

    /data/prompts/{lang}/{train,val,test}.jsonl
    /data/ids/{lang}/{train,val}.pt                  # {input_ids, prompt_len}
    /data/shards/{lang}/{train,val}/shard_NNNN.pt    # + manifest.json
    /data/target_head_embed.pt
    /data/models/{lang}_lora.pt, combined_lora.pt
    /data/results/{lang}_{variant}.jsonl, summary.json

Shard record: `{"input_ids" int32 [T], "prompt_len", "hidden" bf16 [T, 20480]}` —
exactly `OnlineDFlashModel.forward`'s inputs; anchors are resampled per step, so
frozen data still yields fresh block pairs each epoch.

**Sizing**: 40 KB/token → ~20 MB per 500-token conversation → **~22 GB per
language, ~0.9 TB total**. Storage is the bill; GPU stages fan out (8 containers).

    modal run new/exp1-language/pipeline.py::smoke     # 2-language end-to-end, every stage
    modal run --detach new/exp1-language/pipeline.py::launch
    modal run new/exp1-language/pipeline.py::results   # aggregate -> results/summary.json

If fetch 401s, WildChat-4.8M is gated: `modal secret create huggingface
HF_TOKEN=...` and add `secrets=[modal.Secret.from_name("huggingface")]` to `fetch`.
Caveat: "Yoruba" (#6 by share) is largely a language-detector artifact — interpret
its results with suspicion.

## Results (2026-07-21, run ap-ZK0GDpMOJeYkYFMZbkmXhc)

100 held-out test prompts per language, greedy spec decode, r16 LoRAs, 3 epochs.
Pooled acceptance rate per (language x variant); L = mean accept length;
speedup = analytic L/(1+0.44) (exp-08 fit — measured paired subset agrees but is
noisier at 15 prompts).

| lang | base acc | own acc | comb acc | base L | own L | comb L | own spd | comb spd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Polish | 0.036 | 0.046 | 0.044 | 1.54 | 1.70 | 1.66 | 1.18 | 1.15 |
| Hungarian | 0.039 | 0.058 | 0.052 | 1.59 | 1.87 | 1.77 | 1.30 | 1.23 |
| Korean | 0.042 | 0.052 | 0.050 | 1.63 | 1.79 | 1.74 | 1.24 | 1.21 |
| Hebrew | 0.042 | 0.064 | 0.062 | 1.64 | 1.96 | 1.93 | 1.36 | 1.34 |
| Dutch | 0.045 | 0.060 | 0.056 | 1.68 | 1.90 | 1.84 | 1.32 | 1.28 |
| Romanian | 0.046 | 0.058 | 0.055 | 1.69 | 1.87 | 1.83 | 1.30 | 1.27 |
| Estonian | 0.046 | 0.050 | 0.060 | 1.69 | 1.75 | 1.91 | 1.21 | 1.32 |
| Turkish | 0.049 | 0.069 | 0.062 | 1.73 | 2.03 | 1.92 | 1.41 | 1.34 |
| Indonesian | 0.052 | 0.065 | 0.068 | 1.78 | 1.98 | 2.03 | 1.37 | 1.41 |
| Ukrainian | 0.057 | 0.071 | 0.065 | 1.85 | 2.07 | 1.98 | 1.44 | 1.37 |
| Nynorsk | 0.057 | 0.059 | 0.069 | 1.85 | 1.88 | 2.04 | 1.31 | 1.42 |
| Arabic | 0.057 | 0.065 | 0.062 | 1.86 | 1.97 | 1.93 | 1.37 | 1.34 |
| Finnish | 0.058 | 0.060 | 0.070 | 1.87 | 1.91 | 2.04 | 1.32 | 1.42 |
| Catalan | 0.061 | 0.065 | 0.073 | 1.91 | 1.97 | 2.09 | 1.37 | 1.45 |
| Malay | 0.062 | 0.075 | 0.077 | 1.93 | 2.13 | 2.16 | 1.48 | 1.50 |
| Serbian | 0.062 | 0.066 | 0.068 | 1.94 | 1.99 | 2.01 | 1.38 | 1.40 |
| Bulgarian | 0.064 | 0.068 | 0.070 | 1.96 | 2.02 | 2.04 | 1.41 | 1.42 |
| Tswana | 0.064 | 0.067 | 0.078 | 1.97 | 2.01 | 2.17 | 1.39 | 1.51 |
| Tagalog | 0.064 | 0.072 | 0.075 | 1.97 | 2.08 | 2.12 | 1.44 | 1.47 |
| Bokmal | 0.065 | 0.066 | 0.078 | 1.97 | 2.00 | 2.17 | 1.39 | 1.50 |
| Persian | 0.065 | 0.074 | 0.071 | 1.98 | 2.12 | 2.07 | 1.47 | 1.44 |
| Esperanto | 0.066 | 0.068 | 0.078 | 1.99 | 2.02 | 2.17 | 1.40 | 1.50 |
| Portuguese | 0.066 | 0.074 | 0.071 | 1.99 | 2.12 | 2.07 | 1.47 | 1.44 |
| Swedish | 0.067 | 0.076 | 0.074 | 2.01 | 2.14 | 2.11 | 1.49 | 1.47 |
| Vietnamese | 0.073 | 0.088 | 0.084 | 2.10 | 2.32 | 2.27 | 1.61 | 1.57 |
| Chinese | 0.075 | 0.078 | 0.077 | 2.13 | 2.17 | 2.16 | 1.51 | 1.50 |
| Italian | 0.078 | 0.080 | 0.078 | 2.18 | 2.21 | 2.17 | 1.53 | 1.51 |
| Japanese | 0.079 | 0.080 | 0.079 | 2.19 | 2.20 | 2.19 | 1.53 | 1.52 |
| Russian | 0.082 | 0.085 | 0.085 | 2.23 | 2.27 | 2.28 | 1.58 | 1.58 |
| Sotho | 0.083 | 0.086 | 0.093 | 2.25 | 2.28 | 2.40 | 1.59 | 1.67 |
| Yoruba | 0.086 | 0.086 | 0.096 | 2.29 | 2.29 | 2.43 | 1.59 | 1.69 |
| Maori | 0.086 | 0.086 | 0.093 | 2.29 | 2.29 | 2.39 | 1.59 | 1.66 |
| German | 0.088 | 0.090 | 0.089 | 2.32 | 2.35 | 2.33 | 1.63 | 1.62 |
| Welsh | 0.107 | 0.106 | 0.110 | 2.61 | 2.59 | 2.65 | 1.80 | 1.84 |
| French | 0.110 | 0.111 | 0.110 | 2.65 | 2.67 | 2.65 | 1.85 | 1.84 |
| Spanish | 0.115 | 0.116 | 0.113 | 2.72 | 2.74 | 2.70 | 1.90 | 1.87 |
| Somali | 0.115 | 0.114 | 0.123 | 2.73 | 2.71 | 2.84 | 1.89 | 1.97 |
| Latin | 0.118 | 0.119 | 0.119 | 2.77 | 2.79 | 2.79 | 1.94 | 1.94 |
| English | 0.129 | 0.128 | 0.127 | 2.93 | 2.92 | 2.91 | 2.02 | 2.02 |
| Hindi | 0.162 | 0.169 | 0.169 | 3.42 | 3.53 | 3.54 | 2.45 | 2.46 |

### Takeaways

1. **Specialization works, concentrated where the base drafter is weakest**
   (echoes exp-07): Hebrew 4.2%→6.4% (+52% rel), Hungarian 3.9%→5.8%,
   Turkish 4.9%→6.9%. Top-of-table languages (English 12.9%, French, Spanish)
   gain ~nothing — the pretrained drafter already covers them.
2. **One combined LoRA ≈ 40 specialists**: mean acceptance delta vs base is
   +0.79pp for combined vs +0.63pp for own; head-to-head 19-21 across
   languages. No interference at 40-language scale (extends exp-02/05's 5-40).
3. **Cross-lingual transfer rescues data-starved languages**: combined beats
   own on nearly every language that fetched short — Estonian (430 recs,
   5.0%→6.0%), Nynorsk, Bokmal, Tswana, Sotho, Somali, Finnish, Maori,
   Serbian. Own wins on data-rich distinct languages (Turkish, Ukrainian,
   Persian, Vietnamese, Swedish).
4. Wall-clock (analytic): 1.15-2.46x over vanilla; LoRAs add up to +0.3 tokens
   mean accept length on weak languages.
5. Next lever per exp-07: r64 on the weak-base tier (Polish/Hungarian/Korean/
   Hebrew/Dutch), trained from the SAME frozen shards — no teacher re-run.
