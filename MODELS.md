# Trained checkpoints — what exists and where

All adapters/checkpoints (`*.pt`) are **gitignored**. Master copies live on the
Modal volume **`code-sql-pipeline`** (under `/models/...`); local copies sit in
each experiment's `models/` folder. Pull anything back with:

```bash
modal volume get code-sql-pipeline models/<name>.pt experiments/<exp>/models/
```

The exception is `router/results/router_mlp.pt` (the trained router MLP, ~40MB)
which is small enough that it is tracked in git.

| experiment | local `models/` contents | notes |
|---|---|---|
| `01-single-domain-dflash` | `{code_sql,ood_indian_legal}_{lora,full}.pt` | r16 LoRA + full fine-tune of the whole 1B drafter |
| `02-multilingual-dflash` | `{polish,korean,italian,japanese,german,combined}_lora.pt` | r16; the **r64** variants live only on the volume |
| `03-weird-domains` | `{dflash,eagle}_{translation,roleplay,poetry,combined}_lora.pt` | r16; eagle adapters are **v3** (canonical shift_batch alignment, aux 0:std, retrained 2026-07-20); the DFlash **r4/r64** ladder variants live only on the volume |
| `04-multilingual-eagle` | `{polish,korean,italian,japanese,german,combined}_lora.pt` | **v3** (canonical shift_batch alignment, aux 0:std, retrained 2026-07-20); same filenames as 02 — the experiment folder disambiguates |
| `05-interference-ladder` | 10 core specialists + `comb{10,20,40}_lora.pt` | r16, trained on core+distractor mixes |
| `06-independent-drafter` | `{code_sql,lang_polish,lang_korean,ood_legal,task_math_reasoning,combined}_lora.pt` | r16 on the **Qwen3-0.6B independent drafter** (not DFlash); volume path `models/independent/` |
| `00-base-benchmarks`, `07-rank-ladder`, `08-wallclock` | — | no checkpoints of their own: 00 benches base models, 07 collects the r4/r64 results (adapters on the volume under 02/03's paths), 08 measures vanilla decoding |

Base models (never modified, pulled from HF at runtime): target `Qwen/Qwen3-8B`,
DFlash drafter `z-lab/Qwen3-8B-DFlash-b16`, EAGLE3 head
`RedHatAI/Qwen3-8B-speculator.eagle3`, independent drafter `Qwen/Qwen3-0.6B`.
