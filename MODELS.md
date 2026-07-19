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
| `03-weird-domains` | `{dflash,eagle}_{translation,roleplay,poetry,combined}_lora.pt` | r16; the DFlash **r4/r64** ladder variants live only on the volume |
| `04-multilingual-eagle` | `{polish,korean,italian,japanese,german,combined}_lora.pt` | v2 (shadowing bug fixed); same filenames as 02 — the experiment folder disambiguates |
| `05-interference-ladder` | 10 core specialists + `comb{10,20,40}_lora.pt` | r16, trained on core+distractor mixes |

Base models (never modified, pulled from HF at runtime): target `Qwen/Qwen3-8B`,
DFlash drafter `z-lab/Qwen3-8B-DFlash-b16`, EAGLE3 head
`RedHatAI/Qwen3-8B-speculator.eagle3`.
