# 12 — language full fine-tune comparison

Direct language-domain baseline for the blog's LoRA-vs-full-finetune question.

`experiments/01-single-domain-dflash` compared LoRA and full fine-tuning on SQL
and Indian legal. This experiment repeats that comparison on weak WildChat
language lanes from `new/exp1-language`, using the same frozen target-hidden
shards and held-out prompts:

| variant | what |
|---|---|
| `base` | public DFlash drafter |
| `own` | existing per-language rank-16 LoRA |
| `combined` | existing combined rank-16 multilingual LoRA, recorded for routing-context only |
| `full` | full DFlash fine-tune on that language's frozen hidden shards |

Default languages are the weakest base lanes in the existing summary:
`Polish`, `Hungarian`, `Korean`, `Hebrew`, `Dutch`.

Run:

```bash
modal run experiments/12-language-full-finetune/pipeline.py::smoke
modal run --detach experiments/12-language-full-finetune/pipeline.py::launch
modal run experiments/12-language-full-finetune/pipeline.py::results
```

Outputs are written to the shared `exp1-language-hidden` Modal volume under
`/data/exp12_language_full_finetune`.

Completed local outputs:

- `results/summary.json`
- `results/report.md`
- `results/charts/full_finetune_gain.png`
- `results/charts/full_finetune_vs_lora.png`

For the blog's main LoRA-vs-full-finetune section, use `base`, `own`, and
`full`. Keep `combined` in the routing section.
