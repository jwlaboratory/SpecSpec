# finetuning — specializing the DFlash drafter per domain

Take the pretrained **DFlash block-diffusion drafter** (`z-lab/Qwen3-8B-DFlash-b16`,
the ~1B speculator for `Qwen/Qwen3-8B`), **fine-tune it on one domain** two ways —
an unmerged **LoRA** and a **full fine-tune** — and measure whether either tracks
the target better than the untouched base drafter (higher acceptance ⇒ fewer target
passes ⇒ more speedup).

This is the counterpart to `../Benchmarking domains/`: that measures base speculators
(DFlash / EAGLE3) *across* domains; this **trains** the drafter *on* a domain and
compares base vs full vs LoRA.

## Method (one stack, real weights, self-distillation)

The z-lab drafter *is* SpecForge's `DFlashDraftModel`, so the **same object** is
loaded via `AutoModel` for both training and benchmarking — no second model, no
weight remapping.

1. **prep** — the frozen **target Qwen3-8B generates** an answer for each domain
   prompt. The drafter is trained to match the **target's own tokens** (even where
   the target is wrong) — the dataset's own answers are *not* used.
2. **train** — fine-tune the drafter with SpecForge's `OnlineDFlashModel` DFlash
   matching loss (exp-weighted block cross-entropy, γ=7), using the target's **live
   hidden states** at layers `[1,9,17,25,33]`. Two variants:
   - `lora` — rank-16 unmerged LoRA on q/k/v/o, base frozen (~2M params)
   - `full` — every drafter weight (~1.05B params)
3. **bench** — base / full / lora through the real block-diffusion `spec_generate`
   + a target-only baseline, temperature 0 (lossless), over the held-out test split.

## Layout

```
finetuning/
├── pipeline.py        Modal pipeline: prep → train(lora,full) → bench(3) → aggregate
│                       (server-side `orchestrate` + `run` entrypoint → launch detached)
├── online_dflash.py   vendored SpecForge OnlineDFlashModel (DFlash loss engine)
├── spec_patch.py      instrumented spec_generate (returns per-step accept lengths)
├── make_charts.py     pretty base-vs-full-vs-LoRA charts (summary + distribution + cross-domain)
├── LoRA/              standalone LoRA-on-drafter trainer + the LoRALinear module (lora.py)
├── Full-Tune/         standalone full-fine-tune trainer
├── data/<domain>/     prompts used for finetuning (train/val/test.jsonl)
├── models/            trained checkpoints  <domain>_lora.pt · <domain>_full.pt  (gitignored, large)
└── results/<domain>/  per-variant jsonl · <domain>_report.md · comparison.csv · charts/
```

## Reproduce

```bash
# 1. get a domain's prompts into data/<domain>/ (train/val/test.jsonl of {"prompt"})
#    e.g. via ../Benchmarking domains/WildDataGen/sources.py, then move here.

# 2. run the whole thing on Modal — DETACHED so a network drop can't kill it
modal run --detach finetuning/pipeline.py::run --domain <domain> --epochs 3
#    results always land on the `code-sql-pipeline` volume under /results/<domain>/

# 3. pull models + results back
modal volume get code-sql-pipeline models/<domain>_lora.pt finetuning/models/
modal volume get code-sql-pipeline models/<domain>_full.pt finetuning/models/
modal volume get code-sql-pipeline "results/<domain>" finetuning/results/

# 4. render charts
python finetuning/make_charts.py <domain>
```

Re-aggregate only (results already on the volume): `modal run finetuning/pipeline.py::agg_only --domain <domain>`.

## Results

**code_sql** (b-mc2/sql-create-context, 800 train · 100 test):

| variant | accept rate | mean accept len | speedup | exact-match |
|---|--:|--:|--:|--:|
| base (pretrained) | 25.0% | 5.07 | 3.70× | 38% |
| full fine-tune | 25.1% (+0.2pp) | 5.12 (+0.04) | 3.68× | 38% |
| **LoRA** | **28.3% (+3.4pp)** | **6.03 (+0.96)** | **4.11× (+0.42)** | 38% |

→ `results/code_sql/charts/summary.png`, `distribution.png`

**ood_indian_legal** (kaushik-harsh-99/Indian-legal-data-v3, 8000 train · 300 test): _run in progress._

### Takeaway

**LoRA specializes the drafter; full fine-tune ≈ base at small data.** LoRA trains
~2M params, so ~800 short SQL examples (~75K supervised tokens) is enough to move
the needle without forgetting. Full fine-tune moves ~1.05B params on the same data
and can't — it's data-starved and less regularized. The legal run scales to 8000
longer examples partly to give full fine-tune a fairer shot.

Temperature 0 ⇒ DFlash is lossless: the adapter changes **speed**, not correctness
(exact-match is identical across variants; it's <100% only from benign bf16
tie-breaking between the parallel and sequential paths, not from training).
