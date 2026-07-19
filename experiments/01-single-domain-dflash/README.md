# 01 single-domain — specializing the DFlash drafter per domain

Take the pretrained **DFlash block-diffusion drafter** (`z-lab/Qwen3-8B-DFlash-b16`,
the ~1B speculator for `Qwen/Qwen3-8B`), **fine-tune it on one domain** two ways —
an unmerged **LoRA** and a **full fine-tune** — and measure whether either tracks
the target better than the untouched base drafter (higher acceptance ⇒ fewer target
passes ⇒ more speedup).

This is the counterpart to `../../benchmarking/`: that measures base speculators
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
experiments/01-single-domain-dflash/
├── pipeline.py        Modal pipeline: prep → train(lora,full) → bench(3) → aggregate
│                       (server-side `orchestrate` + `run` entrypoint → launch detached)
├── train_lora.py      standalone LoRA-on-drafter trainer
├── full_tune.py       standalone full-fine-tune trainer
├── make_charts.py     pretty base-vs-full-vs-LoRA charts (summary + distribution + cross-domain)
├── models/            trained checkpoints  <domain>_lora.pt · <domain>_full.pt  (gitignored, large)
└── results/<domain>/  per-variant jsonl · <domain>_report.md · comparison.csv · charts/
```

Shared modules live in `../../lib/` (`lora.py` LoRALinear, `online_dflash.py`
vendored SpecForge loss, `spec_patch.py` instrumented spec_generate); prompts in
`../../data/downloaded/<domain>/` (train/val/test.jsonl).

## Reproduce

```bash
# 1. get a domain's prompts into ../../data/downloaded/<domain>/ (train/val/test.jsonl of {"prompt"})
#    e.g. via ../../benchmarking/wilddatagen/sources.py, then move here.

# 2. run the whole thing on Modal — DETACHED so a network drop can't kill it
modal run --detach experiments/01-single-domain-dflash/pipeline.py::run --domain <domain> --epochs 3
#    results always land on the `code-sql-pipeline` volume under /results/<domain>/

# 3. pull models + results back
modal volume get code-sql-pipeline models/<domain>_lora.pt experiments/01-single-domain-dflash/models/
modal volume get code-sql-pipeline models/<domain>_full.pt experiments/01-single-domain-dflash/models/
modal volume get code-sql-pipeline "results/<domain>" experiments/01-single-domain-dflash/results/

# 4. render charts
python experiments/01-single-domain-dflash/make_charts.py <domain>
```

Re-aggregate only (results already on the volume): `modal run experiments/01-single-domain-dflash/pipeline.py::agg_only --domain <domain>`.

## Results

**code_sql** (b-mc2/sql-create-context, 800 train · 100 test):

| variant | accept rate | mean accept len | speedup | exact-match |
|---|--:|--:|--:|--:|
| base (pretrained) | 25.0% | 5.07 | 3.70× | 38% |
| full fine-tune | 25.1% (+0.2pp) | 5.12 (+0.04) | 3.68× | 38% |
| **LoRA** | **28.3% (+3.4pp)** | **6.03 (+0.96)** | **4.11× (+0.42)** | 38% |

→ `results/code_sql/charts/summary.png`, `distribution.png`

**ood_indian_legal** (kaushik-harsh-99/Indian-legal-data-v3, 8000 train · 150 test):

| variant | accept rate | mean accept len | speedup | exact-match |
|---|--:|--:|--:|--:|
| base (pretrained) | 10.9% | 2.67 | 1.94× | 18% |
| full fine-tune | 11.1% (+0.2pp) | 2.70 (+0.03) | 1.96× | 18% |
| **LoRA** | **13.5% (+2.5pp)** | **3.09 (+0.42)** | **2.11× (+0.17)** | 18% |

→ `results/ood_indian_legal/charts/summary.png`, `distribution.png`, `results/charts/cross_domain.png`

### Takeaway

**Same pattern in both domains: LoRA specializes the drafter; full fine-tune ≈ base.**
LoRA trains ~2M params, so it moves the needle (+3.4pp acceptance on SQL, +2.5pp on
legal) without forgetting. Full fine-tune moves ~1.05B params and lands on top of
base in both — even on legal's 8000 longer examples (~2M supervised tokens vs SQL's
~75K), full couldn't pull ahead. Legal's absolute acceptance is much lower than SQL
(~11–13% vs ~25–28%): free-form legal prose is far less predictable than templated
SQL, so the drafter tracks the target on fewer tokens regardless of tuning.

Net: for domain-adapting a speculative-decoding drafter, **LoRA is the better tool**;
full fine-tune only becomes worth it with substantially more data than either run here.

### Pipeline performance (per run)

vLLM prep (continuous batching) + H200 + train batch-12 took the whole legal run
(8000 train, 3 epochs, 150-prompt bench) from a projected ~4–5h to **~35 min** — prep
alone went from ~95 min (HF `generate`) to **89 s** (~60× on an H200).

Temperature 0 ⇒ DFlash is lossless: the adapter changes **speed**, not correctness
(exact-match is identical across variants; it's <100% only from benign bf16
tie-breaking between the parallel and sequential paths, not from training).
