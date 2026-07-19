# interference — does "one combined LoRA = specialists" survive 10/20/40 domains?

`../multilingual` (5 languages) and `../weird-domains` (3 task types) showed a
single combined rank-16 LoRA matches per-domain specialist LoRAs — zero
interference — at 3–5 domains, even down to rank 4. This experiment scales N to
**10 / 20 / 40** to find the phase boundary where a fixed-capacity adapter stops
matching specialists, or prove there isn't one.

Same one-stack method as `../multilingual/pipeline_langs.py`: pretrained z-lab
DFlash drafter (target Qwen3-8B frozen), self-distillation (target generates all
answers via vLLM), SpecForge OnlineDFlashModel loss, rank-16 LoRA (α=32) on
q/k/v/o, 3 epochs, lossless temperature-0 bench with instrumented `spec_generate`.

## Design — core + distractors

40 domains from `Benchmarking domains/data/synthetic` (800 train / 100 val /
100 test each), in a fixed ladder order:

- **core 10** (evaluated; each gets an own specialist): code_python, code_sql,
  lang_polish, lang_korean, lang_german, ood_legal, ood_medical,
  task_math_reasoning, task_summarization, task_roleplay_chat
- **+10 distractors** (→ comb20): code_javascript, code_cpp, code_rust,
  lang_japanese, lang_italian, lang_french, lang_chinese, ood_chemistry,
  ood_regex, task_json_extraction
- **+20 distractors** (→ comb40): code_java, code_go, code_typescript,
  code_ruby, code_haskell, lang_spanish, lang_portuguese, lang_russian,
  lang_arabic, lang_turkish, lang_vietnamese, lang_hindi, ood_poetry, ood_shell,
  ood_ascii_art, ood_customer_support, ood_financial, task_creative_writing,
  task_email_writing, task_logic_reasoning

**13 LoRAs trained:** 10 own specialists + comb10 (core), comb20 (core+10),
comb40 (core+30). Every combined set contains the core, and per-domain data is
constant (800 examples), so the **(combN − own) acceptance gap on core domains
as N grows is a direct read-out of interference** — without training 40
specialists. Epochs fixed at 3 (matches the 5-domain protocol): combN gets N×
more total steps but identical per-domain exposure.

**Bench:** 10×5 matrix on core test splits (n=100/domain):
base | own | comb10 | comb20 | comb40.

## Results (n=100/domain, temperature 0, lossless)

| domain | base | own | comb10 | comb20 | comb40 |
|---|--:|--:|--:|--:|--:|
| code_python | 20.9% | 21.5% (+0.6pp) | 21.1% (+0.1pp) | 21.3% (+0.4pp) | 21.6% (+0.7pp) |
| code_sql | 18.0% | 18.8% (+0.7pp) | 18.4% (+0.4pp) | 18.4% (+0.4pp) | 18.5% (+0.5pp) |
| lang_polish | 3.1% | 4.4% (+1.3pp) | 4.5% (+1.4pp) | 4.3% (+1.1pp) | 4.1% (+1.0pp) |
| lang_korean | 3.5% | 5.1% (+1.6pp) | 5.1% (+1.6pp) | 5.0% (+1.4pp) | 4.8% (+1.2pp) |
| lang_german | 6.8% | 7.2% (+0.4pp) | 7.0% (+0.2pp) | 7.0% (+0.2pp) | 7.0% (+0.2pp) |
| ood_legal | 11.6% | 12.2% (+0.6pp) | 12.0% (+0.4pp) | 11.9% (+0.3pp) | 11.9% (+0.3pp) |
| ood_medical | 13.2% | 13.6% (+0.4pp) | 13.6% (+0.4pp) | 13.5% (+0.3pp) | 13.5% (+0.3pp) |
| task_math_reasoning | 37.6% | 39.5% (+2.0pp) | 38.8% (+1.2pp) | 38.6% (+1.0pp) | 38.5% (+0.9pp) |
| task_summarization | 9.6% | 9.8% (+0.2pp) | 9.7% (+0.1pp) | 9.8% (+0.2pp) | 9.8% (+0.2pp) |
| task_roleplay_chat | 8.1% | 8.5% (+0.4pp) | 8.4% (+0.2pp) | 8.4% (+0.2pp) | 8.3% (+0.2pp) |

**combN − own gap, mean over the 10 core domains (paired per-prompt bootstrap
95% CI, 3000 resamples):**

| N | mean gap | 95% CI |
|--:|--:|:--|
| 10 | **−0.21pp** | [−0.29, −0.13] |
| 20 | **−0.27pp** | [−0.34, −0.20] |
| 40 | **−0.28pp** | [−0.36, −0.19] |

→ `results/charts/ladder.png` (money chart) · `matrix.png` · `delta.png`

## Takeaways

1. **There is no phase boundary through 40 domains.** The combined adapter never
   collapses: it beats base on 10/10 core domains at every N, and the
   combined-vs-specialist gap *saturates* (−0.21 → −0.27 → −0.28pp from N=10→40;
   going 20→40 costs ~0.01pp) instead of growing with N.
2. **But "zero interference" does not survive past N≈5.** With prior points
   (mean gap −0.03pp at N=3 weird-domains, −0.02pp at N=5 multilingual — both
   ≈0 within noise), N=10 is where interference first becomes measurable: the
   CIs at 10/20/40 all exclude zero. The combined adapter retains ~74% of the
   mean specialist gain at N=10 and ~67% at N=40 (own mean +0.82pp vs base;
   comb40 +0.55pp).
3. **Interference concentrates exactly where specialization pays most.** The
   comb40 shortfall is −1.1pp on task_math_reasoning (of its +2.0pp own gain),
   −0.37/−0.29pp on korean/polish (of +1.6/+1.3pp) — and ≈0 on the small-gain
   domains (python +0.08, summarization 0.00, medical −0.13). Domains with a big,
   specific shift compete for the shared low-rank subspace; the broad "steering"
   component is shared for free.
4. **Operationally the single adapter still wins at 40 domains** (one artifact,
   ~2/3 of the gain, never below base). Per-domain routing (`../../router/`)
   is only worth its complexity for the high-gain domains — exactly the ones the
   ladder chart flags.

Caveats: rank fixed at 16 (whether r64 closes the gap is the obvious follow-up);
epochs fixed at 3, so combN gets N× more total steps at identical per-domain
exposure (matches the 5-domain protocol); prior N=3/5 points come from different
domain sets.

## Reproduce

```bash
modal run finetuning/interference/pipeline.py::smoke              # validate paths
modal run --detach finetuning/interference/pipeline.py::launch    # full run, detached
modal volume get code-sql-pipeline results/interference finetuning/interference/results/
modal volume get code-sql-pipeline models/interference finetuning/interference/models/
python3 finetuning/interference/make_charts.py
```

## Files

```
pipeline.py     Modal pipeline: vLLM prep (40 domains) → 13 LoRA trains (parallel)
                → 10×5 bench matrix (parallel) → aggregate. Detached-safe.
make_charts.py  matrix + interference-ladder charts (after fetching results)
results/        50 per-run jsonls · interference_report.md · comparison csv · charts/
models/         13 trained adapters (*.pt, gitignored)
```

(Results section filled in after the run completes.)
