# 10 — English subdomain specialization

This is the canonical "within one language" experiment for the blog.

Question:

> Do weak English subdomains behave like weak languages?

If yes, the specialization story is not just "the drafter is bad at Korean."
It is a broader coverage-gap story: public speculators miss parts of the
target model's behavior even inside English.

## Domains

All seven domains use the local English prompt splits in `data/synthetic`, with
matched 800 / 100 / 100 train / val / test examples:

| domain | why it is here |
|---|---|
| `code_python` | English coding assistance, high-base structured domain |
| `code_sql` | templated code/data domain |
| `ood_legal` | English legal explanations and drafting |
| `ood_medical` | English medical explanations |
| `ood_financial` | English finance/accounting/investing explanations |
| `task_math_reasoning` | English reasoning, usually high acceptance |
| `task_summarization` | English summarization task |

## Variants

| variant | training data |
|---|---|
| `base` | no adapter |
| `own` | one r16 LoRA per domain, 800 train examples/domain |
| `combined` | one r16 LoRA on all seven domains, 5,600 total train examples |
| `combined_equal` | one r16 LoRA on all seven domains, ~800 total train examples, evenly sampled |

`combined_equal` is the data-budget control. It asks whether the combined
adapter transfers across English subdomains, or only wins because it sees more
examples.

## Run

Cheap path validation:

```bash
modal run experiments/10-english-subdomains/pipeline.py::smoke
```

Full run:

```bash
modal run --detach experiments/10-english-subdomains/pipeline.py::launch
```

Aggregate after a run:

```bash
modal run experiments/10-english-subdomains/pipeline.py::agg_only
```

Outputs live on the `code-sql-pipeline` Modal volume:

```text
/work/prep/english_subdomains/
/work/models/english_subdomains/
/work/results/english_subdomains/
```

