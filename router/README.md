# router — automatic LoRA selection from the target's hidden states

An MLP that reads the target model's hidden states and decides **which LoRA
adapter to route a request to** — or none. Classes: the five multilingual DFlash
adapters from `../finetuning/multilingual/` (polish · korean · italian · japanese
· german) plus **other → base drafter (no adapter)**.

Built ahead of need: `../serving/` already has the *mechanism* for per-request
adapters (`AdapterBank.activate(i)` hot-swap, `batched_lora.route(ids)` per-sequence
batched routing) but no *brain* to choose the adapter. This is the brain.

## Why it's free at serve time

DFlash already runs the target's prefill and conditions the drafter on the
target's hidden states at layers `[1, 9, 17, 25, 33]`
(`extract_context_feature`). The router consumes **exactly that tensor**,
mean-pooled over the prompt (20480-dim) → MLP `20480 → 512 → 6`. No extra model,
no extra forward pass — routing costs one tiny MLP matmul on states that are
already in memory.

## Training data

Labels come for free from the dataset folders:

- 5 languages × 800 train prompts (`../finetuning/multilingual/data/lang_*/`)
- **other**: 800 prompts sampled from 8 non-adapter domains
  (`data/other/`), with deliberate hard negatives — English/French/Spanish
  (languages *near* the adapter set), `code_python`, tasks, medical, financial.
  Predicting "other" maps to routing to the plain base drafter.

## Layout

```
router/
├── pipeline_router.py   Modal: extract features (batched prefill, mean-pooled
│                        layer-[1,9,17,25,33] states) → train MLP → eval + confusion
├── router.py            inference module: AdapterRouter.load() / route_hidden() /
│                        route_prompts() / to_adapter_ids() — plugs into
│                        serving/batched_lora.MultiLoRAController.route(ids)
├── data/other/          the sampled "other" negatives (train/val/test.jsonl)
└── results/             router_report.md · router_confusion.json · charts/
```

## Run

```bash
modal run router/pipeline_router.py::smoke              # tiny path validation
modal run router/pipeline_router.py::run                # extract + train + eval
modal run router/pipeline_router.py::run --skip-extract # iterate on MLP only

# pull artifacts
modal volume get code-sql-pipeline router/router_mlp.pt        router/results/
modal volume get code-sql-pipeline router/router_report.md     router/results/
modal volume get code-sql-pipeline router/router_confusion.json router/results/
```

## Serving integration (when needed)

```python
from router import AdapterRouter
router = AdapterRouter.load("router_mlp.pt")

# inside the spec-decode server, prefill already computed target_hidden (B,S,20480):
names, probs = router.route_hidden(target_hidden, attention_mask)
ids = router.to_adapter_ids(names, controller.index)   # "other" -> -1 == base
controller.route(ids)                                   # batched per-sequence LoRA
```

## v2 ideas (documented, not built)

- **Acceptance-aware routing**: train on *measured acceptance* per (prompt,
  adapter) instead of domain labels — routes to whichever adapter actually
  speculates best, not just the domain match.
- **Confidence threshold**: below max-softmax τ, fall back to base even for
  in-set classes (open-set robustness beyond the "other" class).
- **More classes**: `code_sql` / `ood_indian_legal` adapters (from
  `../finetuning/`) drop in by adding their folders to the class list.
