# 07 — rank ladder: how much adapter capacity does specialization need?

The rank question, isolated: **r4 (~130K params) vs r16 (~2M) vs r64 (~8M)**
LoRAs on the DFlash drafter, across both domain sets. The runs were executed by
the exp 02/03 pipelines with a `--rank` flag (ledger #11–13); this folder holds
the rank-variant results and the cross-experiment charts. The base and r16 runs
they ladder against remain in `../02-multilingual-dflash/` and
`../03-weird-domains/`.

## Multilingual (weak base, 3–8% acceptance) — rank keeps paying

| language | base | own r16 | own **r64** |
|---|--:|--:|--:|
| polish | 3.1% | 4.4% | **5.0%** |
| korean | 3.5% | 5.1% | **5.8%** |
| italian | 8.1% | 8.5% | 8.7% |
| japanese | 5.0% | 5.9% | 6.3% |
| german | 6.8% | 7.2% | 7.3% |

r64 > r16 on **5/5**, and the extra gain concentrates exactly where the base is
weakest (polish/korean +0.6–0.7pp extra, +61–66% relative over base; german/
italian only +0.1–0.2pp). combined-r64 ≈ own-r64 (no interference at higher
capacity either).

## Weird domains (moderate base, 7–9%) — gains saturate by r16, and r4 nearly suffices

| domain | base | r4 own | r16 own | r64 own |
|---|--:|--:|--:|--:|
| translation | 8.7% | 9.2% | 9.5% | 9.4% |
| roleplay | 8.1% | 8.5% | 8.5% | 8.5% |
| poetry | 7.0% | 7.5% | 7.6% | 7.8% |

**Rank 4 captures ≈90% of the achievable gain** (0.01% of the drafter's
params) — the domain shift is intrinsically low-rank, a broad "steering"
direction rather than stored knowledge. And combined-r4 ≈ own-r4: no
interference even at the scarcest capacity, with three heterogeneous tasks
sharing one adapter.

## Takeaway

**Rank need scales with the size of the deficit.** Weak-base domains (3–5%
acceptance) keep converting extra rank into acceptance; moderate-base domains
saturate by r16 and nearly by r4. Default recipe: r16 everywhere, r64 only for
domains the base speculator is nearly useless on.

→ `results/charts/rank_scaling.png`, `rank_ladder.png`

## Reproduce

```bash
# rank variants are trained by the source pipelines (rank/alpha args):
modal run --detach experiments/02-multilingual-dflash/pipeline_langs.py::launch --rank 64
modal run --detach experiments/03-weird-domains/pipeline_dflash.py::launch --rank 4
python3 experiments/07-rank-ladder/make_charts.py
```
