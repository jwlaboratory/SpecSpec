# Net wall-clock speedup — spec decode vs target-only decoding

Same prompts, greedy, 256 max new tokens, batch 1, H200. Speedup = pooled spec tok/s ÷ pooled vanilla tok/s, always within one framework (vLLM/vLLM or HF/HF). L = pooled mean accept length (tokens committed per target pass).


## EAGLE3 · multilingual (vLLM)

| domain | vanilla tok/s | base | own | combined |
|---|--:|--:|--:|--:|
| polish | 191 | 1.09× (L=1.27) | 1.15× (L=1.30) | 1.12× (L=1.31) |
| korean | 191 | 1.07× (L=1.24) | 1.08× (L=1.26) | 1.11× (L=1.27) |
| italian | 193 | 1.13× (L=1.42) | 1.22× (L=1.44) | 1.15× (L=1.46) |
| japanese | 199 | 0.95× (L=1.15) | 1.02× (L=1.21) | 1.03× (L=1.21) |
| german | 199 | 1.13× (L=1.37) | 1.15× (L=1.38) | 1.14× (L=1.39) |

## EAGLE3 · weird domains (vLLM)

| domain | vanilla tok/s | base | own | combined |
|---|--:|--:|--:|--:|
| translation | 196 | 1.31× (L=1.62) | 1.30× (L=1.61) | 1.27× (L=1.61) |
| roleplay | 198 | 1.68× (L=2.08) | 1.72× (L=2.06) | 1.51× (L=2.05) |
| poetry | 198 | 1.64× (L=2.00) | 1.55× (L=1.99) | 1.59× (L=1.99) |

## DFlash · weird domains (HF)

| domain | vanilla tok/s | base | own | combined |
|---|--:|--:|--:|--:|
| translation | 41 | 1.71× (L=2.31) | 1.56× (L=2.42) | 1.62× (L=2.41) |
| roleplay | 42 | 1.46× (L=2.22) | 1.42× (L=2.28) | 1.77× (L=2.27) |
| poetry | 43 | 1.56× (L=2.05) | 1.65× (L=2.15) | 1.58× (L=2.14) |

## DFlash · multilingual (HF, paired in-container baselines)

| domain | vanilla tok/s | base | own | combined |
|---|--:|--:|--:|--:|
| polish | 39 | 1.07× (L=1.47) | 1.15× (L=1.66) | 1.16× (L=1.67) |
| korean | 45 | 1.11× (L=1.53) | 1.22× (L=1.76) | 1.22× (L=1.77) |
| italian | 46 | 1.61× (L=2.21) | 1.58× (L=2.28) | 1.54× (L=2.26) |
| japanese | 52 | 1.26× (L=1.75) | 1.30× (L=1.88) | 1.30× (L=1.88) |
| german | 47 | 1.47× (L=2.02) | 1.44× (L=2.07) | 1.42× (L=2.06) |

## Analytic model — speedup ≈ L / (1 + c)

c is the speculator's per-step overhead in units of one target forward, fitted as median(L/speedup − 1) over a section's cells. If c is really a per-speculator constant, the model predicts every cell's wall-clock from its (already-measured) acceptance alone.

| section | fitted c | median |err| | max |err| | cells |
|---|--:|--:|--:|--:|
| EAGLE3 · multilingual (vLLM) | 0.180 | 2.0% | 7.1% | 15 |
| EAGLE3 · weird domains (vLLM) | 0.240 | 1.3% | 9.9% | 9 |
| DFlash · weird domains (HF) | 0.359 | 6.0% | 18.1% | 9 |
| DFlash · multilingual (HF, paired in-container baselines) | 0.445 | 0.3% | 4.9% | 15 |
