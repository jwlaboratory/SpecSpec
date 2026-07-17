# Adapter router — MLP on target hidden states

Features: mean-pooled Qwen3-8B hidden states at layers [1, 9, 17, 25, 33] (20480-dim — the same states the DFlash drafter already consumes, so routing costs one tiny MLP forward at serve time).

MLP 20480→512→6 · train 4800 / val 596 / test 596

**Val accuracy: 100.0%  ·  Test accuracy: 100.0%**

## Per-class test accuracy

| class | accuracy | routes to |
|---|--:|---|
| polish | 100.0% | polish LoRA |
| korean | 100.0% | korean LoRA |
| italian | 100.0% | italian LoRA |
| japanese | 100.0% | japanese LoRA |
| german | 100.0% | german LoRA |
| other | 100.0% | base drafter (no adapter) |

## Confusion matrix (rows = true, cols = predicted)

| | polish | korean | italian | japanese | german | other |
|---|--:|--:|--:|--:|--:|--:|
| **polish** | 100 | 0 | 0 | 0 | 0 | 0 |
| **korean** | 0 | 100 | 0 | 0 | 0 | 0 |
| **italian** | 0 | 0 | 100 | 0 | 0 | 0 |
| **japanese** | 0 | 0 | 0 | 100 | 0 | 0 |
| **german** | 0 | 0 | 0 | 0 | 100 | 0 |
| **other** | 0 | 0 | 0 | 0 | 0 | 96 |
