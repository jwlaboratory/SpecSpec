# 26-language adapter router

Features: filtered from `/data/router40/features`; mean-pooled Qwen3-8B hidden states (20480-dim).

MLP 20480->512->26 · train 26000 / val 2600 / test 2600

**Val accuracy: 84.69% · Test accuracy: 81.58%**

## Per-class test accuracy

| language | accuracy | test n |
|---|---:|---:|
| English | 59.00% | 100 |
| Russian | 90.00% | 100 |
| Chinese | 90.00% | 100 |
| French | 88.00% | 100 |
| Vietnamese | 87.00% | 100 |
| Yoruba | 58.00% | 100 |
| Arabic | 96.00% | 100 |
| Indonesian | 57.00% | 100 |
| Spanish | 93.00% | 100 |
| Portuguese | 94.00% | 100 |
| German | 87.00% | 100 |
| Persian | 96.00% | 100 |
| Tagalog | 57.00% | 100 |
| Turkish | 98.00% | 100 |
| Korean | 96.00% | 100 |
| Italian | 92.00% | 100 |
| Polish | 94.00% | 100 |
| Latin | 58.00% | 100 |
| Japanese | 98.00% | 100 |
| Ukrainian | 95.00% | 100 |
| Malay | 61.00% | 100 |
| Dutch | 82.00% | 100 |
| Esperanto | 27.00% | 100 |
| Romanian | 78.00% | 100 |
| Hungarian | 96.00% | 100 |
| Swedish | 94.00% | 100 |
