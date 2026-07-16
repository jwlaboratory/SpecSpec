# Speculator domain benchmark — dflash

Speculator `z-lab/Qwen3-8B-DFlash-b16` + target `Qwen/Qwen3-8B` · 4 domains · 15 speculative tokens/step.

- **Acceptance rate (pooled):** 13.1%
- **Mean accept length:** 3.34 tokens/pass

## Domains ranked by acceptance rate

| Domain | Accept % | Mean len | Gen tok | n |
|---|---|---|---|---|
| code_sql | 25.0 | 4.75 | 21229 | 100 |
| ood_legal | 14.1 | 3.11 | 12997 | 30 |
| ood_financial | 12.0 | 2.79 | 50730 | 100 |
| ood_medical | 11.3 | 2.69 | 51151 | 100 |
