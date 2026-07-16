# Speculator domain benchmark — eagle3

Speculator `RedHatAI/Qwen3-8B-speculator.eagle3` + target `Qwen/Qwen3-8B` · 4 domains · 3 speculative tokens/step.

- **Acceptance rate (pooled):** 38.9%
- **Mean accept length:** 2.23 tokens/pass

## Domains ranked by acceptance rate

| Domain | Accept % | Mean len | Gen tok | n |
|---|---|---|---|---|
| code_sql | 50.6 | 2.52 | 21170 | 100 |
| ood_legal | 39.9 | 2.20 | 13031 | 30 |
| ood_financial | 38.5 | 2.15 | 50455 | 100 |
| ood_medical | 35.2 | 2.05 | 51164 | 100 |
