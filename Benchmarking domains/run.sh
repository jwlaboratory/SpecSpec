#!/usr/bin/env bash
# Convenience runner for the DFlash domain benchmark.
# Requires a CUDA GPU with ~20GB+ VRAM. See README.md.
set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-dflash_bench}"
LIMIT="${LIMIT:-100}"                 # prompts per category (1..100)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CATEGORIES="${CATEGORIES:-all}"       # all | languages | coding | tasks | "lang_english code_python ..."
EXTRA_ARGS="${EXTRA_ARGS:-}"          # e.g. --no-baseline  or  --resume

echo "== DFlash domain benchmark =="
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'; \
print('GPU:', torch.cuda.get_device_name(0), \
round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

python benchmark.py \
  --run-name "$RUN_NAME" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --categories $CATEGORIES \
  $EXTRA_ARGS

python aggregate.py "results/${RUN_NAME}.jsonl"
echo "== Report: results/${RUN_NAME}_report.md =="
