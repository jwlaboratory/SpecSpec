#!/usr/bin/env bash
# Convenience runner for the DFlash domain benchmark (local GPU box).
# Runs benchmark.py -> aggregate.py -> make_charts.py over the DataGen splits.
# Requires a CUDA GPU with ~20GB+ VRAM. See ../README.md.
set -euo pipefail
cd "$(dirname "$0")"                    # scripts/

RUN_NAME="${RUN_NAME:-dflash_bench}"
LIMIT="${LIMIT:-100}"                   # prompts per domain (1..split size)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SPLIT="${SPLIT:-test}"                  # train | val | test (default: held-out test)
CATEGORIES="${CATEGORIES:-all}"         # all | languages | coding | tasks | ood | "lang_english code_python ..."
EXTRA_ARGS="${EXTRA_ARGS:-}"           # e.g. --no-baseline  or  --resume

RESULTS_DIR="../results"

echo "== DFlash domain benchmark  (split=$SPLIT) =="
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'; \
print('GPU:', torch.cuda.get_device_name(0), \
round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

python benchmark.py \
  --run-name "$RUN_NAME" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt-source datagen \
  --split "$SPLIT" \
  --categories $CATEGORIES \
  $EXTRA_ARGS

python aggregate.py "${RESULTS_DIR}/${RUN_NAME}.jsonl"
python make_charts.py "${RESULTS_DIR}/${RUN_NAME}_by_category.csv" || \
  echo "(charts skipped — install matplotlib to enable)"

echo "== Report:  ${RESULTS_DIR}/${RUN_NAME}_report.md =="
echo "== Charts:  ${RESULTS_DIR}/charts/ =="
