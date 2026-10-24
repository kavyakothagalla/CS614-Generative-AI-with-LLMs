#!/usr/bin/env bash
# Run the six models ONE AFTER ANOTHER for a given chunk of rows.
#
#   bash run_all.sh 0                                  # chunk 0 of the shuffled pool
#   bash run_all.sh 0 --only-clear --balance           # chunk 0 of the balanced clear-only pool
#   bash run_all.sh all                                # whole pool
#
# Any extra arguments after the chunk are passed straight to run_models.py.
# Safe to stop (Ctrl+C) and re-run: finished items are skipped.

CHUNK="${1:?Usage: bash run_all.sh <chunk number | all> [extra run_models.py flags]}"
shift

MODELS=(
  "llama2:7b"
  "llama3.1:8b"
  "qwen:7b"
  "qwen3.5:9b"
  "gemma:7b"
  "gemma4:e4b"
)

for m in "${MODELS[@]}"; do
  echo "################ $m (chunk: $CHUNK) $* ################"
  if [ "$CHUNK" = "all" ]; then
    python3 run_models.py --model "$m" "$@"
  else
    python3 run_models.py --model "$m" --chunk "$CHUNK" "$@"
  fi
done

echo "All models done for chunk: $CHUNK"
