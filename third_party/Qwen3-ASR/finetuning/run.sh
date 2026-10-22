#!/usr/bin/env bash
# Local adaptation for the bundled layout; see THIRD_PARTY_NOTICES.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export PYTHONPATH="$PROJECT_ROOT/third_party/Qwen3-ASR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/qwen3_asr_sft.py" \
  --model_path "${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-ASR-0.6B}" \
  --train_file "${TRAIN_FILE:-$PROJECT_ROOT/datasets/train.jsonl}" \
  --output_dir "${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen3-asr-sft}" \
  --batch_size 4 --grad_acc 1 --lr 2e-5 --epochs 1 \
  --save_steps 200 --save_total_limit 5 "$@"
