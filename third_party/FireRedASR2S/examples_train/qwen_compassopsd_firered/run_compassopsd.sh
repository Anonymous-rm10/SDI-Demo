#!/usr/bin/env bash
# Local adaptation for the bundled layout; see THIRD_PARTY_NOTICES.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
QWEN_PROJECT_DIR="${QWEN_PROJECT_DIR:-$WORKSPACE_DIR/third_party/Qwen3-ASR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
STUDENT_MODEL_DIR="${STUDENT_MODEL_DIR:-$WORKSPACE_DIR/models/Qwen3-ASR-0.6B}"
STUDENT_REFERENCE_MODEL_DIR="${STUDENT_REFERENCE_MODEL_DIR:-$STUDENT_MODEL_DIR}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-$WORKSPACE_DIR/models/FireRedASR2-LLM}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE_DIR/datasets/kespeech/train1_out/Southwestern_opd.jsonl}"
EVAL_FILE="${EVAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/models/Qwen3-ASR-0.6B-compassopsd-firered_rl-Southwestern}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
for path in \
  "$QWEN_PROJECT_DIR" \
  "$STUDENT_MODEL_DIR" \
  "$STUDENT_REFERENCE_MODEL_DIR" \
  "$TEACHER_MODEL_DIR" \
  "$TEACHER_CHECKPOINT"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: required path not found: $path" >&2
    exit 1
  fi
done
if [ ! -f "$TRAIN_FILE" ]; then
  echo "ERROR: training file not found: $TRAIN_FILE" >&2
  exit 1
fi
if [ -n "$EVAL_FILE" ] && [ ! -f "$EVAL_FILE" ]; then
  echo "ERROR: evaluation file not found: $EVAL_FILE" >&2
  exit 1
fi

if [ -z "${TEACHER_DEVICE:-}" ]; then
  GPU_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
  if [ "$GPU_COUNT" -ge 2 ]; then
    TEACHER_DEVICE="cuda:1"
  else
    TEACHER_DEVICE="cuda:0"
  fi
fi

ANCHOR_ALPHA="${ANCHOR_ALPHA:-0.5}"
BETA="${BETA:-0.5}"
TEACHER_SHIFT_CLIP="${TEACHER_SHIFT_CLIP:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK="${TOPK:-}"
SFT_COEFFICIENT="${SFT_COEFFICIENT:-0.0}"
TEACHER_MAX_TEXT_TOKENS="${TEACHER_MAX_TEXT_TOKENS:-1024}"
ASR_LANGUAGE="${ASR_LANGUAGE:-Chinese}"
RESUME_FROM="${RESUME_FROM:-}"

MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
MAX_ROLLOUT_RETRIES="${MAX_ROLLOUT_RETRIES:-3}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-1.0}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.95}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-0}"
LOG_ROLLOUT_TEXT="${LOG_ROLLOUT_TEXT:-1}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-2e-6}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LOG_STEPS="${LOG_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
REPORT_TO="${REPORT_TO:-wandb}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-qwen3-asr-compassopsd-firered_rl-Southwestern}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-asr-compassopsd-$(date +%Y%m%d-%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_MODE
export WANDB_NAME="$WANDB_RUN_NAME"

args=(
  "$SCRIPT_DIR/train_compassopsd.py"
  --student_model_path "$STUDENT_MODEL_DIR"
  --student_reference_model_path "$STUDENT_REFERENCE_MODEL_DIR"
  --teacher_model_path "$TEACHER_MODEL_DIR"
  --teacher_checkpoint "$TEACHER_CHECKPOINT"
  --train_file "$TRAIN_FILE"
  --output_dir "$OUTPUT_DIR"
  --teacher_device "$TEACHER_DEVICE"
  --anchor_alpha "$ANCHOR_ALPHA"
  --beta "$BETA"
  --teacher_shift_clip "$TEACHER_SHIFT_CLIP"
  --temperature "$TEMPERATURE"
  --teacher_max_text_tokens "$TEACHER_MAX_TEXT_TOKENS"
  --language "$ASR_LANGUAGE"
  --sft_coefficient "$SFT_COEFFICIENT"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --max_rollout_retries "$MAX_ROLLOUT_RETRIES"
  --sampling_temperature "$SAMPLING_TEMPERATURE"
  --sampling_top_p "$SAMPLING_TOP_P"
  --sampling_top_k "$SAMPLING_TOP_K"
  --batch_size "$BATCH_SIZE"
  --grad_acc "$GRAD_ACC"
  --lr "$LR"
  --epochs "$EPOCHS"
  --max_steps "$MAX_STEPS"
  --log_steps "$LOG_STEPS"
  --save_steps "$SAVE_STEPS"
  --num_workers "$NUM_WORKERS"
  --report_to "$REPORT_TO"
)
if [ "$LOG_ROLLOUT_TEXT" = "1" ]; then
  args+=(--log_rollout_text)
else
  args+=(--no-log_rollout_text)
fi
if [ "$USE_FLASH_ATTENTION" = "1" ]; then
  args+=(--use_flash_attention)
else
  args+=(--no-use_flash_attention)
fi
if [ -n "$EVAL_FILE" ]; then
  args+=(--eval_file "$EVAL_FILE")
fi
if [ -n "$TOPK" ]; then
  args+=(--topk_logits_k "$TOPK")
fi
if [ -n "$RESUME_FROM" ]; then
  args+=(--resume_from "$RESUME_FROM")
fi
if [ -n "${TEACHER_REFERENCE_TEMPLATE:-}" ]; then
  args+=(--teacher_reference_template "$TEACHER_REFERENCE_TEMPLATE")
fi

echo "Student:             $STUDENT_MODEL_DIR"
echo "Student reference:   $STUDENT_REFERENCE_MODEL_DIR"
echo "Teacher RL policy:   $TEACHER_CHECKPOINT"
echo "Teacher reference:   released FireRed (RL adapter disabled)"
echo "Teacher device:      $TEACHER_DEVICE"
echo "Train data:          $TRAIN_FILE"
echo "Eval data:           ${EVAL_FILE:-disabled}"
echo "Output:              $OUTPUT_DIR"
echo "Alpha / beta / temp: $ANCHOR_ALPHA / $BETA / $TEMPERATURE"
echo "Shift clipping:      ${TEACHER_SHIFT_CLIP} (0 means disabled)"
echo "FlashAttention 2:     $USE_FLASH_ATTENTION"
echo "Method:              reference-conditioned Compass target + generalized JSD"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$QWEN_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "${args[@]}" "$@"
 