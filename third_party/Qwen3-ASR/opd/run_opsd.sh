#!/bin/bash
# Local adaptation for the bundled layout; see THIRD_PARTY_NOTICES.md.
# run_opsd.sh — On-Policy Self-Distillation for Qwen3-ASR
#   Student: Qwen3-ASR-0.6B (trainable)
#   Teacher: Qwen3-ASR-1.7B (frozen, sees privileged reference transcript)
#
# Usage:
#   bash run_opsd.sh                      # train only
#   bash run_opsd.sh train                # same as above
#   bash run_opsd.sh --stage train        # same as above
#   bash run_opsd.sh eval                 # run the checkpoint evaluation stage
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# ── Paths (adjust to your environment) ────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
STUDENT_MODEL_DIR="${STUDENT_MODEL_DIR:-$WORKSPACE_DIR/models/Qwen3-ASR-0.6B}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-$WORKSPACE_DIR/models/Qwen3-ASR-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE_DIR/datasets/WildElder/train_opd.jsonl}"
EVAL_FILE="${EVAL_FILE:-$WORKSPACE_DIR/datasets/kespeech/train1_out/test_opd.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/models/Qwen3-ASR-0.6B-opsd/new_wildelder}"

# ── OPSD hyper-params ─────────────────────────────────────────────────────────
# beta: 0 = forward KL, 1 = reverse KL, 0.5 = symmetric JSD (paper default is 0.5)
BETA="${BETA:-0.5}"
# temperature inside the JSD/KL softmax
TEMPERATURE="${TEMPERATURE:-1.0}"
# sampling temperature for student on-policy generation
SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
# max new tokens per sample in the on-policy rollout
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
# topk for logits distillation (empty = full vocab)
TOPK="${TOPK:-}"

# ── Training hyper-params ──────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_STEPS="${MAX_STEPS:--2}"
LOG_STEPS="${LOG_STEPS:-1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ── Logging ───────────────────────────────────────────────────────────────────
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-asr-opsd}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-asr-0.6b-opsd-$(date +%Y%m%d-%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
WANDB_WATCH="${WANDB_WATCH:-false}"

# ── Stage ─────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--stage" ]; then
  if [ -z "${2:-}" ]; then
    echo "ERROR: --stage requires train or eval" >&2
    exit 2
  fi
  STAGE="$2"
elif [ "$#" -gt 1 ]; then
  echo "Usage: bash run_opsd.sh [train|eval]" >&2
  echo "       bash run_opsd.sh --stage [train|eval]" >&2
  exit 2
else
  STAGE="${1:-train}"
fi

require_file() { if [ ! -f "$1" ]; then echo "ERROR: file not found: $1" >&2; exit 1; fi; }
require_dir()  { if [ ! -d "$1" ]; then echo "ERROR: directory not found: $1" >&2; exit 1; fi; }

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ "$REPORT_TO" = "wandb" ]; then
  case "$WANDB_MODE" in online|offline|disabled) ;; *)
    echo "ERROR: WANDB_MODE must be online, offline, or disabled" >&2; exit 2 ;; esac
  if ! "$PYTHON_BIN" -c 'import wandb' >/dev/null 2>&1; then
    echo "ERROR: wandb is not installed in $PYTHON_BIN" >&2
    echo "Install it with: $PYTHON_BIN -m pip install wandb" >&2
    exit 1
  fi
  mkdir -p "$WANDB_DIR"
  export WANDB_PROJECT WANDB_NAME="$WANDB_RUN_NAME" WANDB_MODE WANDB_DIR WANDB_LOG_MODEL WANDB_WATCH
fi

require_dir "$STUDENT_MODEL_DIR"
require_dir "$TEACHER_MODEL_DIR"
require_file "$TRAIN_FILE"
if [ -n "$EVAL_FILE" ]; then require_file "$EVAL_FILE"; fi

echo "Stage:         $STAGE"
echo "Python:        $PYTHON_BIN"
echo "Student model: $STUDENT_MODEL_DIR"
echo "Teacher model: $TEACHER_MODEL_DIR"
echo "Train data:    $TRAIN_FILE"
echo "Eval data:     ${EVAL_FILE:-disabled}"
echo "Output:        $OUTPUT_DIR"
echo "beta:          $BETA   temperature: $TEMPERATURE"
echo "Logger:        $REPORT_TO"

# =============================================================================
# Train
# =============================================================================
if [ "$STAGE" == "train" ]; then

  train_args=(
    "$SCRIPT_DIR/train_opsd.py"
    --student_model_path "$STUDENT_MODEL_DIR"
    --teacher_model_path "$TEACHER_MODEL_DIR"
    --train_file "$TRAIN_FILE"
    --output_dir "$OUTPUT_DIR"
    --beta "$BETA"
    --temperature "$TEMPERATURE"
    --sampling_temperature "$SAMPLING_TEMP"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --batch_size "$BATCH_SIZE"
    --grad_acc "$GRAD_ACC"
    --lr "$LR"
    --epochs "$EPOCHS"
    --max_steps "$MAX_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 3
    --log_steps "$LOG_STEPS"
    --warmup_ratio 0.02
    --lr_scheduler_type linear
    --seed "$SEED"
    --num_workers "$NUM_WORKERS"
    --pin_memory 1
    --persistent_workers 1
    --prefetch_factor 2
    --report_to "$REPORT_TO"
  )

  if [ -n "$EVAL_FILE" ]; then train_args+=(--eval_file "$EVAL_FILE"); fi
  if [ -n "$TOPK" ]; then train_args+=(--topk_logits_k "$TOPK"); fi

  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" "${train_args[@]}"

# =============================================================================
# Eval (quick ASR WER on a checkpoint)
# =============================================================================
elif [ "$STAGE" == "eval" ]; then

  CKPT_DIR="${CKPT_DIR:-}"
  if [ -z "$CKPT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    CKPT_DIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -1)"
  fi
  if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"; exit 1
  fi
  echo "Evaluating checkpoint: $CKPT_DIR"

  "$PYTHON_BIN" "$PROJECT_DIR/finetuning/qwen3_asr_sft.py" \
    --model_path "$CKPT_DIR" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_DIR/eval_out" \
    --batch_size 1 \
    --grad_acc 1 \
    --lr 2e-5 \
    --epochs 1 \
    --save_steps 999999 \
    --save_total_limit 0

else
  echo "Unknown stage: $STAGE  (known: train, eval)"
  exit 1
fi
