#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-examples_train/llm_sft/sft_config.json}
NUM_GPUS=${NUM_GPUS:-1}

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m fireredasr2s.fireredasr2.train_sft \
  --config "${CONFIG}"
