# FireRedASR2-LLM continued SFT

This training path starts from the released `FireRedASR2-LLM` checkpoint. The
default configuration trains the speech adapter and the released Qwen LoRA,
while keeping the Conformer encoder and Qwen base weights frozen.

## 1. Prepare manifests

Use one JSON object per line:

```json
{"uttid":"utt001","wav":"/data/audio/utt001.wav","text":"你好世界","duration":2.35}
```

Required fields are `wav` (or `wav_path`) and `text`. `uttid` and `duration`
are optional. Missing duration is read from the audio header. Relative audio
paths are resolved against `audio_root` when configured, otherwise against the
manifest directory.

Audio is converted to mono and resampled to 16 kHz by the collator. Records
outside `[min_audio_seconds, max_audio_seconds]` are filtered. The released
model was trained for inputs up to 40 seconds, so increasing the default limit
is not recommended.

## 2. Configure

Copy `sft_config.json` and update at least:

- `model_dir`
- `train_manifest`
- `valid_manifest`
- `output_dir`

The default trainable modules are:

```text
Conformer encoder: frozen
Speech adapter:    trainable, lr=1e-4
Qwen base:         frozen
Released Qwen LoRA:trainable, lr=2e-5
```

LoRA structure is configurable in JSON:

```json
{
  "lora_rank": 64,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "lora_target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj"
  ]
}
```

These defaults match the released FireRedASR2-LLM LoRA and continue updating
the released adapter. If any of the four settings changes, the loader first
merges the released LoRA into the Qwen base and then creates a fresh LoRA with
the requested structure. Consequently, a changed LoRA starts from its normal
PEFT initialization while retaining the released model's behavior in the
merged base weights.

Set `train_encoder_last_n` to a small positive value to tune only the last
Conformer blocks. Set `freeze_encoder=false` and leave
`train_encoder_last_n=0` to tune the entire encoder. Full Qwen fine-tuning is
supported with `full_finetune_llm=true` and `use_lora=false`, but it produces
very large checkpoints and requires substantially more GPU memory.

`max_batch_frames` limits padded 10 ms acoustic frames, not the raw sample
count. Reduce it first when training runs out of memory.

## 3. Train

Single GPU:

```bash
NUM_GPUS=1 bash examples_train/llm_sft/run_sft.sh \
  examples_train/llm_sft/sft_config.json
```

Multi-GPU DDP:

```bash
NUM_GPUS=8 bash examples_train/llm_sft/run_sft.sh \
  examples_train/llm_sft/sft_config.json
```

Resume:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fireredasr2s.fireredasr2.train_sft \
  --config examples_train/llm_sft/sft_config.json \
  --resume_from exp/FireRedASR2-LLM-SFT/checkpoint-step-00000500
```

Checkpoints store the trainable delta plus optional optimizer state. With the
default LoRA setup they do not duplicate the Qwen base weights. Training and
validation metrics are appended to `train_metrics.jsonl` and
`eval_metrics.jsonl`.

## Weights & Biases visualization

Log in once on the training machine:

```bash
wandb login
```

Then enable W&B in the JSON config:

```json
{
  "wandb_enabled": true,
  "wandb_project": "fireredasr2-llm-sft",
  "wandb_entity": "your-team-or-user",
  "wandb_run_name": "domain-sft-r32",
  "wandb_group": "domain-experiments",
  "wandb_tags": ["fireredasr2", "r32"],
  "wandb_mode": "online",
  "wandb_watch": "none"
}
```

Only rank 0 creates and writes the W&B run. Logged charts include training
loss, gradient norm, throughput, every optimizer group's learning rate,
validation loss, MER/CER/WER, epoch duration, and trainable parameter counts.

Set `wandb_mode` to `offline` on machines without outbound network access and
upload later with `wandb sync`. `wandb_watch=gradients` or `all` additionally
tracks gradients/parameters, but is disabled by default because watching a
7B model adds substantial overhead.

The W&B run ID is stored in `output_dir/wandb_run_id.txt`; resuming training
from the same output directory automatically continues the same W&B run. An
explicit `wandb_run_id` can also be supplied in the config.

## 4. Run inference with an SFT checkpoint

```bash
python -m fireredasr2s.fireredasr2.speech2text \
  --asr_type llm \
  --model_dir pretrained_models/FireRedASR2-LLM \
  --sft_checkpoint exp/FireRedASR2-LLM-SFT/checkpoint-step-00000500 \
  --wav_path assets/hello_zh.wav
```

The released checkpoint is loaded first, followed by the compact SFT delta.

## Notes

- SFT targets use the repository's existing normalization: punctuation is
  removed, English is lowercased, Chinese spaces are removed, and English word
  spaces are retained.
- Validation loss uses teacher forcing. MER/CER/WER use autoregressive
  generation and are therefore the primary quality metrics.
- Standard DDP is provided. For full Qwen or full encoder fine-tuning, adding
  FSDP or DeepSpeed ZeRO is recommended.
- Checkpoints saved in the middle of an epoch resume from the stored batch.
  Data order is deterministic, but exact SpecAugment masks are not restored.
