# FireRedASR2-LLM teacher -> Qwen3-ASR-0.6B OPD

This training path freezes FireRedASR2-LLM and fully fine-tunes
Qwen3-ASR-0.6B.

The local model files were compared exhaustively. Their `vocab.json` files are
byte-identical, and all 151,646 common tokens have identical token IDs. Qwen3-ASR
adds 59 ASR/multimodal tokens; FireRed adds `<speech>` at the first ID after the
shared base. At startup the trainer repeats this mapping check and refuses to
train if the tokenizers do not match.

Each OPD step:

1. The fixed Qwen protocol `language ... <asr_text>` is appended to the prompt.
2. Qwen3-ASR samples only the transcript and its termination on-policy.
3. FireRed and Qwen score the exact same shared transcript token IDs, including
   the shared `<|im_end|>` when it terminates the response.
4. Reverse KL `KL(student || teacher)` is computed on those token positions.
5. The frozen teacher receives no gradients; all Qwen3-ASR parameters update.
6. Optional reference CE (`SFT_COEFFICIENT`, default `0.1`) stabilizes training.

Input JSONL accepts `audio` + `text` (standard Qwen OPD format), or
`wav_path` + `reference`.

## Run

Two GPUs are recommended because the frozen FireRed teacher contains a 7B LLM:

```bash
cd /home/ln10/Workspaces/code/FireRedASR2S
TEACHER_DEVICE=cuda:1 \
TRAIN_FILE=/home/ln10/Workspaces/datasets/SeniorTalk/seniortalk_opd.jsonl \
bash examples_train/qwen_opd_firered/run_opd.sh
```

If `TEACHER_DEVICE` is omitted, the launcher chooses `cuda:1` when at least two
GPUs are visible, otherwise `cuda:0`. A single GPU may run out of memory.

Useful overrides include `TEMPERATURE`, `TOPK` (empty means the full ordinary
shared-token vocabulary), `SFT_COEFFICIENT`, `MAX_ROLLOUT_RETRIES`,
`BATCH_SIZE`, `GRAD_ACC`, `LR`, `MAX_STEPS`, `EVAL_FILE`, `REPORT_TO`, and
`OUTPUT_DIR`. Checkpoints contain the complete Qwen3-ASR student and processor
files.

`LOG_ROLLOUT_TEXT=1` (the launcher default) logs every unmodified student
completion with special tokens retained, including each failed retry. Set it to
`0` after diagnosis to reduce log volume.

The OPD term covers ASR text and its termination. Qwen's fixed
`language ... <asr_text>` protocol is part of the student prompt, so malformed
protocol rollouts cannot consume or skip a micro-batch. Student and teacher prompts are used only as their
respective conditioning context; their positions never enter KL. The shared
`<|im_end|>` is retained as the final scored position and KL vocabulary
dimension so the student learns when to stop. PAD, other EOS variants,
chat/audio markers, and all other tokenizer special tokens remain excluded.

If a rollout contains no scoreable shared transcript or termination action, it
is retried `MAX_ROLLOUT_RETRIES` times (default 3). If every retry is still
invalid, that micro-batch is skipped without substituting the reference
transcript, preserving the on-policy objective. Logs report
`empty_rollout_rate` and `rollout_retries`; the final failed completion and
utterance ID are written as a warning for diagnosis.
