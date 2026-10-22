# FireRedASR2-LLM group-relative RL

This training path applies sequence-level reinforcement learning to an SFT or
released FireRedASR2-LLM policy. For every audio prompt it samples a group of
transcripts, computes normalized MER rewards plus an optional JEV semantic
fidelity penalty, constructs group-relative advantages, and
optimizes the policy LoRA with a KL penalty to a frozen copy of the initial
LoRA.

The first implementation intentionally freezes the Conformer encoder, speech
adapter, Qwen base, and reference LoRA. Only the policy LoRA is updated. The
policy and reference adapters share one Qwen base, so RL does not load a second
7B model.

## 1. Data

The manifest format is the same JSONL format used by continued SFT:

```json
{"uttid":"utt001","wav":"/data/audio/utt001.wav","text":"你好世界","duration":2.35}
```

The reference text is never inserted into the rollout prompt. It is used only
after generation to calculate MER and JEV rewards.

## 2. Configuration

Copy `rl_config.json` and set:

- `model_dir`
- `train_manifest`
- `valid_manifest`
- `output_dir`
- `init_checkpoint`

`init_checkpoint` should point to an SFT checkpoint directory or its
`training_state.pt`. Set it to an empty string to start RL directly from the
released checkpoint. Its LoRA rank, alpha, dropout, and target modules must
match the RL configuration.

The main RL controls are:

```json
{
  "num_generations": 4,
  "rollout_temperature": 0.8,
  "rollout_top_p": 0.95,
  "reward_alpha": 2.0,
  "exact_match_bonus": 0.1,
  "repetition_penalty_weight": 0.1,
  "semantic_reward_enabled": true,
  "semantic_api_url": "http://127.0.0.1:8020/v1/systemone",
  "semantic_expected_model": "jev-4b",
  "semantic_reward_weight": 0.1,
  "semantic_batch_size": 4,
  "semantic_timeout_seconds": 120.0,
  "semantic_cache_size": 8192,
  "kl_beta": 0.02
}
```

The bounded ASR reward is `exp(-reward_alpha * MER)`. An exact-match bonus is
added, while missing EOS and repeated n-grams are penalized. Advantages are
normalized independently inside each group. A group with identical rewards
receives zero advantage and contributes only the KL term.

When semantic reward is enabled, training calls the separately deployed
JEV System One service. The old embedding model is no longer loaded. Each
request contains the full reference/hypothesis pair and one `score` question
with exactly six levels, selecting the native JEV score head:

| Level | Fidelity |
| --- | --- |
| 0 | Unrelated content or reversal of the overall main assertion |
| 1 | Only fragments remain; most information is wrong or missing |
| 2 | The central event remains, but multiple independent key facts are wrong |
| 3 | One key fact is wrong, added, or missing, or one substantive clause is missing |
| 4 | All key facts remain correct; only minor non-key meaning is lost |
| 5 | All effective information is equivalent, including harmless formatting or disfluency changes |

The versioned Chinese rubric is defined in
`fireredasr2s/fireredasr2/rl/semantic_rubric.py`. Names, quantities, dates,
negation, conditions, and certainty are key facts. Added unsupported facts
are penalized as well as omissions. The model compares against the reference;
it does not correct labels using outside knowledge. No transcript truncation
is performed by the client.

The expected ordinal level is normalized to `[0, 1]`. It is a reward signal,
not a calibrated probability of a correct transcription:

```text
semantic_score = sum(level * probability[level] for level in range(6)) / 5
semantic_distance = 1 - semantic_score
reward = existing_reward - semantic_reward_weight * semantic_distance
```

The MER reward, exact-match bonus, EOS penalty, repetition penalty, and GRPO
loss are retained. Set `semantic_reward_enabled=false` for MER-only rewards.
Keep `semantic_reward_weight` small initially (the example uses `0.1`): JEV
can still miss errors, particularly omitted content. Its auxiliary Boolean
judgments are not added as additional rewards.

Start the JEV server in its own compatible environment before training. For
example, from a `train-jev` checkout:

```bash
python -m trainjev.serve \
  --base Qwen/Qwen3.5-4B \
  --adapter shibing624/train-jev-4b-lora \
  --name jev-4b --port 8020
```

Use your existing local base/adapter paths instead of repository IDs when
the checkpoint is already downloaded. No JEV model or extra serving packages
are installed into the ASR training environment. Reserve appropriate GPU
capacity for the separate server; each training rank uses only an HTTP client.
Keep the server checkpoint, calibration, and rubric fixed throughout a run.
`semantic_expected_model` checks the response model name, not a weight hash;
do not swap checkpoints behind the same name during training.

From the FireRedASR2S root, verify the actual configured service and reward
calculation without loading the ASR model:

```bash
python -m fireredasr2s.fireredasr2.rl.semantic_reward \
  --config examples_train/llm_rl/rl_config.json
```

This prints rewards for an identical transcript, a year substitution, and an
unrelated transcript. `semantic_batch_size` limits concurrent HTTP requests
**per rank**, not GPU tensor batch size. Duplicate pairs within a rollout are
scored once. `semantic_cache_size` bounds a per-rank LRU cache keyed by both
full texts; `0` disables cross-rollout caching. Calls connect directly to the
configured endpoint without environment HTTP proxies. HTTP errors, timeouts,
model-name mismatches, and malformed probability distributions abort the
rollout before policy optimization; there is no silent zero-reward fallback.
Increase the timeout or reduce concurrency if the shared server is overloaded.

Migration: remove the former `semantic_model_dir`, `semantic_max_length`,
`semantic_dtype`, `semantic_use_flash_attention`, `semantic_instruction`, and
`semantic_cache_references` settings from old configuration files and use the
service settings above. Unknown settings are rejected rather than ignored.

RL memory grows approximately linearly with `num_generations`. Reduce
`max_batch_size`, `max_batch_frames`, `num_generations`, or
`rollout_max_new_tokens` when CUDA runs out of memory. The provided config uses
one audio per GPU and four candidates as a conservative starting point.

## 3. Train

Single GPU:

```bash
NUM_GPUS=1 bash examples_train/llm_rl/run_rl.sh \
  examples_train/llm_rl/rl_config.json
```

Multi-GPU:

```bash
NUM_GPUS=8 bash examples_train/llm_rl/run_rl.sh \
  examples_train/llm_rl/rl_config.json
```

Each rank generates rollouts for its local duration-bucketed batches. LoRA
gradients are averaged across ranks before every optimizer update.

Resume an RL run:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fireredasr2s.fireredasr2.train_rl \
  --config examples_train/llm_rl/rl_config.json \
  --resume_from exp/FireRedASR2-LLM-RL/checkpoint-step-00000100
```

Do not place an RL checkpoint in `init_checkpoint`; use `resume_from` for RL
checkpoints so optimizer, scheduler, sampler position, and RNG state are also
restored.

## 4. Inference

RL checkpoints use the same compact trainable-delta layout as SFT checkpoints:

```bash
python -m fireredasr2s.fireredasr2.speech2text \
  --asr_type llm \
  --model_dir pretrained_models/FireRedASR2-LLM \
  --sft_checkpoint exp/FireRedASR2-LLM-RL/checkpoint-step-00000100 \
  --wav_path assets/hello_zh.wav
```

The option keeps the historical name `--sft_checkpoint`, but accepts both SFT
and RL trainable-delta formats.

## 5. Metrics and stability

Training logs include reward, sampled MER, exact-match rate, group reward
standard deviation, zero-variance group rate, EOS rate, repetition rate, KL,
JEV `semantic_score` (0–1), distance/penalty when enabled, response length, gradient
norm, and learning rate. Validation continues to use greedy generation and
reports MER/CER/WER.

If `zero_variance_group_rate` remains high, increase rollout temperature
slightly. If validation MER degrades while training reward rises, increase
`kl_beta`, reduce the learning rate, and check reward normalization and data
balance. Select the final checkpoint by held-out validation MER, not training
reward.

Logged rollout samples now contain `semantic_scores` instead of the old
`semantic_similarities` field. Existing checkpoints keep their policy/optimizer
format, but switching reward models changes the training objective; use a new
run/output directory when comparing against the embedding reward baseline.
