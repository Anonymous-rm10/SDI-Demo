# Training and standalone model commands

All four local source projects are included under `third_party/`, including their training code, recipes, configurations, preprocessing tools, and local training extensions. Matcha-TTS is bundled as a regular directory, including its training code and Cython alignment extension. No Git submodules are required.

Weights, training/evaluation datasets, checkpoints, logs, and run outputs are not included. Prepare them separately and set recipe arguments or environment variables before starting a job. The local copies include working-tree changes; the manifest identifies these separately from the recorded upstream commit.

## Source inventory

| Model | Inference | Training and data preparation |
| --- | --- | --- |
| Qwen3-ASR | `qwen_asr/`, `examples/` | `finetuning/` SFT; `opd/` OPD and OPSD with trainer/data helpers |
| CosyVoice | `cosyvoice/cli/`, `example.py`, `runtime/` | `cosyvoice/bin/train.py`, datasets, executors, preprocessing tools; LibriTTS CosyVoice 1/2/3 recipes; CosyVoice2 DPO and GRPO |
| train-jev | `trainjev/serve.py`, `examples/` | LoRA training, full-parameter/FSDP2 training, resume, calibration/evaluation, data preparation, 4B/27B launch scripts |
| FireRedASR2S | `fireredasr2s/`, `examples_infer/`, `runtime/` | `examples_train/` and ASR SFT/RL modules; FireRed teacher → Qwen OPD/OPSD; CompassOPD/CompassOPSD; semantic reward and associated tests |
| Matcha-TTS | Flow, decoder, and synthesis components | Full Matcha trainer, configs, preprocessing, and `monotonic_align` extension |

The complete package implementations are restored: training exports in train-jev and Matcha-TTS, ASR CLI/vLLM support, and original training dependency lists are available again. The voice gateway uses the same Qwen-ASR/LLM-API/CosyVoice path as before.

## Shared launcher

Run the launcher with the Python interpreter of the selected model environment:

```bash
.venv-asr/bin/python scripts/model.py --dry-run qwen3-asr -- python finetuning/qwen3_asr_sft.py --help
.venv-jev/bin/python scripts/model.py train-jev -- python -m trainjev.train --help
.venv-firered/bin/python scripts/model.py fireredasr2s -- python -m fireredasr2s.fireredasr2.train_sft --help
```

The launcher changes to the selected model directory and sets `PYTHONPATH` to bundled source roots. It maps `python` to its own interpreter and `torchrun` to that interpreter's `torch.distributed.run`. It also selects the same environment for nested recipe commands and exposes the bundled Qwen source to cross-model distillation.

`--dry-run` prints command details without importing a model or launching a job. `--workdir` selects a recipe subdirectory relative to the chosen model. Place launcher options before the component name. The component names are `qwen3-asr`, `cosyvoice`, `train-jev`, `fireredasr2s`, and `matcha-tts`.

## Environment preparation

Keep the gateway and model environments separate. Install hardware-compatible Torch/Torchaudio/CUDA builds before resolving training dependencies. Training is not supported by the lightweight gateway environment alone.

- **Qwen3-ASR:** install `third_party/Qwen3-ASR` in editable mode, plus the dependencies declared/imported by the chosen SFT or distillation trainer (including `datasets`, and `trl` where used). Its package pins Transformers 4.57.6.
- **CosyVoice:** use the restored `third_party/CosyVoice/requirements.txt`, which includes training and acceleration dependencies. Install/build Matcha-TTS when training requires its Cython monotonic alignment extension; its full `setup.py`, `pyproject.toml`, and source are bundled. Review CUDA-dependent packages for the target platform.
- **train-jev:** use `third_party/train-jev/requirements.txt`; `requirements-tested.txt` records the upstream reference environment. LoRA and full-parameter code use the same package, with different distributed requirements.
- **FireRedASR2S:** use a dedicated environment, for example `python3.11 -m venv .venv-firered`, then install the copied project in editable mode. Its local `pyproject.toml` and legacy `requirements.txt` specify different Torch/Transformers stacks. Choose a compatible stack for the recipe rather than combining both blindly. Cross-model Qwen distillation also needs the bundled Qwen package and a stack compatible with that trainer; consult the copied recipe README before installing.

No model training dependencies are installed by this repository's gateway setup or CI.

## Qwen3-ASR

SFT wrapper with portable paths, retaining the original training options:

```bash
MODEL_PATH="$PWD/models/Qwen3-ASR-0.6B" \
TRAIN_FILE="$PWD/datasets/train.jsonl" \
OUTPUT_DIR="$PWD/outputs/qwen-sft" \
.venv-asr/bin/python scripts/model.py qwen3-asr -- bash finetuning/run.sh
```

OPD/OPSD entry points and recipes:

```bash
.venv-asr/bin/python scripts/model.py qwen3-asr -- python opd/train_opd.py --help
.venv-asr/bin/python scripts/model.py qwen3-asr -- python opd/train_opsd.py --help
# Configure STUDENT_MODEL_DIR, TEACHER_MODEL_DIR, TRAIN_FILE, and OUTPUT_DIR first.
.venv-asr/bin/python scripts/model.py qwen3-asr -- bash opd/run_opd.sh
.venv-asr/bin/python scripts/model.py qwen3-asr -- bash opd/run_opsd.sh
```

The launch scripts now locate sources inside this repository and use the selected Python environment. Training inputs remain user supplied; default dataset paths are examples.

## CosyVoice and Matcha-TTS

The original staged recipe scripts control data preparation, speaker embeddings, speech tokens, and training. Review and set their stage limits and data/model paths before execution; some early stages download public corpora.

```bash
.venv-tts/bin/python scripts/model.py --workdir examples/libritts/cosyvoice3 cosyvoice -- bash run.sh
.venv-tts/bin/python scripts/model.py --workdir examples/libritts/cosyvoice2 cosyvoice -- bash run_dpo.sh
.venv-tts/bin/python scripts/model.py --workdir examples/grpo/cosyvoice2 cosyvoice -- bash run.sh
.venv-tts/bin/python scripts/model.py cosyvoice -- python -m cosyvoice.bin.train --help
```

Internal recipe symlinks are materialized as regular files/directories so they can be uploaded to GitHub without submodules or external links. Their source paths are recorded in the manifest.

## train-jev

```bash
.venv-jev/bin/python scripts/model.py train-jev -- python -m trainjev.train --help
.venv-jev/bin/python scripts/model.py train-jev -- python -m trainjev.train_full --help
# After preparing data and selecting GPUS, BASE, DATA, and OUT:
.venv-jev/bin/python scripts/model.py train-jev -- bash scripts/run_train.sh
.venv-jev/bin/python scripts/model.py train-jev -- bash scripts/run_train_27b.sh
```

`scripts/prepare_data.sh` downloads training data and model weights when run. It is included but not executed during repository setup. Training and full-parameter details remain in `third_party/train-jev/docs/training.en.md`.

## FireRedASR2S

SFT/RL example configurations use paths relative to the FireRed component root, pointing to the main repository's `models/` and `datasets/` directories. Edit them for your actual input manifests and checkpoints.

```bash
.venv-firered/bin/python scripts/model.py fireredasr2s -- torchrun --standalone --nproc_per_node 1 -m fireredasr2s.fireredasr2.train_sft --config examples_train/llm_sft/sft_config.json
.venv-firered/bin/python scripts/model.py fireredasr2s -- torchrun --standalone --nproc_per_node 1 -m fireredasr2s.fireredasr2.train_rl --config examples_train/llm_rl/rl_config.json
```

FireRed teacher → Qwen student recipes:

```bash
.venv-asr/bin/python scripts/model.py fireredasr2s -- bash examples_train/qwen_opd_firered/run_opd.sh
.venv-asr/bin/python scripts/model.py fireredasr2s -- bash examples_train/qwen_opsd_firered/run_opsd.sh
.venv-asr/bin/python scripts/model.py fireredasr2s -- bash examples_train/qwen_compassopd_firered/run_compassopd.sh
.venv-asr/bin/python scripts/model.py fireredasr2s -- bash examples_train/qwen_compassopsd_firered/run_compassopsd.sh
```

These scripts now use bundled source paths and the chosen interpreter. `TEACHER_CHECKPOINT` defaults to empty: set it explicitly to distill from a trained RL teacher rather than silently using an old developer checkpoint. Read the recipe-specific teacher/reference options before training.

Standalone inference uses the canonical package, for example:

```bash
.venv-firered/bin/python scripts/model.py fireredasr2s -- python -m fireredasr2s.fireredasr2.speech2text --help
.venv-firered/bin/python scripts/model.py fireredasr2s -- python -m fireredasr2s.fireredasr2s_cli --help
```

FireRed ASR/AED, VAD, language identification, punctuation, system inference, and Triton runtime sources are included. Example scripts need separately prepared weights and sample audio; upstream sample archives and pretrained directories are not copied.

## Validation scope

The repository validates training source coverage, local import targets, Python/shell syntax, portable launcher behavior, and the existing voice gateway's mock integration. Copying these trainers does not mean their GPU training jobs have been run. Datasets, multi-GPU hardware, compiled training extensions, and model-specific dependency resolution remain prerequisites for actual training.
