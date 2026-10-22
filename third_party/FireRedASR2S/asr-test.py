#!/usr/bin/env python3
"""Evaluate FireRedASR2-LLM on the processed SeniorTalk ASR test set."""

import json
import os
import re
from pathlib import Path

# Match inference_asr_llm.sh. This must be set before importing PyTorch/model code.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cn2an
import kaldiio
import numpy as np
from tqdm import tqdm

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredasr2.utils.wer import (
    WerStats,
    compute_one_wer_info,
    text2tokens,
)


MODEL_DIR = Path("/home/ln10/Workspaces/models/FireRedASR2-LLM")
OUTPUT_ROOT = Path("/home/ln10/Workspaces/code/FireRedASR2S/out/seniortalk_rl_18835")
DATASETS = [
    {
        "name": "WildElder_test",
        "path": Path(
            "/home/ln10/Workspaces/datasets/WildElder/"
            "test.jsonl"
        ),
    },
    {
        "name": "WildElder_validation",
        "path": Path(
            "/home/ln10/Workspaces/datasets/WildElder/"
            "validation.jsonl"
        ),
    },
]

# Decode parameters copied from examples_infer/asr/inference_asr_llm.sh.
BATCH_SIZE = 1
ASR_CONFIG = FireRedAsr2Config(
    use_gpu=True,
    use_half=False,
    beam_size=3,
    decode_max_len=0,
    decode_min_len=0,
    repetition_penalty=3.0,
    llm_length_penalty=1.0,
    temperature=1.0,
    sft_checkpoint=(
        "/home/ln10/Workspaces/code/FireRedASR2S/exp/"
        "FireRedASR2-LLM-RL/checkpoint-step-00018835"
    ),
)


def load_and_validate_dataset(path: Path) -> list[dict]:
    """Load JSONL and validate all fields needed by the ASR evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []
    seen_uttids = set()
    required_fields = {"uttid", "reference", "wav_path"}

    with path.open("r", encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

            missing = required_fields - sample.keys()
            if missing:
                raise ValueError(
                    f"Missing fields {sorted(missing)} at {path}:{line_number}"
                )

            uttid = sample["uttid"]
            if not isinstance(uttid, str) or not uttid:
                raise ValueError(f"Invalid uttid at {path}:{line_number}")
            if uttid in seen_uttids:
                raise ValueError(f"Duplicate uttid at {path}:{line_number}: {uttid}")
            seen_uttids.add(uttid)

            reference = sample["reference"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Invalid reference at {path}:{line_number}")

            wav_path = Path(sample["wav_path"])
            if not wav_path.is_file():
                raise FileNotFoundError(
                    f"Missing audio at {path}:{line_number}: {wav_path}"
                )

            samples.append(sample)

    if not samples:
        raise ValueError(f"Dataset contains no valid samples: {path}")
    return samples


def normalize_for_scoring(text: str) -> str:
    """Apply the same TN and special-token removal as the example WER command."""
    text = " ".join(
        part for part in re.split(r"<\|.*?\|>", text) if part.strip()
    )
    return cn2an.transform(text, "an2cn")


def evaluate_batch(
    model: FireRedAsr2,
    batch: list[dict],
) -> dict[str, dict]:
    uttids = [sample["uttid"] for sample in batch]
    audio_inputs = [load_audio_for_model(sample["wav_path"]) for sample in batch]
    results = model.transcribe(uttids, audio_inputs)
    return {result["uttid"]: result for result in results}


def load_audio_for_model(wav_path: str) -> tuple[int, np.ndarray]:
    """Load PCM audio and downmix multichannel input for FireRedASR2."""
    sample_rate, waveform = kaldiio.load_mat(wav_path)
    waveform = np.asarray(waveform)

    if waveform.ndim == 2:
        if waveform.shape[1] < 1:
            raise ValueError(f"Audio has no channels: {wav_path}")
        # Convert before averaging so int16 channel addition cannot overflow.
        waveform = np.rint(
            waveform.astype(np.float32).mean(axis=1)
        ).astype(np.int16)
    elif waveform.ndim != 1:
        raise ValueError(
            f"Expected mono or multichannel PCM audio, got shape "
            f"{waveform.shape}: {wav_path}"
        )

    if sample_rate != 16000:
        raise ValueError(
            f"FireRedASR2 expects 16 kHz audio, got {sample_rate} Hz: {wav_path}"
        )
    if waveform.size == 0:
        raise ValueError(f"Audio is empty: {wav_path}")

    return sample_rate, np.ascontiguousarray(waveform)


def evaluate_dataset(
    model: FireRedAsr2,
    dataset_name: str,
    dataset_path: Path,
) -> None:
    """Run one complete dataset before returning to the caller."""
    samples = load_and_validate_dataset(dataset_path)
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Validated {len(samples):,} samples from {dataset_path}")

    output_dir = OUTPUT_ROOT / dataset_name
    prediction_path = output_dir / "fireredasr2_llm_predictions.jsonl"
    reference_path = output_dir / "reference.txt"
    hypothesis_path = output_dir / "hypothesis.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    wer_stats = WerStats()
    empty_predictions = 0

    with (
        prediction_path.open("w", encoding="utf-8") as prediction_file,
        reference_path.open("w", encoding="utf-8") as reference_file,
        hypothesis_path.open("w", encoding="utf-8") as hypothesis_file,
    ):
        for start in tqdm(
            range(0, len(samples), BATCH_SIZE),
            desc=f"FireRedASR2-LLM/{dataset_name}",
        ):
            batch = samples[start : start + BATCH_SIZE]
            result_by_uttid = evaluate_batch(model, batch)

            for sample in batch:
                uttid = sample["uttid"]
                result = result_by_uttid.get(uttid, {})
                prediction = result.get("text", "")
                if not prediction:
                    empty_predictions += 1

                output_record = {
                    **sample,
                    "prediction": prediction,
                    "rtf": result.get("rtf"),
                    "model": "FireRedASR2-LLM",
                    "dataset": dataset_name,
                }
                prediction_file.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )
                reference_file.write(f"{uttid}\t{sample['reference']}\n")
                hypothesis_file.write(f"{uttid}\t{prediction}\n")

                reference_tokens = text2tokens(
                    normalize_for_scoring(sample["reference"])
                )
                prediction_tokens = text2tokens(
                    normalize_for_scoring(prediction)
                )
                wer_stats.add(
                    compute_one_wer_info(reference_tokens, prediction_tokens)
                )

            prediction_file.flush()
            reference_file.flush()
            hypothesis_file.flush()

    print(f"Finished dataset: {dataset_name}")
    print(f"Predictions: {prediction_path}")
    print(f"References:  {reference_path}")
    print(f"Hypotheses:  {hypothesis_path}")
    print(f"Empty predictions: {empty_predictions}")
    wer_stats.print()


def main() -> None:
    print(f"Decode config: {ASR_CONFIG}")
    print("Dataset execution order:")
    for index, dataset in enumerate(DATASETS, start=1):
        print(f"  {index}. {dataset['name']}: {dataset['path']}")

    model = FireRedAsr2.from_pretrained("llm", str(MODEL_DIR), ASR_CONFIG)

    # This loop is deliberately sequential: dialogue_test must finish before
    # sentence_test starts. The loaded model is reused for both datasets.
    for dataset in DATASETS:
        evaluate_dataset(model, dataset["name"], dataset["path"])

    print("All datasets finished.")


if __name__ == "__main__":
    main()





