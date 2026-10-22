#!/usr/bin/env python3
"""Generate FireRedASR2-LLM pseudo-labels for the KeSpeech train_phase1 set.

Optimised for throughput over a very large set (~540k utterances):
  * real batching (transcribe() is natively batched)
  * duration-sorted buckets so each batch pads to a similar length
  * fp16/bf16 inference (use_half=True)
  * resume support: already-predicted uttids are skipped on restart
"""

import json
import os
from pathlib import Path

# Match inference_asr_llm.sh. This must be set before importing PyTorch/model code.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import kaldiio
import numpy as np
from tqdm import tqdm

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config


MODEL_DIR = Path("/home/ln10/Workspaces/models/FireRedASR2-LLM")
OUTPUT_ROOT = Path("/home/ln10/Workspaces/code/FireRedASR2S/out/kespeech_rl_18835")
DATASETS = [
    {
        "name": "kespeech_train_phase1",
        "path": Path(
            "/home/ln10/Workspaces/datasets/kespeech/train_phase1.jsonl"
        ),
    },
]

# Number of utterances per forward pass. KeSpeech clips are short (~5 s), so a
# moderate batch fits comfortably; lower this if you hit GPU OOM on the Orin.
BATCH_SIZE = 16
# Flush the prediction file every N batches so a crash loses little progress.
FLUSH_EVERY = 20

# Decode parameters copied from examples_infer/asr/inference_asr_llm.sh.
# use_half=True enables bf16/fp16 for ~2x throughput on the Orin.
ASR_CONFIG = FireRedAsr2Config(
    use_gpu=True,
    use_half=True,
    beam_size=1,
    decode_max_len=0,
    decode_min_len=0,
    repetition_penalty=3.0,
    llm_length_penalty=1.0,
    temperature=1.0,
    sft_checkpoint=(""
    ),
)


def load_dataset(path: Path) -> list[dict]:
    """Stream JSONL and validate the fields needed for inference.

    File existence is checked lazily at audio-load time (a 540k-file stat storm
    at startup is slow and pointless when resume already skips finished ones).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []
    seen_uttids = set()
    required_fields = {"uttid", "text", "wav"}

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

            reference = sample["text"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Invalid text at {path}:{line_number}")

            samples.append(sample)

    if not samples:
        raise ValueError(f"Dataset contains no valid samples: {path}")
    return samples


def load_completed_uttids(prediction_path: Path) -> set[str]:
    """Collect uttids already written so a restart can skip them."""
    done: set[str] = set()
    if not prediction_path.is_file():
        return done
    with prediction_path.open("r", encoding="utf-8") as existing:
        for line in existing:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A torn last line from a previous crash: ignore, it will be
                # re-predicted and simply appended again.
                continue
            uttid = record.get("uttid")
            if uttid:
                done.add(uttid)
    return done


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
    """Run one complete dataset, appending pseudo-labels as they are produced."""
    samples = load_dataset(dataset_path)
    output_dir = OUTPUT_ROOT / dataset_name
    prediction_path = output_dir / "fireredasr2_llm_predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    done = load_completed_uttids(prediction_path)
    pending = [s for s in samples if s["uttid"] not in done]
    # Duration-sorted buckets minimise intra-batch padding (extractor pads to the
    # longest clip in the batch). Missing durations sort last.
    pending.sort(key=lambda s: s.get("duration") or float("inf"))

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Total samples:      {len(samples):,}")
    print(f"Already completed:  {len(done):,}")
    print(f"Pending this run:   {len(pending):,}")
    print(f"Predictions -> {prediction_path}")

    if not pending:
        print("Nothing to do; dataset already complete.")
        return

    empty_predictions = 0
    load_failures = 0

    with prediction_path.open("a", encoding="utf-8") as prediction_file:
        batches = range(0, len(pending), BATCH_SIZE)
        for batch_index, start in enumerate(
            tqdm(batches, desc=f"FireRedASR2-LLM/{dataset_name}")
        ):
            batch = pending[start : start + BATCH_SIZE]

            # Load audio per sample; skip (and record) unreadable/missing files
            # so one bad clip cannot abort the whole batch or the whole run.
            good_uttids, good_audio, good_samples = [], [], []
            for sample in batch:
                try:
                    good_audio.append(load_audio_for_model(sample["wav"]))
                    good_uttids.append(sample["uttid"])
                    good_samples.append(sample)
                except (OSError, ValueError) as exc:
                    load_failures += 1
                    empty_predictions += 1
                    prediction_file.write(
                        json.dumps(
                            {
                                **sample,
                                "prediction": "",
                                "rtf": None,
                                "error": str(exc),
                                "model": "FireRedASR2-LLM",
                                "dataset": dataset_name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if good_uttids:
                results = model.transcribe(good_uttids, good_audio)
                result_by_uttid = {r["uttid"]: r for r in results}

                for sample in good_samples:
                    uttid = sample["uttid"]
                    result = result_by_uttid.get(uttid, {})
                    prediction = result.get("text", "")
                    if not prediction:
                        empty_predictions += 1
                    prediction_file.write(
                        json.dumps(
                            {
                                **sample,
                                "prediction": prediction,
                                "rtf": result.get("rtf"),
                                "model": "FireRedASR2-LLM",
                                "dataset": dataset_name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if batch_index % FLUSH_EVERY == 0:
                prediction_file.flush()

    print(f"Finished dataset: {dataset_name}")
    print(f"Empty predictions: {empty_predictions}")
    print(f"Audio load failures: {load_failures}")


def main() -> None:
    print(f"Decode config: {ASR_CONFIG}")
    print("Dataset execution order:")
    for index, dataset in enumerate(DATASETS, start=1):
        print(f"  {index}. {dataset['name']}: {dataset['path']}")

    model = FireRedAsr2.from_pretrained("llm", str(MODEL_DIR), ASR_CONFIG)

    for dataset in DATASETS:
        evaluate_dataset(model, dataset["name"], dataset["path"])

    print("All datasets finished.")


if __name__ == "__main__":
    main()
