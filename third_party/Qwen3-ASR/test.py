import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from qwen_asr import Qwen3ASRModel


MODEL_PATH = "/home/ln10/Workspace/models/Qwen3-ASR-0.6B-compassopsd-firered_rl-Southwestern/checkpoint-45000"
DEFAULT_INPUT_JSONL = (
    "/home/ln10/Workspace/datasets/kespeech/Tasks/ASR/jsonl/"
    "Southwestern.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    "/home/ln10/Workspace/code/Qwen3-ASR/"
    "Southwestern.jsonl"
)
BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-ASR inference on the SeniorTalk dialogue JSONL dataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Input JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def load_dataset(jsonl_path):
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {jsonl_path}")

    samples = []
    missing_audio = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}: {exc}") from exc

            wav_path = item.get("wav_path")
            if not wav_path:
                raise ValueError(f"Missing wav_path at {jsonl_path}:{line_number}")

            audio_path = Path(wav_path).expanduser()
            if not audio_path.is_absolute():
                audio_path = jsonl_path.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                missing_audio.append((line_number, str(audio_path)))

            # Use the normalized absolute path for inference while preserving all labels,
            # including the reference transcript.
            item["wav_path"] = str(audio_path)
            samples.append(item)

    if missing_audio:
        preview = "\n".join(
            f"  line {line_number}: {audio_path}"
            for line_number, audio_path in missing_audio[:10]
        )
        suffix = "" if len(missing_audio) <= 10 else f"\n  ... and {len(missing_audio) - 10} more"
        raise FileNotFoundError(
            f"Found {len(missing_audio)} missing audio files:\n{preview}{suffix}"
        )
    if not samples:
        raise ValueError(f"No samples found in {jsonl_path}")
    return samples


def save_result(output_file, item):
    output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    output_file.flush()


def transcribe_one(model, item):
    """Retry one sample after a failed batch so one bad file does not lose a full batch."""
    result = model.transcribe(audio=item["wav_path"], language=None)[0]
    item["prediction"] = result.text
    item["language"] = result.language
    item["error"] = ""


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    samples = load_dataset(args.input)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=256,
        dtype=torch.float16,
        # attn_implementation="flash_attention_2",
    )

    error_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for start in tqdm(
            range(0, len(samples), args.batch_size),
            desc="Transcribing SeniorTalk dialogue_test",
        ):
            batch = samples[start : start + args.batch_size]
            try:
                results = model.transcribe(
                    audio=[item["wav_path"] for item in batch],
                    language="Chinese",
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"Model returned {len(results)} results for a batch of {len(batch)}"
                    )
                for item, result in zip(batch, results):
                    item["prediction"] = result.text
                    item["language"] = result.language
                    item["error"] = ""
                    save_result(output_file, item)
            except Exception as batch_error:
                tqdm.write(
                    f"Batch starting at sample {start} failed; retrying one by one: "
                    f"{batch_error}"
                )
                for item in batch:
                    try:
                        transcribe_one(model, item)
                    except Exception as sample_error:
                        error_count += 1
                        item["prediction"] = ""
                        item["language"] = ""
                        item["error"] = str(sample_error)
                        tqdm.write(f"Failed {item.get('uttid', item['wav_path'])}: {sample_error}")
                    save_result(output_file, item)

    print(f"Done: {len(samples)} samples")
    print(f"Failed: {error_count} samples")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()


