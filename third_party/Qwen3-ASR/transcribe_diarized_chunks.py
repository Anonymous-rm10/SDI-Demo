#!/usr/bin/env python3
"""Batch-transcribe diarized meeting chunks and preserve speaker timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR.parent
    / "datawork"
    / "pyannote"
    / "out"
    / "asr_chunks"
    / "manifest.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-ASR over a pyannote chunk manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to transcription.jsonl beside the selected manifest.",
    )
    parser.add_argument(
        "--model",
        default="/home/ln10/Workspaces/models/Qwen3-ASR-1.7B",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Use 'auto' for per-chunk language detection.",
    )
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.manifest.parent / "transcription.jsonl"
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def batched(records: list[dict], size: int) -> Iterable[list[dict]]:
    for offset in range(0, len(records), size):
        yield records[offset : offset + size]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    manifest = load_jsonl(args.manifest)
    completed: set[str] = set()
    if args.output.exists():
        completed = {
            record["chunk_id"]
            for record in load_jsonl(args.output)
            if "chunk_id" in record
        }
        print(f"Resuming: {len(completed)} chunks already exist in {args.output}")

    pending = [record for record in manifest if record["chunk_id"] not in completed]
    if not pending:
        print("No pending chunks.")
        return

    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, args.dtype)
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attention,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    language = None if args.language.strip().lower() == "auto" else args.language
    with args.output.open("a", encoding="utf-8") as target:
        processed = len(completed)
        for batch in batched(pending, args.batch_size):
            audio_paths = [
                str((args.manifest.parent / record["audio_path"]).resolve())
                for record in batch
            ]
            results = model.transcribe(
                audio=audio_paths,
                language=language,
                context=args.context,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Model returned {len(results)} results for a batch of {len(batch)}."
                )

            for source, result in zip(batch, results):
                output = {
                    "chunk_id": source["chunk_id"],
                    "speaker_id": source["speaker_id"],
                    "start": source["start"],
                    "end": source["end"],
                    "audio_path": source["audio_path"],
                    "has_overlap": source.get("has_overlap", False),
                    "overlap_duration": source.get("overlap_duration", 0.0),
                    "language": result.language,
                    "text": result.text,
                }
                target.write(json.dumps(output, ensure_ascii=False) + "\n")
            target.flush()
            processed += len(batch)
            print(f"Transcribed {processed}/{len(manifest)} chunks")

    print(f"Transcription: {args.output}")


if __name__ == "__main__":
    main()
