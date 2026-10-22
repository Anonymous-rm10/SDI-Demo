#!/usr/bin/env python3
"""Split an ASR JSONL file using the level in each audio's TXT annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OUTPUT_NAMES = {
    "Light": "Light.jsonl",
    "Medium": "Medium.jsonl",
    # The annotations use "High". Keep the requested output spelling "Hight".
    "High": "Hight.jsonl",
    "Hight": "Hight.jsonl",
}


def get_level(annotation_path: Path) -> str:
    """Read the second tab-separated field from an annotation file."""
    fields = annotation_path.read_text(encoding="utf-8-sig").strip().split("\t")
    if len(fields) < 2:
        raise ValueError("标注内容不足两列")

    level = fields[1].strip()
    if level not in OUTPUT_NAMES:
        raise ValueError(f"未知等级 {level!r}")
    return level


def split_jsonl(input_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / name for name in set(OUTPUT_NAMES.values())
    }
    output_files = {
        name: path.open("w", encoding="utf-8")
        for name, path in output_paths.items()
    }
    counts = {name: 0 for name in output_paths}

    try:
        with input_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    wav_path = Path(item["wav_path"])
                    annotation_path = wav_path.with_suffix(".txt")
                    level = get_level(annotation_path)
                except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"处理第 {line_number} 行失败: {exc}"
                    ) from exc

                output_name = OUTPUT_NAMES[level]
                # Write the parsed object so every output line is valid JSON.
                output_files[output_name].write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                counts[output_name] += 1
    finally:
        for output_file in output_files.values():
            output_file.close()

    return counts


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="按照音频同名 TXT 标注的第二列拆分 JSONL 文件"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=script_dir / "all.jsonl",
        help="输入 JSONL（默认：脚本目录下的 all.jsonl）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=script_dir,
        help="输出目录（默认：脚本所在目录）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = split_jsonl(args.input.resolve(), args.output_dir.resolve())

    print("拆分完成：")
    for output_name in ("Light.jsonl", "Medium.jsonl", "Hight.jsonl"):
        output_path = args.output_dir.resolve() / output_name
        print(f"  {output_path}: {counts[output_name]} 条")


if __name__ == "__main__":
    main()
