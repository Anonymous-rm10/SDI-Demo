# coding=utf-8
# On-Policy Distillation (OPD) training for Qwen3-ASR.
# Student: Qwen3-ASR-0.6B  |  Teacher: Qwen3-ASR-1.7B (local, same process)
#
# OPD flow per training step:
#   1. Student samples ASR completions (on-policy).
#   2. Teacher scores those completions → produces logits.
#   3. Reverse KL(student || teacher) on completion tokens → backprop through student only.
#
# Reference: west/west/bin/train_knowledge_distillation.py  +  west/west/trainer/kd_trainer.py

import argparse
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (
    AutoProcessor,
    GenerationConfig,
    TrainerCallback,
    TrainingArguments,
)


# ════════════════════════════════════════════════════════════════════════════════
# 1. Data helpers  (mirrors qwen3_asr_sft.py)
# ════════════════════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    """Map a raw dataset row to the columns needed by OPDTrainer."""
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
    return _preprocess


@dataclass
class DataCollatorForASROPD:
    """
    Batch raw features into tensors for OPD.

    Output dict contains:
      - input_ids        : [batch, full_seq_len]  prompt + target tokens
      - attention_mask   : [batch, full_seq_len]
      - input_features   : [batch, n_mels, n_frames]
      - feature_attention_mask : [batch, n_frames]
      - meta_data        : (texts list, audios list)  kept for teacher input prep
    """
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_paths  = [f["audio"]     for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets      = [f["target"]    for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # mask prompt tokens from loss (prefix + audio BOS tokens)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels

        # meta_data: (target texts, audio arrays) — needed by OPDTrainer._rollout / teacher prep
        meta_data = (targets, audios)

        return {
            **full_inputs,
            "meta_data": meta_data,
            # Prefix-only inputs for rollout: generation starts from here (no ground truth)
            "prefix_input_ids": prefix_inputs["input_ids"],
            "prefix_attention_mask": prefix_inputs["attention_mask"],
            "prefix_input_features": prefix_inputs["input_features"],
            "prefix_feature_attention_mask": prefix_inputs["feature_attention_mask"],
        }


# ════════════════════════════════════════════════════════════════════════════════
# 2. Checkpoint callback  (mirrors qwen3_asr_sft.py)
# ════════════════════════════════════════════════════════════════════════════════

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def copy_required_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control


# ════════════════════════════════════════════════════════════════════════════════
# 3. Model patching
# ════════════════════════════════════════════════════════════════════════════════

def patch_forward(model):
    """Redirect model.forward() → model.thinker.forward() so it matches HF Trainer contract."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: no `.thinker.forward` found.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ════════════════════════════════════════════════════════════════════════════════
# 4. CLI args
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR On-Policy Distillation (Student: 0.6B, Teacher: 1.7B)")

    # Paths
    p.add_argument("--student_model_path", type=str, default="Qwen/Qwen3-ASR-0.6B",
                   help="Path to student model (0.6B)")
    p.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen3-ASR-1.7B",
                   help="Path to teacher model (1.7B)")
    p.add_argument("--train_file",   type=str, default="train.jsonl",
                   help="Training JSON/JSONL with fields: audio, text, optional prompt")
    p.add_argument("--eval_file",    type=str, default="",
                   help="Optional eval file")
    p.add_argument("--output_dir",   type=str, default="./qwen3-asr-opd-out")

    # OPD hyper-params
    p.add_argument("--topk_logits_k",    type=int,   default=None,
                   help="Top-k for logits distillation. None = full vocab.")
    p.add_argument("--temperature",        type=float, default=1.0,
                   help="Temperature applied before softmax in KL.")
    p.add_argument("--max_completion_length", type=int, default=256,
                   help="Max tokens for student on-policy sampling.")
    p.add_argument("--sampling_temperature", type=float, default=1.0,
                   help="Sampling temperature for student generation.")

    # Training
    p.add_argument("--batch_size",      type=int, default=4,    help="Per-device batch size")
    p.add_argument("--grad_acc",         type=int, default=4,    help="Gradient accumulation steps")
    p.add_argument("--lr",              type=float, default=1e-5, help="Learning rate")
    p.add_argument("--epochs",          type=float, default=1)
    p.add_argument("--log_steps",       type=int, default=10)
    p.add_argument("--save_steps",       type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--warmup_ratio",     type=float, default=0.02)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--max_steps",       type=int, default=-1,
                   help="Override epochs with max_steps (-1 = use epochs)")

    # DataLoader
    p.add_argument("--sr",               type=int, default=16000, help="Audio sample rate")
    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--pin_memory",        type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor",   type=int, default=2)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume",       type=int, default=0)

    # Misc
    p.add_argument("--report_to",  type=str, default="none",
                   choices=["none", "wandb", "tensorboard"],
                   help="Logging backend")
    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════════
# 5. main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    args_cli = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ── dtype ─────────────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info(f"Using dtype={dtype}, bf16={use_bf16}")

    # ── Load student model ───────────────────────────────────────────────────
    logger.info(f"Loading student model: {args_cli.student_model_path}")
    student_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.student_model_path,
        dtype=dtype,
        device_map=None,
    )
    student_model = student_wrapper.model
    student_processor = student_wrapper.processor
    patch_forward(student_model)
    student_model.generation_config = GenerationConfig.from_model_config(student_model.config)

    # ── Load teacher model (local, same process) ──────────────────────────────
    logger.info(f"Loading teacher model: {args_cli.teacher_model_path}")
    teacher_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.teacher_model_path,
        dtype=dtype,
        device_map=None,
    )
    teacher_model = teacher_wrapper.model
    patch_forward(teacher_model)
    # Teacher does NOT need generation_config (never sampled, only forwarded)

    # Move teacher to same device as student
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    teacher_model = teacher_model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    preprocess_fn = make_preprocess_fn(student_processor)
    ds = raw_ds.map(preprocess_fn, num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    logger.info(f"Train samples: {len(ds['train'])}")

    # ── Collator ─────────────────────────────────────────────────────────────
    collator = DataCollatorForASROPD(processor=student_processor, sampling_rate=args_cli.sr)

    # ── TrainingArguments ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        max_steps=args_cli.max_steps if args_cli.max_steps > 0 else -1,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        save_strategy="steps",
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="no" if not args_cli.eval_file else "steps",
        eval_steps=args_cli.save_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=args_cli.report_to,
        seed=args_cli.seed,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
    )

    # ── Build Trainer ─────────────────────────────────────────────────────────
    # Import OPDTrainer here to keep file structure clean
    from opd_trainer import OPDTrainer

    trainer = OPDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        processor=student_processor,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        topk_logits_k=args_cli.topk_logits_k,
        temperature=args_cli.temperature,
        max_completion_length=args_cli.max_completion_length,
    )

    trainer.add_callback(
        MakeEveryCheckpointInferableCallback(base_model_path=args_cli.student_model_path)
    )

    # ── Resume ─────────────────────────────────────────────────────────────────
    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    # ── Train ─────────────────────────────────────────────────────────────────
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
