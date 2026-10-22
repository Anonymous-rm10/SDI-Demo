# coding=utf-8
# On-Policy Self-Distillation (OPSD) training for Qwen3-ASR.
# Student: Qwen3-ASR (trainable, e.g. 0.6B)  |  Teacher: Qwen3-ASR (frozen, e.g. 1.7B)
#
# OPSD flow per training step:
#   1. Student samples ASR completions (on-policy) from its OWN prefix — no reference.
#   2. Frozen teacher scores those completions, conditioned on a teacher prefix that
#      carries the reference transcript as privileged context.
#   3. JSD(student || teacher) on completion tokens -> backprop through student only.
#
# Teacher prefix: reference is injected into the SYSTEM message (the Qwen3-ASR chat
# template drops text in the user message). See opsd_data_utils.

import argparse
import logging
import os
import re
import shutil

import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, TrainerCallback, TrainingArguments

from opsd_data_utils import DataCollatorForASROPSD, make_opsd_preprocess_fn
from opsd_trainer import OPSDTrainer, patch_outer_forward


# ── checkpoint helpers ─────────────────────────────────────────────────────────

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir):
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


def copy_required_hf_files_src(src_dir, dst_dir):
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
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        copy_required_hf_files_src(self.base_model_path, ckpt_dir)
        return control


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR On-Policy Self-Distillation (Student: 0.6B, Teacher: 1.7B)")
    p.add_argument("--student_model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-opsd-out")

    # OPSD hyper-params
    p.add_argument("--beta", type=float, default=0.5, help="JSD interpolation (0=fwd KL, 1=rev KL, 0.5=JSD)")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature before softmax in JSD/KL.")
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--topk_logits_k", type=int, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--max_steps", type=int, default=-1)

    # DataLoader
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Resume / misc
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args_cli = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info(f"Using dtype={dtype}, bf16={use_bf16}")

    # ── Load student model ─────────────────────────────────────────────────────
    logger.info(f"Loading student model: {args_cli.student_model_path}")
    student_wrapper = Qwen3ASRModel.from_pretrained(args_cli.student_model_path, dtype=dtype, device_map=None)
    student_model = student_wrapper.model
    student_processor = student_wrapper.processor
    patch_outer_forward(student_model)
    student_model.generation_config = GenerationConfig.from_model_config(student_model.config)

    # ── Load teacher model (local, same process, frozen) ───────────────────────
    logger.info(f"Loading teacher model: {args_cli.teacher_model_path}")
    teacher_wrapper = Qwen3ASRModel.from_pretrained(args_cli.teacher_model_path, dtype=dtype, device_map=None)
    teacher_model = teacher_wrapper.model
    patch_outer_forward(teacher_model)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_ds = load_dataset("json", data_files={
        "train": args_cli.train_file,
        **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
    })
    preprocess_fn = make_opsd_preprocess_fn(student_processor)
    ds = raw_ds.map(preprocess_fn, num_proc=1)

    keep = {"prompt", "teacher_prompt", "audio", "target", "prefix_text", "teacher_prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    logger.info(f"Train samples: {len(ds['train'])}")

    collator = DataCollatorForASROPSD(processor=student_processor, sampling_rate=args_cli.sr)

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

    trainer = OPSDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        processor=student_processor,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        beta=args_cli.beta,
        temperature=args_cli.temperature,
        sampling_temperature=args_cli.sampling_temperature,
        max_completion_length=args_cli.max_completion_length,
        topk_logits_k=args_cli.topk_logits_k,
    )
    trainer.add_callback(MakeEveryCheckpointInferableCallback(base_model_path=args_cli.student_model_path))

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
