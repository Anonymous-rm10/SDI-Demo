# coding=utf-8
# On-Policy Knowledge Distillation Trainer for Qwen3-ASR.
# Student: Qwen3-ASR-0.6B  |  Teacher: Qwen3-ASR-1.7B (local, both in same process)

import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from packaging import version
from qwen_asr import Qwen3ASRModel
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)
from trl.models import unwrap_model_for_generation


# ── Best-model tracking ────────────────────────────────────────────────────────

class _BestModelCallback(transformers.TrainerCallback):
    """
    Tracks training loss and copies the best model (lowest loss) to best_model_path.
    Unlike a symlink, this is a real copy so it survives save_total_limit pruning.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_loss = float("inf")
        self.best_ckpt_name = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return

        checkpoints = sorted(
            [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if not checkpoints:
            return
        latest_ckpt = checkpoints[-1]

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_ckpt_name = latest_ckpt
            src = os.path.join(self.output_dir, latest_ckpt)
            dst = os.path.join(self.output_dir, "best")
            # Copy immediately so it survives save_total_limit pruning later
            if os.path.islink(dst) or os.path.exists(dst):
                if os.path.islink(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"\n[Best model updated] loss={loss:.6f}  →  {dst}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_ckpt_name is not None:
            print(f"\n[Training complete] Best model: loss={self.best_loss:.6f}")


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    """Copy processor/tokenizer files into a checkpoint dir so it can be used standalone."""
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


def patch_outer_forward(model):
    """Patch Qwen3ASRForConditionalGeneration so model.forward() calls thinker.forward()."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

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


# ──────────────────────────────────────────────────────────────────────────────
# OPDTrainer
# ──────────────────────────────────────────────────────────────────────────────

class OPDTrainer(Trainer):
    """
    On-Policy Distillation (OPD) Trainer for Qwen3-ASR.

    Training loop per step:
      1. Student (0.6B) samples completions on-policy from the ASR prompt.
      2. Teacher (1.7B) evaluates those exact completions → produces logits.
      3. KL(student || teacher) loss on the completion tokens → backprop through
         student only.

    Key design decisions (mirroring west/west/trainer/kd_trainer.py):
      - Reverse KL  : mode-seeking, consistent with on-policy sampling.
      - Top-k logits: more efficient; teacher and student both run topk before KL.
      - Vocabulary alignment: student/teacher may differ; crop to min vocab size.
      - Teacher forward: no gradient tracking, in inference mode.
      - Both models loaded in same process (local teacher).
    """

    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        processor: ProcessorMixin,
        args: TrainingArguments,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        topk_logits_k: int = 64,
        temperature: float = 1.0,
        max_completion_length: int = 256,
        grad_accum: int = 1,
    ):
        """
        Args:
            student_model:  Pre-initialized 0.6B Qwen3ASRForConditionalGeneration.
            teacher_model:  Pre-initialized 1.7B Qwen3ASRForConditionalGeneration.
            processor:      Qwen3ASRProcessor for the student (tokenizer + audio).
            args:          TrainingArguments (from HuggingFace Trainer).
            train_dataset: Dataset with columns: audio, text, optional prompt.
            eval_dataset:  Optional eval dataset.
            data_collator: Callable that batches features into tensors.
            topk_logits_k: Top-k for logits distillation. None = full vocab.
            temperature:   Temperature applied to logits before softmax in KL.
            max_completion_length: Max tokens for student sampling.
            grad_accum:    Gradient accumulation steps (passed via args).
        """
        super().__init__(
            model=student_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )

        # Register best-model tracking callback
        self.add_callback(_BestModelCallback(args.output_dir))

        set_seed(args.seed, device_specific=True)

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.processor = processor

        self.topk_logits_k = topk_logits_k
        self.temperature = temperature
        self.max_completion_length = max_completion_length

        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        self._metrics = defaultdict(list)

        # Put teacher on same device as student (we manage devices manually here)
        # Accelerator will handle DDP wrapping for the student; teacher stays in
        # its original device_map placement.
        self._teacher_device = None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _rollout(self, inputs: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Student samples ASR completions (on-policy).

        Returns:
            generated_strs : list of decoded strings
            generated_ids : [batch, gen_len]  token IDs (completion only)
            generated_mask: [batch, gen_len]  1=on valid token, 0=padding/EOS
        """
        with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped:
            model_dtype = next(unwrapped.parameters()).dtype
            inputs_student = {
                "input_ids": inputs["prefix_input_ids"].to(unwrapped.device),
                "attention_mask": inputs["prefix_attention_mask"].to(unwrapped.device),
                "input_features": inputs["prefix_input_features"].to(dtype=model_dtype, device=unwrapped.device),
                "feature_attention_mask": inputs["prefix_feature_attention_mask"].to(unwrapped.device),
                "max_new_tokens": self.max_completion_length,
            }
            result = unwrapped.generate(
                **inputs_student,
                generation_config=self.generation_config,
            )
            # generate() returns SampleOutput when do_sample=True; extract sequences tensor
            prompt_completion_ids = getattr(result, "sequences", result[0] if isinstance(result, tuple) else result)

        prompt_len = inputs["prefix_input_ids"].size(1)
        generated_ids = prompt_completion_ids[:, prompt_len:]           # [B, gen_len]

        # Build mask: 1 for every token up to (and including) the first EOS
        device = self.accelerator.device
        is_eos = generated_ids == self.processor.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        generated_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        generated_strs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_strs, generated_ids, generated_mask

    def _get_logits(self, model, inputs: dict) -> torch.Tensor:
        """
        Run a forward pass and return raw logits.
        Handles the Qwen3-ASR model.thinker wrapper.

        Returns:
            logits: [batch, seq_len-1, vocab_size]  (last position dropped → shift)
        """
        forward_model = model
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            forward_model = model.module.thinker
        elif hasattr(model, "thinker"):
            forward_model = model.thinker

        # Move all tensor inputs to the model's device
        device = next(forward_model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        out = forward_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
        logits = out.logits
        return logits[:, :-1, :]    # shift: exclude last logit (no prediction for last token)

    def _prepare_student_inputs(
        self,
        batch_inputs: dict,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        target_dtype: torch.dtype = None,
    ) -> dict:
        """
        Build [prefix + generated] sequence for logit computation.

        For KL loss: student and teacher both evaluate the SAME sequence
        (audio + prefix_text + student_completion_tokens), and we compute
        the KL divergence at each completion-token position.
        """
        prompt_ids = batch_inputs.get("prefix_input_ids", batch_inputs["input_ids"])
        prompt_mask = batch_inputs.get("prefix_attention_mask", batch_inputs["attention_mask"])
        input_features = batch_inputs.get("prefix_input_features", batch_inputs["input_features"])
        feature_mask = batch_inputs.get("prefix_feature_attention_mask", batch_inputs["feature_attention_mask"])

        # Concatenate: [prefix | generated]
        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)

        # Convert input_features to match model dtype (bfloat16 on this model)
        if target_dtype is not None:
            input_features = input_features.to(dtype=target_dtype)

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    def _compute_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        topk: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute reverse KL: KL(student || teacher) per token.
        Reverse KL uses student probability as weights → mode-seeking,
        which is ideal for on-policy distillation.

        student_logits : [batch, seq, vocab]
        teacher_logits : [batch, seq, vocab]
        mask           : [batch, seq]  1 = compute KL here
        topk           : if set, only use student's top-k tokens for efficiency
        temperature    : temperature for softening

        Returns: per_token_kl [batch, seq]
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if topk is None:
            # Full-vocab KL
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_probs     = F.softmax(student_logits, dim=-1)
            per_token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        else:
            # Top-k KL: use student's top-k indices (consistent with reverse-KL weighting)
            _, student_topk_idx = torch.topk(student_logits, k=topk, dim=-1)  # [B, seq, k]
            student_topk_logits  = torch.gather(student_logits, dim=-1, index=student_topk_idx)
            teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=student_topk_idx)

            student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
            teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
            student_topk_probs     = F.softmax(student_topk_logits, dim=-1)
            per_token_kl = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)

        return per_token_kl

    # ── metrics ───────────────────────────────────────────────────────────────

    def _log_metrics(
        self,
        generated_mask: torch.Tensor,
        per_token_kl: torch.Tensor,
        loss: torch.Tensor,
    ):
        """Accumulate OPD-specific metrics for configured logging backends."""
        loss_val = loss.detach().float().mean()
        loss_gathered = self.accelerator.gather_for_metrics(loss_val)
        self._metrics.setdefault("loss_sum", 0.0)
        self._metrics.setdefault("loss_count", 0)
        self._metrics["loss_sum"] += loss_gathered.sum().item()
        self._metrics["loss_count"] += loss_gathered.numel()

        masked_kl = (per_token_kl * generated_mask).sum(dim=1) / generated_mask.sum(dim=1).clamp(min=1)
        kl_gathered = self.accelerator.gather_for_metrics(masked_kl.float().mean())
        self._metrics.setdefault("kl_sum", 0.0)
        self._metrics.setdefault("kl_count", 0)
        self._metrics["kl_sum"] += kl_gathered.sum().item()
        self._metrics["kl_count"] += kl_gathered.numel()

        completion_lens = self.accelerator.gather_for_metrics(generated_mask.sum(1)).float()
        self._metrics.setdefault("completion_length_sum", 0.0)
        self._metrics.setdefault("completion_length_count", 0)
        self._metrics["completion_length_sum"] += completion_lens.sum().item()
        self._metrics["completion_length_count"] += completion_lens.numel()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Override to avoid tuple-unpacking errors on 0-d tensors.
        OPD does not use logits/labels from the standard prediction path.
        """
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def log(self, logs: dict, start_time=None):
        # Aggregate OPD metrics and inject them into Trainer logs (for W&B, etc.).
        if self._metrics.get("loss_count", 0) > 0:
            logs["loss"] = self._metrics["loss_sum"] / self._metrics["loss_count"]
        if self._metrics.get("kl_count", 0) > 0:
            logs["kl"] = self._metrics["kl_sum"] / self._metrics["kl_count"]
        if self._metrics.get("completion_length_count", 0) > 0:
            logs["completion_length"] = (
                self._metrics["completion_length_sum"] / self._metrics["completion_length_count"]
            )
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    # ── checkpoint callback ────────────────────────────────────────────────────

    def _on_save(self, checkpoint_dir: str, model_path: str):
        """Copy processor files into each saved checkpoint directory."""
        copy_required_hf_files_for_qwen_asr(model_path, checkpoint_dir)

    # ── training loop ──────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        One OPD step:

          1. Student samples completion tokens on-policy.
          2. Teacher scores those tokens (no grad, same audio + student output as input).
          3. Reverse KL loss on completion positions → backprop through student only.
        """
        # ── Step 1: student on-policy rollout ──────────────────────────────────
        generated_strs, generated_ids, generated_mask = self._rollout(inputs)

        # ── Step 2: teacher forward (no grad) on [prompt + student completion] ─
        model_dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            teacher_inputs = self._prepare_student_inputs(
                inputs, generated_ids, generated_mask, target_dtype=model_dtype
            )
            teacher_logits = self._get_logits(self.teacher_model, teacher_inputs)

        # ── Step 3: student forward on [prompt + student completion] ───────────
        student_inputs = self._prepare_student_inputs(
            inputs, generated_ids, generated_mask, target_dtype=model_dtype
        )
        student_logits = self._get_logits(model, student_inputs)

        # ── Step 4: align vocab sizes ─────────────────────────────────────────
        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        # ── Step 5: KL loss on completion tokens only ─────────────────────────
        prompt_len = inputs["prefix_input_ids"].size(1)   # length of prefix only
        student_gen_logits  = student_logits[:,  prompt_len - 1:, :]   # [B, gen_len, vocab]
        teacher_gen_logits  = teacher_logits[:,  prompt_len - 1:, :]   # [B, gen_len, vocab]
        completion_mask     = generated_mask.float()                     # [B, gen_len]

        per_token_kl = self._compute_kl(
            student_logits=student_gen_logits,
            teacher_logits=teacher_gen_logits,
            mask=completion_mask,
            topk=self.topk_logits_k,
            temperature=self.temperature,
        )
        loss = ((per_token_kl * completion_mask).sum(dim=1) /
                completion_mask.sum(dim=1).clamp(min=1)).mean()

        # ── Step 6: log ───────────────────────────────────────────────────────
        self._log_metrics(completion_mask, per_token_kl, loss)
        return loss
