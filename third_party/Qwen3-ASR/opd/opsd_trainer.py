# coding=utf-8
# On-Policy Self-Distillation (OPSD) Trainer for Qwen3-ASR.
# Student: Qwen3-ASR (the model being trained, e.g. 0.6B)  |  Teacher: Qwen3-ASR (frozen, e.g. 1.7B)
#
# OPSD flow per training step (mirrors the paper):
#   1. Student samples ASR completions on-policy from its OWN prefix (no reference).
#   2. Frozen teacher scores the SAME completions, but conditioned on a teacher prefix
#      that carries the reference transcript as privileged context.
#   3. JSD(student || teacher) loss on completion tokens -> backprop through student only.
#
# Teacher prefix construction:
#   The Qwen3-ASR chat template renders SYSTEM text but drops text inside the USER
#   message (only the audio placeholder survives). So the reference is injected into
#   the SYSTEM message. See opsd_data_utils.build_teacher_prefix_messages.

import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import set_seed
from packaging import version
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)
from trl.models import unwrap_model_for_generation

logger = logging.getLogger(__name__)


def patch_outer_forward(model):
    """Patch Qwen3ASRForConditionalGeneration so model.forward() calls thinker.forward()."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")
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


# ── checkpoint helpers (mirror opd_trainer.py) ────────────────────────────────

def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
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


class _BestModelCallback(transformers.TrainerCallback):
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
            if os.path.islink(dst) or os.path.exists(dst):
                if os.path.islink(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"\n[Best model updated] loss={loss:.6f}  ->  {dst}")


# ── loss helpers ──────────────────────────────────────────────────────────────

def _kl(x_logp, y_logp):
    """KL(x || y) = sum y? No: this returns per-position KL using x's distribution as weights."""
    # x_prob * (x_logp - y_logp) is KL(x || y) when summed over vocab.
    return (torch.exp(x_logp) * (x_logp - y_logp)).sum(-1)


def jsd_loss(student_logits, teacher_logits, beta, temperature, mask, topk=None):
    """Jensen-Shannon Divergence between student and teacher distributions.

    beta=0 -> forward KL (KL(student || teacher))  [mode-covering, mass-covering]
    beta=1 -> reverse KL (KL(teacher || student))  [mode-seeking, matches on-policy]
    beta in (0,1) -> JSD (symmetric interpolation).
    """
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if topk is None:
        s_logp = F.log_softmax(student_logits, dim=-1)
        t_logp = F.log_softmax(teacher_logits, dim=-1)
        student_dist = torch.exp(s_logp)
        teacher_dist = torch.exp(t_logp)
    else:
        # top-k: use the student's top-k indices for both distributions (consistent weighting).
        _, topk_idx = torch.topk(student_logits, k=topk, dim=-1)
        s_gather = torch.gather(student_logits, -1, topk_idx)
        t_gather = torch.gather(teacher_logits, -1, topk_idx)
        s_logp = F.log_softmax(s_gather, dim=-1)
        t_logp = F.log_softmax(t_gather, dim=-1)
        student_dist = torch.exp(s_logp)
        teacher_dist = torch.exp(t_logp)

    if beta == 0:
        per_tok = _kl(s_logp, t_logp)
    elif beta == 1:
        per_tok = _kl(t_logp, s_logp)
    else:
        # JSD: m = beta*s + (1-beta)*t
        m_dist = beta * student_dist + (1 - beta) * teacher_dist
        m_logp = torch.log(m_dist.clamp_min(1e-12))
        per_tok = beta * _kl(m_logp, s_logp) + (1 - beta) * _kl(m_logp, t_logp)

    per_tok = per_tok * mask
    return per_tok


# ── OPSDTrainer ───────────────────────────────────────────────────────────────

class OPSDTrainer(Trainer):
    """
    On-Policy Self-Distillation (OPSD) Trainer for Qwen3-ASR.

    Per step:
      1. Student samples ASR completions on-policy from its own prefix.
      2. Frozen teacher scores the SAME completions, but sees a teacher prefix that
         carries the reference transcript (privileged context).
      3. JSD(student || teacher) loss on completion positions -> backprop student only.
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
        beta: float = 0.5,
        temperature: float = 1.0,
        sampling_temperature: float = 1.0,
        max_completion_length: int = 256,
        topk_logits_k: int = None,
    ):
        super().__init__(
            model=student_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )
        self.add_callback(_BestModelCallback(args.output_dir))
        set_seed(args.seed, device_specific=True)

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.processor = processor
        self.beta = beta
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.topk_logits_k = topk_logits_k

        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=sampling_temperature,
            num_return_sequences=1,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        self._metrics = defaultdict(list)

        # Freeze teacher: eval mode, no grad. Move to the same device as the student.
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)
            stu_device = next(self.student_model.parameters()).device
            try:
                self.teacher_model.to(stu_device)
            except RuntimeError:
                pass

        self._teacher_device = None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _rollout(self, inputs):
        """Student samples ASR completions on-policy (from its own prefix)."""
        with unwrap_model_for_generation(self.student_model, self.accelerator) as unwrapped:
            model_dtype = next(unwrapped.parameters()).dtype
            inputs_student = {
                "input_ids": inputs["prefix_input_ids"].to(unwrapped.device),
                "attention_mask": inputs["prefix_attention_mask"].to(unwrapped.device),
                "input_features": inputs["prefix_input_features"].to(dtype=model_dtype, device=unwrapped.device),
                "feature_attention_mask": inputs["prefix_feature_attention_mask"].to(unwrapped.device),
                "max_new_tokens": self.max_completion_length,
            }
            result = unwrapped.generate(**inputs_student, generation_config=self.generation_config)
            prompt_completion_ids = getattr(result, "sequences", result[0] if isinstance(result, tuple) else result)

        prompt_len = inputs["prefix_input_ids"].size(1)
        generated_ids = prompt_completion_ids[:, prompt_len:]

        # mask: 1 up to and including first EOS
        device = self.accelerator.device
        is_eos = generated_ids == self.processor.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        generated_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        generated_strs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_strs, generated_ids, generated_mask

    def _get_logits(self, model, inputs):
        """Run a forward pass and return raw logits. Handles the .thinker wrapper."""
        forward_model = model
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            forward_model = model.module.thinker
        elif hasattr(model, "thinker"):
            forward_model = model.thinker

        device = next(forward_model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        out = forward_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        )
        logits = out.logits
        return logits[:, :-1, :]  # drop last position (no prediction for it)

    def _build_sequence_inputs(self, batch_inputs, prefix_key, generated_ids, generated_mask, target_dtype):
        """Concatenate [prefix | generated] for a given side (student or teacher).

        \\``prefix_key`` is one of ``'prefix'`` (student) or ``'teacher'`` (teacher). The
        collator stores teacher inputs under ``teacher_prefix_*`` keys.
        """
        key = "prefix" if prefix_key == "prefix" else "teacher_prefix"
        prompt_ids = batch_inputs[f"{key}_input_ids"]
        prompt_mask = batch_inputs[f"{key}_attention_mask"]
        input_features = batch_inputs[f"{key}_input_features"]
        feature_mask = batch_inputs[f"{key}_feature_attention_mask"]

        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)
        input_features = input_features.to(dtype=target_dtype)

        # The completion region starts right after the (PADDED) prompt. In a batch the
        # probe is left-padded to the batch max, so prompt_ids.size(1) is the uniform
        # padded length — NOT the per-row content length. Slicing against per-row content
        # lengths is wrong for any batch size > 1 (gradients explode / loss garbage).
        padded_len = prompt_ids.size(1)

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
            "prefix_len": torch.full((prompt_ids.size(0),), padded_len, dtype=torch.long, device=prompt_ids.device),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ── Step 1: student on-policy rollout ──────────────────────────────────
        generated_strs, generated_ids, generated_mask = self._rollout(inputs)

        model_dtype = next(model.parameters()).dtype

        # ── Step 2: teacher forward (no grad) on [teacher_prefix + completion] ──
        if self.teacher_model is not None:
            with torch.inference_mode():
                teacher_inputs = self._build_sequence_inputs(
                    inputs, "teacher", generated_ids, generated_mask, target_dtype=model_dtype
                )
                teacher_logits = self._get_logits(self.teacher_model, teacher_inputs)
            teacher_prefix_len = teacher_inputs["prefix_len"]
        else:
            # Self-distillation: student is teacher — compute teacher logits from the same forward.
            teacher_logits = None
            teacher_prefix_len = None

        # ── Step 3: student forward on [prefix + completion] ───────────────────
        student_inputs = self._build_sequence_inputs(
            inputs, "prefix", generated_ids, generated_mask, target_dtype=model_dtype
        )
        student_logits = self._get_logits(model, student_inputs)
        student_prefix_len = student_inputs["prefix_len"]

        # ── Step 4: align against completion region, crop vocab ────────────────
        if teacher_logits is None:
            teacher_logits = student_logits
        else:
            teacher_logits = teacher_logits.to(student_logits.device)

        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        # Each side has its own prefix length (lengths can differ because the teacher
        # prefix carries the reference). Slice each at its own completion start.
        def _slice_completion(logits, prefix_len):
            B = logits.size(0)
            completion_len = generated_mask.size(1)
            # logits is [B, seq_len-1, V] (last dropped). Completion starts at (prefix_len-1).
            idx = torch.arange(B, device=logits.device)
            starts = (prefix_len - 1).clamp(min=0)
            seq = torch.stack([
                logits[b, s:s + completion_len, :]
                for b, s in enumerate(starts)
            ])
            return seq

        student_gen_logits = _slice_completion(student_logits, student_prefix_len)
        teacher_gen_logits = _slice_completion(teacher_logits, teacher_prefix_len)

        completion_mask = generated_mask.float()  # [B, gen_len]

        per_token_jsd = jsd_loss(
            student_gen_logits,
            teacher_gen_logits,
            beta=self.beta,
            temperature=self.temperature,
            mask=completion_mask,
            topk=self.topk_logits_k,
        )

        loss = (
            per_token_jsd.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()

        # ── Step 5: log metrics ────────────────────────────────────────────────
        self._log_metrics(completion_mask, per_token_jsd, loss)
        if return_outputs:
            return (loss, None)
        return loss

    def _log_metrics(self, completion_mask, per_token, loss):
        self._metrics.setdefault("loss_sum", 0.0)
        self._metrics.setdefault("loss_count", 0)
        self._metrics["loss_sum"] += loss.detach().float().item()
        self._metrics["loss_count"] += 1

        masked = (per_token * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        self._metrics.setdefault("kl_sum", 0.0)
        self._metrics.setdefault("kl_count", 0)
        self._metrics["kl_sum"] += masked.mean().detach().float().item()
        self._metrics["kl_count"] += 1

    def log(self, logs, start_time=None):
        if self._metrics.get("loss_count", 0) > 0:
            logs["loss"] = self._metrics["loss_sum"] / self._metrics["loss_count"]
        if self._metrics.get("kl_count", 0) > 0:
            logs["kl"] = self._metrics["kl_sum"] / self._metrics["kl_count"]
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)
