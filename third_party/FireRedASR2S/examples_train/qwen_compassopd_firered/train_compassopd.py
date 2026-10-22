#!/usr/bin/env python3
"""CompassOPD: FireRedASR2-LLM RL shift -> Qwen3-ASR.

This trainer keeps all comparisons inside a model family:

* teacher shift: FireRed RL policy - released FireRed reference;
* student displacement: current Qwen - frozen initial Qwen.

At every student-visited ASR prefix it builds the full-distribution target

    q(v) propto p_student_ref(v) * exp((log p_teacher(v)
                                       - log p_teacher_ref(v)) / alpha)

over the verified shared vocabulary, then minimizes alpha * KL(student || q).
This is the exact reverse-KL form of the reference-regularized CompassOPD
objective, up to a policy-independent constant.

The FireRed RL run used by this repository starts directly from the released
model with a new rank-8 LoRA.  The released LoRA is merged before that adapter
is attached, so disabling the RL adapter recovers the teacher reference without
loading a second 7B model.  The checkpoint validation below deliberately fails
if those assumptions are not true.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, TrainingArguments

from qwen_asr import Qwen3ASRModel

from fireredasr2s.fireredasr2.models.fireredasr_llm import FireRedAsrLlm
from fireredasr2s.fireredasr2.train_qwen_opd import (
    DualFrontendOpdCollator,
    FireRedTeacherOpdTrainer,
    IGNORE_TOKEN_ID,
    InvalidRolloutError,
    OpdJsonlDataset,
    SaveProcessorCallback,
    build_arg_parser,
    load_teacher,
    patch_qwen_forward,
    seed_everything,
    shared_tokenizer_prefix,
    validate_args as validate_base_args,
)


logger = logging.getLogger("fireredasr2s.qwen_compassopd")


def validate_flash_attention_available(enabled: bool) -> None:
    """Fail before model loading when the optional FlashAttention dependency is unusable."""
    if not enabled:
        return
    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "--use_flash_attention requires a flash-attn build compatible with "
            "the active PyTorch/CUDA environment. Install it with "
            "`python -m pip install flash-attn --no-build-isolation`."
        ) from exc


def _validate_same_shape(named_tensors: dict[str, torch.Tensor]) -> None:
    shapes = {name: tuple(value.shape) for name, value in named_tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"CompassOPD logits must have identical shapes: {shapes}")


def compass_reverse_kl(
    student_logits: torch.Tensor,
    student_reference_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_reference_logits: torch.Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 1.0,
    topk: int | None = None,
    shift_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-token KL(student || Compass target) and diagnostics.

    All probability arithmetic is done in float32.  Only ``student_logits``
    receives gradients.  If ``topk`` is set, all distributions are restricted
    to and renormalized on the current student's top-k tokens, matching the
    approximation used by the existing OPD trainer.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if topk is not None and topk <= 0:
        raise ValueError("topk must be positive when set")
    if shift_clip is not None and (
        not math.isfinite(shift_clip) or shift_clip <= 0
    ):
        raise ValueError("shift_clip must be finite and positive when set")

    tensors = {
        "student": student_logits,
        "student_reference": student_reference_logits,
        "teacher": teacher_logits,
        "teacher_reference": teacher_reference_logits,
    }
    _validate_same_shape(tensors)

    # References are fixed even if a caller accidentally supplies a tensor
    # requiring gradients.
    student_reference_logits = student_reference_logits.detach()
    teacher_logits = teacher_logits.detach()
    teacher_reference_logits = teacher_reference_logits.detach()

    if topk is not None:
        k = min(topk, student_logits.size(-1))
        indices = student_logits.detach().topk(k, dim=-1).indices
        student_logits = student_logits.gather(-1, indices)
        student_reference_logits = student_reference_logits.gather(-1, indices)
        teacher_logits = teacher_logits.gather(-1, indices)
        teacher_reference_logits = teacher_reference_logits.gather(-1, indices)

    scale = float(temperature)
    student_log = F.log_softmax(student_logits.float() / scale, dim=-1)
    student_reference_log = F.log_softmax(
        student_reference_logits.float() / scale, dim=-1
    )
    teacher_log = F.log_softmax(teacher_logits.float() / scale, dim=-1)
    teacher_reference_log = F.log_softmax(
        teacher_reference_logits.float() / scale, dim=-1
    )

    teacher_shift = teacher_log - teacher_reference_log
    unclipped_shift = teacher_shift
    if shift_clip is not None:
        teacher_shift = teacher_shift.clamp(-shift_clip, shift_clip)

    target_log = F.log_softmax(
        student_reference_log + teacher_shift / float(alpha), dim=-1
    )
    student_prob = student_log.exp()
    per_token_kl = (student_prob * (student_log - target_log)).sum(dim=-1)

    teacher_prob = teacher_log.exp()
    teacher_reference_kl = (
        teacher_prob * (teacher_log - teacher_reference_log)
    ).sum(dim=-1)
    student_reference_kl = (
        student_prob * (student_log - student_reference_log)
    ).sum(dim=-1)
    diagnostics = {
        "teacher_shift_abs": unclipped_shift.abs().mean(dim=-1),
        "teacher_shift_rms": unclipped_shift.square().mean(dim=-1).sqrt(),
        "teacher_reference_kl": teacher_reference_kl,
        "student_reference_kl": student_reference_kl,
    }
    return per_token_kl, diagnostics


def _checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    return checkpoint / "training_state.pt" if checkpoint.is_dir() else checkpoint


def read_checkpoint_config(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Read and verify checkpoint format/config from training_state.pt."""
    checkpoint = Path(path).expanduser().resolve()
    state_file = _checkpoint_file(checkpoint)
    if not state_file.is_file():
        raise FileNotFoundError(f"Teacher RL checkpoint not found: {state_file}")
    package = torch.load(state_file, map_location="cpu", weights_only=False)
    checkpoint_format = str(package.get("format", ""))
    config = dict(package.get("config", {}))
    del package

    # Prefer the checkpoint payload, but allow a sidecar to fill configuration
    # fields in old checkpoints. The format itself is never inferred from JSON.
    config_file = (
        checkpoint / "config.json"
        if checkpoint.is_dir()
        else checkpoint.parent / "config.json"
    )
    if config_file.is_file():
        with config_file.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
        for key, value in sidecar.items():
            config.setdefault(key, value)
    return checkpoint_format, config


def validate_disabled_adapter_reference(path: str | Path) -> dict[str, Any]:
    """Ensure disabling the loaded RL adapter recovers the RL initialization."""
    checkpoint_format, config = read_checkpoint_config(path)
    if checkpoint_format != "fireredasr2_llm_rl_trainable_v1":
        raise ValueError(
            "CompassOPD teacher must be a FireRed RL checkpoint; got "
            f"format={checkpoint_format!r}"
        )
    if config.get("init_checkpoint"):
        raise ValueError(
            "The RL run started from init_checkpoint, so disabling its adapter "
            "does not recover that SFT initialization. Use an RL checkpoint "
            "trained with init_checkpoint='' or extend this trainer with an "
            "explicit teacher-reference checkpoint."
        )
    if not config.get("use_lora", False) or config.get("full_finetune_llm", False):
        raise ValueError(
            "The memory-efficient teacher reference requires LoRA RL with "
            "use_lora=true and full_finetune_llm=false"
        )

    requested = {
        "rank": int(config.get("lora_rank", FireRedAsrLlm.DEFAULT_LORA_RANK)),
        "alpha": int(config.get("lora_alpha", FireRedAsrLlm.DEFAULT_LORA_ALPHA)),
        "dropout": float(config.get("lora_dropout", FireRedAsrLlm.DEFAULT_LORA_DROPOUT)),
        "targets": set(config.get(
            "lora_target_modules", FireRedAsrLlm.DEFAULT_LORA_TARGET_MODULES
        )),
    }
    released = {
        "rank": FireRedAsrLlm.DEFAULT_LORA_RANK,
        "alpha": FireRedAsrLlm.DEFAULT_LORA_ALPHA,
        "dropout": FireRedAsrLlm.DEFAULT_LORA_DROPOUT,
        "targets": set(FireRedAsrLlm.DEFAULT_LORA_TARGET_MODULES),
    }
    if requested == released:
        raise ValueError(
            "The RL LoRA has the released LoRA structure. The current loader "
            "cannot recover the initial released adapter merely by disabling "
            "the updated adapter. Use a custom/different RL LoRA (the current "
            "rank-8 run is supported)."
        )
    return config


class CompassOpdTrainer(FireRedTeacherOpdTrainer):
    method_name = "CompassOPD"
    distillation_metric_name = "compass_kl"
    distillation_loss_metric_name = "compass_loss"

    def __init__(
        self,
        *args,
        student_reference_model: torch.nn.Module,
        anchor_alpha: float,
        teacher_shift_clip: float | None,
        **kwargs,
    ):
        if not math.isfinite(anchor_alpha) or anchor_alpha <= 0:
            raise ValueError("anchor_alpha must be finite and positive")
        if teacher_shift_clip is not None and (
            not math.isfinite(teacher_shift_clip) or teacher_shift_clip <= 0
        ):
            raise ValueError("teacher_shift_clip must be positive when set")
        self.student_reference_model = student_reference_model
        self.anchor_alpha = anchor_alpha
        self.teacher_shift_clip = teacher_shift_clip
        super().__init__(*args, **kwargs)

        for parameter in self.student_reference_model.parameters():
            parameter.requires_grad = False
        self.student_reference_model.eval()
        self.student_reference_model.to(self.accelerator.device)

        if not hasattr(self.teacher_model.llm, "disable_adapter"):
            raise RuntimeError(
                "Loaded FireRed teacher has no disable_adapter() context; "
                "cannot construct the released-model reference in one 7B instance"
            )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (
            (values * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        ).mean()

    def _align_teacher_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted_logits = logits[:, :-1].index_select(
            -1, self._vocab_ids_for(logits.device)
        )
        shifted_labels = labels[:, 1:]
        token_mask = shifted_labels.ne(IGNORE_TOKEN_ID)
        for row in range(response_ids.size(0)):
            scored_ids = shifted_labels[row][token_mask[row]]
            expected_ids = response_ids[row][response_mask[row].bool()]
            if not torch.equal(scored_ids, expected_ids):
                raise RuntimeError(
                    "FireRed label shift does not align with ASR response IDs"
                )

        rows = [shifted_logits[row][token_mask[row]] for row in range(logits.size(0))]
        max_response = max(row.size(0) for row in rows)
        aligned = shifted_logits.new_zeros(
            (len(rows), max_response, self.distillation_vocab_size)
        )
        aligned_mask = torch.zeros(
            (len(rows), max_response), dtype=torch.bool, device=logits.device
        )
        for row, row_logits in enumerate(rows):
            aligned[row, : row_logits.size(0)] = row_logits
            aligned_mask[row, : row_logits.size(0)] = True
        return aligned, aligned_mask

    def _teacher_response_logits_pair(
        self,
        inputs: dict[str, Any],
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score one response with RL FireRed and its adapter-disabled reference."""
        batch_size = response_ids.size(0)
        teacher_features = inputs["teacher_features"].to(
            self.teacher_device, dtype=self.teacher_dtype
        )
        teacher_lengths = inputs["teacher_feature_lengths"].to(self.teacher_device)

        with torch.inference_mode():
            # The RL run freezes the encoder/projector; both teacher policies can
            # share the same encoded speech and the same merged prompt embeddings.
            speech_features, speech_lengths, _ = self.teacher_model.encode_speech(
                teacher_features, teacher_lengths
            )
            prompt_ids, prompt_mask = self._teacher_prompt_inputs(inputs, batch_size)
            prompt_ids = prompt_ids.to(self.teacher_device)
            prompt_mask = prompt_mask.to(self.teacher_device)
            response_ids_on_teacher = response_ids.to(self.teacher_device)
            response_mask_on_teacher = response_mask.to(self.teacher_device)

            combined_ids: list[torch.Tensor] = []
            combined_labels: list[torch.Tensor] = []
            for row in range(batch_size):
                valid_prompt = prompt_ids[row][prompt_mask[row].bool()]
                valid_response = response_ids_on_teacher[row][
                    response_mask_on_teacher[row].bool()
                ]
                combined_ids.append(torch.cat([valid_prompt, valid_response]))
                combined_labels.append(torch.cat([
                    torch.full_like(valid_prompt, IGNORE_TOKEN_ID), valid_response
                ]))

            max_length = max(ids.numel() for ids in combined_ids)
            teacher_pad = self.teacher_model.llm.config.pad_token_id
            input_ids = prompt_ids.new_full((batch_size, max_length), teacher_pad)
            attention_mask = prompt_mask.new_zeros((batch_size, max_length))
            labels = response_ids_on_teacher.new_full(
                (batch_size, max_length), IGNORE_TOKEN_ID
            )
            for row, (ids, row_labels) in enumerate(zip(combined_ids, combined_labels)):
                length = ids.numel()
                input_ids[row, :length] = ids
                attention_mask[row, :length] = 1
                labels[row, :length] = row_labels

            inputs_embeds = self.teacher_model.llm.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, labels = (
                self.teacher_model._merge_input_ids_with_speech_features(
                    speech_features.to(inputs_embeds.dtype),
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels=labels,
                    speech_lens=speech_lengths,
                    padding_side="right",
                )
            )

            strong_outputs = self.teacher_model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            strong_logits, strong_mask = self._align_teacher_logits(
                strong_outputs.logits,
                labels,
                response_ids_on_teacher,
                response_mask_on_teacher,
            )
            del strong_outputs

            with self.teacher_model.llm.disable_adapter():
                reference_outputs = self.teacher_model.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            reference_logits, reference_mask = self._align_teacher_logits(
                reference_outputs.logits,
                labels,
                response_ids_on_teacher,
                response_mask_on_teacher,
            )

            if not torch.equal(strong_mask, reference_mask):
                raise RuntimeError("Strong/reference FireRed response masks differ")

        # Convert inference tensors to ordinary detached tensors before they are
        # consumed by autograd while differentiating the current student.
        device = self.accelerator.device
        return (
            strong_logits.to(device).clone(),
            reference_logits.to(device).clone(),
            strong_mask.to(device).clone(),
        )

    def _student_reference_response_logits(
        self,
        inputs: dict[str, Any],
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        response_positions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            logits, mask = self._student_response_logits(
                self.student_reference_model,
                inputs,
                generated_ids,
                generated_mask,
                response_positions,
            )
        return logits.clone(), mask.clone()

    def _compass_divergence(
        self,
        student_logits: torch.Tensor,
        student_reference_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_reference_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
        """Return per-token divergence, diagnostics, and its loss multiplier.

        Subclasses can keep the Compass target construction and replace only the
        divergence used to fit it (for example, with an OPSD-style generalized
        JSD).  The base implementation remains the exact CompassOPD objective.
        """
        per_token_kl, diagnostics = compass_reverse_kl(
            student_logits,
            student_reference_logits,
            teacher_logits,
            teacher_reference_logits,
            alpha=self.anchor_alpha,
            temperature=self.distillation_temperature,
            topk=self.topk_logits_k,
            shift_clip=self.teacher_shift_clip,
        )
        # alpha * KL(pi || q) equals
        # alpha * KL(pi || pi_ref) - E_pi[teacher_shift] plus a constant.
        return per_token_kl, diagnostics, self.anchor_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        invalid_reason = None
        for rollout_retry in range(self.max_rollout_retries + 1):
            generated_ids, generated_mask = self._rollout(model, inputs)
            if self.log_rollout_text:
                raw_completions = [
                    self.student_processor.tokenizer.decode(
                        generated_ids[row][generated_mask[row]].detach().cpu().tolist(),
                        skip_special_tokens=False,
                    )
                    for row in range(generated_ids.size(0))
                ]
                logger.info(
                    "Student raw rollout: retry=%d/%d, uttids=%s, completions=%s",
                    rollout_retry,
                    self.max_rollout_retries,
                    inputs.get("uttids", []),
                    raw_completions,
                )
            try:
                response_ids, response_mask, response_positions, truncated = (
                    self._extract_shared_responses(generated_ids, generated_mask)
                )
                break
            except InvalidRolloutError as exc:
                invalid_reason = str(exc)
                if rollout_retry < self.max_rollout_retries:
                    continue
                decoded = self.student_processor.tokenizer.batch_decode(
                    generated_ids.detach().cpu().tolist(), skip_special_tokens=False
                )
                logger.warning(
                    "Skipping %s micro-batch after %d retries: "
                    "uttids=%s, reason=%s, completions=%s",
                    self.method_name,
                    self.max_rollout_retries,
                    inputs.get("uttids", []),
                    invalid_reason,
                    decoded,
                )
                self._opd_metrics["empty_rollout_rate"].append(1.0)
                self._opd_metrics["rollout_retries"].append(
                    float(self.max_rollout_retries)
                )
                zero_loss = torch.zeros(
                    (), device=self.accelerator.device, dtype=torch.float32,
                    requires_grad=torch.is_grad_enabled(),
                )
                return (zero_loss, {"loss": zero_loss.detach()}) if return_outputs else zero_loss

        self._opd_metrics["empty_rollout_rate"].append(0.0)
        self._opd_metrics["rollout_retries"].append(float(rollout_retry))
        for row, positions in enumerate(response_positions):
            selected_ids = generated_ids[row, positions]
            expected_ids = response_ids[row][response_mask[row]]
            if not torch.equal(selected_ids, expected_ids):
                raise RuntimeError("Qwen rollout positions do not align with response IDs")

        teacher_logits, teacher_reference_logits, teacher_mask = (
            self._teacher_response_logits_pair(inputs, response_ids, response_mask)
        )
        student_logits, student_mask = self._student_response_logits(
            model, inputs, generated_ids, generated_mask, response_positions
        )
        student_reference_logits, student_reference_mask = (
            self._student_reference_response_logits(
                inputs, generated_ids, generated_mask, response_positions
            )
        )
        for name, mask in (
            ("teacher", teacher_mask),
            ("student reference", student_reference_mask),
        ):
            if not torch.equal(student_mask, mask):
                raise RuntimeError(f"Student/{name} response masks differ")

        per_token_divergence, diagnostics, loss_multiplier = self._compass_divergence(
            student_logits,
            student_reference_logits,
            teacher_logits,
            teacher_reference_logits,
        )
        compass_divergence = self._masked_mean(per_token_divergence, student_mask)
        distillation_loss = loss_multiplier * compass_divergence

        sft_loss = distillation_loss.new_zeros(())
        if self.sft_coefficient > 0:
            raw_model = self._raw_model(model)
            model_dtype = next(raw_model.parameters()).dtype
            supervised = raw_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"].to(dtype=model_dtype),
                feature_attention_mask=inputs["feature_attention_mask"],
                labels=inputs["labels"],
            )
            sft_loss = supervised.loss
        loss = distillation_loss + self.sft_coefficient * sft_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {self.method_name} loss: "
                f"distillation_loss={distillation_loss}, "
                f"sft={sft_loss}"
            )

        self._opd_metrics[self.distillation_metric_name].append(
            float(compass_divergence.detach().cpu())
        )
        self._opd_metrics[self.distillation_loss_metric_name].append(
            float(distillation_loss.detach().cpu())
        )
        for name, values in diagnostics.items():
            metric = self._masked_mean(values, student_mask)
            self._opd_metrics[name].append(float(metric.detach().cpu()))
        self._opd_metrics["sft_loss"].append(float(sft_loss.detach().cpu()))
        self._opd_metrics["completion_length"].append(
            float(student_mask.sum(dim=1).float().mean().cpu())
        )
        self._opd_metrics["unsupported_special_rate"].append(
            truncated / max(generated_ids.size(0), 1)
        )
        return (loss, {"loss": loss.detach()}) if return_outputs else loss


def build_compass_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = (
        "CompassOPD: transfer the FireRed RL-vs-released likelihood shift "
        "to a Qwen3-ASR student"
    )
    parser.set_defaults(sft_coefficient=0.0, lr=2e-6)
    parser.add_argument(
        "--student_reference_model_path",
        default="",
        help="Frozen Qwen reference; empty means student_model_path",
    )
    parser.add_argument(
        "--anchor_alpha", type=float, default=0.5,
        help="Student reference anchor strength / inverse teacher-shift scale",
    )
    parser.add_argument(
        "--teacher_shift_clip", type=float, default=0.0,
        help="Optional symmetric log-likelihood shift clip; 0 disables clipping",
    )
    parser.add_argument(
        "--use_flash_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FlashAttention 2 for both Qwen models and the FireRed teacher LLM",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_compass_arg_parser().parse_args(argv)


def validate_compass_args(args: argparse.Namespace) -> dict[str, Any]:
    validate_base_args(args)
    validate_flash_attention_available(args.use_flash_attention)
    if not args.teacher_checkpoint:
        raise ValueError("CompassOPD requires --teacher_checkpoint with an RL checkpoint")
    config = validate_disabled_adapter_reference(args.teacher_checkpoint)
    if not math.isfinite(args.anchor_alpha) or args.anchor_alpha <= 0:
        raise ValueError("anchor_alpha must be finite and positive")
    if args.teacher_shift_clip < 0 or not math.isfinite(args.teacher_shift_clip):
        raise ValueError("teacher_shift_clip must be finite and non-negative")
    reference_path = args.student_reference_model_path or args.student_model_path
    if not Path(reference_path).expanduser().is_dir():
        raise FileNotFoundError(f"student_reference_model_path: {reference_path}")
    return config


def run_training(
    args: argparse.Namespace,
    *,
    trainer_class: type[CompassOpdTrainer] = CompassOpdTrainer,
    collator_class: type[DualFrontendOpdCollator] = DualFrontendOpdCollator,
    trainer_kwargs: dict[str, Any] | None = None,
) -> None:
    rl_config = validate_compass_args(args)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    seed_everything(args.seed)

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_load_kwargs = {}
    if args.use_flash_attention:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info(
        "Attention implementation: %s",
        model_load_kwargs.get("attn_implementation", "model default"),
    )

    logger.info("Loading trainable Qwen3-ASR student: %s", args.student_model_path)
    student_wrapper = Qwen3ASRModel.from_pretrained(
        args.student_model_path,
        dtype=dtype,
        device_map=None,
        **model_load_kwargs,
    )
    student_model = student_wrapper.model
    student_processor = student_wrapper.processor
    patch_qwen_forward(student_model)
    student_model.generation_config = GenerationConfig.from_model_config(
        student_model.config
    )

    reference_path = args.student_reference_model_path or args.student_model_path
    logger.info("Loading frozen Qwen3-ASR reference: %s", reference_path)
    student_reference_wrapper = Qwen3ASRModel.from_pretrained(
        reference_path,
        dtype=dtype,
        device_map=None,
        **model_load_kwargs,
    )
    student_reference_model = student_reference_wrapper.model
    if (
        student_reference_wrapper.processor.tokenizer.get_vocab()
        != student_processor.tokenizer.get_vocab()
    ):
        raise RuntimeError(
            "Current and reference Qwen tokenizers must have identical token-ID mappings"
        )
    patch_qwen_forward(student_reference_model)
    for parameter in student_reference_model.parameters():
        parameter.requires_grad = False
    student_reference_model.eval()

    logger.info(
        "Loading FireRed RL teacher %s (reference=disabled RL adapter)",
        args.teacher_checkpoint,
    )
    teacher_model, teacher_tokenizer, teacher_cmvn = load_teacher(args, dtype)
    shared_vocab_size = shared_tokenizer_prefix(
        student_processor.tokenizer, teacher_tokenizer
    )

    asr_text_token_id = student_processor.tokenizer.convert_tokens_to_ids("<asr_text>")
    if asr_text_token_id is None or asr_text_token_id < shared_vocab_size:
        raise RuntimeError(
            f"Unexpected Qwen <asr_text> token ID {asr_text_token_id}; "
            f"shared vocabulary size is {shared_vocab_size}"
        )
    response_end_token_id = student_processor.tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )
    teacher_end_token_id = teacher_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        response_end_token_id != teacher_end_token_id
        or response_end_token_id is None
        or response_end_token_id >= shared_vocab_size
    ):
        raise RuntimeError(
            "Student and teacher must share <|im_end|> inside the verified vocabulary"
        )
    if student_model.thinker.vocab_size < shared_vocab_size:
        raise RuntimeError("Student output vocabulary is smaller than shared prefix")
    if teacher_model.llm.config.vocab_size < shared_vocab_size:
        raise RuntimeError("Teacher output vocabulary is smaller than shared prefix")

    response_stop_token_ids = frozenset(
        int(token_id)
        for token_id in (
            set(student_processor.tokenizer.all_special_ids)
            | set(teacher_tokenizer.all_special_ids)
        )
        if token_id is not None
    )
    logger.info(
        "%s verified shared IDs=%d, alpha=%g, shift_clip=%s, "
        "RL LoRA rank=%s, RL init=%r",
        trainer_class.method_name,
        shared_vocab_size,
        args.anchor_alpha,
        args.teacher_shift_clip or "disabled",
        rl_config.get("lora_rank"),
        rl_config.get("init_checkpoint"),
    )

    train_dataset = OpdJsonlDataset(
        args.train_file, args.language, args.min_audio_seconds, args.max_audio_seconds
    )
    eval_dataset = None
    if args.eval_file:
        eval_dataset = OpdJsonlDataset(
            args.eval_file, args.language, args.min_audio_seconds, args.max_audio_seconds
        )
    collator = collator_class(
        student_processor, teacher_cmvn, args.language
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )
    trainer = trainer_class(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        student_processor=student_processor,
        student_reference_model=student_reference_model,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_device=torch.device(args.teacher_device),
        teacher_dtype=dtype,
        shared_vocab_size=shared_vocab_size,
        asr_text_token_id=asr_text_token_id,
        response_end_token_id=response_end_token_id,
        response_stop_token_ids=response_stop_token_ids,
        max_completion_length=args.max_completion_length,
        max_rollout_retries=args.max_rollout_retries,
        sampling_temperature=args.sampling_temperature,
        sampling_top_p=args.sampling_top_p,
        sampling_top_k=args.sampling_top_k,
        teacher_max_text_tokens=args.teacher_max_text_tokens,
        teacher_instruction=args.teacher_instruction,
        distillation_temperature=args.temperature,
        topk_logits_k=args.topk_logits_k,
        sft_coefficient=args.sft_coefficient,
        log_rollout_text=args.log_rollout_text,
        anchor_alpha=args.anchor_alpha,
        teacher_shift_clip=(args.teacher_shift_clip or None),
        callbacks=[SaveProcessorCallback(student_processor)],
        **(trainer_kwargs or {}),
    )
    resume = args.resume_from.strip() or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    student_processor.save_pretrained(args.output_dir)
    trainer.save_state()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
