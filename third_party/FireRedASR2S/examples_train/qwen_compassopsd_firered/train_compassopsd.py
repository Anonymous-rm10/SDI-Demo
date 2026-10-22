#!/usr/bin/env python3
"""Compass target transfer with OPSD-style privileged context and JSD."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from examples_train.qwen_compassopd_firered.train_compassopd import (
    CompassOpdTrainer,
    _validate_same_shape,
    build_compass_arg_parser,
    run_training,
)
from fireredasr2s.fireredasr2.train_qwen_opsd import (
    DEFAULT_REFERENCE_TEMPLATE,
    DualFrontendOpsdCollator,
    FireRedTeacherOpsdTrainer,
    validate_opsd_options,
)


def compass_generalized_jsd(
    student_logits: torch.Tensor,
    student_reference_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_reference_logits: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    temperature: float = 1.0,
    topk: int | None = None,
    shift_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return JSD(student, Compass target) per token and diagnostics.

    The target is

        q = normalize(student_ref * exp((teacher - teacher_ref) / alpha)).

    ``beta`` follows the repository OPSD convention: 0 is KL(q || student),
    1 is KL(student || q), and intermediate values are generalized JSD.  Only
    ``student_logits`` receives gradients.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
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
    student_reference_logits = student_reference_logits.detach()
    teacher_logits = teacher_logits.detach()
    teacher_reference_logits = teacher_reference_logits.detach()

    if topk is not None:
        indices = student_logits.detach().topk(
            min(topk, student_logits.size(-1)), dim=-1
        ).indices
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
    ).detach()

    student_prob = student_log.exp()
    target_prob = target_log.exp()
    if beta == 0:
        per_token_jsd = (target_prob * (target_log - student_log)).sum(dim=-1)
    elif beta == 1:
        per_token_jsd = (student_prob * (student_log - target_log)).sum(dim=-1)
    else:
        mixture_log = torch.logaddexp(
            target_log + math.log(beta),
            student_log + math.log1p(-beta),
        )
        per_token_jsd = (
            beta * (target_prob * (target_log - mixture_log)).sum(dim=-1)
            + (1 - beta)
            * (student_prob * (student_log - mixture_log)).sum(dim=-1)
        )

    teacher_prob = teacher_log.exp()
    diagnostics = {
        "teacher_shift_abs": unclipped_shift.abs().mean(dim=-1),
        "teacher_shift_rms": unclipped_shift.square().mean(dim=-1).sqrt(),
        "teacher_reference_kl": (
            teacher_prob * (teacher_log - teacher_reference_log)
        ).sum(dim=-1),
        "student_reference_kl": (
            student_prob * (student_log - student_reference_log)
        ).sum(dim=-1),
        "compass_target_entropy": -(target_prob * target_log).sum(dim=-1),
    }
    return per_token_jsd, diagnostics


class CompassOpsdTrainer(CompassOpdTrainer):
    method_name = "CompassOPSD"
    distillation_metric_name = "compass_jsd"
    distillation_loss_metric_name = "compassopsd_loss"

    def __init__(
        self,
        *args,
        beta: float = 0.5,
        teacher_reference_template: str = DEFAULT_REFERENCE_TEMPLATE,
        **kwargs,
    ):
        validate_opsd_options(beta, teacher_reference_template)
        self.beta = beta
        self.teacher_reference_template = teacher_reference_template
        super().__init__(*args, **kwargs)

    def _teacher_prompt_inputs(self, inputs, batch_size):
        # Both FireRed policies see the same privileged reference. Their ratio
        # therefore isolates the RL shift rather than a prompt-content shift.
        return FireRedTeacherOpsdTrainer._teacher_prompt_inputs(
            self, inputs, batch_size
        )

    def _compass_divergence(
        self,
        student_logits,
        student_reference_logits,
        teacher_logits,
        teacher_reference_logits,
    ):
        per_token_jsd, diagnostics = compass_generalized_jsd(
            student_logits,
            student_reference_logits,
            teacher_logits,
            teacher_reference_logits,
            alpha=self.anchor_alpha,
            beta=self.beta,
            temperature=self.distillation_temperature,
            topk=self.topk_logits_k,
            shift_clip=self.teacher_shift_clip,
        )
        # Follow the existing OPSD convention: no temperature^2 or alpha loss
        # multiplier. Alpha shapes the Compass target itself.
        return per_token_jsd, diagnostics, 1.0

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(
            model, self._prepare_inputs(inputs), prediction_loss_only, ignore_keys
        )


def build_compassopsd_arg_parser() -> argparse.ArgumentParser:
    parser = build_compass_arg_parser()
    parser.description = (
        "CompassOPSD: transfer the reference-conditioned FireRed RL shift "
        "to Qwen with generalized JSD"
    )
    parser.set_defaults(teacher_max_text_tokens=1024)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="0: KL(target||student); 1: KL(student||target); otherwise generalized JSD",
    )
    parser.add_argument(
        "--teacher_reference_template",
        default=DEFAULT_REFERENCE_TEMPLATE,
        help="Teacher-only prompt template containing {reference}",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_compassopsd_arg_parser().parse_args(argv)
    validate_opsd_options(args.beta, args.teacher_reference_template)
    return args


def main() -> None:
    args = parse_args()
    run_training(
        args,
        trainer_class=CompassOpsdTrainer,
        collator_class=DualFrontendOpsdCollator,
        trainer_kwargs={
            "beta": args.beta,
            "teacher_reference_template": args.teacher_reference_template,
        },
    )


if __name__ == "__main__":
    main()
