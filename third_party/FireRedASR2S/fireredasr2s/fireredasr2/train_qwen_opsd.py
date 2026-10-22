#!/usr/bin/env python3
"""Reference-conditioned OPSD-style distillation: FireRed teacher -> Qwen student.

The teacher is independent and frozen, so this is privileged-context distillation,
not strict same-model self-distillation. Rollouts never receive the reference.
"""

from __future__ import annotations

import math

import torch

from .train_qwen_opd import (
    DualFrontendOpdCollator,
    FireRedTeacherOpdTrainer,
    IGNORE_TOKEN_ID,
    LlmTokenizerWrapper,
    build_arg_parser,
    pad_sequences,
    run_training,
)


DEFAULT_REFERENCE_TEMPLATE = (
    "以下是音频的参考转录，仅作为识别音频的辅助上下文：\n"
    "<reference>\n{reference}\n</reference>\n"
    "请结合音频和参考转录，直接输出音频的转写结果，不要输出解释。"
)


def generalized_jsd(student_logits, teacher_logits, beta=0.5, temperature=1.0, topk=None):
    """Return per-token divergence in float32, with gradients only to student.

    TRL/ms-swift convention: beta=0 -> KL(teacher || student),
    beta=1 -> KL(student || teacher). Otherwise, m = beta*t + (1-beta)*s
    and loss = beta*KL(t || m) + (1-beta)*KL(s || m).
    No additional temperature-squared scaling (same convention as existing OPD).
    """
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    if topk is not None:
        if topk <= 0:
            raise ValueError("topk must be positive")
        indices = student_logits.topk(min(topk, student_logits.size(-1)), dim=-1).indices
        student_logits = student_logits.gather(-1, indices)
        teacher_logits = teacher_logits.gather(-1, indices)
    s_log = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    t_log = torch.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    if beta == 0:
        return (t_log.exp() * (t_log - s_log)).sum(-1)
    if beta == 1:
        return (s_log.exp() * (s_log - t_log)).sum(-1)
    m_log = torch.logaddexp(t_log + math.log(beta), s_log + math.log1p(-beta))
    return (
        beta * (t_log.exp() * (t_log - m_log)).sum(-1)
        + (1 - beta) * (s_log.exp() * (s_log - m_log)).sum(-1)
    )


class DualFrontendOpsdCollator(DualFrontendOpdCollator):
    def __call__(self, records):
        inputs = super().__call__(records)
        # Full reference inputs are used only by the optional supervised CE.
        # Respect either tokenizer padding side when masking their prompts.
        labels = inputs["input_ids"].clone()
        labels.fill_(IGNORE_TOKEN_ID)
        prefix_mask = inputs["prefix_attention_mask"].bool()
        for row in range(len(records)):
            positions = torch.where(inputs["attention_mask"][row].bool())[0]
            target_positions = positions[int(prefix_mask[row].sum()):]
            labels[row, target_positions] = inputs["input_ids"][row, target_positions]
        inputs["labels"] = labels

        # Decoder-only batched generation must end on a real prompt token.
        # Left-pad only the rollout text; audio features keep their own mask.
        ids = inputs["prefix_input_ids"]
        left_ids = torch.full_like(ids, self.student_processor.tokenizer.pad_token_id)
        left_mask = torch.zeros_like(inputs["prefix_attention_mask"])
        for row in range(len(records)):
            tokens = ids[row][prefix_mask[row]]
            left_ids[row, -tokens.numel():] = tokens
            left_mask[row, -tokens.numel():] = 1
        inputs["prefix_input_ids"] = left_ids
        inputs["prefix_attention_mask"] = left_mask
        return inputs


class FireRedTeacherOpsdTrainer(FireRedTeacherOpdTrainer):
    distillation_metric_name = "jsd"

    def __init__(self, *args, beta=0.5, teacher_reference_template=DEFAULT_REFERENCE_TEMPLATE, **kwargs):
        validate_opsd_options(beta, teacher_reference_template)
        self.beta = beta
        self.teacher_reference_template = teacher_reference_template
        super().__init__(*args, **kwargs)

    def _teacher_prompt_inputs(self, inputs, batch_size):
        references = inputs.get("references")
        if references is None or len(references) != batch_size:
            raise ValueError("OPSD requires one reference per teacher sample")
        prompts = []
        for row, reference in enumerate(references):
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Empty OPSD reference at batch row {row}")
            instruction = (
                self.teacher_instruction + "\n\n"
                + self.teacher_reference_template.format(reference=reference)
            )
            # FireRed's user message preserves text after <speech>. The
            # reference belongs in the prompt, never in assistant labels.
            ids, mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                [""], self.teacher_tokenizer, self.teacher_max_text_tokens,
                decode=True, instruction=instruction,
            )
            prompts.append(ids[0][mask[0].bool()])
        return pad_sequences(prompts, self.teacher_tokenizer.pad_token_id, prompts[0].device)

    def _distillation_loss(self, student_logits, teacher_logits):
        return generalized_jsd(
            student_logits, teacher_logits, self.beta,
            self.distillation_temperature, self.topk_logits_k,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Trainer passes CPU batches to a custom prediction_step.
        return super().prediction_step(
            model, self._prepare_inputs(inputs), prediction_loss_only, ignore_keys,
        )


def validate_opsd_options(beta, template):
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    if "{reference}" not in template:
        raise ValueError("teacher_reference_template must contain {reference}")
    try:
        template.format(reference="example")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Invalid teacher_reference_template; only {reference} is supported") from exc


def parse_args(argv=None):
    parser = build_arg_parser()
    parser.description = "OPSD-style training: reference-conditioned FireRed teacher -> Qwen3-ASR"
    parser.set_defaults(teacher_max_text_tokens=1024, sft_coefficient=0.0)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="0: KL(teacher||student); 1: KL(student||teacher); otherwise weighted JSD")
    parser.add_argument("--teacher_reference_template", default=DEFAULT_REFERENCE_TEMPLATE,
                        help="Teacher-only prompt template containing {reference}")
    args = parser.parse_args(argv)
    validate_opsd_options(args.beta, args.teacher_reference_template)
    if args.max_completion_length <= 0 or args.teacher_max_text_tokens <= 0:
        parser.error("max_completion_length and teacher_max_text_tokens must be positive")
    return args


def main():
    args = parse_args()
    run_training(
        args, trainer_class=FireRedTeacherOpsdTrainer,
        collator_class=DualFrontendOpsdCollator,
        trainer_kwargs={"beta": args.beta, "teacher_reference_template": args.teacher_reference_template},
    )


if __name__ == "__main__":
    main()
