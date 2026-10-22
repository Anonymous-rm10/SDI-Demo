#!/usr/bin/env python3

import json
import math
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import torch

from train_compassopd import (
    CompassOpdTrainer,
    FireRedTeacherOpdTrainer,
    compass_reverse_kl,
    validate_disabled_adapter_reference,
)


class CompassReverseKlTest(unittest.TestCase):
    def test_matches_closed_form_target(self):
        student = torch.tensor([[0.4, 0.35, 0.25]]).log().requires_grad_()
        student_ref = torch.tensor([[0.5, 0.3, 0.2]]).log()
        teacher = torch.tensor([[0.6, 0.1, 0.3]]).log()
        teacher_ref = torch.tensor([[0.3, 0.4, 0.3]]).log()
        alpha = 0.5

        loss, metrics = compass_reverse_kl(
            student, student_ref, teacher, teacher_ref, alpha=alpha
        )
        ratio = teacher.exp() / teacher_ref.exp()
        target = student_ref.exp() * ratio.pow(1 / alpha)
        target = target / target.sum(dim=-1, keepdim=True)
        expected = (student.exp() * (student - target.log())).sum(dim=-1)

        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(metrics["teacher_reference_kl"]), 0)
        loss.sum().backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_references_are_detached(self):
        student = torch.randn(2, 3, 7, requires_grad=True)
        student_ref = torch.randn(2, 3, 7, requires_grad=True)
        teacher = torch.randn(2, 3, 7, requires_grad=True)
        teacher_ref = torch.randn(2, 3, 7, requires_grad=True)
        loss, _ = compass_reverse_kl(
            student, student_ref, teacher, teacher_ref
        )
        loss.mean().backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(student_ref.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_ref.grad)

    def test_identity_reduces_to_student_anchor(self):
        current = torch.randn(4, 9, requires_grad=True)
        initial = torch.randn(4, 9)
        teacher = torch.randn(4, 9)
        loss, metrics = compass_reverse_kl(
            current, initial, teacher, teacher.clone(), alpha=0.5
        )
        current_log = current.log_softmax(-1)
        initial_log = initial.log_softmax(-1)
        expected = (
            current_log.exp() * (current_log - initial_log)
        ).sum(-1)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            metrics["teacher_shift_abs"], torch.zeros_like(loss)
        )

        zero, _ = compass_reverse_kl(
            initial, initial, teacher, teacher, alpha=0.5
        )
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-6, rtol=0)

    def test_topk_clipping_temperature_and_extreme_logits(self):
        student = torch.tensor([[10000.0, -10000.0, 1.0]], requires_grad=True)
        refs = [
            torch.tensor([[-10000.0, 10000.0, -1.0]]),
            torch.tensor([[2.0, -2.0, 0.0]]),
            torch.tensor([[-2.0, 2.0, 0.0]]),
        ]
        loss, metrics = compass_reverse_kl(
            student, refs[0], refs[1], refs[2],
            alpha=1.0, temperature=2.0, topk=2, shift_clip=5.0,
        )
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_invalid_arguments_and_shapes(self):
        value = torch.zeros(1, 3)
        for alpha in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                compass_reverse_kl(value, value, value, value, alpha=alpha)
        with self.assertRaises(ValueError):
            compass_reverse_kl(value, value[:, :2], value, value)
        with self.assertRaises(ValueError):
            compass_reverse_kl(value, value, value, value, topk=0)
        with self.assertRaises(ValueError):
            compass_reverse_kl(value, value, value, value, shift_clip=-1)


class CheckpointValidationTest(unittest.TestCase):
    def _checkpoint(self, config):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        torch.save(
            {
                "format": "fireredasr2_llm_rl_trainable_v1",
                "config": config,
                "model_state_dict": {},
            },
            root / "training_state.pt",
        )
        with (root / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle)
        return temporary, root

    @staticmethod
    def _valid_config():
        return {
            "init_checkpoint": "",
            "use_lora": True,
            "full_finetune_llm": False,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj",
            ],
        }

    def test_current_rank8_rl_layout_is_supported(self):
        temporary, root = self._checkpoint(self._valid_config())
        self.addCleanup(temporary.cleanup)
        result = validate_disabled_adapter_reference(root)
        self.assertEqual(result["lora_rank"], 8)

    def test_nonempty_initial_checkpoint_is_rejected(self):
        config = self._valid_config()
        config["init_checkpoint"] = "sft/checkpoint"
        temporary, root = self._checkpoint(config)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "init_checkpoint"):
            validate_disabled_adapter_reference(root)

    def test_full_finetune_and_released_lora_layout_are_rejected(self):
        config = self._valid_config()
        config["full_finetune_llm"] = True
        temporary, root = self._checkpoint(config)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "LoRA RL"):
            validate_disabled_adapter_reference(root)

        config = self._valid_config()
        config["lora_rank"] = 64
        temporary, root = self._checkpoint(config)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "released LoRA structure"):
            validate_disabled_adapter_reference(root)


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.response_logits = torch.nn.Parameter(torch.tensor([
            [0.3, 0.2, -0.1],
            [0.1, -0.2, 0.4],
        ]))


class GradientAccumulationTest(unittest.TestCase):
    @mock.patch(
        "fireredasr2s.fireredasr2.train_qwen_opd.Trainer.__init__",
        return_value=None,
    )
    def test_custom_loss_uses_trainer_accumulation_scaling(self, _):
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
        trainer = FireRedTeacherOpdTrainer(
            model=object(),
            args=object(),
            student_processor=types.SimpleNamespace(tokenizer=tokenizer),
            teacher_model=object(),
            teacher_tokenizer=object(),
            teacher_device=torch.device("cpu"),
            teacher_dtype=torch.float32,
            shared_vocab_size=3,
            asr_text_token_id=3,
            response_end_token_id=1,
            response_stop_token_ids=frozenset({1}),
            max_completion_length=8,
            max_rollout_retries=0,
            sampling_temperature=1.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            teacher_max_text_tokens=8,
            teacher_instruction="test",
            distillation_temperature=1.0,
            topk_logits_k=None,
            sft_coefficient=0.0,
            log_rollout_text=False,
        )
        self.assertFalse(trainer.model_accepts_loss_kwargs)


class TrainerFlowTest(unittest.TestCase):
    def test_compute_loss_updates_only_current_student(self):
        trainer = object.__new__(CompassOpdTrainer)
        trainer.max_rollout_retries = 0
        trainer.log_rollout_text = False
        trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        trainer.student_processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                decode=lambda *args, **kwargs: "toy",
                batch_decode=lambda *args, **kwargs: ["toy"],
            )
        )
        trainer.anchor_alpha = 0.5
        trainer.distillation_temperature = 1.0
        trainer.topk_logits_k = None
        trainer.teacher_shift_clip = None
        trainer.sft_coefficient = 0.0
        trainer._opd_metrics = defaultdict(list)

        response_ids = torch.tensor([[4, 2]])
        response_mask = torch.tensor([[True, True]])
        response_positions = [torch.tensor([0, 1])]
        trainer._rollout = lambda model, inputs: (
            response_ids.clone(), response_mask.clone()
        )
        trainer._extract_shared_responses = lambda ids, mask: (
            response_ids.clone(), response_mask.clone(), response_positions, 0
        )

        teacher = torch.tensor([[[0.6, 0.1, 0.3], [0.2, 0.2, 0.6]]]).log()
        teacher_ref = torch.tensor([[[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]]).log()
        student_ref = torch.tensor([[[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]]]).log()
        trainer._teacher_response_logits_pair = lambda *args: (
            teacher, teacher_ref, response_mask
        )
        trainer._student_reference_response_logits = lambda *args: (
            student_ref, response_mask
        )
        trainer._student_response_logits = lambda model, *args: (
            model.response_logits.unsqueeze(0), response_mask
        )

        model = TinyStudent()
        raw_kl, _ = compass_reverse_kl(
            model.response_logits.unsqueeze(0), student_ref, teacher, teacher_ref,
            alpha=trainer.anchor_alpha,
        )
        expected = trainer.anchor_alpha * raw_kl.mean()
        loss = trainer.compute_loss(model, {"uttids": ["toy"]})
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(model.response_logits.grad)
        self.assertGreater(float(model.response_logits.grad.abs().sum()), 0)
        self.assertIn("compass_kl", trainer._opd_metrics)
        self.assertIn("compass_loss", trainer._opd_metrics)


if __name__ == "__main__":
    unittest.main()
