#!/usr/bin/env python3

import math
import types
import unittest
from collections import defaultdict

import torch

from train_compassopsd import CompassOpsdTrainer, compass_generalized_jsd


class CompassGeneralizedJsdTest(unittest.TestCase):
    def setUp(self):
        self.student = torch.tensor([[0.4, 0.35, 0.25]]).log().requires_grad_()
        self.student_ref = torch.tensor([[0.5, 0.3, 0.2]]).log()
        self.teacher = torch.tensor([[0.6, 0.1, 0.3]]).log()
        self.teacher_ref = torch.tensor([[0.3, 0.4, 0.3]]).log()

    def _target(self, alpha=0.5):
        target = self.student_ref.exp() * (
            self.teacher.exp() / self.teacher_ref.exp()
        ).pow(1 / alpha)
        return target / target.sum(dim=-1, keepdim=True)

    def test_matches_probability_definition_and_endpoints(self):
        target = self._target()
        student = self.student.exp()
        for beta in (0.0, 0.2, 0.5, 1.0):
            actual, _ = compass_generalized_jsd(
                self.student,
                self.student_ref,
                self.teacher,
                self.teacher_ref,
                beta=beta,
            )
            if beta == 0:
                expected = (target * (target / student).log()).sum(-1)
            elif beta == 1:
                expected = (student * (student / target).log()).sum(-1)
            else:
                mixture = beta * target + (1 - beta) * student
                expected = (
                    beta * target * (target / mixture).log()
                    + (1 - beta) * student * (student / mixture).log()
                ).sum(-1)
            torch.testing.assert_close(actual, expected)

    def test_gradients_only_reach_current_student(self):
        references = [
            self.student_ref.requires_grad_(),
            self.teacher.requires_grad_(),
            self.teacher_ref.requires_grad_(),
        ]
        loss, metrics = compass_generalized_jsd(
            self.student, *references, topk=2, shift_clip=5.0
        )
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        loss.sum().backward()
        self.assertIsNotNone(self.student.grad)
        self.assertTrue(all(value.grad is None for value in references))

    def test_identity_target_is_zero(self):
        value = torch.randn(3, 7)
        loss, _ = compass_generalized_jsd(value, value, value, value)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)

    def test_invalid_options(self):
        value = torch.zeros(1, 3)
        for beta in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                compass_generalized_jsd(value, value, value, value, beta=beta)
        with self.assertRaises(ValueError):
            compass_generalized_jsd(value, value[:, :2], value, value)


class TrainerFlowTest(unittest.TestCase):
    def test_compute_loss_uses_jsd_without_alpha_multiplier(self):
        trainer = object.__new__(CompassOpsdTrainer)
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
        trainer.beta = 0.5
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

        model = torch.nn.Module()
        model.response_logits = torch.nn.Parameter(torch.tensor([
            [0.3, 0.2, -0.1], [0.1, -0.2, 0.4]
        ]))
        expected, _ = compass_generalized_jsd(
            model.response_logits.unsqueeze(0),
            student_ref,
            teacher,
            teacher_ref,
            alpha=trainer.anchor_alpha,
            beta=trainer.beta,
        )
        loss = trainer.compute_loss(model, {"uttids": ["toy"]})
        torch.testing.assert_close(loss, expected.mean())
        loss.backward()
        self.assertGreater(float(model.response_logits.grad.abs().sum()), 0)
        self.assertIn("compass_jsd", trainer._opd_metrics)
        self.assertIn("compassopsd_loss", trainer._opd_metrics)


if __name__ == "__main__":
    unittest.main()
