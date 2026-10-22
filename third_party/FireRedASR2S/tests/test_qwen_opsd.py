import math
import types
import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from fireredasr2s.fireredasr2.models.fireredasr_llm import FireRedAsrLlm
from fireredasr2s.fireredasr2.train_qwen_opsd import (
    DEFAULT_REFERENCE_TEMPLATE,
    DualFrontendOpsdCollator,
    FireRedTeacherOpsdTrainer,
    generalized_jsd,
    parse_args,
)


class DivergenceTest(unittest.TestCase):
    def test_jsd_and_kl_match_probability_definition(self):
        s = torch.tensor([[[0.2, 0.3, 0.5]]], requires_grad=True)
        t = torch.tensor([[[0.6, 0.1, 0.3]]], requires_grad=True)
        for beta in (0.0, 0.2, 0.5, 1.0):
            if beta == 0:
                expected = (t * (t / s).log()).sum(-1)
            elif beta == 1:
                expected = (s * (s / t).log()).sum(-1)
            else:
                m = beta * t + (1 - beta) * s
                expected = (beta * t * (t / m).log() + (1 - beta) * s * (s / m).log()).sum(-1)
            actual = generalized_jsd(s.log(), t.log(), beta)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)

    def test_jsd_symmetry_identity_and_extreme_logits(self):
        s = torch.tensor([[-10000.0, 10000.0]], dtype=torch.bfloat16, requires_grad=True)
        t = -s.detach()
        loss = generalized_jsd(s, t)
        torch.testing.assert_close(loss, torch.tensor([math.log(2)]))
        torch.testing.assert_close(loss, generalized_jsd(t, s))
        torch.testing.assert_close(generalized_jsd(s, s), torch.zeros(1))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_temperature_and_student_topk(self):
        s = torch.tensor([[3.0, 1.0, 2.0]], requires_grad=True)
        t = torch.tensor([[0.0, 9.0, 1.0]])
        expected = generalized_jsd(s[:, [0, 2]] / 2, t[:, [0, 2]] / 2)
        torch.testing.assert_close(generalized_jsd(s, t, temperature=2, topk=2), expected)
        torch.testing.assert_close(generalized_jsd(s, t, topk=100), generalized_jsd(s, t))


class ToyTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        # Keep one speech marker and variable-length user context, ending at
        # the assistant boundary. IDs remain within the toy shared vocabulary.
        return [1, 9] + [10 + ord(c) % 200 for c in messages[0]["content"]] + [3]


class ToyLlm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 4)
        self.head = torch.nn.Linear(4, 256)
        self.config = types.SimpleNamespace(pad_token_id=0, default_speech_token_id=9)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **kwargs):
        return types.SimpleNamespace(logits=self.head(inputs_embeds))


class ToyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = ToyLlm()
        self.requires_grad_(False)
        self.eval()

    def encode_speech(self, features, lengths):
        return features, lengths, None

    def _merge_input_ids_with_speech_features(self, *args, **kwargs):
        result = FireRedAsrLlm._merge_input_ids_with_speech_features(self, *args, **kwargs)
        self.scored_labels = result[2].clone()
        return result


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(256))

    def forward(self, input_ids, **kwargs):
        return types.SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 256))


def toy_trainer():
    trainer = object.__new__(FireRedTeacherOpsdTrainer)
    trainer.teacher_tokenizer = ToyTokenizer()
    trainer.teacher_instruction = "transcribe"
    trainer.teacher_reference_template = DEFAULT_REFERENCE_TEMPLATE
    trainer.teacher_max_text_tokens = 1024
    trainer.teacher_model = ToyTeacher()
    trainer.teacher_device = torch.device("cpu")
    trainer.teacher_dtype = torch.float32
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    trainer.distillation_vocab_ids = torch.tensor([i for i in range(256) if i not in {0, 1, 9}])
    trainer.distillation_vocab_size = len(trainer.distillation_vocab_ids)
    trainer._distillation_vocab_ids_by_device = {}
    trainer.shared_vocab_size = 256
    trainer.response_stop_token_ids = frozenset({0, 1, 2, 9})
    trainer.response_end_token_id = 2
    trainer.student_processor = types.SimpleNamespace(tokenizer=trainer.teacher_tokenizer)
    trainer.max_rollout_retries = 1
    trainer.log_rollout_text = False
    trainer.distillation_temperature = 1.0
    trainer.topk_logits_k = None
    trainer.sft_coefficient = 0.0
    trainer.beta = 0.5
    trainer._opd_metrics = defaultdict(list)
    return trainer


class TrainingFlowTest(unittest.TestCase):
    def test_privileged_prompt_alignment_and_student_only_update(self):
        torch.manual_seed(7)
        trainer = toy_trainer()
        student = ToyStudent()
        teacher_before = [p.clone() for p in trainer.teacher_model.parameters()]
        generated = torch.tensor([[20, 21, 2], [22, 2, 0]])
        trainer._rollout = lambda model, inputs: (generated, generated.ne(0))
        inputs = {
            "references": ["short reference", "a much longer different reference"],
            "teacher_features": torch.randn(2, 3, 4),
            "teacher_feature_lengths": torch.tensor([2, 3]),
            "prefix_input_ids": torch.tensor([[0, 1, 3], [1, 4, 3]]),
            "prefix_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "prefix_input_features": torch.randn(2, 3, 4),
            "prefix_feature_attention_mask": torch.ones(2, 3),
        }
        optimizer = torch.optim.SGD(student.parameters(), lr=1.0)
        initial = trainer.compute_loss(student, inputs)
        initial.backward()
        self.assertGreater(student.logits.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertLess(trainer.compute_loss(student, inputs).item(), initial.item())
        labels = trainer.teacher_model.scored_labels
        self.assertEqual(labels[0][labels[0] != -100].tolist(), [20, 21, 2])
        self.assertEqual(labels[1][labels[1] != -100].tolist(), [22, 2])
        for row, message in enumerate(trainer.teacher_tokenizer.messages[:2]):
            self.assertIn(inputs["references"][row], message[0]["content"])
            self.assertEqual(message[1]["content"], "")
        for before, after in zip(teacher_before, trainer.teacher_model.parameters()):
            torch.testing.assert_close(before, after)
            self.assertIsNone(after.grad)
        self.assertIn("jsd", trainer._opd_metrics)
        self.assertNotIn("kl", trainer._opd_metrics)

    def test_missing_reference_and_overlong_prompt_fail(self):
        trainer = toy_trainer()
        for references in ([], [""], [None]):
            with self.assertRaises(ValueError):
                trainer._teacher_prompt_inputs({"references": references}, 1)
        trainer.teacher_max_text_tokens = 2
        with self.assertRaisesRegex(ValueError, "exceeding max_len"):
            trainer._teacher_prompt_inputs({"references": ["reference"]}, 1)

    def test_collator_left_pads_rollouts_and_masks_reference_ce(self):
        collator = object.__new__(DualFrontendOpsdCollator)
        collator.student_processor = types.SimpleNamespace(tokenizer=ToyTokenizer())
        base_inputs = {
            "input_ids": torch.tensor([[1, 3, 20, 2, 0], [1, 4, 3, 21, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
            "prefix_input_ids": torch.tensor([[1, 3, 0], [1, 4, 3]]),
            "prefix_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
        with patch("fireredasr2s.fireredasr2.train_qwen_opsd.DualFrontendOpdCollator.__call__",
                   return_value=base_inputs):
            result = collator([None, None])
        self.assertEqual(result["prefix_input_ids"].tolist(), [[0, 1, 3], [1, 4, 3]])
        self.assertEqual(result["labels"].tolist(), [[-100, -100, 20, 2, -100], [-100, -100, -100, 21, 2]])

    def test_cli_defaults_and_invalid_beta(self):
        argv = ["--student_model_path", "student", "--teacher_model_path", "teacher",
                "--train_file", "train.jsonl", "--output_dir", "output"]
        args = parse_args(argv)
        self.assertEqual(args.beta, 0.5)
        self.assertEqual(args.sft_coefficient, 0)
        self.assertEqual(args.teacher_max_text_tokens, 1024)
        with self.assertRaises(ValueError):
            parse_args(argv + ["--beta", "nan"])


if __name__ == "__main__":
    unittest.main()
