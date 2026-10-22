import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from fireredasr2s.fireredasr2.models.fireredasr_llm import FireRedAsrLlm
from fireredasr2s.fireredasr2.rl.config import RlConfig
from fireredasr2s.fireredasr2.rl.grpo import group_relative_policy_loss
from fireredasr2s.fireredasr2.rl.rewards import (
    group_relative_advantages, normalized_mixed_error_rate,
    repetition_fraction, transcript_reward)
from fireredasr2s.fireredasr2.train_rl import build_response_mask, make_rollout


class RewardTest(unittest.TestCase):
    def test_mixed_error_and_exact_match_reward(self):
        self.assertEqual(normalized_mixed_error_rate("你好 world", "你好 world"), 0.0)
        self.assertGreater(
            normalized_mixed_error_rate("你好 world", "你 world"), 0.0)
        exact = transcript_reward("你好 world", "你好 world", True)
        mismatch = transcript_reward("你好 world", "你 world", True)
        self.assertGreater(exact["reward"], mismatch["reward"])
        self.assertEqual(exact["exact_match"], 1.0)

    def test_repetition_fraction(self):
        self.assertEqual(repetition_fraction("a b c", 2), 0.0)
        self.assertGreater(repetition_fraction("a b a b a b", 2), 0.0)

    def test_group_advantages(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 5.0, 5.0, 5.0])
        advantages, zero_variance = group_relative_advantages(rewards, 3)
        self.assertAlmostEqual(float(advantages[:3].mean()), 0.0, places=6)
        self.assertFalse(bool(zero_variance[0]))
        self.assertTrue(bool(zero_variance[1]))
        self.assertEqual(advantages[3:].tolist(), [0.0, 0.0, 0.0])

    def test_semantic_distance_is_a_penalty(self):
        exact_semantics = transcript_reward(
            "你好", "您好", True,
            semantic_score=1.0,
            semantic_reward_weight=0.2)
        unrelated_semantics = transcript_reward(
            "你好", "您好", True,
            semantic_score=0.0,
            semantic_reward_weight=0.2)
        self.assertAlmostEqual(exact_semantics["semantic_penalty"], 0.0)
        self.assertAlmostEqual(unrelated_semantics["semantic_penalty"], 0.2)
        self.assertAlmostEqual(
            exact_semantics["reward"] - unrelated_semantics["reward"], 0.2)


class PolicyLossTest(unittest.TestCase):
    def test_loss_has_policy_gradient_and_nonnegative_kl(self):
        policy = torch.tensor(
            [[-1.0, -2.0], [-2.0, -1.0]], requires_grad=True)
        reference = torch.tensor([[-1.2, -2.1], [-1.8, -0.9]])
        mask = torch.ones_like(policy, dtype=torch.bool)
        advantages = torch.tensor([1.0, -1.0])
        loss, metrics = group_relative_policy_loss(
            policy, reference, mask, advantages, kl_beta=0.1)
        loss.backward()
        self.assertIsNotNone(policy.grad)
        self.assertTrue(torch.isfinite(policy.grad).all())
        self.assertGreaterEqual(float(metrics["kl"]), 0.0)


class RolloutRewardTest(unittest.TestCase):
    def test_jev_scores_flow_into_rewards_metrics_and_group_advantages(self):
        responses = torch.tensor([[30, 2], [40, 2]])
        model = types.SimpleNamespace(
            eval=lambda: None,
            llm=types.SimpleNamespace(config=types.SimpleNamespace(
                eos_token_id=2, pad_token_id=0)),
            encode_speech=lambda *args: (torch.zeros(1, 2, 3), torch.tensor([2]), None),
            generate_responses=lambda *args, **kwargs: responses,
            score_responses=lambda *args: (torch.zeros_like(args[4], dtype=torch.float32), args[5]),
        )
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["你好", "您好"]
        scorer = Mock()
        scorer.score_grouped.return_value = torch.tensor([0.8, 0.2])
        config = RlConfig(num_generations=2, mixed_precision="float32",
                          semantic_reward_weight=0.2)
        prompts = torch.tensor([[10, 99, 20]])
        module = "fireredasr2s.fireredasr2.train_rl"
        with patch(module + ".activate_adapter"), patch(
                module + ".LlmTokenizerWrapper.preprocess_texts",
                return_value=(prompts, prompts.ne(0), None, None)):
            rollout = make_rollout(
                model, tokenizer,
                {"texts": ["你好"], "uttids": ["u1"]},
                {"padded_feat": torch.zeros(1, 2, 3), "feat_lengths": torch.tensor([2])},
                config, "policy", "reference", scorer)
        scorer.score_grouped.assert_called_once_with(
            ["你好"], ["你好", "您好"], 2, reference_keys=["u1"])
        self.assertEqual(rollout["sample"]["semantic_scores"], [0.8, 0.2])
        self.assertAlmostEqual(rollout["reward_metrics"]["semantic_score"], 0.5, places=6)
        self.assertAlmostEqual(rollout["reward_metrics"]["semantic_penalty"], 0.1, places=6)
        torch.testing.assert_close(rollout["advantages"], torch.tensor([1.0, -1.0]))


class ResponseMaskTest(unittest.TestCase):
    def test_mask_stops_after_first_eos(self):
        responses = torch.tensor([
            [10, 2, 0, 0],
            [10, 11, 12, 0],
        ])
        mask, has_eos = build_response_mask(responses, 2, 0)
        self.assertEqual(mask.tolist(), [
            [True, True, False, False],
            [True, True, True, False],
        ])
        self.assertEqual(has_eos, [True, False])


class _FakeLlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            default_speech_token_id=99,
            pad_token_id=0,
        )
        self.embedding = nn.Embedding(128, 8)
        self.output = nn.Linear(8, 128, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
        return types.SimpleNamespace(logits=self.output(inputs_embeds))


class ResponseScoringTest(unittest.TestCase):
    def test_only_response_tokens_are_scored(self):
        model = FireRedAsrLlm(
            encoder=nn.Identity(),
            llm=_FakeLlm(),
            encoder_projector=nn.Identity(),
            freeze_encoder=True,
            freeze_llm=False,
        )
        speech = torch.randn(2, 3, 8)
        speech_lens = torch.tensor([1, 3])
        prompts = torch.tensor([
            [10, 99, 20, 0],
            [10, 99, 20, 0],
        ])
        prompt_mask = prompts.ne(0)
        responses = torch.tensor([
            [30, 31, 2, 0],
            [40, 2, 0, 0],
        ])
        response_mask = responses.ne(0)
        logprobs, token_mask = model.score_responses(
            speech, speech_lens, prompts, prompt_mask,
            responses, response_mask)
        self.assertEqual(token_mask.sum(dim=1).tolist(), [3, 2])
        self.assertTrue(torch.isfinite(logprobs[token_mask]).all())
        (-logprobs[token_mask].mean()).backward()
        self.assertIsNotNone(model.llm.output.weight.grad)


if __name__ == "__main__":
    unittest.main()
