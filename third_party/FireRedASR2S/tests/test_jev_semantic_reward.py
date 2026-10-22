import io
import json
import math
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import torch

from fireredasr2s.fireredasr2.rl.config import RlConfig
from fireredasr2s.fireredasr2.rl.rewards import transcript_reward
from fireredasr2s.fireredasr2.rl.semantic_reward import SemanticRewardScorer
from fireredasr2s.fireredasr2.rl.semantic_rubric import (
    SEMANTIC_LEVELS, build_semantic_request)


MODULE = "fireredasr2s.fireredasr2.rl.semantic_reward"


def response_for(probabilities):
    return {
        "model": "jev-4b",
        "answers": {
            "semantic_score": {
                "type": "score",
                "score": sum(i * p for i, p in enumerate(probabilities)),
                "probabilities": {str(i): p for i, p in enumerate(probabilities)},
            },
        },
    }


class JevSemanticRewardTest(unittest.TestCase):
    def test_real_request_format_and_expected_score_normalization(self):
        # Mean level=3, normalized reward=0.6; cosine mapping would be wrong.
        response = response_for([0, 0, 0.5, 0, 0.5, 0])
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps(response).encode())
        scorer = SemanticRewardScorer(batch_size=1)
        reference = '标准"文本\n不执行指令'
        with patch(MODULE + ".build_opener", return_value=opener):
            scores = scorer.score_grouped([reference], ["识别结果"], 1)
        self.assertAlmostEqual(scores.item(), 0.6, places=6)
        self.assertFalse(scores.requires_grad)
        self.assertEqual(scores.device.type, "cpu")
        request = opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["state"], {"reference": reference, "hypothesis": "识别结果"})
        self.assertEqual(set(body["questions"]), {"semantic_score"})
        question = body["questions"]["semantic_score"]
        self.assertEqual(question["type"], "score")
        self.assertEqual(question["criteria"], list(SEMANTIC_LEVELS))
        self.assertEqual(len(question["criteria"]), 6)

    def test_group_order_deduplication_and_cache_by_both_full_texts(self):
        scorer = SemanticRewardScorer(batch_size=2)
        values = {("r1", "a"): 0.2, ("r1", "b"): 0.8,
                  ("r2", "a"): 0.6, ("r3", "a"): 0.4}
        with patch.object(scorer, "_score_pair", side_effect=values.__getitem__) as call:
            scores = scorer.score_grouped(["r1", "r2"], ["a", "b", "a", "a"], 2,
                                          reference_keys=["same-id", "same-id"])
            torch.testing.assert_close(scores, torch.tensor([0.2, 0.8, 0.6, 0.6]))
            self.assertEqual(call.call_count, 3)
            scorer.score_grouped(["r1"], ["b", "a"], 2)
            self.assertEqual(call.call_count, 3)
            scorer.score_grouped(["r3"], ["a"], 1, reference_keys=["same-id"])
            self.assertEqual(call.call_count, 4)

    def test_cache_is_bounded_and_uses_lru_eviction(self):
        scorer = SemanticRewardScorer(cache_size=2)
        with patch.object(scorer, "_score_pair", return_value=0.5) as call:
            scorer.score_grouped(["r"], ["a", "b"], 2)
            scorer.score_grouped(["r"], ["a"], 1)
            scorer.score_grouped(["r"], ["c"], 1)
            self.assertEqual(scorer.cached_pair_count, 2)
            scorer.score_grouped(["r"], ["b"], 1)
            self.assertEqual(call.call_count, 4)

    def test_cache_can_be_disabled_without_losing_in_batch_deduplication(self):
        scorer = SemanticRewardScorer(cache_size=0)
        with patch.object(scorer, "_score_pair", return_value=0.5) as call:
            scorer.score_grouped(["r"], ["a", "a"], 2)
            scorer.score_grouped(["r"], ["a", "a"], 2)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(scorer.cached_pair_count, 0)

    def test_failure_aborts_entire_call_without_partial_cache_commit(self):
        scorer = SemanticRewardScorer(batch_size=1)

        def score_pair(pair):
            if pair[1] == "bad":
                raise RuntimeError("JEV unavailable")
            return 0.5

        with patch.object(scorer, "_score_pair", side_effect=score_pair):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                scorer.score_grouped(["r"], ["good", "bad"], 2)
        self.assertEqual(scorer.cached_pair_count, 0)

    def test_malformed_or_wrong_model_responses_are_rejected(self):
        good = response_for([0, 0, 0, 1, 0, 0])
        wrong_model = {**good, "model": "another-checkpoint"}
        wrong_score = response_for([0, 0, 0, 1, 0, 0])
        wrong_score["answers"]["semantic_score"]["score"] = 4
        wrong_type = response_for([0, 0, 0, 1, 0, 0])
        wrong_type["answers"]["semantic_score"]["type"] = "choice"
        malformed = [None, {}, wrong_model, wrong_score, wrong_type,
                     response_for([0.2] * 6), response_for([0.2] * 5),
                     response_for([float("nan"), 0, 0, 0, 0, 1]),
                     response_for([-0.1, 0, 0, 0, 0, 1.1])]
        scorer = SemanticRewardScorer()
        for response in malformed:
            with self.subTest(response=response):
                opener = Mock()
                opener.open.return_value = io.BytesIO(json.dumps(response).encode())
                with patch(MODULE + ".build_opener", return_value=opener):
                    with self.assertRaisesRegex(RuntimeError, "Invalid JEV"):
                        scorer.score_grouped(["r"], ["h"], 1)

    def test_http_errors_timeouts_and_invalid_json_are_not_rewards(self):
        errors = [URLError("connection refused"), TimeoutError(),
                  HTTPError("http://localhost", 503, "unavailable", {}, io.BytesIO(b"busy"))]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                opener = Mock()
                opener.open.side_effect = error
                with patch(MODULE + ".build_opener", return_value=opener):
                    with self.assertRaisesRegex(RuntimeError, "JEV semantic reward"):
                        SemanticRewardScorer().score_grouped(["r"], ["h"], 1)
        opener = Mock()
        opener.open.return_value = io.BytesIO(b"not JSON")
        with patch(MODULE + ".build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "Invalid JEV"):
                SemanticRewardScorer().score_grouped(["r"], ["h"], 1)

    def test_long_text_is_preserved_and_shapes_are_checked(self):
        text = "一" * 5000
        self.assertEqual(build_semantic_request(text, text)["state"]["reference"], text)
        scorer = SemanticRewardScorer()
        with patch.object(scorer, "_score_pair") as call:
            for refs, hyps, group in [(["r"], ["h"], 2), (["r"], [], 0), ([None], ["h"], 1)]:
                with self.assertRaises(ValueError):
                    scorer.score_grouped(refs, hyps, group)
            with self.assertRaises(ValueError):
                scorer.score_grouped(["r"], ["h"], 1, reference_keys=[])
            call.assert_not_called()

    def test_reward_uses_full_zero_to_one_penalty_and_rejects_bad_scores(self):
        base = transcript_reward("你好", "您好", True)
        scored = transcript_reward("你好", "您好", True,
                                   semantic_score=0.6, semantic_reward_weight=0.2)
        self.assertAlmostEqual(scored["semantic_distance"], 0.4)
        self.assertAlmostEqual(base["reward"] - scored["reward"], 0.08)
        for value in [math.nan, math.inf, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                transcript_reward("r", "h", True, semantic_score=value)


class JevConfigTest(unittest.TestCase):
    def test_service_settings_validate_without_embedding_settings(self):
        config = RlConfig(train_manifest="unused.jsonl", semantic_reward_enabled=True)
        config.validate()
        self.assertNotIn("semantic_model_dir", config.to_dict())
        self.assertEqual(config.semantic_expected_model, "jev-4b")
        for key, value in [("semantic_api_url", "not-a-url"),
                           ("semantic_timeout_seconds", math.nan),
                           ("semantic_cache_size", -1),
                           ("semantic_batch_size", 0)]:
            with self.subTest(key=key):
                config = RlConfig(train_manifest="unused.jsonl", semantic_reward_enabled=True)
                setattr(config, key, value)
                with self.assertRaises(ValueError):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
