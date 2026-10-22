import types
import unittest

import torch

from fireredasr2s.fireredasr2.models.fireredasr_llm import FireRedAsrLlm
from fireredasr2s.fireredasr2.sft.data import DynamicBatchSampler
from fireredasr2s.fireredasr2.tokenizer.llm_tokenizer import IGNORE_TOKEN_ID


class MergeSpeechFeaturesTest(unittest.TestCase):
    def setUp(self):
        config = types.SimpleNamespace(default_speech_token_id=99)
        self.owner = types.SimpleNamespace(
            llm=types.SimpleNamespace(config=config))

    def test_variable_speech_lengths_and_right_padding(self):
        speech = torch.tensor([
            [[100.0], [101.0], [0.0], [0.0]],
            [[200.0], [201.0], [202.0], [203.0]],
        ])
        speech_lens = torch.tensor([2, 4])
        input_ids = torch.tensor([
            [10, 99, 20, 21, 0],
            [10, 99, 20, 0, 0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ], dtype=torch.bool)
        text_embeddings = input_ids.float().unsqueeze(-1)
        labels = torch.tensor([
            [IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, 20, 21, IGNORE_TOKEN_ID],
            [IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, 20, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID],
        ])

        embeddings, mask, merged_labels = \
            FireRedAsrLlm._merge_input_ids_with_speech_features(
                self.owner, speech, text_embeddings, input_ids,
                attention_mask, labels, speech_lens, "right")

        self.assertEqual(tuple(embeddings.shape), (2, 6, 1))
        self.assertEqual(mask.tolist(), [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ])
        self.assertEqual(embeddings[0, :, 0].tolist(), [
            10.0, 100.0, 101.0, 20.0, 21.0, 0.0])
        self.assertEqual(embeddings[1, :, 0].tolist(), [
            10.0, 200.0, 201.0, 202.0, 203.0, 20.0])
        self.assertEqual(merged_labels[0].tolist(), [
            IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID,
            20, 21, IGNORE_TOKEN_ID])

    def test_generation_uses_left_padding(self):
        speech = torch.ones(2, 3, 1)
        input_ids = torch.tensor([[10, 99, 20], [10, 99, 20]])
        text_embeddings = input_ids.float().unsqueeze(-1)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        _, merged_mask, _ = \
            FireRedAsrLlm._merge_input_ids_with_speech_features(
                self.owner, speech, text_embeddings, input_ids, mask,
                labels=None, speech_lens=torch.tensor([1, 3]),
                padding_side="left")
        self.assertEqual(merged_mask.tolist(), [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])


class DynamicBatchSamplerTest(unittest.TestCase):
    def test_all_ranks_have_equal_batch_counts(self):
        lengths = [100, 110, 200, 210, 300, 310, 400]
        rank0 = DynamicBatchSampler(
            lengths, 800, 4, shuffle=False, rank=0, world_size=2)
        rank1 = DynamicBatchSampler(
            lengths, 800, 4, shuffle=False, rank=1, world_size=2)
        self.assertEqual(len(rank0), len(rank1))
        self.assertGreater(len(rank0), 0)


if __name__ == "__main__":
    unittest.main()
