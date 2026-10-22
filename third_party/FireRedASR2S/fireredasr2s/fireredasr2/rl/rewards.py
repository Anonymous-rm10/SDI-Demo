import math

import torch

from ..sft.metrics import edit_distance, mixed_tokens


def normalized_mixed_error_rate(reference, hypothesis):
    reference_tokens = mixed_tokens(reference)
    hypothesis_tokens = mixed_tokens(hypothesis)
    return edit_distance(reference_tokens, hypothesis_tokens) / max(
        len(reference_tokens), 1)


def repetition_fraction(text, ngram_size=3):
    tokens = mixed_tokens(text)
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(tokens[index:index + ngram_size])
              for index in range(len(tokens) - ngram_size + 1)]
    return (len(ngrams) - len(set(ngrams))) / max(len(ngrams), 1)


def transcript_reward(reference, hypothesis, has_eos, reward_alpha=2.0,
                      exact_match_bonus=0.1, missing_eos_penalty=0.05,
                      repetition_ngram_size=3,
                      repetition_penalty_weight=0.1,
                      semantic_score=None,
                      semantic_reward_weight=0.0):
    error_rate = normalized_mixed_error_rate(reference, hypothesis)
    asr_reward = math.exp(-reward_alpha * error_rate)
    exact_bonus = exact_match_bonus if error_rate == 0.0 else 0.0
    repetition = repetition_fraction(hypothesis, repetition_ngram_size)
    eos_penalty = 0.0 if has_eos else missing_eos_penalty
    normalized_semantic_score = 0.0
    semantic_distance = 0.0
    semantic_penalty = 0.0
    if semantic_score is not None:
        normalized_semantic_score = float(semantic_score)
        if (not math.isfinite(normalized_semantic_score) or
                not 0.0 <= normalized_semantic_score <= 1.0):
            raise ValueError("JEV semantic_score must be finite and in [0, 1]")
        semantic_distance = 1.0 - normalized_semantic_score
        semantic_penalty = semantic_reward_weight * semantic_distance
    reward = (
        asr_reward + exact_bonus -
        repetition_penalty_weight * repetition - eos_penalty -
        semantic_penalty
    )
    return {
        "reward": reward,
        "asr_reward": asr_reward,
        "error_rate": error_rate,
        "exact_match": float(error_rate == 0.0),
        "repetition_fraction": repetition,
        "has_eos": float(has_eos),
        "semantic_score": normalized_semantic_score,
        "semantic_distance": semantic_distance,
        "semantic_penalty": semantic_penalty,
    }


def group_relative_advantages(rewards, group_size, epsilon=1.0e-4):
    if rewards.ndim != 1 or rewards.numel() % group_size != 0:
        raise ValueError("rewards must be flat and divisible by group_size")
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - means) / stds.clamp_min(epsilon)
    zero_variance = stds.lt(epsilon)
    advantages = torch.where(
        zero_variance.expand_as(advantages),
        torch.zeros_like(advantages),
        advantages,
    )
    return advantages.reshape(-1), zero_variance.squeeze(1)
