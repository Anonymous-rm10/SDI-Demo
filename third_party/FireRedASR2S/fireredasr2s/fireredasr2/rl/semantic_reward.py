import json
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import torch

from .semantic_rubric import build_semantic_request


class SemanticRewardScorer:
    """JEV six-level fidelity scores via a separate System One server.

    Returns E[level] / 5 in [0, 1], without allocating a reward model on the
    training GPU. Failed/malformed requests abort the rollout; they must never
    become fabricated low rewards or partially scored GRPO groups.
    """

    def __init__(self, api_url="http://127.0.0.1:8020/v1/systemone", *,
                 expected_model="jev-4b", timeout_seconds=120.0,
                 batch_size=4, cache_size=8192):
        url = urlsplit(api_url)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("JEV api_url must be an HTTP(S) endpoint")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("JEV timeout_seconds must be positive")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("JEV batch_size must be a positive integer")
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("JEV cache_size must be a non-negative integer")
        self.api_url = api_url
        self.expected_model = expected_model
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def _parse_response(self, response):
        if not isinstance(response, dict):
            raise ValueError("JEV response must be a JSON object")
        if (self.expected_model and
                response.get("model") != self.expected_model):
            raise ValueError(
                f"JEV model mismatch: expected {self.expected_model!r}, "
                f"received {response.get('model')!r}")
        try:
            answer = response["answers"]["semantic_score"]
            probabilities = answer["probabilities"]
            reported_score = answer["score"]
        except (KeyError, TypeError) as error:
            raise ValueError("JEV response is missing semantic_score fields") from error
        if answer.get("type") != "score":
            raise ValueError("JEV semantic_score must have type=score")
        if (not isinstance(probabilities, dict) or
                set(probabilities) != {str(i) for i in range(6)}):
            raise ValueError("JEV must return all six probabilities (0 through 5)")
        values = [probabilities[str(i)] for i in range(6)]
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
               not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in values):
            raise ValueError("JEV probabilities must be finite numbers in [0, 1]")
        total = sum(values)
        if not math.isclose(total, 1.0, abs_tol=1.0e-5):
            raise ValueError("JEV probabilities must sum to 1")
        expectation = sum(i * p for i, p in enumerate(values))
        if (isinstance(reported_score, bool) or
                not isinstance(reported_score, (int, float)) or
                not math.isfinite(reported_score) or
                not math.isclose(reported_score, expectation, abs_tol=1.0e-4)):
            raise ValueError("JEV score does not match its level probabilities")
        # Divide by probability mass to remove floating-point summation drift.
        # Do not use confidence, argmax, or the obsolete cosine transformation.
        return expectation / (5.0 * total)

    def _score_pair(self, pair):
        request = Request(
            self.api_url,
            data=json.dumps(build_semantic_request(*pair),
                            ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # A local/private reward endpoint should not go through an environment
        # HTTP proxy. Each worker owns its opener; no shared HTTP client state.
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_seconds) as reply:
                response = json.load(reply)
            return self._parse_response(response)
        except HTTPError as error:
            detail = error.read(512).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"JEV semantic reward HTTP {error.code} at {self.api_url}: "
                f"{detail}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"JEV semantic reward request failed at {self.api_url}: "
                "check the server and semantic_timeout_seconds") from error
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Invalid JEV semantic reward response: {error}") from error

    @torch.inference_mode()
    def score_grouped(self, references, hypotheses, group_size,
                      reference_keys=None):
        """Return CPU scores in sample-major order, deduplicating exact pairs.

        reference_keys is accepted for the rollout interface. Cache identity
        includes both full texts, so reusing an utterance ID cannot reuse a
        score for different content. Text is never silently truncated.
        """
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if len(hypotheses) != len(references) * group_size:
            raise ValueError(
                "hypotheses length must equal references * group_size")
        if reference_keys is not None and len(reference_keys) != len(references):
            raise ValueError("reference_keys and references lengths differ")
        if any(not isinstance(text, str) for text in (*references, *hypotheses)):
            raise ValueError("JEV references and hypotheses must be strings")
        pairs = [(reference, hypotheses[i * group_size + j])
                 for i, reference in enumerate(references)
                 for j in range(group_size)]
        unique_pairs = list(dict.fromkeys(pairs))
        scores = {}
        missing = []
        for pair in unique_pairs:
            if pair in self._cache:
                scores[pair] = self._cache[pair]
                self._cache.move_to_end(pair)
            else:
                missing.append(pair)
        if missing:
            with ThreadPoolExecutor(
                    max_workers=min(self.batch_size, len(missing))) as pool:
                # Only commit new cache entries after every request succeeds.
                computed = list(pool.map(self._score_pair, missing))
            scores.update(zip(missing, computed))
            for pair, score in zip(missing, computed):
                if self.cache_size:
                    self._cache[pair] = score
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return torch.tensor([scores[pair] for pair in pairs], dtype=torch.float32)

    @property
    def cached_pair_count(self):
        return len(self._cache)


def main():
    """Smoke-test the configured service without loading/training the ASR model."""
    import argparse

    from .config import RlConfig
    from .rewards import transcript_reward
    from .semantic_rubric import SEMANTIC_RUBRIC_VERSION

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = RlConfig.from_json(args.config)
    scorer = SemanticRewardScorer(
        config.semantic_api_url,
        expected_model=config.semantic_expected_model,
        timeout_seconds=config.semantic_timeout_seconds,
        batch_size=config.semantic_batch_size,
        cache_size=config.semantic_cache_size,
    )
    reference = "拿小泉开涮也是在二零零二年愚人节"
    hypotheses = [reference, "拿小泉开涮也是在二零一二年愚人节",
                  "请把桌上的蓝色雨伞拿给我"]
    scores = scorer.score_grouped([reference], hypotheses, len(hypotheses))
    for hypothesis, score in zip(hypotheses, scores.tolist()):
        detail = transcript_reward(
            reference, hypothesis, True,
            reward_alpha=config.reward_alpha,
            exact_match_bonus=config.exact_match_bonus,
            missing_eos_penalty=config.missing_eos_penalty,
            repetition_ngram_size=config.repetition_ngram_size,
            repetition_penalty_weight=config.repetition_penalty_weight,
            semantic_score=score,
            semantic_reward_weight=config.semantic_reward_weight,
        )
        print(json.dumps({"rubric": SEMANTIC_RUBRIC_VERSION,
                          "reference": reference, "hypothesis": hypothesis,
                          **detail}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
