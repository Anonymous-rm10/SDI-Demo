from dataclasses import dataclass
import math
from urllib.parse import urlsplit

from ..sft.config import SftConfig


@dataclass
class RlConfig(SftConfig):
    """Configuration for LoRA-only group-relative ASR RL."""

    output_dir: str = "exp/FireRedASR2-LLM-RL"
    init_checkpoint: str = ""

    # RL v1 intentionally updates only the policy LoRA. The frozen encoder,
    # projector, Qwen base, and reference LoRA are shared efficiently.
    train_adapter: bool = False
    max_batch_frames: int = 2400
    max_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_learning_rate: float = 5.0e-6
    epochs: int = 1

    # Rollout sampling
    num_generations: int = 4
    rollout_max_new_tokens: int = 160
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    rollout_top_k: int = 0
    rollout_repetition_penalty: float = 1.0

    # Reward and loss
    reward_alpha: float = 2.0
    exact_match_bonus: float = 0.1
    missing_eos_penalty: float = 0.05
    repetition_ngram_size: int = 3
    repetition_penalty_weight: float = 0.1
    semantic_reward_enabled: bool = False
    semantic_api_url: str = "http://127.0.0.1:8020/v1/systemone"
    semantic_expected_model: str = "jev-4b"
    semantic_reward_weight: float = 0.1
    semantic_batch_size: int = 4  # Maximum concurrent HTTP requests per rank.
    semantic_timeout_seconds: float = 120.0
    semantic_cache_size: int = 8192  # Bounded (reference, hypothesis) LRU cache.
    advantage_epsilon: float = 1.0e-4
    kl_beta: float = 0.02

    # RL evaluation/tracking defaults
    eval_generate: bool = True
    eval_num_beams: int = 1
    wandb_project: str = "fireredasr2-llm-rl"

    def validate(self):
        super().validate()
        if not self.use_lora or self.full_finetune_llm:
            raise ValueError(
                "RL v1 requires use_lora=true and full_finetune_llm=false")
        if not self.freeze_encoder or self.train_encoder_last_n != 0:
            raise ValueError(
                "RL v1 requires freeze_encoder=true and train_encoder_last_n=0")
        if self.train_adapter:
            raise ValueError("RL v1 requires train_adapter=false")
        if self.dither != 0.0 or self.specaug_time_masks or self.specaug_freq_masks:
            raise ValueError(
                "RL rollouts require deterministic audio features: disable "
                "dither and SpecAugment")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.rollout_max_new_tokens <= 0:
            raise ValueError("rollout_max_new_tokens must be positive")
        if self.rollout_temperature <= 0.0:
            raise ValueError("rollout_temperature must be positive")
        if not 0.0 < self.rollout_top_p <= 1.0:
            raise ValueError("rollout_top_p must be in (0, 1]")
        if self.rollout_top_k < 0:
            raise ValueError("rollout_top_k must be non-negative")
        if self.reward_alpha <= 0.0:
            raise ValueError("reward_alpha must be positive")
        if self.exact_match_bonus < 0.0:
            raise ValueError("exact_match_bonus must be non-negative")
        if self.missing_eos_penalty < 0.0:
            raise ValueError("missing_eos_penalty must be non-negative")
        if self.repetition_ngram_size < 1:
            raise ValueError("repetition_ngram_size must be positive")
        if self.repetition_penalty_weight < 0.0:
            raise ValueError("repetition_penalty_weight must be non-negative")
        semantic_url = urlsplit(self.semantic_api_url)
        if self.semantic_reward_enabled and (
                semantic_url.scheme not in ("http", "https") or
                not semantic_url.netloc):
            raise ValueError("semantic_api_url must be an HTTP(S) endpoint")
        if (not math.isfinite(self.semantic_reward_weight) or
                self.semantic_reward_weight < 0.0):
            raise ValueError("semantic_reward_weight must be non-negative")
        if (not isinstance(self.semantic_batch_size, int) or
                self.semantic_batch_size <= 0):
            raise ValueError("semantic_batch_size must be positive")
        if (not math.isfinite(self.semantic_timeout_seconds) or
                self.semantic_timeout_seconds <= 0.0):
            raise ValueError("semantic_timeout_seconds must be positive")
        if (not isinstance(self.semantic_cache_size, int) or
                self.semantic_cache_size < 0):
            raise ValueError("semantic_cache_size must be non-negative")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")
        if self.kl_beta < 0.0:
            raise ValueError("kl_beta must be non-negative")
