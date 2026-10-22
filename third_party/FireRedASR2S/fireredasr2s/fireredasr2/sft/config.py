import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class SftConfig:
    # Paths
    model_dir: str = "pretrained_models/FireRedASR2-LLM"
    train_manifest: str = ""
    valid_manifest: str = ""
    output_dir: str = "exp/FireRedASR2-LLM-SFT"
    resume_from: str = ""
    audio_root: str = ""

    # Data
    min_audio_seconds: float = 0.2
    max_audio_seconds: float = 40.0
    max_text_tokens: int = 192
    max_batch_frames: int = 6000
    max_batch_size: int = 8
    bucket_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    instruction: str = "请转写音频为文字"
    dither: float = 0.0
    specaug_time_masks: int = 0
    specaug_time_width: int = 40
    specaug_freq_masks: int = 0
    specaug_freq_width: int = 8

    # Trainable modules
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ])
    full_finetune_llm: bool = False
    freeze_encoder: bool = True
    train_encoder_last_n: int = 0
    train_adapter: bool = True

    # Runtime
    seed: int = 42
    epochs: int = 3
    max_steps: int = 0
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"  # bf16, fp16, or float32
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    find_unused_parameters: bool = False

    # Optimizer
    adapter_learning_rate: float = 1.0e-4
    lora_learning_rate: float = 2.0e-5
    encoder_learning_rate: float = 2.0e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Logging, validation, and checkpointing
    log_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    keep_last_checkpoints: int = 3
    max_eval_batches: int = 0
    eval_generate: bool = True
    eval_max_new_tokens: int = 160
    eval_num_beams: int = 1
    eval_repetition_penalty: float = 1.0
    log_eval_samples: int = 5
    save_optimizer: bool = True

    # Weights & Biases (rank zero only)
    wandb_enabled: bool = False
    wandb_project: str = "fireredasr2-llm-sft"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_group: str = ""
    wandb_tags: list = field(default_factory=list)
    wandb_mode: str = "online"  # online, offline, or disabled
    wandb_run_id: str = ""
    wandb_watch: str = "none"  # none, gradients, parameters, or all
    wandb_watch_freq: int = 100

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as fin:
            values = json.load(fin)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown SFT config fields: {unknown}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if not self.train_manifest:
            raise ValueError("train_manifest is required")
        if self.mixed_precision not in ("bf16", "fp16", "float32"):
            raise ValueError("mixed_precision must be bf16, fp16, or float32")
        if self.use_flash_attention and self.mixed_precision == "float32":
            raise ValueError(
                "FlashAttention requires mixed_precision=bf16 or fp16")
        if self.use_lora and self.full_finetune_llm:
            raise ValueError("use_lora and full_finetune_llm are mutually exclusive")
        if not isinstance(self.lora_rank, int) or self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if not isinstance(self.lora_alpha, int) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if (not isinstance(self.lora_target_modules, list) or
                not self.lora_target_modules or
                not all(isinstance(name, str) and name
                        for name in self.lora_target_modules)):
            raise ValueError("lora_target_modules must be a non-empty string list")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("lora_target_modules must not contain duplicates")
        if self.train_encoder_last_n < 0:
            raise ValueError("train_encoder_last_n must be non-negative")
        if self.max_batch_frames <= 0 or self.max_batch_size <= 0:
            raise ValueError("Batch limits must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive; max_steps is an optional cap")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.log_steps <= 0:
            raise ValueError("log_steps must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.eval_max_new_tokens <= 0:
            raise ValueError("eval_max_new_tokens must be positive")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.wandb_watch not in ("none", "gradients", "parameters", "all"):
            raise ValueError(
                "wandb_watch must be none, gradients, parameters, or all")
        if self.wandb_watch_freq <= 0:
            raise ValueError("wandb_watch_freq must be positive")
        if (not isinstance(self.wandb_tags, list) or
                not all(isinstance(tag, str) for tag in self.wandb_tags)):
            raise ValueError("wandb_tags must be a string list")
        if not any((
            self.train_adapter,
            self.use_lora,
            self.full_finetune_llm,
            not self.freeze_encoder,
            self.train_encoder_last_n > 0,
        )):
            raise ValueError("The configuration leaves every model component frozen")

    def to_dict(self):
        return asdict(self)
