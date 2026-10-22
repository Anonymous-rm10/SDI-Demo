import glob
import json
import os
import random
import shutil

import numpy as np
import torch

from ..sft.checkpoint import trainable_state_dict, unwrap_model


RL_CHECKPOINT_FORMAT = "fireredasr2_llm_rl_trainable_v1"


def save_rl_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                       config, global_step, epoch, batch_in_epoch,
                       keep_last=3, save_optimizer=True,
                       static_init_state_dict=None):
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-step-{global_step:08d}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Keep the checkpoint standalone for inference. If RL started from an SFT
    # delta that also changed the speech projector or encoder, retain those
    # frozen initial weights and overwrite its LoRA entries with the current
    # trainable policy weights.
    model_state_dict = dict(static_init_state_dict or {})
    model_state_dict.update(trainable_state_dict(model))
    state = {
        "format": RL_CHECKPOINT_FORMAT,
        "model_state_dict": model_state_dict,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "config": config.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if save_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

    temporary = os.path.join(checkpoint_dir, "training_state.pt.tmp")
    target = os.path.join(checkpoint_dir, "training_state.pt")
    torch.save(state, temporary)
    os.replace(temporary, target)
    with open(os.path.join(checkpoint_dir, "config.json"), "w",
              encoding="utf-8") as fout:
        json.dump(config.to_dict(), fout, ensure_ascii=False, indent=2)

    checkpoints = sorted(glob.glob(
        os.path.join(output_dir, "checkpoint-step-*")))
    if keep_last > 0:
        for old_checkpoint in checkpoints[:-keep_last]:
            shutil.rmtree(old_checkpoint)
    return checkpoint_dir


def load_rl_checkpoint(path, model, optimizer=None, scheduler=None,
                       scaler=None, expected_config=None):
    if os.path.isdir(path):
        path = os.path.join(path, "training_state.pt")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != RL_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported RL checkpoint format: {path}")
    if expected_config is not None:
        saved = package.get("config", {})
        for key in (
                "init_checkpoint", "use_lora", "lora_rank", "lora_alpha",
                "lora_dropout", "lora_target_modules", "freeze_encoder",
                "train_encoder_last_n", "train_adapter", "num_generations"):
            expected = expected_config.to_dict()[key]
            if saved.get(key) != expected:
                raise ValueError(
                    f"Resume config mismatch for {key}: checkpoint has "
                    f"{saved.get(key)!r}, current config has {expected!r}")

    incompatible = unwrap_model(model).load_state_dict(
        package["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in package:
        optimizer.load_state_dict(package["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in package:
        scheduler.load_state_dict(package["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in package:
        scaler.load_state_dict(package["scaler_state_dict"])
    if "python_rng_state" in package:
        random.setstate(package["python_rng_state"])
    if "numpy_rng_state" in package:
        np.random.set_state(package["numpy_rng_state"])
    if "torch_rng_state" in package:
        torch.set_rng_state(package["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in package:
        torch.cuda.set_rng_state_all(package["cuda_rng_state_all"])
    return package, incompatible