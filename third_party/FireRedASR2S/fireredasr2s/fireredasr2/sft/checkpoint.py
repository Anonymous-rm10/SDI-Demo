import glob
import json
import os
import shutil

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def trainable_state_dict(model):
    model = unwrap_model(model)
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                    config, global_step, epoch, batch_in_epoch,
                    keep_last=3, save_optimizer=True):
    checkpoint_dir = os.path.join(
        output_dir, f"checkpoint-step-{global_step:08d}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "format": "fireredasr2_llm_sft_trainable_v1",
        "model_state_dict": trainable_state_dict(model),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "config": config.to_dict(),
    }
    if save_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
    temporary = os.path.join(checkpoint_dir, "training_state.pt.tmp")
    target = os.path.join(checkpoint_dir, "training_state.pt")
    torch.save(state, temporary)
    os.replace(temporary, target)
    with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as fout:
        json.dump(config.to_dict(), fout, ensure_ascii=False, indent=2)

    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-step-*")))
    if keep_last > 0:
        for old_checkpoint in checkpoints[:-keep_last]:
            shutil.rmtree(old_checkpoint)
    return checkpoint_dir


def load_training_checkpoint(path, model, optimizer=None,
                             scheduler=None, scaler=None,
                             expected_config=None):
    if os.path.isdir(path):
        path = os.path.join(path, "training_state.pt")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "fireredasr2_llm_sft_trainable_v1":
        raise ValueError(f"Unsupported SFT checkpoint format: {path}")
    if expected_config is not None:
        saved_config = package.get("config", {})
        for key in (
            "use_lora", "lora_rank", "lora_alpha", "lora_dropout",
            "lora_target_modules", "full_finetune_llm", "freeze_encoder",
            "train_encoder_last_n", "train_adapter",
        ):
            expected_value = expected_config.to_dict()[key]
            if saved_config.get(key) != expected_value:
                raise ValueError(
                    f"Resume config mismatch for {key}: checkpoint has "
                    f"{saved_config.get(key)!r}, current config has "
                    f"{expected_value!r}")
    incompatible = unwrap_model(model).load_state_dict(
        package["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in package:
        optimizer.load_state_dict(package["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in package:
        scheduler.load_state_dict(package["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in package:
        scaler.load_state_dict(package["scaler_state_dict"])
    return package, incompatible
