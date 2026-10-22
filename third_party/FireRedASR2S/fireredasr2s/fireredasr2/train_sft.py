#!/usr/bin/env python3

import argparse
import contextlib
import gc
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models.fireredasr_llm import FireRedAsrLlm
from .sft.checkpoint import (load_training_checkpoint, save_checkpoint,
                             unwrap_model)
from .sft.config import SftConfig
from .sft.data import DynamicBatchSampler, SftCollator, SftDataset
from .sft.metrics import ErrorRateStats
from .sft.tracking import WandbTracker
from .tokenizer.llm_tokenizer import LlmTokenizerWrapper

logger = logging.getLogger("fireredasr2s.llm_sft")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue SFT from a released FireRedASR2-LLM checkpoint")
    parser.add_argument("--config", required=True, help="Path to SFT JSON config")
    parser.add_argument(
        "--resume_from", default="",
        help="Optional checkpoint directory or training_state.pt; overrides config")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("FireRedASR2-LLM SFT requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def is_main_process(rank):
    return rank == 0


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def seed_everything(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_released_model(config):
    model_path = os.path.join(config.model_dir, "model.pth.tar")
    encoder_path = os.path.join(config.model_dir, "asr_encoder.pth.tar")
    llm_dir = os.path.join(config.model_dir, "Qwen2-7B-Instruct")
    for path in (model_path, encoder_path, llm_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    package = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = package["args"]
    model_args.encoder_path = encoder_path
    model_args.llm_dir = llm_dir
    model_args.use_flash_attn = config.use_flash_attention
    model_args.torch_dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "float32": "float32",
    }[config.mixed_precision]

    # Always reconstruct the released LoRA first so its weights can be loaded.
    # A custom LoRA is attached only after the released delta is merged.
    released_uses_lora = bool(getattr(model_args, "use_lora", False))
    model_args.use_lora = released_uses_lora
    model_args.lora_rank = FireRedAsrLlm.DEFAULT_LORA_RANK
    model_args.lora_alpha = FireRedAsrLlm.DEFAULT_LORA_ALPHA
    model_args.lora_dropout = FireRedAsrLlm.DEFAULT_LORA_DROPOUT
    model_args.lora_target_modules = list(
        FireRedAsrLlm.DEFAULT_LORA_TARGET_MODULES)
    model_args.freeze_llm = False
    model_args.freeze_encoder = config.freeze_encoder and \
        config.train_encoder_last_n == 0

    model = FireRedAsrLlm.from_args(model_args)
    incompatible = model.load_state_dict(package["model_state_dict"], strict=False)
    logger.info(
        "Loaded released checkpoint; missing=%d unexpected=%d",
        len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected released checkpoint keys (first 20): %s",
            incompatible.unexpected_keys[:20])
    del package
    gc.collect()

    requested_lora_spec = {
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": list(config.lora_target_modules),
    }
    released_lora_spec = FireRedAsrLlm.lora_spec_from_args(model_args)
    same_lora_structure = (
        requested_lora_spec["rank"] == released_lora_spec["rank"] and
        requested_lora_spec["alpha"] == released_lora_spec["alpha"] and
        requested_lora_spec["dropout"] == released_lora_spec["dropout"] and
        set(requested_lora_spec["target_modules"]) ==
        set(released_lora_spec["target_modules"])
    )

    if config.full_finetune_llm:
        if not hasattr(model.llm, "merge_and_unload"):
            raise RuntimeError(
                "full_finetune_llm requires the released PEFT LoRA model")
        logger.info("Merging released LoRA weights into the Qwen base model")
        model.llm = model.llm.merge_and_unload()
        model.llm_config = model.llm.config
    elif config.use_lora and (not released_uses_lora or not same_lora_structure):
        if released_uses_lora:
            if not hasattr(model.llm, "merge_and_unload"):
                raise RuntimeError(
                    "A custom LoRA requires a released PEFT LoRA to merge first")
            logger.info(
                "Custom LoRA differs from the released LoRA. Merging released "
                "weights, then creating a new LoRA: %s", requested_lora_spec)
            model.llm = model.llm.merge_and_unload()
        else:
            logger.info("Creating a new LoRA: %s", requested_lora_spec)
        model_args.lora_rank = config.lora_rank
        model_args.lora_alpha = config.lora_alpha
        model_args.lora_dropout = config.lora_dropout
        model_args.lora_target_modules = list(config.lora_target_modules)
        model.llm = FireRedAsrLlm.add_lora(model.llm, model_args)
        model.llm_config = model.llm.config

    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(
        llm_dir, use_flash_attn=False)
    return model, tokenizer


def configure_trainable_parameters(model, config):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if config.train_adapter:
        for parameter in model.encoder_projector.parameters():
            parameter.requires_grad = True

    if config.full_finetune_llm:
        for parameter in model.llm.parameters():
            parameter.requires_grad = True
    elif config.use_lora:
        lora_count = 0
        for name, parameter in model.llm.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
                lora_count += parameter.numel()
        if lora_count == 0:
            raise RuntimeError("use_lora=True but no LoRA parameters were found")

    if config.train_encoder_last_n > 0:
        layers = model.encoder.layer_stack
        if config.train_encoder_last_n > len(layers):
            raise ValueError(
                f"train_encoder_last_n={config.train_encoder_last_n} exceeds "
                f"the encoder depth {len(layers)}")
        for layer in layers[-config.train_encoder_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    elif not config.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

    model.freeze_encoder = not any(
        parameter.requires_grad for parameter in model.encoder.parameters())
    model.freeze_llm = not any(
        parameter.requires_grad for parameter in model.llm.parameters())
    model.llm.config.use_cache = False

    if config.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()

    counts = {}
    for component, module in (
        ("encoder", model.encoder),
        ("adapter", model.encoder_projector),
        ("llm", model.llm),
    ):
        counts[component] = sum(
            parameter.numel() for parameter in module.parameters()
            if parameter.requires_grad)
    counts["total"] = sum(counts.values())
    if counts["total"] == 0:
        raise RuntimeError("No trainable parameters")
    logger.info("Trainable parameter counts: %s", counts)
    return counts


def set_sft_train_mode(model, config):
    """Enter train mode without enabling dropout in frozen encoder blocks."""
    model.train()
    raw_model = unwrap_model(model)
    if raw_model.freeze_encoder:
        raw_model.encoder.eval()
    elif config.train_encoder_last_n > 0:
        raw_model.encoder.eval()
        for layer in raw_model.encoder.layer_stack[-config.train_encoder_last_n:]:
            layer.train()


def build_optimizer(model, config):
    parameter_groups = []
    seen = set()

    def add_module(module, learning_rate, label):
        decay = []
        no_decay = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if decay:
            parameter_groups.append({
                "params": decay,
                "lr": learning_rate,
                "weight_decay": config.weight_decay,
                "name": f"{label}_decay",
            })
        if no_decay:
            parameter_groups.append({
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "name": f"{label}_no_decay",
            })

    add_module(model.encoder_projector, config.adapter_learning_rate, "adapter")
    add_module(model.llm, config.lora_learning_rate, "llm")
    add_module(model.encoder, config.encoder_learning_rate, "encoder")
    optimizer = torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.95), eps=1.0e-8)
    logger.info(
        "Optimizer groups: %s",
        [(group["name"], group["lr"], len(group["params"]))
         for group in parameter_groups])
    return optimizer


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_train_loader(config, tokenizer, rank, world_size):
    dataset = SftDataset(
        config.train_manifest,
        config.min_audio_seconds,
        config.max_audio_seconds,
        config.audio_root,
    )
    sampler = DynamicBatchSampler(
        dataset.frame_lengths,
        config.max_batch_frames,
        config.max_batch_size,
        config.bucket_size,
        shuffle=True,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )
    collator = SftCollator(
        tokenizer,
        os.path.join(config.model_dir, "cmvn.ark"),
        config.max_text_tokens,
        config.instruction,
        training=True,
        dither=config.dither,
        specaug_time_masks=config.specaug_time_masks,
        specaug_time_width=config.specaug_time_width,
        specaug_freq_masks=config.specaug_freq_masks,
        specaug_freq_width=config.specaug_freq_width,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    return loader, sampler


def build_valid_loader(config, tokenizer):
    if not config.valid_manifest:
        return None
    dataset = SftDataset(
        config.valid_manifest,
        config.min_audio_seconds,
        config.max_audio_seconds,
        config.audio_root,
    )
    sampler = DynamicBatchSampler(
        dataset.frame_lengths,
        config.max_batch_frames,
        config.max_batch_size,
        config.bucket_size,
        shuffle=False,
        seed=config.seed,
    )
    collator = SftCollator(
        tokenizer,
        os.path.join(config.model_dir, "cmvn.ark"),
        config.max_text_tokens,
        config.instruction,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    return loader


def move_batch(batch, device):
    tensor_keys = (
        "padded_feat", "feat_lengths", "input_ids",
        "attention_mask", "labels",
    )
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in tensor_keys
    }


def autocast_context(config):
    if config.mixed_precision == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, tokenizer, valid_loader, config, device, global_step):
    if valid_loader is None:
        return {}
    model = unwrap_model(model)
    model.eval()
    total_loss = 0.0
    loss_batches = 0
    error_stats = ErrorRateStats()
    samples = []
    total_batches = len(valid_loader)
    if config.max_eval_batches > 0:
        total_batches = min(total_batches, config.max_eval_batches)
    with tqdm(
            total=total_batches,
            desc=f"Validation step {global_step}",
            unit="batch",
            dynamic_ncols=True,
            leave=False) as progress:
        for batch_index, batch in enumerate(valid_loader):
            if (config.max_eval_batches > 0 and
                    batch_index >= config.max_eval_batches):
                break
            device_batch = move_batch(batch, device)
            with autocast_context(config):
                outputs = model(**device_batch)
            total_loss += float(outputs.loss.detach().cpu())
            loss_batches += 1

            if config.eval_generate:
                prompt_ids, prompt_mask, _, _ = \
                    LlmTokenizerWrapper.preprocess_texts(
                        [""] * len(batch["texts"]),
                        tokenizer,
                        max_len=config.max_text_tokens,
                        decode=True,
                        instruction=config.instruction,
                    )
                generated_ids = model.transcribe(
                    device_batch["padded_feat"],
                    device_batch["feat_lengths"],
                    prompt_ids.to(device),
                    prompt_mask.to(device),
                    beam_size=config.eval_num_beams,
                    decode_max_len=config.eval_max_new_tokens,
                    repetition_penalty=config.eval_repetition_penalty,
                )
                hypotheses = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                hypotheses = [
                    LlmTokenizerWrapper.clean_text(text) for text in hypotheses]
                for uttid, reference, hypothesis in zip(
                        batch["uttids"], batch["texts"], hypotheses):
                    error_stats.add(reference, hypothesis)
                    if len(samples) < config.log_eval_samples:
                        samples.append({
                            "uttid": uttid,
                            "reference": reference,
                            "hypothesis": hypothesis,
                        })

            progress.set_postfix(
                loss=f"{total_loss / loss_batches:.4f}", refresh=False)
            progress.update(1)

    metrics = {
        "step": global_step,
        "loss": total_loss / max(loss_batches, 1),
    }
    if config.eval_generate:
        metrics.update(error_stats.as_dict())
    logger.info("Validation: %s", json.dumps(metrics, ensure_ascii=False))
    for sample in samples:
        logger.info("Validation sample: %s", json.dumps(sample, ensure_ascii=False))
    return metrics


def write_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(value, ensure_ascii=False) + "\n")


def train(config, cli_resume=""):
    rank, world_size, local_rank, device = setup_distributed()
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format=f"%(asctime)s rank={rank} %(levelname)s %(message)s")
    seed_everything(config.seed, rank)
    if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed_precision=bf16 requires a BF16-capable CUDA GPU")

    if is_main_process(rank):
        with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as fout:
            json.dump(config.to_dict(), fout, ensure_ascii=False, indent=2)
        logger.info("SFT config: %s", json.dumps(config.to_dict(), ensure_ascii=False))

    model, tokenizer = load_released_model(config)
    trainable_counts = configure_trainable_parameters(model, config)
    model.to(device)
    tracker = WandbTracker(
        config, model=model, enabled=is_main_process(rank))
    if is_main_process(rank):
        tracker.log_config_snapshot(config.to_dict())

    train_loader, train_sampler = build_train_loader(
        config, tokenizer, rank, world_size)
    valid_loader = build_valid_loader(config, tokenizer) if is_main_process(rank) else None

    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps)
    total_steps = config.max_steps or updates_per_epoch * config.epochs
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.mixed_precision == "fp16")

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.find_unused_parameters,
        )

    global_step = 0
    start_epoch = 0
    start_batch = 0
    resume_path = cli_resume or config.resume_from
    if resume_path:
        package, incompatible = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, config)
        global_step = int(package["global_step"])
        start_epoch = int(package["epoch"])
        start_batch = int(package["batch_in_epoch"])
        logger.info(
            "Resumed %s at step=%d epoch=%d batch=%d; missing=%d unexpected=%d",
            resume_path, global_step, start_epoch, start_batch,
            len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    if is_main_process(rank):
        tracker.log(
            trainable_counts, "model/trainable_parameters", global_step)

    optimizer.zero_grad(set_to_none=True)
    stop_training = bool(
        config.max_steps > 0 and global_step >= config.max_steps)
    if stop_training:
        logger.info(
            "Resume step %d already reached max_steps=%d; no updates needed",
            global_step, config.max_steps)
    last_epoch = start_epoch
    last_batch = start_batch
    log_loss = 0.0
    log_micro_batches = 0
    log_audio_seconds = 0.0
    log_utterances = 0
    log_start_time = time.time()
    train_log_path = os.path.join(config.output_dir, "train_metrics.jsonl")
    eval_log_path = os.path.join(config.output_dir, "eval_metrics.jsonl")

    for epoch in range(start_epoch, config.epochs):
        if stop_training:
            break
        train_sampler.set_epoch(epoch)
        set_sft_train_mode(model, config)
        epoch_start = time.time()
        accumulation_count = 0
        completed_batches = start_batch if epoch == start_epoch else 0
        with tqdm(
                total=len(train_loader),
                initial=min(completed_batches, len(train_loader)),
                desc=f"Train epoch {epoch + 1}/{config.epochs}",
                unit="batch",
                dynamic_ncols=True,
                disable=not is_main_process(rank)) as progress:
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                device_batch = move_batch(batch, device)
                last_epoch = epoch
                last_batch = batch_index + 1
                accumulation_count += 1
                is_last_batch = batch_index == len(train_loader) - 1
                should_update = (
                    accumulation_count >= config.gradient_accumulation_steps or
                    is_last_batch)
                sync_context = contextlib.nullcontext()
                if world_size > 1 and not should_update:
                    sync_context = model.no_sync()

                with sync_context:
                    with autocast_context(config):
                        outputs = model(**device_batch)
                        loss = outputs.loss / config.gradient_accumulation_steps
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch={epoch} "
                            f"batch={batch_index}: {loss}")
                    scaler.scale(loss).backward()

                batch_loss = float(outputs.loss.detach().cpu())
                log_loss += batch_loss
                log_micro_batches += 1
                log_audio_seconds += sum(batch["durations"])
                log_utterances += len(batch["uttids"])

                if not should_update:
                    progress.set_postfix(
                        step=global_step,
                        loss=f"{batch_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        refresh=False)
                    progress.update(1)
                    continue

                scaler.unscale_(optimizer)
                if accumulation_count < config.gradient_accumulation_steps:
                    correction = (
                        config.gradient_accumulation_steps / accumulation_count)
                    for parameter in unwrap_model(model).parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in unwrap_model(model).parameters()
                     if parameter.requires_grad],
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                global_step += 1
                progress.set_postfix(
                    step=global_step,
                    loss=f"{batch_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    refresh=False)
                progress.update(1)

                if is_main_process(rank) and global_step % config.log_steps == 0:
                    log_elapsed = max(time.time() - log_start_time, 1.0e-6)
                    metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_index,
                        "loss": log_loss / max(log_micro_batches, 1),
                        "grad_norm": float(grad_norm),
                        "utterances_per_second": (
                            log_utterances * world_size / log_elapsed),
                        "audio_seconds_per_second": (
                            log_audio_seconds * world_size / log_elapsed),
                        "learning_rates": {
                            group.get("name", str(index)): group["lr"]
                            for index, group in enumerate(optimizer.param_groups)
                        },
                    }
                    logger.info(
                        "Train: %s", json.dumps(metrics, ensure_ascii=False))
                    write_jsonl(train_log_path, metrics)
                    tracker.log(metrics, "train", global_step)
                    log_loss = 0.0
                    log_micro_batches = 0
                    log_audio_seconds = 0.0
                    log_utterances = 0
                    log_start_time = time.time()

                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    barrier(world_size)
                    if is_main_process(rank):
                        metrics = evaluate(
                            model, tokenizer, valid_loader, config, device,
                            global_step)
                        if metrics:
                            write_jsonl(eval_log_path, metrics)
                            tracker.log(metrics, "eval", global_step)
                    barrier(world_size)
                    set_sft_train_mode(model, config)

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    barrier(world_size)
                    if is_main_process(rank):
                        checkpoint_dir = save_checkpoint(
                            config.output_dir, model, optimizer, scheduler, scaler,
                            config, global_step, epoch, batch_index + 1,
                            config.keep_last_checkpoints, config.save_optimizer)
                        logger.info("Saved checkpoint: %s", checkpoint_dir)
                    barrier(world_size)

                if config.max_steps > 0 and global_step >= config.max_steps:
                    stop_training = True
                    break

        start_batch = 0
        if is_main_process(rank):
            epoch_metrics = {
                "epoch": epoch,
                "duration_seconds": time.time() - epoch_start,
            }
            logger.info(
                "Finished epoch %d in %.1f seconds",
                epoch, epoch_metrics["duration_seconds"])
            tracker.log(epoch_metrics, "epoch", global_step)
        if stop_training:
            break

        barrier(world_size)
        if is_main_process(rank):
            metrics = evaluate(
                model, tokenizer, valid_loader, config, device, global_step)
            if metrics:
                write_jsonl(eval_log_path, metrics)
                tracker.log(metrics, "eval", global_step)
            checkpoint_dir = save_checkpoint(
                config.output_dir, model, optimizer, scheduler, scaler,
                config, global_step, epoch + 1, 0,
                config.keep_last_checkpoints, config.save_optimizer)
            logger.info("Saved epoch checkpoint: %s", checkpoint_dir)
        barrier(world_size)

    barrier(world_size)
    if is_main_process(rank):
        checkpoint_dir = save_checkpoint(
            config.output_dir, model, optimizer, scheduler, scaler,
            config, global_step,
            last_epoch if stop_training else min(config.epochs, last_epoch + 1),
            last_batch if stop_training else 0,
            config.keep_last_checkpoints, config.save_optimizer)
        logger.info("Training complete. Final checkpoint: %s", checkpoint_dir)
        tracker.finish()
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    config = SftConfig.from_json(args.config)
    train(config, args.resume_from)


if __name__ == "__main__":
    main()
