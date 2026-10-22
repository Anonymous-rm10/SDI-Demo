#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .rl.checkpoint import load_rl_checkpoint, save_rl_checkpoint
from .rl.config import RlConfig
from .rl.grpo import group_relative_policy_loss
from .rl.reference import activate_adapter, clone_reference_adapter
from .rl.rewards import group_relative_advantages, transcript_reward
from .rl.semantic_reward import SemanticRewardScorer
from .rl.semantic_rubric import SEMANTIC_RUBRIC_VERSION
from .sft.checkpoint import unwrap_model
from .sft.tracking import WandbTracker
from .tokenizer.llm_tokenizer import LlmTokenizerWrapper
from .train_sft import (autocast_context, barrier, build_optimizer,
                        build_scheduler, build_train_loader,
                        build_valid_loader, configure_trainable_parameters,
                        evaluate, is_main_process, load_released_model,
                        seed_everything, setup_distributed, write_jsonl)

logger = logging.getLogger("fireredasr2s.llm_rl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group-relative RL for FireRedASR2-LLM")
    parser.add_argument("--config", required=True, help="Path to RL JSON config")
    parser.add_argument(
        "--resume_from", default="",
        help="Optional RL checkpoint directory or training_state.pt")
    return parser.parse_args()


def _checkpoint_file(path):
    return os.path.join(path, "training_state.pt") if os.path.isdir(path) else path


def load_initial_delta(path, model, config):
    """Load an SFT policy delta before cloning the frozen RL reference."""
    if not path:
        return {}
    checkpoint_path = _checkpoint_file(path)
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if package.get("format") != "fireredasr2_llm_sft_trainable_v1":
        raise ValueError(
            "init_checkpoint must be an SFT checkpoint; use resume_from for "
            f"an RL checkpoint: {checkpoint_path}")
    saved = package.get("config", {})
    for key in (
            "use_lora", "lora_rank", "lora_alpha", "lora_dropout",
            "lora_target_modules", "full_finetune_llm"):
        expected = config.to_dict()[key]
        if saved.get(key) != expected:
            raise ValueError(
                f"Initial SFT config mismatch for {key}: checkpoint has "
                f"{saved.get(key)!r}, RL config has {expected!r}")
    incompatible = model.load_state_dict(package["model_state_dict"], strict=False)
    logger.info(
        "Loaded initial SFT delta %s; missing=%d unexpected=%d",
        checkpoint_path, len(incompatible.missing_keys),
        len(incompatible.unexpected_keys))
    return {
        name: value.detach().cpu()
        for name, value in package["model_state_dict"].items()
    }


def build_response_mask(response_ids, eos_token_id, pad_token_id):
    """Keep generated tokens through the first EOS and drop generation padding."""
    mask = response_ids.ne(pad_token_id)
    has_eos = []
    for row in range(response_ids.size(0)):
        eos_positions = torch.where(
            response_ids[row].eq(eos_token_id) & mask[row])[0]
        found = eos_positions.numel() > 0
        has_eos.append(found)
        if found:
            first_eos = int(eos_positions[0].item())
            mask[row, first_eos + 1:] = False
    if torch.any(mask.sum(dim=1) == 0):
        raise RuntimeError("Generation returned an empty response")
    return mask, has_eos


def synchronize_gradients(model, world_size):
    if world_size <= 1:
        return
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def reduce_metric_sums(metric_sums, count, device, world_size):
    keys = sorted(metric_sums)
    values = [metric_sums[key] for key in keys] + [float(count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[-1].item()), 1.0)
    return {
        key: float(tensor[index].item()) / total_count
        for index, key in enumerate(keys)
    }


def add_metrics(sums, values):
    for key, value in values.items():
        if torch.is_tensor(value):
            value = float(value.detach().cpu())
        sums[key] = sums.get(key, 0.0) + float(value)


def make_rollout(model, tokenizer, batch, device_batch, config,
                 policy_adapter, reference_adapter, semantic_scorer=None):
    raw_model = unwrap_model(model)
    activate_adapter(raw_model, policy_adapter, policy_adapter)
    raw_model.eval()
    batch_size = len(batch["texts"])
    group_size = config.num_generations

    prompt_ids, prompt_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
        [""] * batch_size,
        tokenizer,
        max_len=config.max_text_tokens,
        decode=True,
        instruction=config.instruction,
    )
    prompt_ids = prompt_ids.to(device_batch["padded_feat"].device)
    prompt_mask = prompt_mask.to(device_batch["padded_feat"].device)

    with torch.no_grad(), autocast_context(config):
        speech_features, speech_lens, _ = raw_model.encode_speech(
            device_batch["padded_feat"], device_batch["feat_lengths"])
        response_ids = raw_model.generate_responses(
            speech_features,
            speech_lens,
            prompt_ids,
            prompt_mask,
            num_return_sequences=group_size,
            max_new_tokens=config.rollout_max_new_tokens,
            temperature=config.rollout_temperature,
            top_p=config.rollout_top_p,
            top_k=config.rollout_top_k,
            repetition_penalty=config.rollout_repetition_penalty,
        )

    expected = batch_size * group_size
    if response_ids.size(0) != expected:
        raise RuntimeError(
            f"Expected {expected} sampled responses, got {response_ids.size(0)}")
    response_mask, has_eos = build_response_mask(
        response_ids,
        raw_model.llm.config.eos_token_id,
        raw_model.llm.config.pad_token_id,
    )
    decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    hypotheses = [LlmTokenizerWrapper.clean_text(text) for text in decoded]
    references = [reference for reference in batch["texts"]
                  for _ in range(group_size)]
    if semantic_scorer is not None:
        semantic_scores = semantic_scorer.score_grouped(
            batch["texts"],
            hypotheses,
            group_size,
            reference_keys=batch["uttids"],
        ).detach().cpu().tolist()
    else:
        semantic_scores = [None] * expected

    reward_details = [
        transcript_reward(
            reference,
            hypothesis,
            eos,
            reward_alpha=config.reward_alpha,
            exact_match_bonus=config.exact_match_bonus,
            missing_eos_penalty=config.missing_eos_penalty,
            repetition_ngram_size=config.repetition_ngram_size,
            repetition_penalty_weight=config.repetition_penalty_weight,
            semantic_score=semantic_score,
            semantic_reward_weight=config.semantic_reward_weight,
        )
        for reference, hypothesis, eos, semantic_score in zip(
            references, hypotheses, has_eos, semantic_scores)
    ]
    rewards = torch.tensor(
        [detail["reward"] for detail in reward_details],
        dtype=torch.float32,
        device=response_ids.device,
    )
    advantages, zero_variance = group_relative_advantages(
        rewards, group_size, config.advantage_epsilon)

    repeated_speech = speech_features.repeat_interleave(group_size, dim=0)
    repeated_speech_lens = speech_lens.repeat_interleave(group_size, dim=0)
    repeated_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    repeated_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)

    activate_adapter(raw_model, reference_adapter, policy_adapter)
    with torch.no_grad(), autocast_context(config):
        reference_logprobs, reference_token_mask = raw_model.score_responses(
            repeated_speech,
            repeated_speech_lens,
            repeated_prompt_ids,
            repeated_prompt_mask,
            response_ids,
            response_mask,
        )
    activate_adapter(raw_model, policy_adapter, policy_adapter)

    reward_metrics = {
        "reward": sum(d["reward"] for d in reward_details) / expected,
        "asr_reward": sum(d["asr_reward"] for d in reward_details) / expected,
        "mer": sum(d["error_rate"] for d in reward_details) / expected,
        "exact_match_rate": sum(d["exact_match"] for d in reward_details) / expected,
        "repetition_fraction": sum(
            d["repetition_fraction"] for d in reward_details) / expected,
        "eos_rate": sum(d["has_eos"] for d in reward_details) / expected,
        "reward_std": float(rewards.view(
            batch_size, group_size).std(dim=1, unbiased=False).mean().cpu()),
        "zero_variance_group_rate": float(zero_variance.float().mean().cpu()),
    }
    if semantic_scorer is not None:
        reward_metrics.update({
            "semantic_score": sum(
                d["semantic_score"] for d in reward_details) / expected,
            "semantic_distance": sum(
                d["semantic_distance"] for d in reward_details) / expected,
            "semantic_penalty": sum(
                d["semantic_penalty"] for d in reward_details) / expected,
        })
    return {
        "speech_features": repeated_speech.detach(),
        "speech_lens": repeated_speech_lens,
        "prompt_ids": repeated_prompt_ids,
        "prompt_mask": repeated_prompt_mask,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "reference_logprobs": reference_logprobs.detach(),
        "reference_token_mask": reference_token_mask,
        "advantages": advantages,
        "reward_metrics": reward_metrics,
        "sample": {
            "uttid": batch["uttids"][0],
            "reference": references[0],
            "hypotheses": hypotheses[:group_size],
            "rewards": [round(d["reward"], 6)
                        for d in reward_details[:group_size]],
            "semantic_scores": [
                None if score is None else round(float(score), 6)
                for score in semantic_scores[:group_size]
            ],
        },
    }


def train(config, cli_resume=""):
    rank, world_size, _, device = setup_distributed()
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format=f"%(asctime)s rank={rank} %(levelname)s %(message)s")
    seed_everything(config.seed, rank)
    if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed_precision=bf16 requires a BF16-capable CUDA GPU")

    if is_main_process(rank):
        with open(os.path.join(config.output_dir, "config.json"), "w",
                  encoding="utf-8") as fout:
            json.dump(config.to_dict(), fout, ensure_ascii=False, indent=2)
        logger.info("RL config: %s", json.dumps(
            config.to_dict(), ensure_ascii=False))

    model, tokenizer = load_released_model(config)
    static_init_state_dict = load_initial_delta(
        config.init_checkpoint, model, config)
    trainable_counts = configure_trainable_parameters(model, config)
    policy_adapter, reference_adapter, policy_parameters = \
        clone_reference_adapter(model)
    logger.info(
        "Cloned policy adapter %s to frozen reference %s (%d trainable policy "
        "parameters)", policy_adapter, reference_adapter, policy_parameters)
    model.to(device)
    model.eval()

    semantic_scorer = None
    if config.semantic_reward_enabled:
        logger.info(
            "Using JEV semantic reward endpoint %s (model=%s, rubric=%s)",
            config.semantic_api_url, config.semantic_expected_model,
            SEMANTIC_RUBRIC_VERSION)
        semantic_scorer = SemanticRewardScorer(
            config.semantic_api_url,
            expected_model=config.semantic_expected_model,
            timeout_seconds=config.semantic_timeout_seconds,
            batch_size=config.semantic_batch_size,
            cache_size=config.semantic_cache_size,
        )

    tracker = WandbTracker(
        config, model=model, enabled=is_main_process(rank))
    if is_main_process(rank):
        tracker.log_config_snapshot(config.to_dict())

    train_loader, train_sampler = build_train_loader(
        config, tokenizer, rank, world_size)
    valid_loader = build_valid_loader(
        config, tokenizer) if is_main_process(rank) else None
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps)
    total_steps = config.max_steps or updates_per_epoch * config.epochs
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, total_steps, config.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.mixed_precision == "fp16")

    global_step = 0
    start_epoch = 0
    start_batch = 0
    resume_path = cli_resume or config.resume_from
    if resume_path:
        package, incompatible = load_rl_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, config)
        global_step = int(package["global_step"])
        start_epoch = int(package["epoch"])
        start_batch = int(package["batch_in_epoch"])
        logger.info(
            "Resumed %s at step=%d epoch=%d batch=%d; missing=%d unexpected=%d",
            resume_path, global_step, start_epoch, start_batch,
            len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    activate_adapter(model, policy_adapter, policy_adapter)
    if is_main_process(rank):
        tracker.log(trainable_counts, "model/trainable_parameters", global_step)

    optimizer.zero_grad(set_to_none=True)
    stop_training = config.max_steps > 0 and global_step >= config.max_steps
    last_epoch = start_epoch
    last_batch = start_batch
    train_log_path = os.path.join(config.output_dir, "train_metrics.jsonl")
    eval_log_path = os.path.join(config.output_dir, "eval_metrics.jsonl")
    metric_sums = {}
    metric_count = 0
    log_start_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        if stop_training:
            break
        train_sampler.set_epoch(epoch)
        accumulation_count = 0
        completed_batches = start_batch if epoch == start_epoch else 0
        with tqdm(
                total=len(train_loader),
                initial=min(completed_batches, len(train_loader)),
                desc=f"RL epoch {epoch + 1}/{config.epochs}",
                unit="rollout",
                dynamic_ncols=True,
                disable=not is_main_process(rank)) as progress:
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                device_batch = {
                    "padded_feat": batch["padded_feat"].to(
                        device, non_blocking=True),
                    "feat_lengths": batch["feat_lengths"].to(
                        device, non_blocking=True),
                }
                rollout = make_rollout(
                    model, tokenizer, batch, device_batch, config,
                    policy_adapter, reference_adapter, semantic_scorer)

                with autocast_context(config):
                    policy_logprobs, token_mask = model.score_responses(
                        rollout["speech_features"],
                        rollout["speech_lens"],
                        rollout["prompt_ids"],
                        rollout["prompt_mask"],
                        rollout["response_ids"],
                        rollout["response_mask"],
                    )
                    if not torch.equal(
                            token_mask, rollout["reference_token_mask"]):
                        raise RuntimeError("Policy/reference token masks differ")
                    loss, loss_metrics = group_relative_policy_loss(
                        policy_logprobs,
                        rollout["reference_logprobs"],
                        token_mask,
                        rollout["advantages"],
                        config.kl_beta,
                    )
                    scaled_loss = loss / config.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite RL loss at epoch={epoch} "
                        f"batch={batch_index}: {loss}")
                scaler.scale(scaled_loss).backward()
                accumulation_count += 1
                last_epoch = epoch
                last_batch = batch_index + 1
                add_metrics(metric_sums, rollout["reward_metrics"])
                add_metrics(metric_sums, {"loss": loss.detach(), **loss_metrics})
                metric_count += 1

                is_last_batch = batch_index == len(train_loader) - 1
                should_update = (
                    accumulation_count >= config.gradient_accumulation_steps or
                    is_last_batch)
                if not should_update:
                    progress.set_postfix(
                        step=global_step,
                        reward=f"{rollout['reward_metrics']['reward']:.3f}",
                        mer=f"{rollout['reward_metrics']['mer']:.3f}",
                        refresh=False)
                    progress.update(1)
                    continue

                scaler.unscale_(optimizer)
                if accumulation_count < config.gradient_accumulation_steps:
                    correction = (
                        config.gradient_accumulation_steps / accumulation_count)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                synchronize_gradients(model, world_size)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters()
                     if parameter.requires_grad],
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                global_step += 1

                metrics = reduce_metric_sums(
                    metric_sums, metric_count, device, world_size)
                metrics.update({
                    "step": global_step,
                    "epoch": epoch,
                    "batch": batch_index,
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds_per_step": (
                        time.time() - log_start_time) /
                        max(metric_count, 1),
                })
                metric_sums = {}
                metric_count = 0
                log_start_time = time.time()
                progress.set_postfix(
                    step=global_step,
                    reward=f"{metrics['reward']:.3f}",
                    mer=f"{metrics['mer']:.3f}",
                    refresh=False)
                progress.update(1)

                if is_main_process(rank) and global_step % config.log_steps == 0:
                    logger.info(
                        "RL train: %s", json.dumps(metrics, ensure_ascii=False))
                    logger.info(
                        "RL sample: %s", json.dumps(
                            rollout["sample"], ensure_ascii=False))
                    write_jsonl(train_log_path, metrics)
                    tracker.log(metrics, "train", global_step)

                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    barrier(world_size)
                    if is_main_process(rank):
                        activate_adapter(model, policy_adapter, policy_adapter)
                        eval_metrics = evaluate(
                            model, tokenizer, valid_loader, config, device,
                            global_step)
                        if eval_metrics:
                            write_jsonl(eval_log_path, eval_metrics)
                            tracker.log(eval_metrics, "eval", global_step)
                    barrier(world_size)
                    activate_adapter(model, policy_adapter, policy_adapter)
                    model.eval()

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    barrier(world_size)
                    if is_main_process(rank):
                        activate_adapter(model, policy_adapter, policy_adapter)
                        checkpoint_dir = save_rl_checkpoint(
                            config.output_dir, model, optimizer, scheduler,
                            scaler, config, global_step, epoch,
                            batch_index + 1, config.keep_last_checkpoints,
                            config.save_optimizer,
                            static_init_state_dict)
                        logger.info("Saved RL checkpoint: %s", checkpoint_dir)
                    barrier(world_size)

                if config.max_steps > 0 and global_step >= config.max_steps:
                    stop_training = True
                    break

        start_batch = 0
        if stop_training:
            break
        barrier(world_size)
        if is_main_process(rank):
            activate_adapter(model, policy_adapter, policy_adapter)
            eval_metrics = evaluate(
                model, tokenizer, valid_loader, config, device, global_step)
            if eval_metrics:
                write_jsonl(eval_log_path, eval_metrics)
                tracker.log(eval_metrics, "eval", global_step)
            checkpoint_dir = save_rl_checkpoint(
                config.output_dir, model, optimizer, scheduler, scaler,
                config, global_step, epoch + 1, 0,
                config.keep_last_checkpoints, config.save_optimizer,
                static_init_state_dict)
            logger.info("Saved epoch RL checkpoint: %s", checkpoint_dir)
        barrier(world_size)

    barrier(world_size)
    if is_main_process(rank):
        activate_adapter(model, policy_adapter, policy_adapter)
        checkpoint_dir = save_rl_checkpoint(
            config.output_dir, model, optimizer, scheduler, scaler,
            config, global_step,
            last_epoch if stop_training else min(config.epochs, last_epoch + 1),
            last_batch if stop_training else 0,
            config.keep_last_checkpoints, config.save_optimizer,
            static_init_state_dict)
        logger.info("RL training complete. Final checkpoint: %s", checkpoint_dir)
        tracker.finish()
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    config = RlConfig.from_json(args.config)
    train(config, args.resume_from)


if __name__ == "__main__":
    main()
