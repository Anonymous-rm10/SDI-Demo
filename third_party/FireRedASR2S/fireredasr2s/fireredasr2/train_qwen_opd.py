#!/usr/bin/env python3
"""On-policy distillation from FireRedASR2-LLM to Qwen3-ASR.

Student: Qwen3-ASR-0.6B (trainable, full-parameter fine-tuning)
Teacher: FireRedASR2-LLM (frozen)

The released local models share the same 151,646-token Qwen2 BPE base
vocabulary with identical token IDs. Qwen3-ASR appends ASR/multimodal special
tokens, while FireRed appends ``<speech>`` at the first non-shared ID. This
trainer therefore computes token-level OPD only on the verified shared prefix:

1. The fixed Qwen ``language ... <asr_text>`` protocol is placed in the prompt.
2. Qwen3-ASR samples only the transcript and its termination on-policy.
3. FireRed and Qwen score the same remaining shared text IDs plus the shared
   ``<|im_end|>`` response terminator.
4. Prompt positions and unsupported special-token positions/vocabulary
   dimensions are excluded from reverse KL(student || teacher), which updates
   Qwen3-ASR only.
5. An optional supervised CE term anchors training to the reference text.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

from qwen_asr import Qwen3ASRModel

from .asr import load_firered_llm_model_and_tokenizer
from .data.asr_feat import ASRFeatExtractor
from .tokenizer.llm_tokenizer import IGNORE_TOKEN_ID, LlmTokenizerWrapper


logger = logging.getLogger("fireredasr2s.qwen_opd")
QWEN_ASR_PREFIX_RE = re.compile(r"^language\s+[^<]+<asr_text>", re.IGNORECASE)


class InvalidRolloutError(RuntimeError):
    """A sampled completion contains no usable on-policy ASR text."""


@dataclass(frozen=True)
class OpdRecord:
    uttid: str
    audio: str
    target: str
    reference: str
    prompt: str
    duration: float


def plain_transcript(text: str) -> str:
    """Remove Qwen3-ASR's output protocol prefix for FireRed scoring."""
    return QWEN_ASR_PREFIX_RE.sub("", str(text).strip(), count=1).strip()


def qwen_target(text: str, language: str) -> str:
    text = str(text).strip()
    if "<asr_text>" in text:
        return text
    return f"language {language}<asr_text>{text}"


class OpdJsonlDataset(Dataset):
    """Read OPD JSONL while accepting common ASR field aliases."""

    def __init__(
        self,
        manifest: str,
        language: str,
        min_audio_seconds: float,
        max_audio_seconds: float,
    ) -> None:
        self.path = Path(manifest).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        records: list[OpdRecord] = []
        filtered = 0
        seen = set()
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    audio = item.get("audio") or item.get("wav") or item.get("wav_path")
                    text = item.get("text")
                    if text is None:
                        text = item.get("reference")
                    if not audio or text is None or not str(text).strip():
                        raise ValueError("requires audio/wav/wav_path and text/reference")
                    audio_path = Path(str(audio)).expanduser()
                    if not audio_path.is_absolute():
                        audio_path = self.path.parent / audio_path
                    audio_path = audio_path.resolve()
                    if not audio_path.is_file():
                        raise FileNotFoundError(audio_path)
                    duration = float(item.get("duration") or sf.info(audio_path).duration)
                    if not min_audio_seconds <= duration <= max_audio_seconds:
                        filtered += 1
                        continue
                    # Standard OPD manifests contain only audio/text.  A stem is
                    # not globally unique (many corpora contain audio_001.wav in
                    # different directories), so include the stable source line.
                    uttid = str(
                        item.get("uttid")
                        or f"{audio_path.stem}-line-{line_number:09d}"
                    )
                    if uttid in seen:
                        raise ValueError(f"duplicate uttid: {uttid}")
                    seen.add(uttid)
                    target = qwen_target(str(text), language)
                    reference = plain_transcript(target)
                    if not reference:
                        raise ValueError("empty transcript after removing ASR prefix")
                    records.append(
                        OpdRecord(
                            uttid=uttid,
                            audio=str(audio_path),
                            target=target,
                            reference=reference,
                            prompt=str(item.get("prompt", "")),
                            duration=duration,
                        )
                    )
                except Exception as exc:
                    raise ValueError(f"Invalid entry {self.path}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"No usable records in {self.path}")
        self.records = records
        logger.info("Loaded %d records from %s (%d duration-filtered)", len(records), self.path, filtered)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OpdRecord:
        return self.records[index]


def build_qwen_prefix(processor: Any, prompt: str, language: str) -> str:
    messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    chat_prefix = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )[0]
    return chat_prefix + f"language {language}<asr_text>"


def read_audio_16k(path: str) -> np.ndarray:
    wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1, dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)
    if sample_rate != 16000:
        wav_tensor = torch.from_numpy(wav)
        wav = torchaudio.functional.resample(wav_tensor, sample_rate, 16000).numpy()
    if wav.ndim != 1 or wav.size == 0:
        raise ValueError(f"Invalid audio shape for {path}: {wav.shape}")
    return np.ascontiguousarray(wav, dtype=np.float32)


class DualFrontendOpdCollator:
    def __init__(self, student_processor: Any, teacher_cmvn: str, language: str):
        self.student_processor = student_processor
        self.teacher_features = ASRFeatExtractor(teacher_cmvn)
        self.language = language

    def __call__(self, records: list[OpdRecord]) -> dict[str, Any]:
        audio_arrays = [read_audio_16k(record.audio) for record in records]
        prefix_texts = [
            build_qwen_prefix(self.student_processor, record.prompt, self.language)
            for record in records
        ]
        eos = self.student_processor.tokenizer.eos_token or ""
        full_texts = [
            prefix + record.reference + eos
            for prefix, record in zip(prefix_texts, records)
        ]
        full_inputs = self.student_processor(
            text=full_texts,
            audio=audio_arrays,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.student_processor(
            text=prefix_texts,
            audio=audio_arrays,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = full_inputs["input_ids"].clone()
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        for row, prefix_length in enumerate(prefix_lengths):
            labels[row, : int(prefix_length)] = IGNORE_TOKEN_ID
        pad_id = self.student_processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = IGNORE_TOKEN_ID
        full_inputs["labels"] = labels

        # FireRed's released frontend was trained on int16-scale waveforms.
        teacher_wavs = [(16000, wav * 32768.0) for wav in audio_arrays]
        teacher_feat, teacher_lengths, _, _, returned_ids = self.teacher_features(
            teacher_wavs, [record.uttid for record in records]
        )
        if teacher_feat is None or len(returned_ids) != len(records):
            raise ValueError("FireRed feature extraction dropped one or more samples")

        return {
            **full_inputs,
            "prefix_input_ids": prefix_inputs["input_ids"],
            "prefix_attention_mask": prefix_inputs["attention_mask"],
            "prefix_input_features": prefix_inputs["input_features"],
            "prefix_feature_attention_mask": prefix_inputs["feature_attention_mask"],
            "teacher_features": teacher_feat,
            "teacher_feature_lengths": teacher_lengths,
            "references": [record.reference for record in records],
            "uttids": [record.uttid for record in records],
        }


def patch_qwen_forward(model: torch.nn.Module) -> None:
    """Make the outer Qwen3-ASR module compatible with Trainer.forward."""
    cls = model.__class__
    if getattr(cls, "_cross_opd_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("Qwen3-ASR model has no .thinker module")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._cross_opd_forward_patched = True


def first_eos_mask(ids: torch.Tensor, eos_id: int | None, pad_id: int | None) -> torch.Tensor:
    mask = torch.ones_like(ids, dtype=torch.bool)
    if pad_id is not None:
        mask &= ids.ne(pad_id)
    if eos_id is not None and ids.numel():
        is_eos = ids.eq(eos_id)
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        first = torch.full((ids.size(0),), ids.size(1), device=ids.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        first[has_eos] = is_eos.long().argmax(dim=1)[has_eos]
        mask &= positions <= first.unsqueeze(1)
    return mask


def pad_sequences(sequences: Iterable[torch.Tensor], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [sequence.to(device=device, dtype=torch.long) for sequence in sequences]
    max_length = max(sequence.numel() for sequence in sequences)
    ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        ids[row, : sequence.numel()] = sequence
        mask[row, : sequence.numel()] = True
    return ids, mask


def shared_tokenizer_prefix(student_tokenizer: Any, teacher_tokenizer: Any) -> int:
    """Return the largest contiguous token-ID prefix with identical semantics."""
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_by_id = {token_id: token for token, token_id in student_vocab.items()}
    teacher_by_id = {token_id: token for token, token_id in teacher_vocab.items()}
    limit = min(max(student_by_id, default=-1), max(teacher_by_id, default=-1)) + 1
    shared = 0
    while shared < limit and student_by_id.get(shared) == teacher_by_id.get(shared):
        shared += 1
    if shared <= 0:
        raise RuntimeError("Student and teacher tokenizers have no shared ID prefix")
    return shared


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: Any):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.processor.save_pretrained(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        return control


class FireRedTeacherOpdTrainer(Trainer):
    distillation_metric_name = "kl"

    def __init__(
        self,
        *args,
        student_processor: Any,
        teacher_model: torch.nn.Module,
        teacher_tokenizer: Any,
        teacher_device: torch.device,
        teacher_dtype: torch.dtype,
        shared_vocab_size: int,
        asr_text_token_id: int,
        response_end_token_id: int,
        response_stop_token_ids: frozenset[int],
        max_completion_length: int,
        max_rollout_retries: int,
        sampling_temperature: float,
        sampling_top_p: float,
        sampling_top_k: int,
        teacher_max_text_tokens: int,
        teacher_instruction: str,
        distillation_temperature: float,
        topk_logits_k: int | None,
        sft_coefficient: float,
        log_rollout_text: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # ``patch_qwen_forward`` accepts ``**kwargs``, which makes recent
        # Transformers versions assume that the model consumes
        # ``num_items_in_batch`` and normalizes a custom loss across gradient
        # accumulation micro-batches.  Our OPD loss is already normalized per
        # response and deliberately ignores that argument, so leaving the
        # inferred flag enabled makes both the reported loss and accumulated
        # gradients larger by ``gradient_accumulation_steps``.  Tell Trainer to
        # apply its normal accumulation scaling instead.
        self.model_accepts_loss_kwargs = False
        self.student_processor = student_processor
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_device = teacher_device
        self.teacher_dtype = teacher_dtype
        self.shared_vocab_size = shared_vocab_size
        self.asr_text_token_id = asr_text_token_id
        self.response_end_token_id = response_end_token_id
        self.response_stop_token_ids = response_stop_token_ids
        self.distillation_vocab_ids = torch.tensor(
            [
                token_id
                for token_id in range(shared_vocab_size)
                if (
                    token_id not in response_stop_token_ids
                    or token_id == response_end_token_id
                )
            ],
            dtype=torch.long,
        )
        if self.distillation_vocab_ids.numel() == 0:
            raise RuntimeError("No ordinary shared tokens remain for OPD")
        self.distillation_vocab_size = int(self.distillation_vocab_ids.numel())
        self._distillation_vocab_ids_by_device: dict[str, torch.Tensor] = {}
        self.max_completion_length = max_completion_length
        self.max_rollout_retries = max_rollout_retries
        self.teacher_max_text_tokens = teacher_max_text_tokens
        self.teacher_instruction = teacher_instruction
        self.distillation_temperature = distillation_temperature
        self.topk_logits_k = topk_logits_k
        self.sft_coefficient = sft_coefficient
        self.log_rollout_text = log_rollout_text
        self.rollout_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=sampling_temperature,
            top_p=sampling_top_p,
            top_k=sampling_top_k,
            num_return_sequences=1,
            pad_token_id=student_processor.tokenizer.pad_token_id,
            eos_token_id=student_processor.tokenizer.eos_token_id,
        )
        self._opd_metrics: dict[str, list[float]] = defaultdict(list)

    def _vocab_ids_for(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        ids = self._distillation_vocab_ids_by_device.get(key)
        if ids is None:
            # This helper is first called by the frozen teacher from inside
            # torch.inference_mode().  Explicitly disable inference mode while
            # materializing the cached index tensor; otherwise a same-device
            # student reuses an inference tensor that index_select cannot save
            # for backward.
            with torch.inference_mode(False):
                ids = self.distillation_vocab_ids.to(device).clone()
            self._distillation_vocab_ids_by_device[key] = ids
        return ids

    @staticmethod
    def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
        while hasattr(model, "module"):
            model = model.module
        return model

    def _rollout(self, model: torch.nn.Module, inputs: dict[str, Any]):
        raw_model = self._raw_model(model)
        model_dtype = next(raw_model.parameters()).dtype
        generation_inputs = {
            "input_ids": inputs["prefix_input_ids"],
            "attention_mask": inputs["prefix_attention_mask"],
            "input_features": inputs["prefix_input_features"].to(dtype=model_dtype),
            "feature_attention_mask": inputs["prefix_feature_attention_mask"],
        }
        was_training = raw_model.training
        raw_model.eval()
        with torch.no_grad():
            generated = raw_model.generate(
                **generation_inputs,
                generation_config=self.rollout_config,
                max_new_tokens=self.max_completion_length,
            )
        if was_training:
            raw_model.train()
        sequences = getattr(generated, "sequences", generated)
        prompt_length = inputs["prefix_input_ids"].size(1)
        generated_ids = sequences[:, prompt_length:]
        generated_mask = first_eos_mask(
            generated_ids,
            self.student_processor.tokenizer.eos_token_id,
            self.student_processor.tokenizer.pad_token_id,
        )
        return generated_ids, generated_mask

    def _extract_shared_responses(
        self,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], int]:
        """Select text tokens generated after the forced Qwen ASR prefix.

        The returned response IDs are valid in both vocabularies. Positions map
        each response token back to its location in the student's completion.
        The shared ``<|im_end|>`` token is included as the final scored action so
        the student learns when to stop. Any other student/teacher special token
        terminates the text span without being scored. Non-shared IDs terminate
        it as well rather than being assigned a false meaning in the other
        tokenizer.
        """
        sequences = []
        positions = []
        truncated = 0
        for row in range(generated_ids.size(0)):
            valid_positions = torch.where(generated_mask[row])[0]
            valid_ids = generated_ids[row, valid_positions]
            # ``language {language}<asr_text>`` is part of the prompt, so every
            # generated token is already an ASR response action.
            response_ids = valid_ids
            response_positions = valid_positions
            end_mask = response_ids.eq(self.response_end_token_id)
            stop_mask = response_ids.ge(self.shared_vocab_size)
            for token_id in self.response_stop_token_ids:
                if token_id != self.response_end_token_id:
                    stop_mask |= response_ids.eq(token_id)
            # Include the first shared end token, but exclude any other boundary.
            stop_mask |= end_mask
            stops = torch.where(stop_mask)[0]
            if stops.numel():
                stop = int(stops[0].item())
                stopped_id = int(response_ids[stop].item())
                include_stop = stopped_id == self.response_end_token_id
                if (
                    stopped_id >= self.shared_vocab_size
                    and stopped_id not in self.response_stop_token_ids
                ):
                    truncated += 1
                end = stop + int(include_stop)
                response_ids = response_ids[:end]
                response_positions = response_positions[:end]
            if response_ids.numel() == 0:
                raise InvalidRolloutError(
                    "Qwen rollout contained no shared transcript or end token"
                )
            if response_ids.ge(self.shared_vocab_size).any():
                raise RuntimeError("Non-shared token leaked into the ASR text span")
            forbidden_special_ids = self.response_stop_token_ids - {
                self.response_end_token_id
            }
            if any(response_ids.eq(token_id).any() for token_id in forbidden_special_ids):
                raise RuntimeError("Unsupported special token leaked into the ASR text span")
            end_locations = torch.where(response_ids.eq(self.response_end_token_id))[0]
            if end_locations.numel() and int(end_locations[0].item()) != response_ids.numel() - 1:
                raise RuntimeError("Response end token must be the final scored token")
            sequences.append(response_ids)
            positions.append(response_positions)
        padded_ids, padded_mask = pad_sequences(
            sequences,
            self.student_processor.tokenizer.pad_token_id,
            generated_ids.device,
        )
        return padded_ids, padded_mask, positions, truncated

    def _teacher_prompt_inputs(self, inputs: dict[str, Any], batch_size: int):
        prompt_ids, prompt_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
            [""] * batch_size,
            self.teacher_tokenizer,
            self.teacher_max_text_tokens,
            decode=True,
            instruction=self.teacher_instruction,
        )
        return prompt_ids, prompt_mask

    def _teacher_response_logits(
        self,
        inputs: dict[str, Any],
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score shared response IDs with FireRed and return aligned logits."""
        batch_size = response_ids.size(0)

        teacher_features = inputs["teacher_features"].to(
            self.teacher_device, dtype=self.teacher_dtype
        )
        teacher_lengths = inputs["teacher_feature_lengths"].to(self.teacher_device)
        with torch.inference_mode():
            speech_features, speech_lengths, _ = self.teacher_model.encode_speech(
                teacher_features, teacher_lengths
            )

            prompt_ids, prompt_mask = self._teacher_prompt_inputs(inputs, batch_size)
            prompt_ids = prompt_ids.to(self.teacher_device)
            prompt_mask = prompt_mask.to(self.teacher_device)
            response_ids = response_ids.to(self.teacher_device)
            response_mask = response_mask.to(self.teacher_device)

            combined_ids = []
            combined_labels = []
            for row in range(batch_size):
                valid_prompt = prompt_ids[row][prompt_mask[row].bool()]
                valid_response = response_ids[row][response_mask[row].bool()]
                combined_ids.append(torch.cat([valid_prompt, valid_response]))
                combined_labels.append(torch.cat([
                    torch.full_like(valid_prompt, IGNORE_TOKEN_ID),
                    valid_response,
                ]))
            max_length = max(ids.numel() for ids in combined_ids)
            teacher_pad = self.teacher_model.llm.config.pad_token_id
            input_ids = prompt_ids.new_full((batch_size, max_length), teacher_pad)
            attention_mask = prompt_mask.new_zeros((batch_size, max_length))
            labels = response_ids.new_full((batch_size, max_length), IGNORE_TOKEN_ID)
            for row, (ids, row_labels) in enumerate(zip(combined_ids, combined_labels)):
                length = ids.numel()
                input_ids[row, :length] = ids
                attention_mask[row, :length] = 1
                labels[row, :length] = row_labels

            inputs_embeds = self.teacher_model.llm.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, labels = (
                self.teacher_model._merge_input_ids_with_speech_features(
                    speech_features.to(inputs_embeds.dtype),
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels=labels,
                    speech_lens=speech_lengths,
                    padding_side="right",
                )
            )
            outputs = self.teacher_model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            shifted_logits = outputs.logits[:, :-1].index_select(
                -1, self._vocab_ids_for(outputs.logits.device)
            )
            shifted_labels = labels[:, 1:]
            token_mask = shifted_labels.ne(IGNORE_TOKEN_ID)
            for row in range(batch_size):
                scored_ids = shifted_labels[row][token_mask[row]]
                expected_ids = response_ids[row][response_mask[row].bool()]
                if not torch.equal(scored_ids, expected_ids):
                    raise RuntimeError(
                        "FireRed label shift does not align with the ASR text token IDs"
                    )
            rows = [shifted_logits[row][token_mask[row]] for row in range(batch_size)]
            max_response = max(row.size(0) for row in rows)
            aligned = shifted_logits.new_zeros(
                (batch_size, max_response, self.distillation_vocab_size)
            )
            aligned_mask = torch.zeros(
                (batch_size, max_response), dtype=torch.bool, device=self.teacher_device
            )
            for row, logits in enumerate(rows):
                aligned[row, : logits.size(0)] = logits
                aligned_mask[row, : logits.size(0)] = True
        # clone() outside inference_mode creates a normal detached tensor that
        # autograd may safely use while differentiating the student KL.
        return (
            aligned.to(self.accelerator.device).clone(),
            aligned_mask.to(self.accelerator.device).clone(),
        )

    def _student_response_logits(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        response_positions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = inputs["prefix_input_ids"]
        prompt_mask = inputs["prefix_attention_mask"]
        audio_features = inputs["prefix_input_features"]
        audio_mask = inputs["prefix_feature_attention_mask"]
        input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, generated_mask.to(prompt_mask.dtype)], dim=1)

        raw_model = self._raw_model(model)
        model_dtype = next(raw_model.parameters()).dtype
        forward_model = raw_model.thinker if hasattr(raw_model, "thinker") else raw_model
        outputs = forward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=audio_features.to(dtype=model_dtype),
            feature_attention_mask=audio_mask,
        )
        logits = outputs.logits[:, :-1]
        prompt_length = prompt_ids.size(1)
        action_logits = logits[:, prompt_length - 1 : prompt_length - 1 + generated_ids.size(1)]
        vocab_ids = self._vocab_ids_for(action_logits.device)
        rows = [
            action_logits[row, positions].index_select(-1, vocab_ids)
            for row, positions in enumerate(response_positions)
        ]
        max_response = max(row.size(0) for row in rows)
        aligned = action_logits.new_zeros(
            (len(rows), max_response, self.distillation_vocab_size)
        )
        aligned_mask = torch.zeros(
            (len(rows), max_response), dtype=torch.bool, device=action_logits.device
        )
        for row, row_logits in enumerate(rows):
            aligned[row, : row_logits.size(0)] = row_logits
            aligned_mask[row, : row_logits.size(0)] = True
        return aligned, aligned_mask

    def _reverse_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.distillation_temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        if self.topk_logits_k is None:
            student_log_probs = torch.log_softmax(student_logits, dim=-1)
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            student_probs = torch.softmax(student_logits, dim=-1)
        else:
            k = min(self.topk_logits_k, self.distillation_vocab_size)
            top_indices = torch.topk(student_logits, k=k, dim=-1).indices
            student_logits = student_logits.gather(-1, top_indices)
            teacher_logits = teacher_logits.gather(-1, top_indices)
            student_log_probs = torch.log_softmax(student_logits, dim=-1)
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            student_probs = torch.softmax(student_logits, dim=-1)
        return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    def _distillation_loss(self, student_logits, teacher_logits):
        return self._reverse_kl(student_logits, teacher_logits)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        invalid_reason = None
        for rollout_retry in range(self.max_rollout_retries + 1):
            generated_ids, generated_mask = self._rollout(model, inputs)
            if self.log_rollout_text:
                raw_completions = [
                    self.student_processor.tokenizer.decode(
                        generated_ids[row][generated_mask[row]].detach().cpu().tolist(),
                        skip_special_tokens=False,
                    )
                    for row in range(generated_ids.size(0))
                ]
                logger.info(
                    "Student raw rollout: retry=%d/%d, uttids=%s, completions=%s",
                    rollout_retry,
                    self.max_rollout_retries,
                    inputs.get("uttids", []),
                    raw_completions,
                )
            try:
                response_ids, response_mask, response_positions, truncated = (
                    self._extract_shared_responses(generated_ids, generated_mask)
                )
                break
            except InvalidRolloutError as exc:
                invalid_reason = str(exc)
                if rollout_retry < self.max_rollout_retries:
                    continue

                decoded = self.student_processor.tokenizer.batch_decode(
                    generated_ids.detach().cpu().tolist(),
                    skip_special_tokens=False,
                )
                logger.warning(
                    "Skipping OPD micro-batch after %d rollout retries: "
                    "uttids=%s, reason=%s, completions=%s",
                    self.max_rollout_retries,
                    inputs.get("uttids", []),
                    invalid_reason,
                    decoded,
                )
                self._opd_metrics["empty_rollout_rate"].append(1.0)
                self._opd_metrics["rollout_retries"].append(
                    float(self.max_rollout_retries)
                )
                # A standalone differentiable zero lets Trainer/Accelerate
                # finish this micro-batch without fabricating a reference-based
                # objective or allocating gradients for a giant model tensor.
                zero_loss = torch.zeros(
                    (),
                    device=self.accelerator.device,
                    dtype=torch.float32,
                    requires_grad=torch.is_grad_enabled(),
                )
                if return_outputs:
                    return zero_loss, {"loss": zero_loss.detach()}
                return zero_loss

        self._opd_metrics["empty_rollout_rate"].append(0.0)
        self._opd_metrics["rollout_retries"].append(float(rollout_retry))
        for row, positions in enumerate(response_positions):
            selected_ids = generated_ids[row, positions]
            expected_ids = response_ids[row][response_mask[row]]
            if not torch.equal(selected_ids, expected_ids):
                raise RuntimeError(
                    "Qwen rollout positions do not align with the ASR text token IDs"
                )
        teacher_logits, teacher_mask = self._teacher_response_logits(
            inputs, response_ids, response_mask
        )
        student_logits, student_mask = self._student_response_logits(
            model, inputs, generated_ids, generated_mask, response_positions
        )
        if not torch.equal(student_mask, teacher_mask):
            raise RuntimeError("Student/teacher response masks differ after token alignment")
        per_token_kl = self._distillation_loss(student_logits, teacher_logits)
        distillation_loss = (
            (per_token_kl * student_mask).sum(dim=1)
            / student_mask.sum(dim=1).clamp(min=1)
        ).mean()

        sft_loss = distillation_loss.new_zeros(())
        if self.sft_coefficient > 0:
            raw_model = self._raw_model(model)
            model_dtype = next(raw_model.parameters()).dtype
            supervised = raw_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"].to(dtype=model_dtype),
                feature_attention_mask=inputs["feature_attention_mask"],
                labels=inputs["labels"],
            )
            sft_loss = supervised.loss
        loss = distillation_loss + self.sft_coefficient * sft_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite OPD loss: kl={distillation_loss}, sft={sft_loss}"
            )

        self._opd_metrics[self.distillation_metric_name].append(float(distillation_loss.detach().cpu()))
        self._opd_metrics["sft_loss"].append(float(sft_loss.detach().cpu()))
        self._opd_metrics["completion_length"].append(
            float(student_mask.sum(dim=1).float().mean().cpu())
        )
        self._opd_metrics["unsupported_special_rate"].append(
            truncated / max(generated_ids.size(0), 1)
        )
        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        # Evaluation measures the same sampled token-level OPD objective. There are
        # no fixed-shape logits to concatenate because every rollout is sampled.
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def log(self, logs: dict[str, float], start_time=None) -> None:
        for name, values in self._opd_metrics.items():
            if values:
                logs[name] = sum(values) / len(values)
        self._opd_metrics.clear()
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token-level OPD: FireRedASR2-LLM teacher -> Qwen3-ASR student"
    )
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--teacher_device", default="cuda:0")
    parser.add_argument("--teacher_instruction", default="请转写音频为文字")
    parser.add_argument("--teacher_max_text_tokens", type=int, default=256)
    parser.add_argument("--min_audio_seconds", type=float, default=0.2)
    parser.add_argument("--max_audio_seconds", type=float, default=40.0)

    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_rollout_retries", type=int, default=3)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_top_p", type=float, default=0.95)
    parser.add_argument("--sampling_top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_logits_k", type=int, default=None)
    parser.add_argument("--sft_coefficient", type=float, default=0.1)
    parser.add_argument(
        "--log_rollout_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log every raw student completion, including every retry",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", choices=["none", "wandb", "tensorboard"], default="wandb")
    parser.add_argument("--resume_from", default="")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("student_model_path", "teacher_model_path"):
        if not Path(getattr(args, name)).expanduser().is_dir():
            raise FileNotFoundError(f"{name}: {getattr(args, name)}")
    for name in ("train_file",):
        if not Path(getattr(args, name)).expanduser().is_file():
            raise FileNotFoundError(f"{name}: {getattr(args, name)}")
    if args.eval_file and not Path(args.eval_file).expanduser().is_file():
        raise FileNotFoundError(f"eval_file: {args.eval_file}")
    if args.batch_size < 1 or args.grad_acc < 1:
        raise ValueError("batch_size and grad_acc must be positive")
    if args.sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if args.max_rollout_retries < 0:
        raise ValueError("max_rollout_retries must be non-negative")
    if not 0 < args.sampling_top_p <= 1:
        raise ValueError("sampling_top_p must be in (0, 1]")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.topk_logits_k is not None and args.topk_logits_k <= 0:
        raise ValueError("topk_logits_k must be positive when set")
    if args.sft_coefficient < 0:
        raise ValueError("sft_coefficient must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen/FireRed OPD requires CUDA")
    teacher_device = torch.device(args.teacher_device)
    if teacher_device.type != "cuda":
        raise ValueError("teacher_device must be a CUDA device")
    if teacher_device.index is not None and teacher_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"teacher_device={teacher_device} but only {torch.cuda.device_count()} CUDA devices exist"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(args: argparse.Namespace, dtype: torch.dtype):
    teacher_root = Path(args.teacher_model_path).expanduser().resolve()
    model, tokenizer = load_firered_llm_model_and_tokenizer(
        str(teacher_root / "model.pth.tar"),
        str(teacher_root / "asr_encoder.pth.tar"),
        str(teacher_root / "Qwen2-7B-Instruct"),
        args.teacher_checkpoint,
        use_flash_attn=getattr(args, "use_flash_attention", None),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model.to(device=torch.device(args.teacher_device), dtype=dtype)
    model.llm.config.use_cache = False
    return model, tokenizer, str(teacher_root / "cmvn.ark")


def main() -> None:
    run_training(parse_args())


def run_training(
    args: argparse.Namespace,
    trainer_class=FireRedTeacherOpdTrainer,
    collator_class=DualFrontendOpdCollator,
    trainer_kwargs: dict[str, Any] | None = None,
) -> None:
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info("Loading Qwen3-ASR student: %s", args.student_model_path)
    student_wrapper = Qwen3ASRModel.from_pretrained(
        args.student_model_path,
        dtype=dtype,
        device_map=None,
    )
    student_model = student_wrapper.model
    student_processor = student_wrapper.processor
    patch_qwen_forward(student_model)
    student_model.generation_config = GenerationConfig.from_model_config(student_model.config)

    logger.info("Loading frozen FireRedASR2-LLM teacher: %s", args.teacher_model_path)
    teacher_model, teacher_tokenizer, teacher_cmvn = load_teacher(args, dtype)
    shared_vocab_size = shared_tokenizer_prefix(
        student_processor.tokenizer, teacher_tokenizer
    )
    asr_text_token_id = student_processor.tokenizer.convert_tokens_to_ids("<asr_text>")
    if asr_text_token_id is None or asr_text_token_id < shared_vocab_size:
        raise RuntimeError(
            f"Unexpected Qwen <asr_text> token ID: {asr_text_token_id}; "
            f"shared vocabulary size is {shared_vocab_size}"
        )
    response_end_token_id = student_processor.tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )
    teacher_end_token_id = teacher_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        response_end_token_id != teacher_end_token_id
        or response_end_token_id is None
        or response_end_token_id >= shared_vocab_size
    ):
        raise RuntimeError(
            "Student and teacher must share the <|im_end|> token ID inside the "
            f"verified vocabulary: student={response_end_token_id}, "
            f"teacher={teacher_end_token_id}, shared={shared_vocab_size}"
        )
    if student_model.thinker.vocab_size < shared_vocab_size:
        raise RuntimeError("Student output vocabulary is smaller than shared tokenizer prefix")
    if teacher_model.llm.config.vocab_size < shared_vocab_size:
        raise RuntimeError("Teacher output vocabulary is smaller than shared tokenizer prefix")
    response_stop_token_ids = frozenset(
        int(token_id)
        for token_id in (
            set(student_processor.tokenizer.all_special_ids)
            | set(teacher_tokenizer.all_special_ids)
        )
        if token_id is not None
    )
    distillation_vocab_size = sum(
        token_id not in response_stop_token_ids or token_id == response_end_token_id
        for token_id in range(shared_vocab_size)
    )
    logger.info(
        "Student tokenizer=%d, teacher tokenizer=%d, verified shared IDs=%d, "
        "text-plus-EOS KL vocabulary=%d, <asr_text>=%d, <|im_end|>=%d; using "
        "ASR-text-plus-termination "
        "token-level distillation (%s)",
        len(student_processor.tokenizer),
        len(teacher_tokenizer),
        shared_vocab_size,
        distillation_vocab_size,
        asr_text_token_id,
        response_end_token_id,
        trainer_class.distillation_metric_name,
    )

    train_dataset = OpdJsonlDataset(
        args.train_file, args.language, args.min_audio_seconds, args.max_audio_seconds
    )
    eval_dataset = None
    if args.eval_file:
        eval_dataset = OpdJsonlDataset(
            args.eval_file, args.language, args.min_audio_seconds, args.max_audio_seconds
        )
    collator = collator_class(student_processor, teacher_cmvn, args.language)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )
    trainer = trainer_class(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        student_processor=student_processor,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_device=torch.device(args.teacher_device),
        teacher_dtype=dtype,
        shared_vocab_size=shared_vocab_size,
        asr_text_token_id=asr_text_token_id,
        response_end_token_id=response_end_token_id,
        response_stop_token_ids=response_stop_token_ids,
        max_completion_length=args.max_completion_length,
        max_rollout_retries=args.max_rollout_retries,
        sampling_temperature=args.sampling_temperature,
        sampling_top_p=args.sampling_top_p,
        sampling_top_k=args.sampling_top_k,
        teacher_max_text_tokens=args.teacher_max_text_tokens,
        teacher_instruction=args.teacher_instruction,
        distillation_temperature=args.temperature,
        topk_logits_k=args.topk_logits_k,
        sft_coefficient=args.sft_coefficient,
        log_rollout_text=args.log_rollout_text,
        callbacks=[SaveProcessorCallback(student_processor)],
        **(trainer_kwargs or {}),
    )
    resume = args.resume_from.strip() or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    student_processor.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
