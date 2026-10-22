# coding=utf-8
# OPSD data helpers for Qwen3-ASR.
# Student prefix: base ASR prompt (audio + instruction only).
# Teacher prefix: same base prompt, but the reference transcript is injected into
#   the SYSTEM message as privileged context. The student never sees the reference.
#
# This mirrors the paper (*On-Policy Self-Distillation for Multi-Dialect ASR*):
#   - student is trained on its OWN decoded prefixes (on-policy rollout);
#   - a frozen teacher, conditioned on the reference as privileged context,
#     provides soft token-level targets.

from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch

MSR = "AsrText"  # sentinel only used to split text fields, see below
ASR_MARKER = "<asr_text>"


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def split_text_for_asr(text: str) -> Dict[str, str]:
    """Split a Qwen3-ASR text field into (base_instruction, reference).

    The dataset stores text as ``language Chinese<asr_text>transcript``.
    The ASR instruction (everything up to and including ``<asr_text>``) is what the
    student should be conditioned on. The transcript after ``<asr_text>`` is treated
    as the privileged reference for the teacher.
    """
    if ASR_MARKER in text:
        instruction, reference = text.split(ASR_MARKER, 1)
        base_instruction = instruction + ASR_MARKER
        reference = reference.strip()
    else:
        # No marker: treat the whole text as the base instruction, no reference.
        base_instruction = text
        reference = None
    return {"base_instruction": base_instruction, "reference": reference}


def build_prefix_messages(base_instruction: str, audio_array) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": base_instruction or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_teacher_prefix_messages(base_instruction: str, reference: str, audio_array) -> List[Dict[str, Any]]:
    """Teacher sees the reference placed in the SYSTEM message (privileged context).

    The Qwen3-ASR chat template renders system text but DROPS text in the user
    message (only the ``type: audio`` placeholder survives), so the reference must
    go in the system message to be visible to the teacher.
    """
    teacher_system = base_instruction
    if reference:
        teacher_system = f"{base_instruction}\n\n[Reference transcript]\n{reference}"
    return [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_opsd_preprocess_fn(processor):
    """Map a raw row to columns used by OPSDTrainer.

    Expected input row fields: ``audio`` (path), ``text`` (base_instruction + asr_text + reference).
    Produces:
      - prefix_text        : student prefix (base instruction only)
      - teacher_prefix_text: teacher prefix (base instruction + reference in system)
      - target             : the reference transcript (used as the on-policy completion target)
      - audio, prompt
    """
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        instruction = ex.get("prompt", "")          # optional extra prompt
        text = ex.get("text", "")
        split = split_text_for_asr(text)
        base_instruction = split["base_instruction"]
        reference = split["reference"] or ""

        dummy_audio = None
        student_msgs = build_prefix_messages(base_instruction, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [student_msgs], add_generation_prompt=True, tokenize=False
        )[0]

        teacher_msgs = build_teacher_prefix_messages(base_instruction, reference, dummy_audio)
        teacher_prefix_text = processor.apply_chat_template(
            [teacher_msgs], add_generation_prompt=True, tokenize=False
        )[0]

        return {
            "prompt": base_instruction,
            "teacher_prompt": reference,
            "audio": ex["audio"],
            "target": reference,
            "prefix_text": prefix_text,
            "teacher_prefix_text": teacher_prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForASROPSD:
    """Batch raw features into tensors for OPSD.

    Output dict contains (in addition to the standard full inputs used by the student
    SFT path):
      - prefix_input_ids / prefix_attention_mask / prefix_input_features /
        prefix_feature_attention_mask : student prefix (for on-policy rollout + logits)
      - teacher_prefix_input_ids / ... : teacher prefix (for teacher logits over the
        same on-policy completion)
      - prefix_len / teacher_prefix_len : per-sample prefix lengths (to align logits)
      - meta_data: (targets, refs) kept for reference.
    """
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        teacher_prefix_texts = [f["teacher_prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        refs = [f["teacher_prompt"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # The audio tokens are identical for student and teacher; encoding full_texts
        # gives us input_features aligned to the audio. We re-encode the teacher prefix
        # separately because its text differs (reference in system).
        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )
        teacher_prefix_inputs = self.processor(
            text=teacher_prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )

        # Mask prompt tokens from loss (prefix + audio BOS tokens)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels

        meta_data = (targets, refs)

        return {
            **full_inputs,
            "meta_data": meta_data,
            "prefix_input_ids": prefix_inputs["input_ids"],
            "prefix_attention_mask": prefix_inputs["attention_mask"],
            "prefix_input_features": prefix_inputs["input_features"],
            "prefix_feature_attention_mask": prefix_inputs["feature_attention_mask"],
            "teacher_prefix_input_ids": teacher_prefix_inputs["input_ids"],
            "teacher_prefix_attention_mask": teacher_prefix_inputs["attention_mask"],
            "teacher_prefix_input_features": teacher_prefix_inputs["input_features"],
            "teacher_prefix_feature_attention_mask": teacher_prefix_inputs["feature_attention_mask"],
            "prefix_len": torch.tensor(prefix_lens, dtype=torch.long),
            "teacher_prefix_len": torch.tensor(
                teacher_prefix_inputs["attention_mask"].sum(dim=1).tolist(), dtype=torch.long,
            ),
        }
