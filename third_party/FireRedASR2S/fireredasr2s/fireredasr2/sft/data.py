import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from ..data.asr_feat import CMVN, KaldifeatFbank
from ..tokenizer.llm_tokenizer import LlmTokenizerWrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SftRecord:
    uttid: str
    wav_path: str
    text: str
    duration: float


class SftDataset(Dataset):
    """JSONL speech/text dataset with duration-based filtering.

    Accepted audio keys are ``wav`` and ``wav_path``. Relative paths are
    resolved against ``audio_root`` when set, otherwise against the manifest
    directory. ``duration`` is optional and is read from the audio header when
    absent.
    """

    def __init__(self, manifest_path, min_audio_seconds=0.2,
                 max_audio_seconds=40.0, audio_root=""):
        self.manifest_path = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(self.manifest_path)
        base_dir = os.path.abspath(audio_root) if audio_root else manifest_dir
        records = []
        filtered = 0
        seen_uttids = set()
        with open(self.manifest_path, "r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    wav_path = item.get("wav") or item.get("wav_path")
                    text = str(item["text"]).strip()
                    if (not wav_path or not text or
                            not LlmTokenizerWrapper.clean_text(text)):
                        filtered += 1
                        continue
                    wav_path = os.path.expanduser(str(wav_path))
                    if not os.path.isabs(wav_path):
                        wav_path = os.path.join(base_dir, wav_path)
                    wav_path = os.path.abspath(wav_path)
                    if not os.path.isfile(wav_path):
                        raise FileNotFoundError(wav_path)
                    duration = item.get("duration")
                    if duration is None:
                        duration = sf.info(wav_path).duration
                    duration = float(duration)
                    if not min_audio_seconds <= duration <= max_audio_seconds:
                        filtered += 1
                        continue
                    uttid = str(item.get("uttid") or Path(wav_path).stem)
                    if uttid in seen_uttids:
                        raise ValueError(f"duplicate uttid: {uttid}")
                    seen_uttids.add(uttid)
                    records.append(SftRecord(uttid, wav_path, text, duration))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid manifest entry {self.manifest_path}:"
                        f"{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"No usable records found in {self.manifest_path}")
        self.records = records
        self.frame_lengths = [max(1, math.ceil(r.duration * 100)) for r in records]
        logger.info(
            "Loaded %d SFT records from %s (%d filtered)",
            len(records), self.manifest_path, filtered)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class DynamicBatchSampler(Sampler):
    """Length-bucketed batches constrained by padded acoustic frame cost."""

    def __init__(self, frame_lengths, max_batch_frames, max_batch_size,
                 bucket_size=128, shuffle=True, seed=42,
                 rank=0, world_size=1):
        self.frame_lengths = list(frame_lengths)
        self.max_batch_frames = int(max_batch_frames)
        self.max_batch_size = int(max_batch_size)
        self.bucket_size = max(int(bucket_size), self.max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if not self.frame_lengths:
            raise ValueError("frame_lengths must not be empty")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid distributed sampler rank")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.frame_lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        ordered = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda idx: self.frame_lengths[idx])
            ordered.extend(bucket)

        batches = []
        batch = []
        longest = 0
        for index in ordered:
            frames = self.frame_lengths[index]
            next_longest = max(longest, frames)
            next_size = len(batch) + 1
            padded_cost = next_longest * next_size
            if batch and (padded_cost > self.max_batch_frames or
                          next_size > self.max_batch_size):
                batches.append(batch)
                batch = []
                longest = 0
            batch.append(index)
            longest = max(longest, frames)
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _rank_batches(self):
        batches = self._global_batches()
        if self.world_size > 1:
            remainder = len(batches) % self.world_size
            if remainder:
                needed = self.world_size - remainder
                batches.extend(batches[i % len(batches)] for i in range(needed))
            batches = batches[self.rank::self.world_size]
        return batches

    def __iter__(self):
        yield from self._rank_batches()

    def __len__(self):
        return len(self._rank_batches())


class SftCollator:
    def __init__(self, tokenizer, cmvn_path, max_text_tokens=192,
                 instruction="请转写音频为文字", training=True, dither=0.0,
                 specaug_time_masks=0, specaug_time_width=40,
                 specaug_freq_masks=0, specaug_freq_width=8):
        self.tokenizer = tokenizer
        self.cmvn = CMVN(cmvn_path)
        self.fbank = KaldifeatFbank(
            num_mel_bins=80, frame_length=25, frame_shift=10,
            dither=dither)
        self.max_text_tokens = int(max_text_tokens)
        self.instruction = instruction
        self.training = bool(training)
        self.specaug_time_masks = int(specaug_time_masks)
        self.specaug_time_width = int(specaug_time_width)
        self.specaug_freq_masks = int(specaug_freq_masks)
        self.specaug_freq_width = int(specaug_freq_width)

    @staticmethod
    def _read_audio(path):
        wav, sample_rate = sf.read(path, dtype="int16", always_2d=False)
        if wav.ndim == 2:
            wav = wav.astype(np.float32).mean(axis=1)
        if sample_rate != 16000:
            tensor = torch.as_tensor(wav, dtype=torch.float32)
            tensor = torchaudio.functional.resample(tensor, sample_rate, 16000)
            wav = tensor.cpu().numpy()
            sample_rate = 16000
        return sample_rate, np.asarray(wav)

    def _specaugment(self, feat):
        if not self.training:
            return feat
        feat = feat.clone()
        time_len, freq_len = feat.shape
        for _ in range(self.specaug_time_masks):
            width = random.randint(0, min(self.specaug_time_width, time_len))
            if width:
                start = random.randint(0, time_len - width)
                feat[start:start + width] = 0
        for _ in range(self.specaug_freq_masks):
            width = random.randint(0, min(self.specaug_freq_width, freq_len))
            if width:
                start = random.randint(0, freq_len - width)
                feat[:, start:start + width] = 0
        return feat

    def __call__(self, records):
        feats = []
        uttids = []
        texts = []
        durations = []
        for record in records:
            sample_rate, wav = self._read_audio(record.wav_path)
            feat = self.fbank((sample_rate, wav), is_train=self.training)
            if feat.shape[0] < 1:
                raise ValueError(f"Audio is too short for FBank: {record.wav_path}")
            feat = torch.from_numpy(self.cmvn(feat)).float()
            feats.append(self._specaugment(feat))
            uttids.append(record.uttid)
            texts.append(record.text)
            durations.append(record.duration)

        feat_lengths = torch.tensor([feat.size(0) for feat in feats], dtype=torch.long)
        max_feat_len = int(feat_lengths.max().item())
        padded_feat = feats[0].new_zeros((len(feats), max_feat_len, feats[0].size(1)))
        for idx, feat in enumerate(feats):
            padded_feat[idx, :feat.size(0)] = feat

        input_ids, attention_mask, labels, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                texts,
                self.tokenizer,
                self.max_text_tokens,
                decode=False,
                instruction=self.instruction,
            )
        return {
            "padded_feat": padded_feat,
            "feat_lengths": feat_lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "uttids": uttids,
            "texts": clean_texts,
            "durations": durations,
        }
