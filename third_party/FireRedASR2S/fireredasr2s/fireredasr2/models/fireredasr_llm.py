# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..models.fireredasr_aed import FireRedAsrAed
from ..models.module.adapter import Adapter
from ..models.param import count_model_parameters
from ..tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID
from ..tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsrLlm(nn.Module):
    DEFAULT_LORA_RANK = 64
    DEFAULT_LORA_ALPHA = 16
    DEFAULT_LORA_DROPOUT = 0.05
    DEFAULT_LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]

    @classmethod
    def lora_spec_from_args(cls, args):
        return {
            "rank": int(getattr(args, "lora_rank", cls.DEFAULT_LORA_RANK)),
            "alpha": int(getattr(args, "lora_alpha", cls.DEFAULT_LORA_ALPHA)),
            "dropout": float(getattr(
                args, "lora_dropout", cls.DEFAULT_LORA_DROPOUT)),
            "target_modules": list(getattr(
                args, "lora_target_modules",
                cls.DEFAULT_LORA_TARGET_MODULES)),
        }

    @classmethod
    def add_lora(cls, llm, args):
        from peft import LoraConfig, get_peft_model

        spec = cls.lora_spec_from_args(args)
        logging.info("Build LoRA: %s", spec)
        lora_config = LoraConfig(
            r=spec["rank"],
            lora_alpha=spec["alpha"],
            target_modules=spec["target_modules"],
            lora_dropout=spec["dropout"],
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
        return llm

    @classmethod
    def load_encoder(cls, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = FireRedAsrAed.from_args(package["args"])
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim

    @classmethod
    def from_args(cls, args):
        logging.info(args)
        logging.info("Build FireRedAsrLlm")
        # Build Speech Encoder
        encoder, encoder_dim = cls.load_encoder(args.encoder_path)
        count_model_parameters(encoder)
        if args.freeze_encoder:
            logging.info(f"Frezee encoder")
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()

        # The released checkpoint used bfloat16. ``torch_dtype`` is optional
        # so older inference checkpoints keep their original behaviour.
        requested_dtype = getattr(args, "torch_dtype", "")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if requested_dtype:
            if requested_dtype not in dtype_map:
                raise ValueError(f"Unsupported torch_dtype: {requested_dtype}")
            torch_dtype = dtype_map[requested_dtype]
        elif args.use_fp16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        if args.use_flash_attn:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"

        # Build LLM
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        count_model_parameters(llm)

        # LLM Freeze or LoRA
        llm_dim = llm.config.hidden_size
        if args.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in llm.named_parameters():
                param.requires_grad = False
            llm.eval()
        else:
            if args.use_lora:
                llm = cls.add_lora(llm, args)

        tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        assert tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids("<|endoftext|>")
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )

        # Build projector
        encoder_projector = Adapter(
            encoder_dim, llm_dim, args.encoder_downsample_rate)
        count_model_parameters(encoder_projector)

        return cls(encoder, llm, encoder_projector,
                   args.freeze_encoder, args.freeze_llm)

    def __init__(self, encoder, llm, encoder_projector,
                 freeze_encoder, freeze_llm):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        # args
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.llm_config = llm.config

    def _encode_speech(self, padded_feat, feat_lengths):
        """Encode audio while keeping a frozen encoder deterministic."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                encoder_outs, enc_lengths, enc_mask = self.encoder(
                    padded_feat, feat_lengths)
        else:
            encoder_outs, enc_lengths, enc_mask = self.encoder(
                padded_feat, feat_lengths)
        speech_features, speech_lens = self.encoder_projector(
            encoder_outs, enc_lengths)
        return speech_features, speech_lens, enc_mask

    def encode_speech(self, padded_feat, feat_lengths):
        """Public speech-encoding API shared by generation and RL scoring."""
        return self._encode_speech(padded_feat, feat_lengths)

    def forward(self, padded_feat, feat_lengths, input_ids,
                attention_mask, labels):
        """Teacher-forcing forward pass used by FireRedASR2-LLM SFT.

        Prompt and speech positions are expected to be masked with
        ``IGNORE_TOKEN_ID`` in ``labels``. Hugging Face's causal LM performs
        the one-token label shift and cross-entropy calculation.
        """
        speech_features, speech_lens, _ = self._encode_speech(
            padded_feat, feat_lengths)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype),
                inputs_embeds,
                input_ids,
                attention_mask,
                labels=labels,
                speech_lens=speech_lens,
                padding_side="right",
            )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    def transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                   beam_size=1, decode_max_len=0, decode_min_len=0,
                   repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
        speech_features, speech_lens, _ = self._encode_speech(
            padded_feat, feat_lengths)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens, padding_side="left"
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids

    def generate_responses(
            self, speech_features, speech_lens, prompt_input_ids,
            prompt_attention_mask, num_return_sequences=1,
            max_new_tokens=160, temperature=0.8, top_p=0.95,
            top_k=0, repetition_penalty=1.0):
        """Sample response token IDs for pre-computed speech features.

        ``transformers.generate`` expands the multimodal prompt internally for
        ``num_return_sequences``. Because only ``inputs_embeds`` are supplied,
        the returned sequences contain generated response IDs rather than the
        text/speech prompt IDs.
        """
        if num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        inputs_embeds = self.llm.get_input_embeddings()(prompt_input_ids)
        inputs_embeds, merged_attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype),
                inputs_embeds,
                prompt_input_ids,
                prompt_attention_mask,
                speech_lens=speech_lens,
                padding_side="left",
            )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
            use_cache=True,
        )

    def score_responses(
            self, speech_features, speech_lens, prompt_input_ids,
            prompt_attention_mask, response_input_ids,
            response_attention_mask=None):
        """Return log-probabilities only for generated response tokens.

        Labels are used solely as an alignment map. Prompt and speech-prefix
        positions are set to ``IGNORE_TOKEN_ID``; after the causal one-token
        shift, ``token_mask`` therefore identifies exactly the generated
        action tokens optimized by sequence-level RL.
        """
        batch_size = prompt_input_ids.size(0)
        if response_input_ids.size(0) != batch_size:
            raise ValueError("Prompt and response batch sizes do not match")
        if speech_features.size(0) != batch_size:
            raise ValueError("Speech and response batch sizes do not match")
        if response_attention_mask is None:
            response_attention_mask = response_input_ids.ne(
                self.llm.config.pad_token_id)

        combined_ids = []
        combined_labels = []
        for index in range(batch_size):
            prompt_ids = prompt_input_ids[index][
                prompt_attention_mask[index].bool()]
            response_ids = response_input_ids[index][
                response_attention_mask[index].bool()]
            if response_ids.numel() == 0:
                raise ValueError(f"Response {index} contains no valid tokens")
            combined_ids.append(torch.cat([prompt_ids, response_ids], dim=0))
            combined_labels.append(torch.cat([
                torch.full_like(prompt_ids, IGNORE_TOKEN_ID),
                response_ids,
            ], dim=0))

        max_length = max(ids.numel() for ids in combined_ids)
        input_ids = prompt_input_ids.new_full(
            (batch_size, max_length), self.llm.config.pad_token_id)
        attention_mask = prompt_attention_mask.new_zeros(
            (batch_size, max_length))
        labels = response_input_ids.new_full(
            (batch_size, max_length), IGNORE_TOKEN_ID)
        for index, (ids, sample_labels) in enumerate(zip(
                combined_ids, combined_labels)):
            length = ids.numel()
            input_ids[index, :length] = ids
            attention_mask[index, :length] = 1
            labels[index, :length] = sample_labels

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype),
                inputs_embeds,
                input_ids,
                attention_mask,
                labels=labels,
                speech_lens=speech_lens,
                padding_side="right",
            )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        shift_logits = outputs.logits[:, :-1]
        shift_labels = labels[:, 1:]
        token_mask = shift_labels.ne(IGNORE_TOKEN_ID)
        safe_labels = shift_labels.masked_fill(~token_mask, 0)
        # Selected logit minus logsumexp avoids materializing a second full
        # [batch, sequence, vocabulary] log-softmax tensor during RL.
        selected_logits = shift_logits.gather(
            dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = (
            selected_logits - torch.logsumexp(shift_logits, dim=-1)).float()
        token_logprobs = token_logprobs.masked_fill(~token_mask, 0.0)
        return token_logprobs, token_mask


    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None, padding_side=None
    ):
        """Replace one ``<speech>`` token per sample with valid audio frames.

        The old inference-only implementation expanded every sample to the
        batch maximum speech length. That made padded audio visible to the LLM
        and caused repetition for mixed-duration batches. This implementation
        first packs every sample using its own ``speech_lens`` and only then
        pads the merged batch. Training uses right padding; autoregressive
        generation uses left padding so the last prompt token is never padding.
        """
        if padding_side is None:
            padding_side = "right" if labels is not None else "left"
        if padding_side not in ("left", "right"):
            raise ValueError(f"Unsupported padding_side: {padding_side}")

        batch_size, max_speech_len, embed_dim = speech_features.shape
        if input_ids.size(0) != batch_size or inputs_embeds.size(0) != batch_size:
            raise ValueError("Speech and text batch sizes do not match")
        if speech_lens is None:
            speech_lens = torch.full(
                (batch_size,), max_speech_len, dtype=torch.long,
                device=speech_features.device)
        speech_lens = speech_lens.to(device=speech_features.device, dtype=torch.long)
        if torch.any(speech_lens < 1) or torch.any(speech_lens > max_speech_len):
            raise ValueError(
                f"speech_lens must be within [1, {max_speech_len}], got "
                f"{speech_lens.detach().cpu().tolist()}")

        merged_embeddings = []
        merged_labels = []
        merged_lengths = []
        speech_token_id = self.llm.config.default_speech_token_id

        for batch_idx in range(batch_size):
            valid_text = attention_mask[batch_idx].bool()
            text_ids = input_ids[batch_idx][valid_text]
            text_embeddings = inputs_embeds[batch_idx][valid_text]
            speech_positions = torch.where(text_ids == speech_token_id)[0]
            if speech_positions.numel() != 1:
                raise ValueError(
                    "Each SFT prompt must contain exactly one <speech> token; "
                    f"sample {batch_idx} has {speech_positions.numel()}")
            speech_pos = int(speech_positions.item())
            speech_len = int(speech_lens[batch_idx].item())
            valid_speech = speech_features[batch_idx, :speech_len]
            sample_embedding = torch.cat([
                text_embeddings[:speech_pos],
                valid_speech.to(text_embeddings.dtype),
                text_embeddings[speech_pos + 1:],
            ], dim=0)
            merged_embeddings.append(sample_embedding)
            merged_lengths.append(sample_embedding.size(0))

            if labels is not None:
                text_labels = labels[batch_idx][valid_text]
                speech_labels = torch.full(
                    (valid_speech.size(0),), IGNORE_TOKEN_ID,
                    dtype=text_labels.dtype, device=text_labels.device)
                merged_labels.append(torch.cat([
                    text_labels[:speech_pos],
                    speech_labels,
                    text_labels[speech_pos + 1:],
                ], dim=0))

        max_merged_len = max(merged_lengths)
        final_embedding = inputs_embeds.new_zeros(
            (batch_size, max_merged_len, embed_dim))
        final_attention_mask = attention_mask.new_zeros(
            (batch_size, max_merged_len))
        final_labels = None
        if labels is not None:
            final_labels = labels.new_full(
                (batch_size, max_merged_len), IGNORE_TOKEN_ID)

        for batch_idx, sample_embedding in enumerate(merged_embeddings):
            sample_len = sample_embedding.size(0)
            start = 0 if padding_side == "right" else max_merged_len - sample_len
            end = start + sample_len
            final_embedding[batch_idx, start:end] = sample_embedding
            final_attention_mask[batch_idx, start:end] = 1
            if final_labels is not None:
                final_labels[batch_idx, start:end] = merged_labels[batch_idx]

        return final_embedding, final_attention_mask, final_labels
