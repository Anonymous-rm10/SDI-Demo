# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import re

import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

DEFAULT_SPEECH_TOKEN = "<speech>"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class LlmTokenizerWrapper:
    TRAIN_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    DECODE_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

    @classmethod
    def build_llm_tokenizer(cls, llm_path, use_flash_attn=False):
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if use_flash_attn:
            tokenizer.padding_side = "left"
        else:
            tokenizer.padding_side = "right"
        special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    @classmethod
    def clean_text(cls, origin_text):
        """remove punc, remove space between Chinese and keep space between English"""
        # remove punc
        text = re.sub(r"[，。？！,.!?《》（）·“”、\\/]", "", origin_text)
        # merge space
        text = re.sub(r"\s+", " ", text)

        # remove space between Chinese and keep space between English
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')  # Chinese
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        text = "".join(parts)
        text = text.strip()

        text = text.lower()
        return text

    @classmethod
    def preprocess_texts(cls, origin_texts, tokenizer, max_len, decode=False,
                         instruction="请转写音频为文字"):
        """Build inference prompts or SFT teacher-forcing tensors.

        For SFT, the prompt length is measured by tokenizing the matching
        decode prompt instead of searching for a token whose surface form is
        ``assistant``. This remains correct when tokenizer versions split the
        role name into multiple pieces.
        """
        if not origin_texts:
            raise ValueError("origin_texts must not be empty")

        texts = []
        prompt_lengths = []
        clean_texts = []
        for sample_idx, origin_text in enumerate(origin_texts):
            text = cls.clean_text(origin_text)
            clean_texts.append(text)
            message = [
                {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}{instruction}"},
                {"role": "assistant", "content": "" if decode else text},
            ]
            template = cls.DECODE_TEMPLATE if decode else cls.TRAIN_TEMPLATE
            token_ids = tokenizer.apply_chat_template(
                message,
                tokenize=True,
                chat_template=template,
                add_generation_prompt=False,
                truncation=False,
                return_dict=False,
            )
            if len(token_ids) > max_len:
                raise ValueError(
                    f"Tokenized sample {sample_idx} has {len(token_ids)} tokens, "
                    f"exceeding max_len={max_len}")
            texts.append(list(token_ids))

            if decode:
                prompt_lengths.append(len(token_ids))
            else:
                prompt_message = [
                    {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}{instruction}"},
                    {"role": "assistant", "content": ""},
                ]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_message,
                    tokenize=True,
                    chat_template=cls.DECODE_TEMPLATE,
                    add_generation_prompt=False,
                    truncation=False,
                    return_dict=False,
                )
                prompt_lengths.append(len(prompt_ids))

        if not decode:
            for idx, (token_ids, prompt_len) in enumerate(zip(texts, prompt_lengths)):
                if len(token_ids) <= prompt_len:
                    raise ValueError(
                        f"SFT target for sample {idx} was fully truncated; "
                        f"increase max_len={max_len} or shorten the transcript")

        # Padding texts
        text_lengths = [len(text) for text in texts]
        max_len_texts = max([len(text) for text in texts])
        if tokenizer.padding_side == "right":
            texts = [
                list(text) + [tokenizer.pad_token_id] * (max_len_texts - len(text))
                for text in texts
            ]
        else:
            texts = [
                [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
                for text in texts
            ]
        input_ids = torch.tensor(texts, dtype=torch.int)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

        # Supervise only the assistant transcript and its <|im_end|> token.
        for row, prompt_len in enumerate(prompt_lengths):
            if tokenizer.padding_side == "right":
                target_ids[row, :prompt_len] = IGNORE_TOKEN_ID
            else:
                pad_len = input_ids.size(1) - text_lengths[row]
                target_ids[row, :pad_len + prompt_len] = IGNORE_TOKEN_ID

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        target_ids = target_ids.type(torch.LongTensor)
        input_ids = input_ids.type(torch.LongTensor)
        return input_ids, attention_mask, target_ids, clean_texts
