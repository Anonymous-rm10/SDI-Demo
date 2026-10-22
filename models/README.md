# Model weights

Use this layout, or point `.env` at existing local model directories:

```text
models/
├── Qwen3-ASR-0.6B/
├── Fun-CosyVoice3-0.5B/
├── Qwen3.5-4B/
└── train-jev-4b-lora/
```

With the Hugging Face CLI installed in your download environment, run from the repository root:

```bash
hf download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/Fun-CosyVoice3-0.5B
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
hf download shibing624/train-jev-4b-lora --local-dir models/train-jev-4b-lora
```

CosyVoice needs its complete directory, including CosyVoice-BlankEN, tokenizer, configuration, and acoustic components. The JEV adapter needs LoRA weights, `head.pt`, and `calibration.json`; loading only a standard PEFT adapter is insufficient. Follow the upstream checkpoint's instructions for experimental attention flags.

Weights are not downloaded automatically or committed to this source repository. Model artifacts and resources retain their publishers' licenses.
