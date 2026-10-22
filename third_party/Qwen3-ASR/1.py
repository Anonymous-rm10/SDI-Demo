import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "/home/ln10/Workspace/models/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio="/home/ln10/Workspace/datasets/kespeech/Audio/1000001/phase1/1000001_0b1a33a3.wav",
    language="Chinese", # set "English" to force the language
)

print(results[0].language)
print(results[0].text)

