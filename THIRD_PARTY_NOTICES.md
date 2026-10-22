# Third-party source notices

This independent project bundles inference and training source from the local working trees listed below. Original licenses and copyright notices are retained. Local extensions are not presented as official upstream releases.

| Component | Upstream | Recorded upstream commit | License |
| --- | --- | --- | --- |
| Qwen3-ASR | https://github.com/QwenLM/Qwen3-ASR | `c17a131fe028b2e428b6e80a33d30bb4fa57b8df` | [Apache-2.0](third_party/Qwen3-ASR/LICENSE) |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | [Apache-2.0](third_party/CosyVoice/LICENSE) |
| train-jev | https://github.com/shibing624/train-jev | `928137ef70bcea690d5d561e5990dfba313a225a` | [Apache-2.0](third_party/train-jev/LICENSE) |
| FireRedASR2S | https://github.com/FireRedTeam/FireRedASR2S | `4e7d9aaf4482a47cec1724807026b9b151926eb5` | [Apache-2.0](third_party/FireRedASR2S/LICENSE) |
| Matcha-TTS | https://github.com/shivammehta25/Matcha-TTS | `dd9105b34bf2be2230f4aa1e4769fb586a3c824e` | [MIT](third_party/CosyVoice/third_party/Matcha-TTS/LICENSE) |
| HiFi-GAN within Matcha-TTS | Included in the Matcha-TTS snapshot | Same snapshot | [MIT](third_party/CosyVoice/third_party/Matcha-TTS/matcha/hifigan/LICENSE) |

## Provenance

`third_party/manifest.json` lists copied files and final SHA-256 digests. `source_sha256` identifies the source working-tree bytes before integration changes. `upstream_sha256` records the tracked HEAD bytes when local source differs. `local_only` identifies local extensions absent from the upstream index, `local_modified` identifies tracked local edits, and `modified` identifies integration changes. `materialized_from` records the internal source of an expanded recipe symlink.

The copied code includes local Qwen OPD/OPSD trainers, local FireRed SFT/RL and Qwen distillation extensions, their configurations, and associated tests. Their copyright and authorship remain with their respective contributors; upstream commit IDs alone do not describe these local additions.

## Integration changes

- Restored all supplied training families, package exports, inference demos, and original dependency lists after the earlier inference-only reduction.
- Corrected Qwen3-ASR dictionary package data and adapted the local SFT/distillation launch paths.
- Adapted FireRed training configurations and cross-model scripts to the bundled layout; removed implicit references to old local teacher checkpoints.
- Kept canonical FireRed packages instead of stale duplicate copies in upstream inference convenience directories, and set example scripts to use the canonical package paths.
- Materialized CosyVoice's internal recipe links. Matcha-TTS, its training code, distributed logging, and extension build sources are included as regular files.

Application code and the shared launcher live outside the upstream directories. Project-level documentation and interface strings are English. Upstream documentation, training prompts, dictionaries, tokenizers, and reference resources retain their source contents and languages.

Weights, datasets, generated predictions, experiment state, caches, logs, Git metadata, upstream CI configuration, and unrelated publication assets are excluded. Downloaded model/data resources retain their publishers' licenses. The voice architecture was informed by [qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent); its source is not bundled.
