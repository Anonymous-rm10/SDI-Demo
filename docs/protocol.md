# Audio and WebSocket protocol

Connect to `/ws` on the gateway, default `http://127.0.0.1:8090`. Each connection has independent history and turn generations; reconnecting starts a new session.

## Microphone input

Send binary messages containing **mono signed PCM16 little-endian at 16000 Hz**, without a WAV header. Do not send WebM/Opus. Recommended packet size: 100 ms / 3200 bytes. Each message must have an even byte count and contain no more than one second / 32000 bytes.

The gateway uses 20 ms energy VAD frames. The browser worklet resamples the actual device rate to 16 kHz. `audio.commit` submits the current detected utterance, subject to the minimum speech duration; an incomplete VAD frame and unsent worklet packet may be discarded.

## Client JSON messages

| Type | Fields | Action |
| --- | --- | --- |
| `text` | `text`: 1–2000 characters | Submit an utterance through the same JEV decision path |
| `audio.commit` | — | Submit the current detected audio segment |
| `interrupt` | — | Cancel the reply, clear queued input, and reset VAD |
| `reset` | — | Cancel and clear conversation history |
| `playback.start` | `turn_id` | Audio for this turn is playing or scheduled |
| `playback.end` | `turn_id` | This turn's playback queue is empty |
| `ping` | — | Request a keepalive response |

Report playback state: server generation can finish before the speakers finish playing. Stale turn IDs are ignored.

## Server events

| Type | Fields | Meaning |
| --- | --- | --- |
| `ready` | `sample_rate`, `format`, `mock` | Session ready |
| `vad.start`, `vad.end` | — | Detected speech boundaries |
| `transcript` | `text` | Final ASR or submitted text |
| `decision` | `respond`, `interrupt`, `source`, probability fields | Turn decision |
| `input.ignored` | `reason` | `keep_speaking` or `no_reply_needed` |
| `turn.cancelled` | `turn_id`, `reason` | Clear this turn's queued playback |
| `turn.start` | `turn_id` | New reply |
| `text.delta` | `turn_id`, `text` | Incremental LLM text |
| `audio.chunk` | `turn_id`, `sample_rate`, `audio` | Base64 mono PCM16LE, without a container header |
| `turn.done` | `turn_id` | All text/audio sent; playback may remain |
| `turn.failed` | `turn_id` | Reply failed; clear remaining playback |
| `error` | `message`, optional `turn_id` | Displayable error; details are logged server-side |
| `pong` | — | Keepalive response |

A new turn also emits `turn.cancelled` for the previous generation, including an already completed generation. Only display/play deltas for the active turn. Play audio chunks in order at their declared sample rate.

Each session queues at most two pending inputs and four LLM sentence chunks. The TTS worker queues at most four audio chunks. ASR/TTS HTTP requests default to a 120-second timeout; LLM/JEV timeouts are configurable. Disconnecting cancels the session. History records sentences whose audio transmission began, without claiming that the user heard all of them.

## Model worker endpoints

`POST :8091/transcribe`: raw mono PCM16 WAV, 8–48 kHz, no more than 30 seconds and approximately 964 KB. Prefer 16 kHz audio. Returns `{"text":"...","language":"English"}`. Inference is serialized in a worker thread.

`POST :8092/synthesize`: JSON `{"text":"Hello"}`, 1–500 characters. Returns streaming raw PCM16LE with `X-Sample-Rate`, `X-Audio-Format: pcm_s16le`, and `X-Channels: 1`. Cancellation stops transmission; an in-flight GPU call finishes before the model lock is released.

Both workers provide `GET /health` and accept `--mock` when launched separately.

## LLM and JEV

The gateway calls `${LLM_BASE_URL}/chat/completions` with Bearer authentication, `stream=true`, and standard `messages`. It consumes SSE `choices[0].delta.content`, not reasoning or tool calls. API keys remain server-side.

JEV uses `${JEV_URL}/v1/systemone` with two `noul` questions, reading `answers.respond.noul` and `answers.interrupt.noul`. Decision sources are `jev`, `fallback`, `disabled`, and `mock`.
