# Kiki: plan for a table-top conversational agent

A device that sits on a table, listens to a group conversation, knows who is speaking,
and joins in when that makes sense. Fully self-hosted: open models on the Pi and on a
local server.

> The research section below was written for the first, cloud-based plan. Cloud STT,
> Claude and cloud TTS have since been dropped (see Architecture).

## Hardware (confirmed 2026-11-21 by `scripts/pi_check.sh`)

- Raspberry Pi 5 Model B, 8 GB, Debian 13 (trixie) 64-bit. **Currently boots from the
  SD card; the NVMe isn't detected yet.**
- Jabra SPEAK 510 USB speakerphone (ALSA card `USB`). Mic: 16 kHz mono S16_LE natively.
  Speaker: stereo, 8–48 kHz. Echo test: +15 dB above a quiet floor, so the Jabra's own
  AEC works partially. Half-duplex stays the default; full-duplex/barge-in needs tuning.

## What the research says

1. **Deciding when to speak is the hard part, and it is still unsolved.** Recent papers find that
   LLMs used zero-shot are bad at telling who an utterance is meant for and whether to speak at
   all in a group. GPT-4o was barely better than chance at three-person addressee recognition, and
   most models jump in far too often ([arXiv 2603.11409](https://arxiv.org/html/2603.11409),
   [arXiv 2501.16643](https://arxiv.org/html/2501.16643v2), [arXiv 2610.07732](https://arxiv.org/pdf/2610.07732),
   [survey](https://arxiv.org/pdf/2505.18845)). So we should not leave the whole decision to one
   prompt. The design below combines hard signals (its name, a direct question, an active
   conversation thread with Kiki) with an LLM gate that defaults to staying silent, and logs every
   decision so the gate can be tuned on real sessions.
2. **Detecting end of turn:** Pipecat **Smart Turn v3** is open source (BSD-2). It reads raw 16 kHz
   audio, covers 23 languages, and takes about 65 ms per inference. It runs after Silero VAD with
   `stop_secs≈0.2` ([repo](https://github.com/pipecat-ai/smart-turn),
   [docs](https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview)). Its output
   answers "has the speaker finished?". It says nothing about whether Kiki should talk.
3. **Diarization (who spoke when):**
   - Full streaming diarization on the Pi itself is not proven. I found no project that does it,
     and the usual pattern sends audio to a server ([UB report](https://cse.buffalo.edu/tech-reports/2026-29.pdf)).
     diart and FastDiar exist but would have to be benchmarked on the Pi.
   - Cloud STT services with streaming diarization: AssemblyAI Universal-3 Pro Streaming (6
     languages, no Swedish) ([docs](https://assemblyai.com/docs/streaming/streaming-diarization)),
     Speechmatics (55+ real-time languages, diarization included at no extra cost)
     ([comparison](https://www.speechmatics.com/product/how-we-compare/speechmatics-vs-assemblyai)), and
     Deepgram (streaming diarization returns only a speaker index and uses the older v1 diarizer)
     ([docs](https://developers.deepgram.com/docs/diarization)). Every benchmark comes from a
     vendor, so we will run our own bake-off.
   - **Speaker *identity*** (names that stay the same across sessions) needs our own voiceprints
     either way. sherpa-onnx provides speaker embeddings, identification, and verification on ARM64.
4. **Local STT:** Moonshine streaming takes about 240–800 ms on a Pi 5, against about 6–10 s for
   Whisper tiny/small ([benchmarks](https://moonshine-voice.readthedocs.io/en/latest/using/benchmarks/)).
   Moonshine is mainly English. Swedish locally would mean whisper.cpp with a KB-Whisper model,
   which is too slow on a Pi for live conversation.
5. **TTS:** Piper runs in real time on a Pi 5 CPU. I found no published Pi benchmark for Kokoro.
   Cloud TTS sounds more natural but adds latency and cost.
6. **Echo is the main practical risk.** If Kiki hears its own voice, it transcribes itself and
   barge-in breaks ([example](https://community.openai.com/t/realtime-api-starts-to-answer-itself-with-mic-speaker-setup/977801)).
   Many USB speakerphones (Jabra Speak, Anker PowerConf, eMeet) cancel echo in hardware, and that
   is the best case. Otherwise PipeWire's WebRTC `echo-cancel` module works best when the mic and
   speaker are the same device, which ours are ([man page](https://man.archlinux.org/man/libpipewire-module-echo-cancel.7.en)).
7. **Wake word (optional):** openWakeWord runs cheaply on a Pi, but training a custom model is
   painful. I propose detecting the name "Kiki" in the transcript instead. If that turns out too
   slow, we add a wake-word model.

## Architecture (fully self-hosted, decided 2026-11-21)

No cloud services and no Claude: every model is open-weight and runs either on the
Pi or on a **Kiki server** the Pi connects to. The server is the laptop for now
(Ryzen 7 7735HS, 26 GB, Radeon 680M) and moves later to the box with 256 GB RAM
and an RTX A4000 (16 GB). Everything is containerized, so the move means a CUDA
image and a different compose file.

```
 Pi (edge)                                        Kiki server (laptop → A4000 box)
 ─────────                                        ───────────────────────────────
 USB speakerphone ─► AEC ─► Silero VAD            websocket :8765
                         └► Smart Turn v3 ──────► segments (int16 PCM) + turn_end
                                                    │
                                                    ├► Parakeet TDT 0.6B v2 (STT)     ✅
                                                    ├► WeSpeaker ResNet34 voiceprints ✅
                                                    │    → registry: enrolled + guests
                                                    ├► conversation log (JSONL)       ✅
                                                    ├► floor policy + gate (local LLM)
                                                    ├► responder (local LLM, llama.cpp)
                                                    └► Kokoro TTS ◄─ sentence chunks   ✅ (module)
 speaker ◄── playback ◄─────────────────────────── say + PCM chunks / stop
```

**Edge ↔ server protocol** (`src/kiki/protocol.py`): JSON control messages, with
audio sent as a binary frame. The edge sends one segment per contiguous run of
speech, so the server transcribes and identifies the speaker *while the person is
still talking*. When `turn_end` arrives, the text is usually ready.

### How it decides to speak (floor policy)

Same layered design as before, but the gate and the responder are local models
served by llama.cpp (OpenAI-compatible API; `json_schema` constrained output for
the gate):

1. **Hard rules** (instant): its name, mute, cooldowns, and the engagement window.
2. **Gate:** a small, fast instruct model (~3–4B on the laptop) reads the recent
   transcript and returns `{action, addressed_to_kiki, reason}`. Proactive
   `interject` is enabled, with a confidence threshold, a cooldown, and the
   "chime in less/more" voice control.
3. **Speculative responder start**, run in parallel with the gate.
4. Every decision is logged for later tuning.

### Responder

A local instruct model via llama.cpp (Vulkan on the 680M for now, CUDA on the A4000
later), streaming tokens → sentence chunker → Kokoro → Pi. Model size per host:
~4–8B Q4 on the laptop, ~14–32B on the A4000 (16 GB VRAM). With 256 GB of RAM,
larger MoE models with CPU offload are possible too. The model is pinned in the
config, not hard-coded.

### Who's talking (done, needs real-voice calibration)

- Each segment gets a 256-d WeSpeaker embedding and is compared with voices by
  cosine similarity. Clips under 1 s never create or update a voice. A clearly new
  voice, or a long ambiguous clip (≥ 2.5 s), becomes `Guest-N`. Guests that drift
  onto each other get merged.
- Threshold 0.68, calibrated on Kokoro voices (same voice ≥ 0.74 on > 1 s clips,
  different voices up to ~0.8). It **must be re-tuned on real voices from the Pi
  mic**.
- Enrollment: `kiki enroll NAME a.wav b.wav`. Conversational naming ("Kiki, I'm
  Anna" → rename a guest) comes with the brain.

### Measured on the laptop (CPU, in Docker)

| Stage | Time |
|---|---|
| Silero VAD per 32 ms frame | 0.1 ms |
| Smart Turn v3.2 (8 s window) | ~90 ms (features 10 ms) |
| Parakeet STT per segment (1.5–4 s of speech) | 120–380 ms |
| Kokoro TTS | ~0.27× real time (1 s of speech in ~0.27 s) |

## Tech choices

- Python 3.12 + asyncio, `uv`, Docker everywhere (the image runs on amd64 and arm64).
- sherpa-onnx 2.x for STT, speaker embeddings, and TTS. Kokoro's phonemes come from
  our own lexicon + espeak-ng fallback (`server/phonemize.py`), because
  sherpa-onnx 2.0 dropped the built-in espeak and silently skipped unknown words.
- llama.cpp server for the LLMs (`server-vulkan` image now, `server-cuda` later).
- Websockets between the Pi and the server.

## Repo layout (current)

```
src/kiki/
  config.py events.py protocol.py models.py cli.py
  audio/   sources.py vad.py turn.py features.py segmenter.py   # runs on the Pi
  edge/    client.py                                             # Pi → server link
  server/  app.py stt.py speakers.py tts.py phonemize.py         # Kiki server
scripts/   pi_check.sh make_conversation.py
tests/     VAD/turn segmentation, feature parity, speakers, TTS↔STT round trip
```

## Milestones

0. **Hardware bring-up:** run `scripts/pi_check.sh` on the Pi (board, NVMe boot,
   speakerphone, echo test). ⏳ waiting on the Pi.
1. **Ears:** VAD + Smart Turn on the edge; local STT + speaker ID on the server;
   transcript log. ✅ works with laptop mic or WAV replay.
2. **Voice:** llama.cpp service + responder, triggered by its name (and by follow-ups
   in the engagement window) → Kokoro → playback on the edge. ✅ on the laptop:
   ~3.5 s from end of speech to first audio (STT 0.4 s, LLM first token +0.8 s,
   first clause +0.6 s, TTS of that clause ~1.6 s under CPU contention). The Pi
   edge config is in `compose.pi.yaml`.
3. **Who:** conversational naming, calibration on real voices, merge-aware history.
4. **Manners:** floor policy (hard rules, engagement window, local gate,
   speculative start), barge-in, decision logging.
5. **Personality and memory:** persona, per-person memory (SQLite), local tools.
6. **Tuning:** label logged decisions, build an eval set, tune the gate. Move the
   server to the A4000 box.

## Decisions

- **English only.**
- **Fully self-hosted:** no cloud STT/TTS/LLM, and no Claude (2026-11-21). This
  replaces the earlier cloud plan (AssemblyAI / Claude / cloud TTS).
- **Proactive from day one,** with a threshold, a cooldown, and voice controls.
- **Run in Docker** on the dev laptop.

## Still open

- Exact Pi model and RAM, and the speakerphone model (hardware AEC?). Run
  `bash scripts/pi_check.sh` on the Pi.
- LLM: **Qwen3-4B-Instruct-2507 Q4_K_M on CPU** for now (~10 tok/s). Findings:
  - The Radeon 680M via Vulkan was ~30% faster but hung the GPU under load
    (amdgpu reset, "device wedged"), even with small batches. Don't use it.
  - Hybrid-attention models (Qwen3.5) defeat llama.cpp's partial KV-cache reuse,
    so every reply re-processed the whole transcript. Use pure-attention models.
  - Ministral 3 3B was faster but its answers were worse.
  The A4000 should cut latency to ~1.5 s: an 8–14B model on CUDA, with TTS no longer
  competing for the CPU.
- NVMe not detected on the Pi.
