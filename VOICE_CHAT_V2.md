# Voice Chat Pipeline v2 — Architecture & Documentation

> *Cartographer's Map — Védis Eikleið*
> *Date: 2026-06-07*
> *Status: Operational (Pi 5 — audio I/O requires USB peripherals)*

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Runa Voice Chat v2                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────┐    ┌────────────┐    ┌─────────┐           │
│  │  Mic    │───▶│  Silero    │───▶│ faster- │           │
│  │  Input   │    │  VAD       │    │ whisper │           │
│  │(PyAudio)│    │(or energy) │    │  STT    │           │
│  └─────────┘    └────────────┘    └────┬────┘           │
│                                         │                │
│                                    Transcribed Text       │
│                                         │                │
│                                    ┌────▼────┐           │
│                                    │ Hermes  │           │
│                                    │   API   │           │
│                                    │(LLM)    │           │
│                                    └────┬────┘           │
│                                         │                │
│                                      Response Text        │
│                                         │                │
│                                    ┌────▼────┐           │
│                                    │ Piper   │           │
│                                    │   TTS   │           │
│                                    └────┬────┘           │
│                                         │                │
│                                    ┌────▼────┐           │
│                                    │ aplay   │           │
│                                    │(Speaker)│           │
│                                    └─────────┘           │
└──────────────────────────────────────────────────────────┘
```

## File Map

| File | Purpose | Status |
|------|---------|--------|
| `voice_chat_v2.py` | **Primary pipeline** — rewritten with hardening | ✅ Active |
| `voice_llm.py` | Original PyAudio+LLM pipeline | ⚠️ Legacy (hardcoded API key) |
| `voice_chat.py` | Original sounddevice echo mode | ⚠️ Legacy (no LLM) |
| `wake_runa.py` | Wake word daemon (VAD + whisper base) | ⚠️ Legacy (energy VAD only) |
| `voice_llm_arecord.py` | arecord-based alternative | ⚠️ Legacy (hardcoded hw:3,0) |
| `wake_runa_oww.py` | openWakeWord version | ❌ Experimental (uses "alexa" wake word) |

## Configuration

### Environment Variables (or `~/.hermes/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MIC_SAMPLE_RATE` | 48000 | USB mic native rate |
| `MIC_DEVICE_INDEX` | -1 (auto) | PyAudio input device index |
| `STT_MODEL_SIZE` | base | faster-whisper model (tiny/base/small) |
| `STT_DEVICE` | cpu | Whisper device |
| `STT_COMPUTE` | int8 | Whisper compute type |
| `STT_LANGUAGE` | en | Whisper language |
| `HERMES_API_URL` | http://localhost:8642/v1/chat/completions | LLM endpoint |
| `API_SERVER_KEY` | (from .env) | API authentication key |
| `HERMES_API_MODEL` | kimi-k2.6 | LLM model name |
| `HERMES_API_MAX_TOKENS` | 250 | Max response tokens |
| `HERMES_API_TIMEOUT` | 30 | Request timeout (seconds) |
| `HERMES_API_MAX_RETRIES` | 3 | Retry count with backoff |
| `PIPER_VOICE` | cori | Default TTS voice |
| `VAD_MODE` | silero | VAD mode (silero or energy) |
| `VAD_SILENCE_SEC` | 1.5 | Silence timeout before end-of-speech |
| `VAD_ENERGY_THRESHOLD` | 0.003 | Energy VAD threshold (if not using Silero) |
| `CONV_MAX_MESSAGES` | 20 | Max messages in conversation history |
| `CONV_MAX_CHARS` | 4000 | Max total characters in conversation |

### Available Voices

| Short Name | Full Model | Quality | Size |
|-----------|-----------|---------|------|
| `cori` | en_GB-cori-medium | ⭐ Best balance | 61MB |
| `cori-high` | en_GB-cori-high | ⭐⭐ Highest quality | 109MB |
| `alba` | en_GB-alba-medium | Good | 61MB |
| `jenny` | en_GB-jenny_dioco-medium | Good (Irish warmth) | 61MB |
| `aru` | en_GB-aru-medium | Good | 74MB |
| `alan` | en_GB-alan-medium | Male UK | 61MB |
| `vctk` | en_GB-vctk-medium | Multi-speaker UK | 74MB |
| `ljspeech` | en_US-ljspeech-medium | US female | 61MB |

## Usage

```bash
# VAD auto-detect mode (recommended)
cd ~/voice-pipeline
python3 voice_chat_v2.py

# Push-to-talk
python3 voice_chat_v2.py --mode ptt

# One-shot (5 seconds)
python3 voice_chat_v2.py --mode once -d 5

# Text only (no TTS)
python3 voice_chat_v2.py --no-speak

# High-quality voice
python3 voice_chat_v2.py --voice cori-high

# Wake word daemon mode
python3 voice_chat_v2.py --wake

# List audio devices
python3 voice_chat_v2.py --list-devices

# Debug mode
python3 voice_chat_v2.py --debug

# Reset conversation
python3 voice_chat_v2.py --reset

# Specify device
python3 voice_chat_v2.py --device 2
```

## Improvements over v1

### Bug Fixes

| # | Severity | Description | Fix | Date |
|---|----------|-------------|-----|------|
| 1 | P0 | API key hardcoded in legacy scripts | Load from env var `API_SERVER_KEY` or `.env` | 2026-06-07 |
| 2 | P0 | API key hardcoded in arecord script | Same env var loading as v2 | 2026-06-08 |
| 3 | P0 | Subprocess orphan on crash | `try/finally` with `terminate()` + `kill()` | 2026-06-07 |
| 4 | P0 | Piper hang (no timeout) | 30s timeout on `player.wait()` | 2026-06-07 |
| 5 | P1 | Missing Piper restart budget check | Pre-flight check with cooldown recovery | 2026-06-08 |
| 6 | P1 | `piper_proc.wait()` timeout not handled | Added `TimeoutExpired` with `terminate()` → `kill()` | 2026-06-08 |
| 7 | P1 | `player_proc.wait()` zombie after terminate | Added `kill()` fallback on second timeout | 2026-06-08 |
| 8 | P1 | `piper_proc.stderr.read()` None risk | Guarded with `if piper_proc.stderr else ""` | 2026-06-08 |
| 9 | P1 | Duplicate `HERMES_API`/`API_URL` | Single `API_URL` config | 2026-06-07 |
| 10 | P1 | No Silero VAD (installed but unused) | Integrated Silero with energy fallback | 2026-06-07 |
| 11 | P1 | No API retry on failure | 3 retries with exponential backoff | 2026-06-07 |
| 12 | P1 | Unbounded conversation history | Capped by message count + character budget | 2026-06-07 |
| 13 | P1 | Hardcoded Mic device index | Auto-detect with manual override | 2026-06-07 |
| 14 | P1 | Missing voices in voice map | Unified voice map with 16 voices (was 8) | 2026-06-08 |
| 15 | P2 | Piper stderr swallowed | Capture stderr for debugging | 2026-06-07 |
| 16 | P2 | No graceful TTS fallback | Continue with text output if TTS fails | 2026-06-07 |
| 17 | P2 | Inconsistent gain normalization | Always normalize audio before STT | 2026-06-07 |
| 18 | P2 | SSL_CERT_FILE in kokoro venv | Clear before torch/huggingface calls | 2026-06-07 |
| 19 | P2 | Wake word false positives | Removed single-syllable words from main list | 2026-06-07 |
| 20 | P2 | No noise suppression | Added scipy spectral gating + `suppress_noise()` | 2026-06-08 |

### Hardening Features

- **Auto-degrading VAD**: Falls back from Silero to energy-based if torch unavailable
- **Audio device auto-detection**: Scans PyAudio devices for USB mic, falls back to default
- **Subprocess health monitoring**: Piper process exit status checked, auto-restart tracked
- **Structured logging**: All events logged to `~/.hermes/logs/voice_chat.log`
- **Configurable via environment**: Every parameter overridable without code changes
- **Wake word daemon**: Built-in `--wake` mode with Silero VAD + whisper base

## Piper TTS

- Binary: `~/piper/piper/piper` (aarch64, v2023.11.14-2)
- Wrapper: `/usr/local/bin/piper` → `/home/pi/piper/piper-wrapper.sh`
- Voices: `~/piper/voices/` (16 models)
- Default: `en_GB-cori-medium` (RTF ~0.14 on Pi 5)
- **Critical**: Must use wrapper script (sets LD_LIBRARY_PATH for bundled libespeak-ng)

## Known Limitations

1. **USB mic currently detached** — No capture device detected. Scripts will fail until mic is reconnected.
2. **Speaker output HDMI only** — No USB speaker detected. Piper outputs raw PCM; aplay defaults to HDMI audio.
3. **No streaming TTS** — Piper generates entire utterance before playback starts. Latency ~0.5-1s for short phrases.
4. **No streaming STT** — faster-whisper processes complete utterances. Real-time streaming would require whisper-streaming or Vosk.
5. **No noise suppression** — rnnoise not installed. Background noise can cause false VAD triggers. Silero VAD is more robust than energy VAD but still benefits from clean audio.
6. **Wake word accuracy** — "Hey Runa" detection uses whisper base model. Short phrases can be misheard. Recommended use: push-to-talk for reliability, wake word for convenience.

## Future Improvements (Research Phase)

### TTS Candidates

| Engine | Quality | Speed (Pi 5) | RAM | Offline | Notes |
|--------|---------|---------------|-----|---------|-------|
| **Piper** | ⭐⭐⭐⭐ | Very fast (RTF 0.14) | ~200MB | ✅ | Currently in use. Best Pi 5 option |
| **Kokoro 82M** | ⭐⭐⭐⭐⭐ | Fast (35-100x RT on GPU) | ~500MB | ✅ | Installed in `~/.venvs/kokoro/`. Better prosody than Piper |
| **Chatterbox Turbo** | ⭐⭐⭐⭐⭐ | Medium (needs GPU) | ~2GB | ✅ | Emotion control + zero-shot voice cloning. Too heavy for Pi CPU |
| **Voxtral TTS** | ⭐⭐⭐⭐⭐ | Slow on CPU | ~8GB | ✅ | Mistral's 4B TTS model. Streaming capable. Needs GPU |
| **Dia 1.6B** | ⭐⭐⭐⭐⭐ | Medium | ~4GB | ✅ | Multi-character dialogue generation. Apache 2.0 |
| **F5-TTS** | ⭐⭐⭐⭐ | Slow | ~3GB | ✅ | Voice cloning from short samples |
| **MeloTTS** | ⭐⭐⭐ | Fast | ~800MB | ✅ | Multi-language. Lower quality |

**Recommendation**: Keep Piper as primary. Investigate Kokoro for higher-quality output with Kokoro FastAPI server (already installed in venv). Use Chatterbox/Voxtral on Mjölnir (GPU) for voice cloning.

### STT Candidates

| Engine | Latency | Accuracy | Streaming | Notes |
|--------|---------|----------|-----------|-------|
| **faster-whisper** | ~300ms-1s | ⭐⭐⭐⭐⭐ | No | Currently in use. Best accuracy |
| **whisper-streaming** | ~200ms | ⭐⭐⭐⭐⭐ | ✅ | Real-time partial transcription. Boosts responsiveness |
| **Vosk** | ~100ms | ⭐⭐⭐ | ✅ | Streaming, lightweight. Good for commands, less for natural speech |
| **Silero VAD** | ~20ms | N/A (VAD) | ✅ | Already integrated in v2. Best-in-class VAD |

**Recommendation**: Integrate whisper-streaming for partial transcription display. This gives the user real-time feedback while they speak.

### Real-Time Pipeline Architecture

The best path to low-latency voice conversation:

1. **Phase 1 (Current)**: VAD → full STT → LLM → full TTS → playback. Latency: 5-15s.
2. **Phase 2**: Silero VAD → streaming STT (whisper-streaming) → LLM (streaming) → chunked TTS → playback. Target latency: 2-5s.
3. **Phase 3**: LiveKit WebRTC → streaming audio → server-side pipeline → streaming TTS → WebRTC playback. Target latency: 0.5-2s.

**LiveKit Agents** (Python SDK): Most promising framework for Phase 3. Provides WebRTC transport, VAD, STT/TTS integration, and interruption handling as built-in features. Runs on Mjölnir with GPU, Pi as client.

### Noise Suppression

- **rnnoise** not installed. Recommended for Phase 2: `pip install rnnoise-noisy` — neural noise suppression that runs in real-time on CPU.
- **scipy spectral gating** — Can be added as intermediate step between mic capture and STT.
- **Silero VAD already handles noise** implicitly by only passing speech segments to STT.

## Troubleshooting

### "No audio input device"
- USB mic not connected. Check with `aplay -l` and `python3 -c "import pyaudio; ..."`
- Use `--list-devices` to see available devices
- Use `--device N` to specify device index

### "Piper symbol lookup error"
- Always use `/usr/local/bin/piper` (wrapper script), not the raw binary
- The wrapper sets `LD_LIBRARY_PATH` for the bundled libespeak-ng

### "SSL_CERT_FILE path too long"
- The kokoro venv sets a very long SSL_CERT_FILE that causes OSError
- Fixed in v2 by clearing SSL_CERT_FILE before torch/huggingface calls
- If it recurs: `unset SSL_CERT_FILE` before running

### "Silero VAD failed to load"
- Check torch installation: `python3 -c "import torch; print(torch.__version__)"`
- Falls back to energy-based VAD automatically
- Less accurate but functional

### Slow model loading
- First run downloads whisper models (~75MB for base, ~40MB for tiny)
- Subsequent runs use cache from `~/.cache/huggingface/`
- Pre-load with: `python3 -c "from faster_whisper import WhisperModel; WhisperModel('base')"`

### API connection failure
- Check Hermes server: `curl http://localhost:8642/v1/models`
- Verify API key in `~/.hermes/.env` as `API_SERVER_KEY=...`
- v2 retries 3 times with exponential backoff

## Dependencies

```
faster-whisper==1.2.1
silero-vad==6.2.1
torch==2.11.0+cpu
torchaudio==2.11.0+cpu
numpy==1.26.4
scipy==1.10.1
PyAudio==0.2.13
sounddevice==0.5.5
piper-tts==1.4.2 (via Hermes venv)
kokoro==0.9.4 (in ~/.venvs/kokoro/)
```