# Runa Voice Chat Pipeline

> *Resilient voice interaction on Raspberry Pi 5 — STT → LLM → TTS*

## What It Does

Real-time voice conversation pipeline running entirely on a Raspberry Pi 5:

1. **Listen** — PyAudio captures microphone input
2. **Detect speech** — Silero VAD (neural) or energy-based detection
3. **Transcribe** — faster-whisper (base/tiny on CPU, <1s latency)
4. **Think** — Hermes API (OpenAI-compatible LLM endpoint)
5. **Speak** — Piper TTS (cori-medium, 22kHz, RTF ~0.14)

## Quick Start

```bash
cd ~/voice-pipeline

# VAD auto-detect mode (recommended)
python3 voice_chat_v2.py

# Push-to-talk
python3 voice_chat_v2.py --mode ptt

# One-shot (5 seconds)
python3 voice_chat_v2.py --mode once -d 5

# Text-only (no TTS)
python3 voice_chat_v2.py --no-speak

# Wake word daemon
python3 voice_chat_v2.py --wake

# List audio devices
python3 voice_chat_v2.py --list-devices
```

## Voices

| Name | Description | Quality |
|------|-------------|---------|
| `cori` | British RP female (default) | ⭐⭐⭐⭐ |
| `cori-high` | British RP high quality | ⭐⭐⭐⭐⭐ |
| `alba` | Scottish female | ⭐⭐⭐⭐ |
| `jenny` | UK English with Irish warmth | ⭐⭐⭐⭐ |
| `aru` | Official Piper UK female | ⭐⭐⭐⭐ |
| `alan` | UK male | ⭐⭐⭐⭐ |
| `vctk` | UK multi-speaker | ⭐⭐⭐ |
| `ljspeech` | US female (public domain) | ⭐⭐⭐ |

## Architecture

```
Mic → Silero VAD → faster-whisper → Hermes API → Piper TTS → Speaker
         ↓               ↓                ↓               ↓
    (or energy VAD)  (base/tiny)    (kimi-k2.6)    (cori-medium)
```

## v2 Improvements (2026-06-07)

- **Silero VAD** — Neural voice activity detection, 10x more accurate than energy thresholds
- **API retry** — 3 attempts with exponential backoff on LLM failures
- **Auto-detect audio** — No more hardcoded device indices
- **Subprocess cleanup** — No orphan Piper/aplay processes on crash
- **Environment config** — All settings via `.env` or env vars (no hardcoded secrets)
- **Conversation limits** — Capped by message count (20) and character budget (4000)
- **Structured logging** — All events to `~/.hermes/logs/voice_chat.log`
- **Wake word mode** — Built-in `--wake` for "Hey Runa" continuous listening
- **8 voice options** — Unified voice map across all scripts

## Configuration

All settings configurable via environment variables or `~/.hermes/.env`:

```bash
# In ~/.hermes/.env or as environment variables
API_SERVER_KEY=your-api-key-here
HERMES_API_URL=http://localhost:8642/v1/chat/completions
HERMES_API_MODEL=kimi-k2.6
MIC_DEVICE_INDEX=-1          # -1 = auto-detect
STT_MODEL_SIZE=base           # tiny/base/small
VAD_MODE=silero               # silero or energy
PIPER_VOICE=cori              # default TTS voice
```

## Dependencies

```
faster-whisper==1.2.1
silero-vad==6.2.1
torch==2.11.0+cpu
numpy==1.26.4
PyAudio==0.2.13
```

Plus Piper TTS binary at `~/piper/piper/piper` with voice models in `~/piper/voices/`.

## Legacy Scripts (Pre-v2)

- `voice_llm.py` — Original PyAudio LLM pipeline (hardcoded API key, energy VAD)
- `voice_chat.py` — Original sounddevice echo mode (no LLM)
- `wake_runa.py` — Wake word daemon (PyAudio + whisper base)
- `voice_llm_arecord.py` — arecord-based alternative (hardcoded hw:3,0)
- `wake_runa_oww.py` — Experimental openWakeWord version (uses "alexa")

These are kept for reference but superseded by `voice_chat_v2.py`.

## License

MIT

## Authors

Runa Gridweaver Freyjasdottir — Voice Pipeline v2 (Mythic Engineering session, 2026-06-07)