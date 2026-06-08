#!/usr/bin/env python3
"""
Runa's Voice Chat Pipeline — Raspberry Pi 5
============================================
Record mic → faster-whisper STT → (optional LLM) → Piper TTS

Modes:
  push-to-talk   Press Enter to start/stop recording
  vad            Energy-based voice activity detection (auto)
  once           Record N seconds, transcribe, speak, exit

Usage:
  python voice_chat.py                     # push-to-talk (default)
  python voice_chat.py --mode vad          # auto VAD
  python voice_chat.py --mode once -d 5    # 5-second recording
  python voice_chat.py --no-tts            # transcription only, no speech
  python voice_chat.py --voice alba        # use alba-medium voice
"""

import argparse
import io
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ── Configuration ──────────────────────────────────────────────────
MIC_SAMPLE_RATE = 48000  # onn USB mic native rate
MIC_CHANNELS = 1
MIC_DEVICE = 2  # sounddevice index for "onn USB Microphone"
MIC_DEVICE_ALSA = "hw:3,0"  # ALSA device for arecord fallback

MODEL_SIZE = "tiny"  # tiny/base/small for faster-whisper
MODEL_DEVICE = "cpu"
MODEL_COMPUTE = "int8"

PIPER_VOICE_DIR = Path.home() / "piper/voices"
PIPER_DEFAULT_VOICE = "en_GB-cori-medium.onnx"
PIPER_SAMPLE_RATE = 22050

# Energy-based VAD
VAD_ENERGY_THRESHOLD = 0.003   # RMS threshold for speech
VAD_SILENCE_SEC = 1.5           # seconds of silence before auto-stop
VAD_CHUNK_SEC = 0.1             # audio chunk size for VAD
VAD_BUFFER_SEC = 0.3            # pre-speech buffer

# ── Piper TTS ───────────────────────────────────────────────────────

def find_piper_voice(voice_name: str = None) -> Path:
    """Resolve voice name to .onnx path."""
    if voice_name and os.path.isfile(voice_name):
        return Path(voice_name)

    voice_map = {
        "cori": "en_GB-cori-medium.onnx",
        "cori-high": "en_GB-cori-high.onnx",
        "alba": "en_GB-alba-medium.onnx",
        "jenny": "en_GB-jenny_dioco-medium.onnx",
        "aru": "en_GB-aru-medium.onnx",
    }
    filename = voice_map.get(voice_name, PIPER_DEFAULT_VOICE)
    
    # Search known locations
    search_paths = [
        PIPER_VOICE_DIR / filename,
        Path.home() / ".local/share/piper/voices" / filename,
    ]
    for p in search_paths:
        if p.exists():
            return p
    
    return PIPER_VOICE_DIR / filename  # fallback, let piper error


def speak(text: str, voice_name: Optional[str] = None, blocking: bool = True):
    """Synthesize text and play through speaker via PipeWire."""
    if not text.strip():
        return
    
    voice_path = find_piper_voice(voice_name or "cori")
    
    print(f"  🔊 Runa says: \"{text}\"")
    
    try:
        # Piper → raw PCM → aplay
        cmd_piper = [
            "piper",
            "--model", str(voice_path),
            "--output_file", "-",
        ]
        cmd_play = [
            "aplay",
            "-r", str(PIPER_SAMPLE_RATE),
            "-f", "S16_LE",
            "-c", "1",
            "-t", "raw",
            "-",
        ]
        
        piper_proc = subprocess.Popen(
            cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        play_proc = subprocess.Popen(
            cmd_play, stdin=piper_proc.stdout, stderr=subprocess.DEVNULL
        )
        
        if piper_proc.stdin:
            piper_proc.stdin.write(text.encode())
            piper_proc.stdin.close()
        
        if blocking:
            play_proc.wait()
        
    except Exception as e:
        print(f"  ⚠️ TTS failed: {e}")


# ── Transcription ───────────────────────────────────────────────────

_model_cache = {}

def get_model():
    """Lazy-load faster-whisper model."""
    if MODEL_SIZE not in _model_cache:
        print(f"  ⏳ Loading faster-whisper {MODEL_SIZE}...", end=" ", flush=True)
        _model_cache[MODEL_SIZE] = WhisperModel(
            MODEL_SIZE, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE
        )
        print("ready.")
    return _model_cache[MODEL_SIZE]


def transcribe(audio_data: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> str:
    """Transcribe audio with faster-whisper."""
    model = get_model()
    
    # Convert to float32, mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # faster-whisper resamples internally, we just pass the array
    start = time.time()
    segments, info = model.transcribe(audio_float, language="en", beam_size=1)
    text = ""
    for seg in segments:
        text += seg.text
    elapsed = time.time() - start
    
    text = text.strip()
    if text:
        print(f"  🎤 {text}")
    print(f"  ⏱️  Transcribed in {elapsed*1000:.0f}ms (lang_prob={info.language_probability:.2f})")
    return text


# ── Recording ───────────────────────────────────────────────────────

def record_via_sounddevice(duration: float) -> np.ndarray:
    """Record audio via sounddevice."""
    print(f"  🔴 Recording ({duration}s)... ", end="", flush=True)
    audio = sd.rec(
        int(duration * MIC_SAMPLE_RATE),
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        device=MIC_DEVICE,
    )
    sd.wait()
    print("done.")
    return audio.flatten()


def record_via_arecord(duration: float) -> np.ndarray:
    """Fallback recording via arecord."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmpfile = f.name
    
    subprocess.run([
        "arecord", "-D", MIC_DEVICE_ALSA,
        "-f", "S16_LE", "-r", str(MIC_SAMPLE_RATE),
        "-c", "1", "-d", str(int(duration)),
        tmpfile
    ], check=True, stderr=subprocess.DEVNULL)
    
    import wave
    with wave.open(tmpfile, "rb") as wf:
        data = wf.readframes(wf.getnframes())
    audio = np.frombuffer(data, dtype=np.int16)
    os.unlink(tmpfile)
    return audio


def record_audio(duration: float) -> np.ndarray:
    """Record audio, preferring sounddevice, falling back to arecord."""
    try:
        return record_via_sounddevice(duration)
    except Exception as e:
        print(f"  ⚠️ sounddevice failed ({e}), using arecord fallback")
        return record_via_arecord(duration)


# ── VAD Mode ────────────────────────────────────────────────────────

def record_with_vad() -> Optional[np.ndarray]:
    """
    Record audio using energy-based VAD.
    Starts recording when speech energy exceeds threshold,
    stops after VAD_SILENCE_SEC of silence.
    Returns audio array or None if nothing detected.
    """
    print("  👂 Listening... (Ctrl+C to stop)", flush=True)
    
    chunk_samples = int(VAD_CHUNK_SEC * MIC_SAMPLE_RATE)
    pre_buffer = deque(maxlen=int(VAD_BUFFER_SEC / VAD_CHUNK_SEC))
    
    recording = False
    audio_chunks = []
    silent_chunks = 0
    max_silent = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
    
    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        device=MIC_DEVICE,
        blocksize=chunk_samples,
    )
    
    try:
        with stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) / 32768.0
                
                if not recording:
                    pre_buffer.append(chunk)
                    if rms > VAD_ENERGY_THRESHOLD:
                        recording = True
                        # Include pre-buffer for context
                        audio_chunks = list(pre_buffer)
                        silent_chunks = 0
                        print("  🟢 Speech detected...", end="", flush=True)
                else:
                    audio_chunks.append(chunk)
                    if rms < VAD_ENERGY_THRESHOLD:
                        silent_chunks += 1
                        if silent_chunks >= max_silent:
                            print(" done.")
                            break
                    else:
                        silent_chunks = 0
                        
    except KeyboardInterrupt:
        print()
        return None
    
    if not recording or len(audio_chunks) == 0:
        return None
    
    audio = np.concatenate(audio_chunks).flatten()
    duration = len(audio) / MIC_SAMPLE_RATE
    print(f"  📼 Captured {duration:.1f}s of audio")
    return audio


# ── Push-to-Talk ────────────────────────────────────────────────────

def record_push_to_talk() -> Optional[np.ndarray]:
    """Press Enter to start recording, press Enter to stop."""
    print("  ⏎  Press ENTER to start recording...", flush=True)
    input()
    
    print("  🔴 Recording — press ENTER to stop...", flush=True)
    
    chunks = []
    stop_event = threading.Event()
    
    def wait_for_stop():
        input()
        stop_event.set()
    
    waiter = threading.Thread(target=wait_for_stop, daemon=True)
    waiter.start()
    
    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        device=MIC_DEVICE,
        blocksize=int(0.1 * MIC_SAMPLE_RATE),
    )
    
    with stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(int(0.1 * MIC_SAMPLE_RATE))
            chunks.append(chunk)
    
    if not chunks:
        return None
    
    audio = np.concatenate(chunks).flatten()
    duration = len(audio) / MIC_SAMPLE_RATE
    print(f"  📼 Captured {duration:.1f}s")
    return audio


# ── Main Loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Runa Voice Chat Pipeline")
    parser.add_argument("--mode", choices=["ptt", "vad", "once"], default="ptt",
                        help="Recording mode (default: ptt = push-to-talk)")
    parser.add_argument("-d", "--duration", type=float, default=5.0,
                        help="Recording duration in seconds for 'once' mode")
    parser.add_argument("--no-tts", action="store_true",
                        help="Skip TTS output (transcription only)")
    parser.add_argument("--voice", choices=["cori", "cori-high", "alba", "jenny", "aru"],
                        default="cori", help="Piper TTS voice")
    parser.add_argument("--no-echo", action="store_true",
                        help="Don't play back transcription as audio")
    args = parser.parse_args()
    
    print("╔══════════════════════════════════╗")
    print("║  Runa Voice Pipeline v1.0       ║")
    print(f"║  Mode: {args.mode:<22s}  ║")
    print(f"║  Voice: {args.voice:<21s}  ║")
    print(f"║  TTS: {'on' if not args.no_tts else 'off':<23s}  ║")
    print("╚══════════════════════════════════╝")
    print()
    
    handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    
    while True:
        try:
            # ── Capture ──
            if args.mode == "ptt":
                audio = record_push_to_talk()
            elif args.mode == "vad":
                audio = record_with_vad()
            elif args.mode == "once":
                audio = record_audio(args.duration)
            
            if audio is None or len(audio) < MIC_SAMPLE_RATE * 0.5:
                print("  ⚠️ Too short, skipping\n")
                if args.mode == "once":
                    break
                continue
            
            # ── Transcribe ──
            text = transcribe(audio)
            
            if not text:
                print("  🤫 (no speech detected)\n")
                if args.mode == "once":
                    break
                continue
            
            # ── Speak response ──
            if not args.no_tts and not args.no_echo:
                speak(text, voice_name=args.voice)
            
            print()
            
            if args.mode == "once":
                break
                
        except KeyboardInterrupt:
            print("\n  👋 Farewell!")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            if args.mode == "once":
                break


if __name__ == "__main__":
    main()
