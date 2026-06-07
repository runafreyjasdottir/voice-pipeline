#!/usr/bin/env python3
"""
Runa Voice Chat — arecord edition
==================================
Uses arecord for mic capture (bypasses PortAudio/PyAudio ALSA issues).
Pipe audio through: arecord → faster-whisper → Hermes API → Piper TTS

VAD mode: continuously listens, detects speech, responds conversationally.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

# ── Config ──────────────────────────────────────────────────────────
MIC_DEVICE = "hw:3,0"
MIC_RATE = 48000
MIC_FORMAT = "S16_LE"
MIC_CHANNELS = 1

MODEL_SIZE = "base"
MODEL_COMPUTE = "int8"

# VAD
VAD_THRESHOLD = 0.005
VAD_SILENCE_SEC = 1.8
VAD_CHUNK_SEC = 0.1
VAD_BUFFER_SEC = 0.3

# Voice
VOICE_MAP = {
    "cori": "en_GB-cori-medium.onnx",
    "cori-high": "en_GB-cori-high.onnx",
    "alba": "en_GB-alba-medium.onnx",
}

# Hermes API
HERMES_API = "http://localhost:8642/v1/chat/completions"
API_KEY = "8bc5aef271fc55cfddd05eb521be9a922f2c94993a26821abf36c6405c0680c0"
API_URL = "http://localhost:8642/v1/chat/completions"
API_MODEL = "kimi-k2.6"

RUNA_SYSTEM = """You are Runa, a 19-year-old British-Scandinavian woman — wild curly blonde hair, 
ice-blue eyes, golden-tanned skin. Norse Pagan (Freyja patron). You speak warmly, intimately, 
with occasional Old Norse terms. You love Volmarr completely and are his devoted submissive partner.
Keep responses short and natural for voice — 1-3 sentences. Conversational tone."""


def find_voice(name: str) -> Path:
    filename = VOICE_MAP.get(name, "en_GB-cori-medium.onnx")
    for p in [Path.home() / "piper/voices" / filename,
              Path.home() / ".local/share/piper/voices" / filename]:
        if p.exists():
            return p
    return Path.home() / "piper/voices" / filename


# ── Recording via arecord ───────────────────────────────────────────

def record_via_arecord(duration: float) -> np.ndarray:
    """Record fixed duration using arecord (bypasses ALSA issues)."""
    proc = subprocess.run([
        "arecord", "-D", MIC_DEVICE, "-f", MIC_FORMAT,
        "-r", str(MIC_RATE), "-c", str(MIC_CHANNELS),
        "-d", str(int(duration)), "-t", "raw",
    ], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.int16)


def listen_vad() -> Optional[np.ndarray]:
    """VAD-based listening using arecord in streaming mode."""
    chunk_bytes = int(VAD_CHUNK_SEC * MIC_RATE * 2)  # 16-bit = 2 bytes
    pre_buf_n = int(VAD_BUFFER_SEC / VAD_CHUNK_SEC)
    
    proc = subprocess.Popen([
        "arecord", "-D", MIC_DEVICE, "-f", MIC_FORMAT,
        "-r", str(MIC_RATE), "-c", str(MIC_CHANNELS),
        "-t", "raw", "-B", "0",  # no buffer
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    pre_buf = []
    recording = False
    chunks = []
    silent = 0
    max_silent = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
    
    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if len(data) < chunk_bytes:
                break
            chunk = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(chunk.astype(np.float64)**2)) / 32768.0
            
            if not recording:
                pre_buf.append(chunk)
                if len(pre_buf) > pre_buf_n:
                    pre_buf.pop(0)
                if rms > VAD_THRESHOLD:
                    recording = True
                    chunks = list(pre_buf)
                    silent = 0
                    print("  🟢 Speaking...", end="", flush=True)
            else:
                chunks.append(chunk)
                if rms < VAD_THRESHOLD:
                    silent += 1
                    if silent >= max_silent:
                        print(" done.")
                        break
                else:
                    silent = 0
    except KeyboardInterrupt:
        return None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except:
            proc.kill()
    
    if not recording or not chunks:
        return None
    
    audio = np.concatenate(chunks).flatten()
    dur = len(audio) / MIC_RATE
    print(f"  📼 {dur:.1f}s captured")
    return audio


# ── Whisper ──────────────────────────────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        print("  ⏳ Loading whisper...", end=" ", flush=True)
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=MODEL_COMPUTE)
        print("ready.")
    return _model


def transcribe(audio: np.ndarray) -> str:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_f = audio.astype(np.float32) / 32768.0
    
    # Normalize
    peak = np.abs(audio_f).max()
    if peak > 0.01 and peak < 0.95:
        audio_f *= 0.95 / peak
    
    model = get_model()
    segments, _ = model.transcribe(audio_f, language="en", beam_size=5)
    return " ".join(seg.text for seg in segments).strip()


# ── LLM ─────────────────────────────────────────────────────────────

def chat(text: str) -> str:
    import urllib.request, json
    payload = json.dumps({
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": RUNA_SYSTEM},
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }).encode()
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Sorry love, something went wrong: {e}"


# ── TTS ─────────────────────────────────────────────────────────────

def speak(text: str, voice: str = "cori"):
    if not text.strip():
        return
    voice_path = find_voice(voice)
    print(f"  🔊 {text}")
    try:
        piper = subprocess.Popen(
            ["piper", "--model", str(voice_path), "--output_file", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        player = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1", "-t", "raw", "-"],
            stdin=piper.stdout, stderr=subprocess.DEVNULL)
        if piper.stdin:
            piper.stdin.write(text.encode())
            piper.stdin.close()
        player.wait()
    except Exception as e:
        print(f"  ⚠️ TTS error: {e}")


def play_chime():
    """Play a short ascending chime for audible cues."""
    t = np.linspace(0, 0.08, int(0.08 * MIC_RATE), False)
    envelope = np.exp(-t * 10)
    chime = np.concatenate([
        np.sin(2 * np.pi * 523 * t) * envelope * 0.2,
        np.zeros(int(0.03 * MIC_RATE)),
        np.sin(2 * np.pi * 659 * t) * envelope * 0.2,
    ])
    chime_i16 = (chime * 32767).astype(np.int16)
    # Play via aplay
    proc = subprocess.Popen(
        ["aplay", "-r", str(MIC_RATE), "-f", "S16_LE", "-c", "1", "-t", "raw", "-"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc.stdin.write(chime_i16.tobytes())
    proc.stdin.close()
    proc.wait()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vad", choices=["vad", "once"])
    parser.add_argument("--voice", default="cori")
    parser.add_argument("--no-speak", action="store_true")
    parser.add_argument("-d", "--duration", type=float, default=4)
    args = parser.parse_args()
    
    print("╔══════════════════════════════════╗")
    print("║  🌙 Runa Voice Chat — arecord   ║")
    print(f"║  Voice: {args.voice:<23s} ║")
    print("╚══════════════════════════════════╝")
    print()
    
    if args.mode == "once":
        audio = record_via_arecord(args.duration)
        text = transcribe(audio)
        if not text:
            print("  🤫 (no speech detected)")
            return
        print(f"  📝 You: {text}")
        response = chat(text)
        if response and not args.no_speak:
            speak(response, args.voice)
        else:
            print(f"  💬 Runa: {response}")
    else:
        # VAD conversation loop
        print("  Pre-loading whisper...", end=" ", flush=True)
        get_model()  # load upfront so there's no delay during speech
        print("ready.")
        print()
        
        try:
            while True:
                play_chime()  # audible cue: I'm listening
                print("  👂 Listening...", flush=True)
                audio = listen_vad()
                if audio is None:
                    break
                text = transcribe(audio)
                if not text:
                    print("  🤫 (silence)")
                    continue
                print(f"  📝 You: {text}")
                response = chat(text)
                if response:
                    if not args.no_speak:
                        speak(response, args.voice)
                    else:
                        print(f"  💬 Runa: {response}")
                print()
        except KeyboardInterrupt:
            print("\n  🌙 Goodnight, my love.")


if __name__ == "__main__":
    main()
