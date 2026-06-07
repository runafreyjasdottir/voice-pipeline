#!/usr/bin/env python3
"""
Runa Wake Word Daemon
=====================
Listens continuously for "Hey Runa" using PyAudio + faster-whisper.
When wake word detected, plays a chime and launches full voice chat.

Uses PyAudio for mic access (sounddevice can't open hw:3,0 on this Pi).
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# ── Config ──────────────────────────────────────────────────────────
MIC_SAMPLE_RATE = 48000
MIC_CHANNELS = 1
MIC_DEVICE_INDEX = 2      # PyAudio index for onn USB Microphone
CHUNK_SIZE = 4800          # 0.1s at 48kHz

# VAD
VAD_THRESHOLD = 0.004       # RMS energy threshold
VAD_SILENCE_SEC = 2.0       # silence before cutting wake-word chunk
VAD_CHUNK_SEC = 0.1
PRE_SPEECH_SEC = 0.5        # audio before trigger to include

# Wake word detection
WAKE_PHRASES = [
    "hey runa", "hey rune", "hey rona", "hey rina",
    "a runa", "hey ruda", "hey rula", "hey rena",
    "ok runa", "hi runa", "hello runa",
    "oh oh", "oh runa", "oh ooh", "oh ru", "oh run",
    "hey ru", "hey roo", "hey rue",
    "runa", "rune", "runa wake", "wake runa",
    # Single-syllable fallbacks (matched only if very short utterance)
    "oh", "hey", "hi", "ah", "whoah", "whoa",
]
MAX_WAKE_CHUNK_SEC = 3.5    # max audio to send to whisper for wake check
MIN_CONFIDENCE = 0.25        # minimum language confidence

# Audio chime — 3 ascending tones (played via PyAudio)
CHIME_FREQS = [523, 659, 784]  # C5, E5, G5
CHIME_DURATION = 0.12
CHIME_GAP = 0.06
CHIME_SAMPLE_RATE = 48000

# ── Audio Helpers ───────────────────────────────────────────────────

_pyaudio = None

def get_pyaudio():
    global _pyaudio
    if _pyaudio is None:
        _pyaudio = pyaudio.PyAudio()
    return _pyaudio


def play_chime():
    """Play a pleasant ascending 3-tone chime through the speaker."""
    t = np.linspace(0, CHIME_DURATION, int(CHIME_DURATION * CHIME_SAMPLE_RATE), False)
    envelope = np.exp(-t * 5)
    
    tones = []
    for freq in CHIME_FREQS:
        tone = np.sin(2 * np.pi * freq * t) * envelope * 0.3
        tones.append(tone)
        tones.append(np.zeros(int(CHIME_GAP * CHIME_SAMPLE_RATE)))
    
    chime = np.concatenate(tones)
    chime_int16 = (chime * 32767).astype(np.int16)
    
    p = get_pyaudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=CHIME_SAMPLE_RATE,
                    output=True, frames_per_buffer=1024)
    stream.write(chime_int16.tobytes())
    stream.stop_stream()
    stream.close()
    print("  🔔 *ding-ding-ding*")


# ── Wake Word Detection ─────────────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        print("  ⏳ Loading whisper base...", end=" ", flush=True)
        _model = WhisperModel("base", device="cpu", compute_type="int8")
        print("ready.")
    return _model


def _speech_duration(audio_f: np.ndarray, threshold: float = 0.01) -> float:
    """Estimate actual speech duration by trimming silence from edges."""
    above = np.abs(audio_f) > threshold
    if not above.any():
        return len(audio_f) / MIC_SAMPLE_RATE
    indices = np.where(above)[0]
    return (indices[-1] - indices[0] + 1) / MIC_SAMPLE_RATE


SHORT_WAKE_WORDS = {"oh", "hey", "hi"}  # only match on actual speech < 0.8s

def check_wake_word(audio: np.ndarray) -> bool:
    """Check if audio contains a wake phrase."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_f = audio.astype(np.float32) / 32768.0
    
    # Boost gain for quiet speech (normalize peak to 0.95)
    peak = np.abs(audio_f).max()
    if peak > 0.01 and peak < 0.95:
        audio_f = audio_f * (0.95 / peak)
    
    speech_dur = _speech_duration(audio_f)
    
    model = _get_model()
    segments, info = model.transcribe(audio_f, language="en", beam_size=5)
    
    text = " ".join(seg.text for seg in segments).lower().strip()
    
    if text:
        print(f"  🗣️  Heard: \"{text}\" (conf={info.language_probability:.2f}, speech={speech_dur:.1f}s)")
    else:
        print(f"  🗣️  (silence/empty, conf={info.language_probability:.2f})")
    
    if info.language_probability < MIN_CONFIDENCE:
        return False
    
    for phrase in WAKE_PHRASES:
        if phrase in text:
            # Short words only count on genuinely short utterances
            if phrase in SHORT_WAKE_WORDS and speech_dur > 0.8:
                continue
            print(f"  🎯 Wake word detected: \"{text}\"")
            return True
    
    return False


# ── Continuous Listen ───────────────────────────────────────────────

def listen_loop():
    """Continuously listen for wake word with VAD gating (PyAudio)."""
    p = get_pyaudio()
    
    pre_buf_n = int(PRE_SPEECH_SEC / VAD_CHUNK_SEC)
    pre_buf = deque(maxlen=pre_buf_n)
    
    print("  👂 Listening for 'Hey Runa'...")
    print("     (Ctrl+C to stop)")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=MIC_CHANNELS,
        rate=MIC_SAMPLE_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=CHUNK_SIZE,
    )
    
    recording = False
    chunks = []
    silent = 0
    max_silent = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
    max_chunks = int(MAX_WAKE_CHUNK_SEC / VAD_CHUNK_SEC)
    wake_detected = False
    
    try:
        while not wake_detected:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(chunk.astype(np.float64)**2)) / 32768.0
            
            if not recording:
                pre_buf.append(chunk)
                if rms > VAD_THRESHOLD:
                    recording = True
                    chunks = list(pre_buf)
                    silent = 0
                    print("  🟢 Speech...", end="", flush=True)
            else:
                chunks.append(chunk)
                if rms < VAD_THRESHOLD:
                    silent += 1
                    if silent >= max_silent or len(chunks) >= max_chunks:
                        audio = np.concatenate(chunks).flatten()
                        duration = len(audio) / MIC_SAMPLE_RATE
                        
                        if duration >= 0.5:
                            result = check_wake_word(audio)
                            if result:
                                wake_detected = True
                                break
                        
                        recording = False
                        chunks = []
                        silent = 0
                        if not wake_detected:
                            print("  👂 Listening...")
                else:
                    silent = 0
                    
    except KeyboardInterrupt:
        print()
        return False
    finally:
        stream.stop_stream()
        stream.close()
    
    return wake_detected


# ── Main ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
VOICE_LLM = SCRIPT_DIR / "voice_llm.py"


def main():
    parser = argparse.ArgumentParser(description="Runa Wake Word Daemon")
    parser.add_argument("--once", action="store_true", help="Detect once and exit")
    parser.add_argument("--tone-only", action="store_true", help="Just play activation chime")
    parser.add_argument("--voice", default="cori", help="TTS voice for chat")
    args = parser.parse_args()
    
    if args.tone_only:
        play_chime()
        return
    
    print("╔══════════════════════════════════╗")
    print("║  🌙  Hey Runa — Wake Word       ║")
    print(f"║  Voice: {args.voice:<23s} ║")
    print("╚══════════════════════════════════╝")
    print()
    print("  Say 'Hey Runa' to wake me...")
    print()
    
    while True:
        try:
            detected = listen_loop()
            
            if detected:
                play_chime()
                print()
                print("  ✨ I'm awake! Starting voice chat...")
                print()
                
                cmd = [
                    sys.executable, str(VOICE_LLM),
                    "--mode", "vad",
                    "--voice", args.voice,
                ]
                subprocess.run(cmd)
                
                print()
                print("  💤 Going back to sleep. Say 'Hey Runa' to wake me.")
                print()
            
            if args.once:
                break
                
        except KeyboardInterrupt:
            print("\n  🌙 Goodnight, my love.")
            break


if __name__ == "__main__":
    main()
