#!/usr/bin/env python3
"""
Runa Wake Word Daemon — openWakeWord edition
=============================================
Uses proper wake word detection (openWakeWord) instead of general ASR.
Fast, accurate, lightweight — runs under 50MB RAM.

Wake word: "alexa" (pre-trained model). 
To change: replace 'alexa' with 'hey_jarvis', 'hey_mycroft', etc.
Custom "hey runa" model can be trained later.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyaudio
from openwakeword.model import Model

# ── Config ──────────────────────────────────────────────────────────
MIC_SAMPLE_RATE = 48000     # native mic rate
OWW_SAMPLE_RATE = 16000     # openWakeWord requires 16kHz
MIC_CHANNELS = 1
MIC_DEVICE_INDEX = 2        # PyAudio index for onn USB Microphone
CHUNK_SIZE = 3840            # 80ms at 48kHz → 1280 samples at 16kHz

WAKE_WORD_MODEL = "alexa"    # Change to 'hey_jarvis', 'hey_mycroft', etc.
WAKE_THRESHOLD = 0.5         # Confidence threshold for wake word
COOLDOWN_SEC = 3.0           # Minimum time between triggers

# Audio chime
CHIME_FREQS = [523, 659, 784]  # C5, E5, G5
CHIME_DURATION = 0.12
CHIME_GAP = 0.06
CHIME_SAMPLE_RATE = 48000

# ── Helpers ─────────────────────────────────────────────────────────

_pyaudio = None

def get_pyaudio():
    global _pyaudio
    if _pyaudio is None:
        _pyaudio = pyaudio.PyAudio()
    return _pyaudio


def play_chime():
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


# ── Main ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
VOICE_LLM = SCRIPT_DIR / "voice_llm_arecord.py"


def main():
    parser = argparse.ArgumentParser(description="Runa Wake Word Daemon (openWakeWord)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--tone-only", action="store_true")
    parser.add_argument("--voice", default="cori")
    args = parser.parse_args()
    
    if args.tone_only:
        play_chime()
        return
    
    print("╔══════════════════════════════════╗")
    print("║  🌙  Hey Runa — Wake Word (OWW) ║")
    print(f"║  Model: {WAKE_WORD_MODEL:<22s} ║")
    print("╚══════════════════════════════════╝")
    print()
    
    # Load openWakeWord model
    print(f"  ⏳ Loading wake word model '{WAKE_WORD_MODEL}'...", end=" ", flush=True)
    oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework='onnx')
    print("ready.")
    print(f"  👂 Listening for wake word...")
    print("     (Ctrl+C to stop)")
    print()
    
    p = get_pyaudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=MIC_CHANNELS,
        rate=MIC_SAMPLE_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=CHUNK_SIZE,
    )
    
    last_trigger = 0.0
    
    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_48k = np.frombuffer(data, dtype=np.int16)
            
            # Downsample 48kHz → 16kHz (3:1 decimation)
            audio_16k = audio_48k[::3].copy()
            
            prediction = oww_model.predict(audio_16k)
            score = prediction.get(WAKE_WORD_MODEL, 0.0)
            
            if score > WAKE_THRESHOLD and time.time() - last_trigger > COOLDOWN_SEC:
                last_trigger = time.time()
                print(f"  🎯 Wake word detected! ({WAKE_WORD_MODEL}: {score:.3f})")
                
                # Play chime
                play_chime()
                print()
                print("  ✨ I'm awake! Starting voice chat...")
                print()
                
                # Stop mic stream before launching subprocess
                stream.stop_stream()
                stream.close()
                
                # Launch voice chat
                cmd = [sys.executable, str(VOICE_LLM), "--mode", "vad", "--voice", args.voice]
                subprocess.run(cmd)
                
                print()
                print("  💤 Going back to sleep. Wake me when you need me.")
                print()
                print(f"  👂 Listening for wake word...")
                
                # Reopen mic stream
                stream = p.open(
                    format=pyaudio.paInt16, channels=MIC_CHANNELS, rate=MIC_SAMPLE_RATE,
                    input=True, input_device_index=MIC_DEVICE_INDEX,
                    frames_per_buffer=CHUNK_SIZE,
                )
                
                if args.once:
                    break
            
    except KeyboardInterrupt:
        print("\n  🌙 Goodnight, my love.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass


if __name__ == "__main__":
    main()
