"""
Milestone 1 — transcribe ONE audio file with a local Whisper model.

Usage:
    python3 src/transcribe_file.py data/audio/clip_01.mp3
    python3 src/transcribe_file.py data/audio/clip_01.mp3 --model base

Key ideas to understand:
  1. A "model" is a file of learned numbers (weights). Whisper ships several
     sizes: tiny, base, small, ... Bigger = more accurate but slower.
  2. We load the model ONCE, then ask it to transcribe audio.
  3. Everything runs locally on your CPU — no API key, no internet needed
     at inference time.
"""

import argparse          # standard library: parses command-line arguments
import time              # standard library: we use it to measure processing time

import whisper           # OpenAI's Whisper speech-to-text library (runs on PyTorch)


def transcribe(audio_path: str, model_size: str = "tiny") -> dict:
    """Transcribe one audio file and return the result with timing info."""

    # --- Step 1: load the model -------------------------------------------
    # whisper.load_model() reads the weights file (e.g. tiny.pt, ~75 MB)
    # into memory and builds the neural network.
    # device="cpu" forces it to run on the processor, not a GPU.
    model = whisper.load_model(model_size, device="cpu")

    # --- Step 2: transcribe ------------------------------------------------
    # We start a timer right before inference and stop it right after.
    # perf_counter() is the most precise wall-clock timer in Python.
    start = time.perf_counter()
    result = model.transcribe(audio_path, fp16=False)  # fp16=False: CPUs don't support half precision well
    elapsed = time.perf_counter() - start

    # --- Step 3: return text + timing --------------------------------------
    return {
        "text": result["text"].strip(),
        "processing_time_s": elapsed,
    }


def main():
    # argparse lets the user pass the audio path (and optionally a model size)
    # on the command line, e.g.  python3 src/transcribe_file.py clip.mp3 --model base
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Whisper.")
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, ...)")
    parser.add_argument("--model", default="tiny", help="Whisper model size: tiny, base, small, ...")
    args = parser.parse_args()

    out = transcribe(args.audio, args.model)

    print("Transcript :", out["text"])
    print(f"Time taken : {out['processing_time_s']:.2f} seconds")


if __name__ == "__main__":
    # This block only runs when you execute the file directly,
    # not when another script imports `transcribe` from it.
    main()
