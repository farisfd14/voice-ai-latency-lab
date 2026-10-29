"""
Milestone 2 + 3 - benchmark a whole folder of audio clips.

For every audio file we measure:
    - audio_duration_s  : how long the recording is
    - processing_time_s : how long the model took to transcribe it
    - rtf               : Real-time Factor = processing_time / audio_duration
    - wer               : Word Error Rate against the reference transcript

Results are appended to a CSV file, so you can compare runs and models.

Usage:
    python3 src/benchmark.py --model tiny
    python3 src/benchmark.py --model base
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import whisper

# Make "import wer" work when this script is run from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer import wer # our from-scratch Word Error Rate function

def audio_duration_seconds(path: str) -> float:
    """
    Get the duration of an audio file WITHOUT decoding all of it.

    We call ffprobe (part of ffmpeg) which just reads the file header.
    Works for mp3, wav, m4a, ... anything ffmpeg understands.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def load_references(csv_path: str) -> dict:
    """Read data/transcripts.csv into a dict: {filename: reference_text}."""
    refs = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[row["file"]] = row["reference"]
        return refs


def benchmark(audio_dir: str, model_size: str, references: dict) -> list:
    """Transcrive every audio file in a folder and collect measurements."""

    # Load the model ONCE, outside the Loop. Loading takes seconds and
    # would otherwise pollute our per-clip timing measurements.
    print(f"Loading Whisper '{model_size}' model on CPU ...")
    model = whisper.load_model(model_size, device="cpu")

    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    files = sorted(
        p for p in Path(audio_dir).iterdir()
        if p.suffix.lower() in audio_exts
    )
    if not files:
        raise SystemExit(f"No audio files found in {audio_dir}")

    rows = []
    for path in files:
        duration = audio_duration_seconds(str(path))

        start = time.perf_counter()
        result = model.transcribe(str(path), fp16=False)
        elapsed = time.perf_counter() - start

        hypothesis = result["text"].strip()
        reference = references.get(path.name, "")
        error_rate = wer(reference, hypothesis) if reference else None

        rows.append({
            "model": model_size,
            "file": path.name,
            "audio_duration_s": round(duration, 3),
            "processing_time_s": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3),
            "wer": round(error_rate, 3) if error_rate is not None else "",
            "hypothesis": hypothesis,
        })
        print(f" {path.name:14s} dur={duration:5.2f}s time={elapsed:5.2f}s "
              f"RTF={elapsed / duration:.3f} WER={error_rate if error_rate is not None else 'n/a'}")

    return rows

def save_csv(rows: list, out_path: str) -> None:
    """Append rows to the results CSV (creating it with a header if needed)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper on a folder of audio clips."
    )
    parser.add_argument("--model", default="tiny",
                        help="Whisper model size (tiny, base, small, ...)")
    parser.add_argument("--audior-dir", default="data/audio",
                        help="Folder with audio clips")
    parser.add_argument("--references", default="data/transcripts.csv",
                        help="CSV with reference transcripts")
    parser.add_argument("--out", default="results/benchmark_results.csv",
                        help="Output CSV file")
    args = parser.parse_args()
    
    references = load_references(args.references)
    rows = benchmark(args.audio_dir, args.model, references)
    save_csv(rows, args.out)

if __name__ == "__main__":
    main()