# Voice AI Latency Lab

A small tool that measures the **speed** and **accuracy** of a local
speech-to-text model — not just "can it transcribe?", but "how fast, and how
well?"

Built with Python, OpenAI Whisper, pandas, and matplotlib. Runs entirely on
CPU — no GPU, no API keys, no paid services.

---

## What it measures

| Metric | Meaning |
|---|---|
| `processing_time_s` | Wall-clock time to transcribe one clip |
| `audio_duration_s` | Length of the recording |
| `rtf` | **Real-Time Factor** = processing time ÷ audio duration. RTF < 1 means faster than real time |
| `wer` | **Word Error Rate** = (substitutions + deletions + insertions) ÷ reference words. 0.0 = perfect |

## Project structure

```
voice-ai-latency-lab/
├── src/
│   ├── transcribe_file.py   # Milestone 1: transcribe one file
│   ├── benchmark.py         # Milestone 2: benchmark a folder -> CSV
│   ├── wer.py               # Milestone 3: Word Error Rate (from scratch)
│   └── make_charts.py       # Milestone 3: speed / accuracy / trade-off charts
├── data/
│   ├── audio/               # 8 short test clips
│   └── transcripts.csv      # reference (correct) transcripts
├── results/
│   └── benchmark_results.csv
├── charts/                  # generated PNG charts
├── tests/
│   └── test_wer.py          # Milestone 4: unit tests
└── requirements.txt
```

## Installation

```bash
# 1. System dependency: ffmpeg (for reading audio files)
sudo apt install ffmpeg        # Debian/Ubuntu
brew install ffmpeg            # macOS

# 2. Python dependencies
pip install -r requirements.txt
```

The first time you use a Whisper model, it is downloaded automatically
(`tiny` ≈ 75 MB, `base` ≈ 145 MB) and cached in `~/.cache/whisper/`.

## Usage

### Milestone 1 — transcribe one file

```bash
python3 src/transcribe_file.py data/audio/clip_01.mp3
# Transcript : The quick brown fox jumps over the lazy dog near the riverbank.
# Time taken : 2.07 seconds
```

### Milestone 2 — benchmark a folder

```bash
python3 src/benchmark.py --model tiny
python3 src/benchmark.py --model base     # compare another model
```

Appends one row per clip to `results/benchmark_results.csv`.

### Milestone 3 — charts

```bash
python3 src/make_charts.py
```

Creates `charts/chart_speed.png`, `charts/chart_accuracy.png`, and
`charts/chart_tradeoff.png`.

### Milestone 4 — tests

```bash
python3 -m pytest tests/ -v
```

## Example results (AMD Ryzen 7 8745HS w/ Radeon 780M Graphics (3.80 GHz), 8 synthetic English clips, ~4–7 s each)

| Model | Avg processing time | Avg RTF | Avg WER |
|---|---------------------|------|---|
| tiny (39 M params) | 1.27 s              | 0.25 | 0.028 |
| base (74 M params) | 2.44 s              | 0.49 | 0.028 |

## What I learned

1. **RTF is the metric that matters for latency.** Both models transcribe
   faster than real time on a plain CPU (RTF < 1), but `base` is exactly 2×
   slower than `tiny`. If your app needs live captions, that 2× decides
   whether you keep up with the speaker.

2. **On clean, easy audio, the bigger model bought us nothing.** WER is
   identical (0.028) for `tiny` and `base` here — both only stumbled on
   "river bank" → "riverbank" and "nine" → "9". Bigger models pay off on
   noisy, accented, or overlapping speech, which these clips don't have.
   That's a real engineering insight: *benchmark on audio that matches your
   real use case.*

3. **Some "errors" are formatting choices, not listening mistakes.** The
   model wrote "9" while the reference says "nine". A stricter text
   normalizer (e.g. expanding numbers to words) would lower measured WER
   without the model changing at all — so WER always depends on your
   normalization rules.

4. **Load the model once, outside the timing loop.** Model loading takes
   seconds; if you loaded it per clip, your "processing time" would measure
   disk I/O, not inference speed.

5. **Measure duration without decoding.** `ffprobe` reads the duration from
   the file header in milliseconds instead of decoding the whole clip.

## Next steps

- Record 5–10 clips of *my own voice* (different accents, background noise)
  and re-run the benchmark — expect WER to rise and `base` to pull ahead.
- Add `small` model to the comparison.
- Optional: a simple Streamlit UI (`streamlit run app.py`) wrapping
  `benchmark.py`.
- GitHub Actions: run `pytest` on every push.
