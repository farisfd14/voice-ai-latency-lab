"""
Milestone 3 — turn the results CSV into charts.

Chart 1 (speed):    processing time per clip, grouped by model.
Chart 2 (accuracy): Word Error Rate per clip, grouped by model.
Chart 3 (trade-off): average RTF vs. average WER per model — the classic
                    speed-vs-accuracy trade-off picture.

Usage:
    python3 src/make_charts.py
    python3 src/make_charts.py --csv results/benchmark_results.csv --out-dir charts
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")          # render to files, no display needed (works on servers)
import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Average any repeated runs of the same model+file, so charts stay clean.
    df = df.groupby(["model", "file"], as_index=False).agg(
        {
            "audio_duration_s": "mean",
            "processing_time_s": "mean",
            "rtf": "mean",
            "wer": "mean",
        }
    )
    return df


def chart_speed(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: how long did each model take on each clip?"""
    pivot = df.pivot(index="file", columns="model", values="processing_time_s")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Processing time per clip (lower = faster)")
    ax.set_ylabel("seconds")
    ax.set_xlabel("audio clip")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "chart_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def chart_accuracy(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: Word Error Rate per clip, per model (lower = better)."""
    pivot = df.pivot(index="file", columns="model", values="wer")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Word Error Rate per clip (lower = more accurate)")
    ax.set_ylabel("WER")
    ax.set_xlabel("audio clip")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "chart_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def chart_tradeoff(df: pd.DataFrame, out_dir: str) -> None:
    """
    Scatter: average speed (RTF) vs. average accuracy (WER), one point per model.
    The interesting engineering story lives here: bigger models are usually
    slower (higher RTF) but more accurate (lower WER).
    """
    summary = df.groupby("model").agg(avg_rtf=("rtf", "mean"), avg_wer=("wer", "mean"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary["avg_rtf"], summary["avg_wer"], s=120)
    for model, row in summary.iterrows():
        ax.annotate(model, (row["avg_rtf"], row["avg_wer"]),
                    textcoords="offset points", xytext=(10, 8), fontsize=11)
    ax.set_title("Speed vs. accuracy trade-off")
    ax.set_xlabel("average Real-Time Factor (lower = faster)")
    ax.set_ylabel("average Word Error Rate (lower = more accurate)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "chart_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Make charts from benchmark results.")
    parser.add_argument("--csv", default="results/benchmark_results.csv")
    parser.add_argument("--out-dir", default="charts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_results(args.csv)
    chart_speed(df, args.out_dir)
    chart_accuracy(df, args.out_dir)
    chart_tradeoff(df, args.out_dir)


if __name__ == "__main__":
    main()
