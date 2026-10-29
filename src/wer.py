"""
wer.py — Word Error Rate, implemented from scratch.

Word Error Rate (WER) is the standard way to measure speech recognition accuracy:

    WER = (substitutions + deletions + insertions) / (words in the reference)

To count substitutions/deletions/insertions we need the smallest number of
single-word edits that turns the hypothesis (what the model heard) into the
reference (what was actually said). That number is called the *edit distance*
or *Levenshtein distance*, and the classic way to compute it is a small
dynamic program.

Example:
    reference : "the cat sat on the mat"
    hypothesis: "the cat sat on a mat"
    -> 1 substitution ("a" instead of "the"), so WER = 1 / 6 = 0.167

A WER of 0.0 means a perfect transcript. Lower is better.
"""

import re
from typing import List, Tuple


def normalize(text: str) -> List[str]:
    """
    Turn a sentence into a list of comparable word tokens.

    Normalization matters: "Hello," and "hello" should count as the same word,
    and "riverbank" should be compared to "riverbank", not to "river bank".
    We lowercase, remove punctuation, and split on whitespace.
    """
    text = text.lower()                      # ignore letter case
    text = re.sub(r"[^\w\s]", " ", text)     # replace punctuation with spaces
    return text.split()                      # split into words, dropping extra spaces


def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, Tuple[int, int, int]]:
    """
    Levenshtein edit distance between two word lists, plus the breakdown
    (substitutions, deletions, insertions).

    Dynamic programming idea:
      dp[i][j] = minimum edits to turn ref[:i] into hyp[:j]
      - if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]          (match, free)
      - else: dp[i][j] = 1 + min(
            dp[i-1][j-1],   # substitution
            dp[i-1][j],     # deletion (a reference word was missed)
            dp[i][j-1],     # insertion (an extra word was hallucinated)
        )

    We then walk BACK through the table to count how many edits of each
    kind were used. This is the same algorithm used by the `jiwer` library.
    """
    n, m = len(ref), len(hyp)

    # dp table of size (n+1) x (m+1), all zeros to start
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases: turning i reference words into nothing = i deletions,
    # and turning nothing into j hypothesis words = j insertions.
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    # Fill the table row by row
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]                      # words match
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1],             # substitution
                                   dp[i - 1][j],                 # deletion
                                   dp[i][j - 1])                 # insertion

    # Backtrack from dp[n][m] to dp[0][0] to count edit types
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1                                               # match: move diagonally
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1                                               # substitution
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1                                               # deletion
        else:
            ins += 1
            j -= 1                                               # insertion

    return dp[n][m], (subs, dels, ins)


def wer(reference: str, hypothesis: str) -> float:
    """Compute the Word Error Rate between a reference and a hypothesis string."""
    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)
    if not ref_words:
        # An empty reference is undefined territory; treat it as 0 errors
        # only if the hypothesis is also empty.
        return 0.0 if not hyp_words else 1.0
    distance, _ = edit_distance(ref_words, hyp_words)
    return distance / len(ref_words)


if __name__ == "__main__":
    # A tiny self-check you can run with:  python3 src/wer.py
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumped over a lazy dog"
    print(f"WER({hyp!r} vs. {ref!r}) = {wer(ref, hyp):.3f}")  # expect 2/9 = 0.222
