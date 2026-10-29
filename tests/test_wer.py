"""
Milestone 4 — automated tests for the parts we can test without a GPU/model.

The WER function is pure Python, so it's cheap to test. Model inference is
slow, so we don't test it here — a common pattern is to test your *logic*
hard and keep *model calls* behind a thin wrapper.

Run with:  python3 -m pytest tests/ -v
"""

import os
import sys

# Let the test file find src/wer.py no matter where pytest is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wer import wer, normalize


def test_perfect_transcription():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_one_substitution():
    # 1 wrong word out of 4 -> WER 0.25
    assert wer("the cat sat down", "the dog sat down") == 0.25


def test_one_deletion():
    # model missed a word: 1 deletion out of 4 reference words
    assert wer("the cat sat down", "the cat down") == 0.25


def test_one_insertion():
    # model hallucinated a word: 1 insertion out of 3 reference words
    assert abs(wer("the cat sat", "the big cat sat") - 1 / 3) < 1e-9


def test_empty_hypothesis():
    # model said nothing at all -> every reference word was deleted
    assert wer("hello world", "") == 1.0


def test_normalization_ignores_case_and_punctuation():
    assert normalize("Hello, WORLD!") == ["hello", "world"]
    assert wer("Hello, world!", "hello world") == 0.0
