"""Tests for the sharp (one-hot) text / large-alphabet path.

These prove that large-alphabet search is now as discriminating as the
DNA path: an exact match scores 1.0 and a *total* non-match scores 0.0,
regardless of alphabet size.
"""

from __future__ import annotations

import numpy as np
import pytest

from selly_fft import (
    TextAssociativeMemory,
    text_match,
    holographic_match,
    encode_orthogonal,
    normalized_xcorr_multichannel,
)
from selly_fft.text import TEXT_ALPHABET


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------
def test_onehot_symbols_mutually_orthogonal():
    ab = "ABCDEFGHIJ"
    codes = encode_orthogonal(list("ABCDE"), ab)
    assert codes.shape == (5, len(ab))
    # distinct symbols are exactly orthogonal
    assert float(np.dot(codes[0], codes[1])) == 0.0
    # a symbol is self-correlated to 1
    assert float(np.dot(codes[0], codes[0])) == 1.0


def test_onehot_unit_norm():
    codes = encode_orthogonal(list("ABCXYZ"), TEXT_ALPHABET)
    assert np.allclose(np.linalg.norm(codes, axis=1), 1.0)


def test_onehot_symbol_not_in_alphabet_raises():
    with pytest.raises(ValueError):
        encode_orthogonal(list("AB!"), "AB")  # '!' not in alphabet


def test_multichannel_correlation_equals_match_fraction():
    ref = "THE BROWN CAT"
    q = "BROWN"
    scores = normalized_xcorr_multichannel(
        encode_orthogonal(list(q), TEXT_ALPHABET),
        encode_orthogonal(list(ref), TEXT_ALPHABET),
    )
    # one exact occurrence at index 4 -> 1.0; everywhere else 0.0
    assert float(scores[4]) == pytest.approx(1.0)
    assert float(np.max(np.delete(scores, 4))) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TextAssociativeMemory — sharpness
# ---------------------------------------------------------------------------
def test_total_nonmatch_scores_zero():
    """The original bug: AAAAAAAAAA vs BBBBB scored 0.9848 under the
    unit-circle encoding.  The sharp path must score 0.0."""
    mem = TextAssociativeMemory(case_sensitive=False)
    assert mem.find_matches("BBBBB", "AAAAAAAAAA") == []


def test_exact_text_match_is_one():
    mem = TextAssociativeMemory(case_sensitive=False)
    matches = mem.find_matches("BROWN", "THE QUICK BROWN FOX")
    assert [(m.position, round(m.score, 4)) for m in matches] == [(10, 1.0)]


def test_lowercase_and_punctuation_work():
    mem = TextAssociativeMemory(case_sensitive=False)
    matches = mem.find_matches("world", "Hello, World!")
    assert matches and matches[0].position == 7 and round(matches[0].score, 3) == 1.0


def test_case_insensitive_by_default():
    mem = TextAssociativeMemory(case_sensitive=False)
    assert mem.find_matches("World", "hello world")  # default folds case


def test_case_sensitive_distinguishes():
    mem = TextAssociativeMemory(case_sensitive=True)
    # 'World' (mixed) vs 'hello world' (lower): only 'orld' aligns after the
    # W/w mismatch, so it scores < 1.0 (≈0.8), not a full match.
    matches = mem.find_matches("World", "hello world")
    assert matches and round(matches[0].score, 4) < 1.0
    # same probe in a case-matching target is exact
    assert mem.find_matches("world", "hello world")


def test_partial_text_match_fraction():
    mem = TextAssociativeMemory(case_sensitive=False)
    # 4 of 5 chars match ('BROWX' vs 'BROWN')
    matches = mem.find_matches("BROWX", "THE BROWN FOX")
    assert matches and round(matches[0].score, 4) == pytest.approx(0.8)


def test_find_matches_returns_positions_in_order():
    mem = TextAssociativeMemory(case_sensitive=False)
    matches = mem.find_matches("AB", "AB AB AB")
    positions = [m.position for m in matches]
    assert positions == [0, 3, 6]


def test_unicode_via_build_alphabet():
    mem = TextAssociativeMemory(case_sensitive=False).build_alphabet("UN CAFÉ NAÏVE 日本語")
    matches = mem.find_matches("café", "UN CAFÉ NAÏVE")
    assert matches and matches[0].position == 3 and round(matches[0].score, 3) == 1.0


def test_whitespace_included_in_default_alphabet():
    assert " " in TEXT_ALPHABET


# ---------------------------------------------------------------------------
# holographic_match routing
# ---------------------------------------------------------------------------
def test_holographic_match_text_is_sharp():
    assert holographic_match("BROWN", "THEQUICKBROWNFOX") == pytest.approx(1.0)
    assert holographic_match("ZZZZZ", "THEQUICKBROWNFOX") == pytest.approx(0.0)
    assert holographic_match("BBBBB", "AAAAAAAAAA") == pytest.approx(0.0)
    assert holographic_match("World", "Hello, World!") == pytest.approx(1.0)


def test_holographic_match_small_alphabet_still_dna_exact():
    # 4-symbol alphabet routes to unit-circle (exact for 90° separation)
    assert holographic_match("ACGT", "ACGTACGT", alphabet="ACGT") == pytest.approx(1.0)
    assert holographic_match("GGGG", "AAAAAAAA", alphabet="AGCT") == pytest.approx(0.0)


def test_holographic_match_empty_returns_zero():
    assert holographic_match("abc", "") == 0.0
    assert holographic_match("", "abc") == 0.0


def test_text_match_functional_shortcut():
    matches = text_match("fox", "the quick brown fox jumps")
    assert matches and matches[0].position == 16 and round(matches[0].score, 3) == 1.0
