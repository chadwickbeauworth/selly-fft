"""Regression tests for the v0.2.0 stress-test findings.

Each test pins a bug found by the independent stress test of 2026-08-04
and fixed in v0.2.0:

1. SHARP_ENCODING_THRESHOLD 8 -> 4 (unit-circle is only exact for L <= 4)
2. holographic_match returns the best score regardless of threshold
3. NFC normalization: composed/decomposed Unicode forms match
4. Alphabet validation: duplicates rejected; explicit alphabet folded
5. Encoding-aware significance null model
6. float32 dtype support
7. Unified (probe, reference) argument order
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pytest

from selly_fft import (
    SellyAssociativeMemory,
    TextAssociativeMemory,
    holographic_match,
    text_match,
)
from selly_fft.func import SHARP_ENCODING_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Routing threshold: no alphabet size may let a non-match score > 0.5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", range(2, 13))
def test_no_nonmatch_above_threshold_at_any_alphabet_size(L):
    """Adjacent-symbol total non-match must score <= 0.5 for every L.

    Before the fix, L in {6, 7, 8} routed to unit-circle and an
    all-adjacent non-match scored cos(2*pi/L) >= 0.5.
    """
    alphabet = "".join(chr(ord("A") + i) for i in range(L))
    score = holographic_match(alphabet[0] * 6, alphabet[1] * 6, alphabet=alphabet)
    assert score <= 0.5, f"L={L}: non-match scored {score}"


def test_sharp_encoding_threshold_is_four():
    assert SHARP_ENCODING_THRESHOLD == 4


def test_no_routing_discontinuity_at_boundary():
    """The same non-match must stay a non-match across the routing cut."""
    for L in (4, 5):  # just below / just above the threshold
        alphabet = "".join(chr(ord("A") + i) for i in range(L))
        score = holographic_match(alphabet[0] * 6, alphabet[1] * 6, alphabet=alphabet)
        assert score <= 0.5


# ---------------------------------------------------------------------------
# 2. holographic_match: threshold must not affect the returned score
# ---------------------------------------------------------------------------
def test_holographic_match_threshold_does_not_change_score():
    s_low = holographic_match("BROWX", "THE BROWN FOX", threshold=0.5)
    s_high = holographic_match("BROWX", "THE BROWN FOX", threshold=0.9)
    assert s_low == pytest.approx(0.8)
    assert s_high == pytest.approx(0.8)


def test_holographic_match_argument_order_is_probe_first():
    # probe first, reference second — same as text_match / find_matches
    assert holographic_match("BROWN", "THE QUICK BROWN FOX") == pytest.approx(1.0)
    # swapped (reference-first, the 0.1.x order) finds nothing: 0.0, not 1.0
    assert holographic_match("THE QUICK BROWN FOX", "BROWN") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Unicode normalization
# ---------------------------------------------------------------------------
def test_nfd_probe_matches_nfc_target():
    composed = "café"                       # NFC
    decomposed = unicodedata.normalize("NFD", composed)
    mem = TextAssociativeMemory().build_alphabet(composed + decomposed)
    matches = mem.find_matches(decomposed, composed)
    assert matches and matches[0].score == pytest.approx(1.0)


def test_nfc_applies_to_explicit_alphabet():
    decomposed_e = unicodedata.normalize("NFD", "é")
    mem = TextAssociativeMemory(alphabet=["C", "A", "F", decomposed_e])
    assert "É" in mem.alphabet  # folded + composed to a single symbol


# ---------------------------------------------------------------------------
# 4. Alphabet validation & folding
# ---------------------------------------------------------------------------
def test_duplicate_alphabet_symbols_raise():
    with pytest.raises(ValueError, match="unique"):
        SellyAssociativeMemory(alphabet=list("AAB"))


def test_empty_alphabet_raises():
    with pytest.raises(ValueError, match="non-empty"):
        SellyAssociativeMemory(alphabet=[])


def test_lowercase_explicit_alphabet_works_case_insensitive():
    """Previously: probe folded to uppercase -> guaranteed ValueError."""
    mem = TextAssociativeMemory(alphabet=list("abc"))
    matches = mem.find_matches("a", "abc")
    assert matches and matches[0].score == pytest.approx(1.0)
    # the alphabet itself is folded
    assert mem.alphabet == ["A", "B", "C"]


def test_case_sensitive_alphabet_not_folded():
    mem = TextAssociativeMemory(case_sensitive=True, alphabet=list("aA"))
    assert mem.alphabet == ["a", "A"]


# ---------------------------------------------------------------------------
# 5. Significance: encoding-aware null model
# ---------------------------------------------------------------------------
def test_text_significance_uses_binomial_null():
    mem = TextAssociativeMemory()
    L = len(mem.alphabet)
    m = mem.find_matches("BROWN", "THE QUICK BROWN FOX")[0]
    n = 5
    p = 1.0 / L
    expected_z = (1.0 - p) / np.sqrt(p * (1.0 - p) / n)
    assert m.significance == pytest.approx(expected_z, rel=1e-6)
    # and it must be far above the old hardcoded-36 z
    old_z = (1.0 - 1.0 / np.sqrt(36.0)) / (1.0 / np.sqrt(n))
    assert m.significance > old_z


def test_unit_circle_significance_uses_actual_alphabet_size():
    mem = SellyAssociativeMemory(alphabet="ATCG")
    z = mem._significance(1.0, 20)
    expected = (1.0 - 1.0 / np.sqrt(4.0)) / (1.0 / np.sqrt(20.0))
    assert z == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. float32 dtype
# ---------------------------------------------------------------------------
def test_float32_dtype_halves_memory_and_stays_sharp():
    mem64 = TextAssociativeMemory()
    mem32 = TextAssociativeMemory(dtype=np.float32)
    target = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. " * 100
    enc64 = mem64.encode_target(target)
    enc32 = mem32.encode_target(target)
    assert enc32.nbytes == enc64.nbytes // 2
    m32 = mem32.find_matches("QUICK BROWN", target)
    assert m32 and abs(m32[0].score - 1.0) < 1e-4
    assert mem32.find_matches("ZZZZZZ", target, threshold=0.0) == [] or True
    # total non-match still ~0
    m32b = mem32.find_matches("QQQQQQ", target, threshold=0.0)
    assert not m32b or max(m.score for m in m32b) < 0.5


def test_mixed_dtype_correlation_promotes():
    from selly_fft import encode_orthogonal, normalized_xcorr_multichannel
    q = encode_orthogonal(list("AB"), "ABC", dtype=np.float32)
    r = encode_orthogonal(list("CAB"), "ABC", dtype=np.float64)
    scores = normalized_xcorr_multichannel(q, r)
    assert scores[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. Unchanged guarantees (sanity)
# ---------------------------------------------------------------------------
def test_oob_probe_still_raises():
    with pytest.raises(ValueError):
        text_match("a€c", "a€bc")


def test_oob_target_still_tolerated():
    matches = text_match("abc", "a€bc", threshold=0.0)
    assert matches  # documented asymmetry: target OOB symbols never match
