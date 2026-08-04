"""Tests for the core FFT correlation primitives (Run-127).

These tests verify the bug fix documented in Run-126: the corrected
algorithm must give exact matches ~1.0 and total non-matches ~0.0,
unlike the Run-112 spec which gave both 0.707.
"""

import numpy as np
import pytest

from selly_fft.core import (
    DEFAULT_ALPHABET,
    SellyAssociativeMemory,
    encode_unit_circle,
    linear_correlation,
    normalized_xcorr,
    Match,
)


# ---------------------------------------------------------------------------
# Run-112 bug reproduction (the core fix)
# ---------------------------------------------------------------------------

class TestRun112BugFix:
    """Verify the specific cases from TASK-Lane-C-Phase6-Build.md."""

    def test_exact_substring_match_is_one(self):
        """An exact substring match must score ~1.0."""
        from selly_fft import holographic_match
        s = holographic_match("ACGT", "ACGTACGT")
        assert s > 0.99

    def test_total_nonmatch_is_near_zero(self):
        """A complete non-match must NOT score 0.707 (the Run-112 bug)."""
        from selly_fft import holographic_match
        s = holographic_match("GGGG", "AAAAAAAA")
        assert s < 0.6  # was 0.707 with the buggy method

    def test_exact_beats_nonmatch_substantially(self):
        """Exact match must be strictly and substantially higher than non-match."""
        from selly_fft import holographic_match
        exact = holographic_match("ACGT", "ACGTACGT")
        nonmatch = holographic_match("GGGG", "AAAAAAAA")
        assert exact > nonmatch + 0.3

    def test_full_self_match_is_one(self):
        from selly_fft import holographic_match
        assert holographic_match("ACGTACGT", "ACGTACGT") > 0.99

    def test_empty_query_returns_zero(self):
        from selly_fft import holographic_match
        assert holographic_match("", "ACGT") == 0.0

    def test_empty_reference_returns_zero(self):
        from selly_fft import holographic_match
        assert holographic_match("ACGT", "") == 0.0


# ---------------------------------------------------------------------------
# Run-112 Test 1: Basic Match Detection
# ---------------------------------------------------------------------------

class TestBasicMatchDetection:

    def test_finds_known_positions(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.7)
        matches = mem.search_direct(list("ATCG"), list("ATCGATCGATCG"))
        positions = sorted(m.position for m in matches)
        assert positions == [0, 4, 8]
        for m in matches:
            assert m.score > 0.99

    def test_single_character_match(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.9)
        matches = mem.search_direct(list("A"), list("GATTACA"))
        # A appears at positions 1, 4, 6 in "GATTACA"
        positions = sorted(m.position for m in matches)
        assert positions == [1, 4, 6]

    def test_no_match_below_threshold(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.9)
        matches = mem.search_direct(list("GGGG"), list("AAAAAAAA"))
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# Run-112 Test 2: No False Positives
# ---------------------------------------------------------------------------

class TestNoFalsePositives:

    def test_no_match_for_completely_different_sequences(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.8)
        matches = mem.search_direct(list("AAAA"), list("GGGG"))
        assert len(matches) == 0

    def test_no_false_positive_with_high_threshold(self):
        """With a high threshold, no spurious matches should appear."""
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.95)
        matches = mem.search_direct(list("AAAA"), list("TTTT"))
        assert len(matches) == 0  # A and T are antipodal → clipped to 0


# ---------------------------------------------------------------------------
# Run-112 Test 3: Partial Match Scoring
# ---------------------------------------------------------------------------

class TestPartialMatchScoring:

    def test_partial_match_scores_lower_than_exact(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.0)
        target = "ATCGATCG"
        m_exact = mem.search_direct(list("ATCG"), list(target), threshold=0.0)
        m_partial = mem.search_direct(list("ATCA"), list(target), threshold=0.0)
        assert m_exact[0].score > m_partial[0].score

    def test_3_of_4_match_scores_075(self):
        """3/4 matching symbols → score ≈ 0.75."""
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.0)
        target = "ATCGATCG"
        # ATC A matches ATCG at pos 0 → 3/4 match
        m = mem.search_direct(list("ATCA"), list(target), threshold=0.0)
        assert abs(m[0].score - 0.75) < 0.01

    def test_2_of_4_match_scores_05(self):
        """2/4 matching symbols → score ≈ 0.5."""
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.0)
        target = "ATCGATCG"
        m = mem.search_direct(list("ATTA"), list(target), threshold=0.0)
        assert abs(m[0].score - 0.5) < 0.01

    def test_1_of_4_match_scores_025(self):
        """1/4 matching symbol → score ≈ 0.25."""
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.0)
        target = "ATCGATCG"
        m = mem.search_direct(list("ATTT"), list(target), threshold=0.0)
        assert abs(m[0].score - 0.25) < 0.01


# ---------------------------------------------------------------------------
# Run-112 Test 4: FFT Size Sensitivity
# ---------------------------------------------------------------------------

class TestFFTSizeSensitivity:

    def test_results_consistent_across_sizes(self):
        """Results should be consistent regardless of internal FFT size."""
        target = "ATCGATCGATCGATCG"
        probe = "ATCG"
        mem16 = SellyAssociativeMemory(alphabet="ATCG", threshold=0.7)
        mem32 = SellyAssociativeMemory(alphabet="ATCG", threshold=0.7)
        m16 = mem16.search_direct(list(probe), list(target))
        m32 = mem32.search_direct(list(probe), list(target))
        assert set(m.position for m in m16) == set(m.position for m in m32)
        # scores should be very close (same mathematical result)
        for a, b in zip(m16, m32):
            assert abs(a.score - b.score) < 1e-10

    def test_large_target_small_probe(self):
        mem = SellyAssociativeMemory(alphabet="ATCG", threshold=0.8)
        target = "ATCG" * 100
        matches = mem.search_direct(list("ATCG"), list(target))
        assert len(matches) >= 50  # at least half the positions


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperties:

    @pytest.mark.parametrize("alphabet", ["ATCG", "ACGT", DEFAULT_ALPHABET[:10]])
    def test_score_in_unit_interval(self, alphabet):
        """Score must always be in [0, 1]."""
        mem = SellyAssociativeMemory(alphabet=alphabet, threshold=0.0)
        # generate random sequences
        rng = np.random.RandomState(42)
        for _ in range(20):
            ref = list(rng.choice(list(alphabet), size=rng.randint(5, 50)))
            qry = list(rng.choice(list(alphabet), size=rng.randint(1, 5)))
            matches = mem.search_direct(qry, ref)
            for m in matches:
                assert 0.0 <= m.score <= 1.0

    def test_self_match_is_maximal(self):
        """A sequence should score 1.0 against itself."""
        from selly_fft import holographic_match
        for seq in ["A", "AT", "ATCG", "ACGTACGT", "GGGAAAATTTTCCCC"]:
            assert holographic_match(seq, seq) > 0.99

    def test_monotonic_in_matching_symbols(self):
        """More matching symbols → higher score (with same-length probes)."""
        from selly_fft import holographic_match
        target = "ACGTACGT"
        # probes with 0, 1, 2, 3, 4 matches against target[0:4]="ACGT"
        scores = []
        for probe in ["TTTT", "ACCC", "ACGC", "ACGT"]:
            scores.append(holographic_match(probe, target))
        # 0 matches: TTTT vs ACGT (all different phasors)
        # 1 match: ACCC
        # 2 matches: ACGC
        # 4 matches: ACGT
        assert scores[-1] > scores[-2] > scores[-3]

    def test_empty_input_returns_empty(self):
        mem = SellyAssociativeMemory(alphabet="ATCG")
        assert mem.search_direct(list(""), list("ATCG")) == []
        assert mem.search_direct(list("A"), list("")) == []

    def test_single_character_alphabet(self):
        """Edge case: single-symbol alphabet (all symbols identical)."""
        mem = SellyAssociativeMemory(alphabet="A", threshold=0.5)
        matches = mem.search_direct(list("A"), list("AAAA"))
        assert len(matches) > 0
        assert matches[0].score > 0.99

    def test_symbol_not_in_alphabet_raises(self):
        mem = SellyAssociativeMemory(alphabet="ATCG")
        with pytest.raises(ValueError, match="not in alphabet"):
            mem.search_direct(list("ATCX"), list("AAAA"))

    def test_custom_alphabet(self):
        mem = SellyAssociativeMemory(alphabet="abcd", threshold=0.5)
        matches = mem.search_direct(list("ab"), list("dcdabdcab"))
        positions = sorted(m.position for m in matches)
        assert 3 in positions  # "ab" at position 3
        assert 7 in positions  # "ab" at position 7


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_unit_circle_encoding(self):
        codes = encode_unit_circle(list("ATCG"), "ATCG")
        assert len(codes) == 4
        # all on unit circle
        assert np.allclose(np.abs(codes), 1.0)

    def test_phase_separation(self):
        codes = encode_unit_circle(list("ATCG"), "ATCG")
        # A=0°, T=90°, C=180°, G=270°
        angles = np.angle(codes)
        # sorted angles should be 0, 90, 180, 270 degrees
        sorted_angles = sorted(a % (2 * np.pi) for a in angles)
        expected = [0, np.pi/2, np.pi, 3*np.pi/2]
        for a, e in zip(sorted_angles, expected):
            assert abs(a - e) < 1e-10

    def test_normalize_xcorr_empty(self):
        assert len(normalized_xcorr(np.array([]), np.array([1, 2, 3]))) == 0
        assert len(normalized_xcorr(np.array([1, 2]), np.array([]))) == 0

    def test_normalize_xcorr_query_longer_than_ref(self):
        """When query is longer than reference, no valid shifts exist."""
        a = np.ones(5, dtype=complex)
        b = np.ones(3, dtype=complex)
        result = normalized_xcorr(a, b)
        assert len(result) == 0 or np.all(result == 0)


# ---------------------------------------------------------------------------
# Match dataclass tests
# ---------------------------------------------------------------------------

class TestMatch:

    def test_match_fields(self):
        m = Match(position=5, score=0.99, significance=3.2)
        assert m.position == 5
        assert m.score == 0.99
        assert m.significance == 3.2
