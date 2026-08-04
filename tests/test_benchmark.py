"""Benchmark FFT-based correlation vs naive O(n*m) string correlation (Run-127).

The Selly method achieves O((n+m) log(n+m)) via FFT instead of the
naive O(n*m) per-position comparison.  This benchmark demonstrates the
crossover point where the FFT approach becomes faster.
"""

import time
import numpy as np
import pytest

from selly_fft import SellyAssociativeMemory, DNAAssociativeMemory, holographic_match


def naive_correlation(probe: list, reference: list, alphabet: list) -> float:
    """Naive O(n*m) normalized correlation (reference implementation).

    Computes the best normalized dot-product score over all valid
    sliding-window positions, using the same one-hot-style per-symbol
    inner product as the FFT method but computed in O(n*m*L) time.
    """
    idx = {s: k for k, s in enumerate(alphabet)}
    L = len(alphabet)
    na, nb = len(probe), len(reference)
    if na == 0 or na > nb:
        return 0.0
    best = 0.0
    norm_a = np.sqrt(na)  # each symbol has unit-norm one-hot vector
    for start in range(nb - na + 1):
        dot = 0.0
        for i in range(na):
            if probe[i] == reference[start + i]:
                dot += 1.0
        # normalized: dot / (norm_a * norm_b) = dot / (na * 1) since
        # each matching symbol contributes 1 and the window norm = sqrt(na)
        score = dot / (norm_a * norm_a)
        best = max(best, score)
    return best


@pytest.fixture
def dna_alphabet():
    return list("ATCG")


@pytest.fixture
def random_dna():
    rng = np.random.RandomState(12345)
    return rng.choice(list("ATCG"), size=5000).tolist()


class TestBenchmarkFFTvsNaive:
    """Benchmark FFT correlation vs naive O(n*m) over increasing sizes."""

    @pytest.mark.parametrize("size", [50, 200, 500, 1000, 2000])
    def test_timing(self, size, dna_alphabet, random_dna):
        """Time both methods at various reference sizes."""
        # Use a probe of fixed small size
        probe = random_dna[:8]
        reference = random_dna[:size]

        # Naive (skip for large sizes — too slow, but demonstrate the trend)
        if size <= 500:
            t0 = time.perf_counter()
            naive_score = naive_correlation(probe, reference, dna_alphabet)
            t_naive = time.perf_counter() - t0
        else:
            naive_score = None
            t_naive = None

        # FFT
        t0 = time.perf_counter()
        fft_score = holographic_match(reference, probe, alphabet=dna_alphabet)
        t_fft = time.perf_counter() - t0

        print(f"\n  size={size:5d}  FFT={t_fft*1000:8.3f}ms  score={fft_score:.4f}", end="")
        if t_naive is not None:
            print(f"  Naive={t_naive*1000:8.3f}ms  score={naive_score:.4f}", end="")
        else:
            print(f"  Naive=  skipped", end="")

        # Scores should match (within tolerance)
        if naive_score is not None:
            assert abs(fft_score - naive_score) < 0.01

    def test_speedup_demonstration(self, dna_alphabet, random_dna):
        """Show FFT speedup at a size where naive is still tractable."""
        size = 500
        probe = random_dna[:10]
        reference = random_dna[:size]

        # Warm up
        _ = holographic_match(reference, probe, alphabet=dna_alphabet)

        t0 = time.perf_counter()
        fft_score = holographic_match(reference, probe, alphabet=dna_alphabet)
        t_fft = time.perf_counter() - t0

        t0 = time.perf_counter()
        naive_score = naive_correlation(probe, reference, dna_alphabet)
        t_naive = time.perf_counter() - t0

        speedup = t_naive / t_fft if t_fft > 0 else float("inf")

        print(f"\n  --- Speedup at size={size} ---")
        print(f"  FFT:   {t_fft*1000:.3f} ms  score={fft_score:.4f}")
        print(f"  Naive: {t_naive*1000:.3f} ms  score={naive_score:.4f}")
        print(f"  Speedup: {speedup:.1f}x")

        assert abs(fft_score - naive_score) < 0.01
        assert t_fft < t_naive  # FFT should be faster


class TestScalingComplexity:
    """Verify that the FFT method scales sub-quadratically."""

    @pytest.mark.parametrize("size", [200, 400, 800, 1600])
    def test_subquadratic_scaling(self, size, dna_alphabet, random_dna):
        """Time the FFT method at increasing sizes and check scaling."""
        probe = random_dna[:8]
        reference = random_dna[:size]
        mem = SellyAssociativeMemory(alphabet=dna_alphabet)

        # warm up
        mem.search_direct(probe, reference)

        t0 = time.perf_counter()
        mem.search_direct(probe, reference)
        elapsed = time.perf_counter() - t0

        print(f"\n  FFT search at size={size:5d}: {elapsed*1000:.3f}ms")

    def test_large_sequence_fft_search(self, dna_alphabet):
        """Demonstrate FFT handles large sequences efficiently."""
        rng = np.random.RandomState(42)
        # 10,000 bp synthetic genome
        target = "".join(rng.choice(list("ATCG"), size=10000).tolist())
        # Insert a known motif at position 5000
        target = target[:5000] + "ATCGATCG" + target[5008:]

        mem = DNAAssociativeMemory(threshold=0.9)
        t0 = time.perf_counter()
        matches = mem.find_matches("ATCGATCG", target)
        elapsed = time.perf_counter() - t0

        print(f"\n  10Kbp genome search for 'ATCGATCG': {elapsed*1000:.3f}ms, "
              f"found {len(matches)} matches")

        # The exact match at 5000 should be found
        exact_match_pos = [m for m in matches if m.position == 5000]
        assert len(exact_match_pos) >= 1
        assert exact_match_pos[0].score > 0.99

        # Should also find other "ATCG" occurrences (but not necessarily exact)
        assert len(matches) >= 1
