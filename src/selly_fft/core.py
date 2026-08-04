"""Core FFT-based associative memory implementation (corrected).

This module implements the core methodology of US8832139B2 with two
corrections over the Run-112 design spec, both documented in Run-126:

1. **Linear (not circular) correlation** — both sequences are zero-padded
   to ``len(ref) + len(query) - 1`` and only the *valid* sliding-window
   region is retained.  This eliminates circular wrap-around artefacts
   that manufacture phantom peaks.

2. **Match-filter normalization + real-part scoring** — the raw linear
   correlation is divided by ``||query|| * ||ref_window||`` (the product
   of L2 norms, not ``sqrt(len_a * len_b)``), and the *real part* is
   taken.  For complex unit-circle phasor encodings this computes
   ``cos(Δθ)`` per-symbol:

   * identical symbols   (Δθ = 0°)   → cos = +1 → score 1.0
   * orthogonal symbols  (Δθ = 90°)  → cos =  0 → score 0.0
   * antipodal symbols   (Δθ = 180°) → cos = −1 → clipped to 0.0

   This makes the score discriminating: an exact substring match scores
   1.0, a total non-match scores ≈ 0.0, and partial matches fall
   monotonically in between.

   The Run-112 bug arose because the spec used *circular* correlation
   with ``sqrt(len)`` normalization and took the *real part* of a
   constant-modulus signal — by Parseval's theorem every symbol
   contributes ``|c|^2 = 1`` to the peak regardless of content, so an
   exact match and a total non-match produced identical scores
   (0.707).  Switching to *linear* correlation with *match-filter*
   normalization removes the content-independence.

The term "quantum" in the patents is **metaphorical**.  This is a
classical FFT signal-processing algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

def _alphabet_index(alphabet: Sequence) -> dict:
    alphabet = list(alphabet)
    if len(alphabet) == 0:
        raise ValueError("alphabet must be non-empty")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must contain unique symbols")
    return {sym: k for k, sym in enumerate(alphabet)}


def encode_unit_circle(symbols: Sequence, alphabet: Sequence) -> np.ndarray:
    """Encode symbols as complex unit-circle phasors (patent-faithful).

    Each symbol maps to ``exp(2j*pi*k/L)`` where ``k`` is the symbol's
    index in ``alphabet`` and ``L = len(alphabet)``.  This is the encoding
    described in US8832139B2 (cols 4-5): "superposition representations"
    and "wavefunction encoding."

    The *discriminability* of this encoding (vs. the Run-112 bug) comes
    not from the encoding itself but from the corrected normalization
    and real-part scoring in :func:`normalized_xcorr` — see the module
    docstring and Run-126 for the full analysis.

    Parameters
    ----------
    symbols : sequence
        Ordered symbols to encode.
    alphabet : sequence
        Ordered set defining the symbol→index mapping.

    Returns
    -------
    np.ndarray
        Complex128 array of shape ``(len(symbols),)``.
    """
    idx = _alphabet_index(alphabet)
    L = len(idx)
    codes = np.empty(len(symbols), dtype=np.complex128)
    for i, sym in enumerate(symbols):
        if sym not in idx:
            raise ValueError(f"symbol {sym!r} not in alphabet")
        codes[i] = np.exp(2j * np.pi * idx[sym] / L)
    return codes


# Deprecated alias kept for backwards reference (not in public API).
encode_symbols = encode_unit_circle


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------

def _pad_to(a: np.ndarray, length: int) -> np.ndarray:
    """Right-pad a 1-D array with zeros to ``length``."""
    out = np.zeros(length, dtype=np.complex128)
    n = min(len(a), length)
    out[:n] = a[:n]
    return out


def linear_correlation(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute the *linear* (non-circular) cross-correlation via FFT.

    Zero-pads both inputs to the next power of two >=
    ``len(query) + len(reference) - 1`` and retains only the *valid*
    region (length ``len(reference) - len(query) + 1``), eliminating
    circular wrap-around artefacts.

    Parameters
    ----------
    query : np.ndarray
        Shorter sequence (the "probe").
    reference : np.ndarray
        Longer sequence (the "target database").

    Returns
    -------
    np.ndarray
        Complex cross-correlation values, one per valid shift.
    """
    a = np.asarray(query, dtype=np.complex128)
    b = np.asarray(reference, dtype=np.complex128)
    na, nb = len(a), len(b)
    n = na + nb - 1
    nfft = 1
    while nfft < n:
        nfft <<= 1
    A = _pad_to(a, nfft)
    B = _pad_to(b, nfft)
    # Cross-correlation: corr[k] = sum_i a[i] * conj(b[k+i])
    # FFT: ifft(conj(FFT(A)) * FFT(B)) gives this (real part) after
    # zero-padding to >= na+nb-1.
    corr = np.fft.ifft(np.conj(np.fft.fft(A)) * np.fft.fft(B))
    valid = corr[: nb - na + 1]
    return valid


def normalized_xcorr(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    r"""Normalized linear cross-correlation (match-filter normalization).

    Divides the raw linear correlation by ``||query|| * ||ref_window||``
    for each valid sliding window and takes the **real part**, so the
    score is bounded in ``[0, 1]``.

    Using the real part (rather than the magnitude) is essential for
    complex phasor encodings: for a constant sequence of symbol *s*
    correlated against a constant sequence of a *different* symbol
    *t*, the correlation is ``k * exp(i(θ_s − θ_t))``.  The real part
    is ``k * cos(θ_s − θ_t)``:

    * identical symbols   (Δθ = 0°)   → cos = +1 → score 1.0
    * orthogonal symbols  (Δθ = 90°)  → cos =  0 → score 0.0
    * antipodal symbols   (Δθ = 180°) → cos = −1 → clipped to 0.0

    This makes the normalized cross-correlation a *discriminating*
    match score.  Taking the magnitude instead would treat
    anti-correlation as a positive match — the Run-112 bug, compounded
    by circular correlation.

    Parameters
    ----------
    query, reference : np.ndarray
        1-D sequences (real or complex).

    Returns
    -------
    np.ndarray
        Scores in ``[0, 1]`` over the valid region.
    """
    a = np.asarray(query, dtype=np.complex128)
    b = np.asarray(reference, dtype=np.complex128)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0 or na > nb:
        return np.zeros(0, dtype=np.float64)
    raw = linear_correlation(a, b)
    norm_a = np.linalg.norm(a)
    if norm_a == 0:
        return np.zeros(nb - na + 1, dtype=np.float64)
    # sliding L2 norm of each reference window (O(n) via cumsum)
    b_sq = np.abs(b) ** 2
    csum = np.concatenate(([0.0], np.cumsum(b_sq)))
    win_sums = csum[na:nb + 1] - csum[: nb - na + 1]
    win_norms = np.sqrt(win_sums)
    denom = norm_a * win_norms
    scores = np.zeros_like(raw, dtype=np.float64)
    nz = denom > 0
    # real part = cosine similarity in the complex plane
    scores[nz] = np.real(raw[nz]) / denom[nz]
    # clamp: anti-correlated (cos = -1) → 0; exact match (cos = +1) → 1
    scores = np.clip(scores, 0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Match:
    """A single match result.

    Attributes
    ----------
    position : int
        Start index of the match within ``reference`` (0-based, in symbol
        positions).
    score : float
        Normalized correlation score in ``[0, 1]``.
    significance : float
        Heuristic z-score against a uniform-random null model.
    """

    position: int
    score: float
    significance: float


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class SellyAssociativeMemory:
    """FFT-based associative memory search engine (classical).

    Implements the core methodology of US8832139B2 with corrections over
    the Run-112 design spec (see module docstring and Run-126):

    * **Linear (not circular) correlation** — zero-padded, valid region
      only.
    * **Match-filter normalization + real-part scoring** — the score is
      ``Re(<query, ref_window>) / (||query|| · ||ref_window||)``, which
      equals ``cos(Δθ)`` per symbol.  An exact match scores 1.0; a total
      non-match scores ≈ 0.0.

    The term "quantum" in the patents is **metaphorical**.  This is a
    classical FFT signal-processing algorithm — no quantum hardware or
    quantum-mechanical effects are used.

    Parameters
    ----------
    alphabet : str or sequence, optional
        Symbols defining the encoding.  Defaults to alphanumeric.
    threshold : float, default 0.5
        Minimum normalized score to report as a match.
    """

    def __init__(
        self,
        alphabet: Optional[Sequence] = None,
        threshold: float = 0.5,
    ) -> None:
        self.alphabet = list(alphabet) if alphabet is not None else list(DEFAULT_ALPHABET)
        self.threshold = float(threshold)
        self._L = len(self.alphabet)

    # -- encoding ------------------------------------------------------
    def encode(self, symbols: Sequence) -> np.ndarray:
        """Encode a symbol sequence onto the unit circle."""
        return encode_unit_circle(symbols, self.alphabet)

    def encode_target(self, target_data: Sequence) -> np.ndarray:
        """Encode the target/reference data."""
        return self.encode(target_data)

    def encode_probe(self, probe_data: Sequence) -> np.ndarray:
        """Encode the probe/query data."""
        return self.encode(probe_data)

    # -- search --------------------------------------------------------
    def search(
        self,
        probe_data: Sequence,
        target_encoded: np.ndarray,
        *,
        threshold: Optional[float] = None,
    ) -> List[Match]:
        """Search for ``probe_data`` within a pre-encoded target.

        Parameters
        ----------
        probe_data : sequence
            The query symbols.
        target_encoded : np.ndarray
            Output of :meth:`encode_target` (complex unit-circle codes).
        threshold : float, optional
            Override the instance threshold.

        Returns
        -------
        list of Match
            Matches sorted by score descending.
        """
        thr = threshold if threshold is not None else self.threshold
        probe = self.encode(probe_data)
        scores = normalized_xcorr(probe, target_encoded)
        matches: List[Match] = []
        for pos, s in enumerate(scores):
            if s >= thr:
                matches.append(
                    Match(
                        position=int(pos),
                        score=float(s),
                        significance=float(self._significance(s, len(probe_data))),
                    )
                )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def search_direct(
        self,
        probe_data: Sequence,
        target_data: Sequence,
        *,
        threshold: Optional[float] = None,
    ) -> List[Match]:
        """One-shot search without pre-encoding the target."""
        target_enc = self.encode_target(target_data)
        return self.search(probe_data, target_enc, threshold=threshold)

    # -- internals -----------------------------------------------------
    @staticmethod
    def _significance(score: float, n: int) -> float:
        """Heuristic z-score against a uniform-random null model.

        For L equally-likely unit-circle symbols the expected normalized
        correlation magnitude under the null is ``1/sqrt(L)``; the
        standard error scales as ``1/sqrt(n)``.
        """
        if n <= 0:
            return 0.0
        expected = 1.0 / np.sqrt(36.0)
        se = 1.0 / np.sqrt(n)
        if se == 0:
            return 0.0
        z = (score - expected) / se
        return float(z)
