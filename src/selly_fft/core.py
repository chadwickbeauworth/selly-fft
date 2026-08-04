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

import math
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


def encode_orthogonal(
    symbols: Sequence, alphabet: Sequence, dtype: "np.typing.DTypeLike" = np.float64
) -> np.ndarray:
    """Encode symbols as mutually orthogonal (one-hot) vectors.

    Each symbol maps to a unit basis vector ``e_k`` of dimension
    ``L = len(alphabet)``.  Distinct symbols are *exactly* orthogonal
    (inner product 0) regardless of alphabet size, so the normalized
    cross-correlation becomes the true fraction of matching positions.

    Why this exists
    ---------------
    :func:`encode_unit_circle` packs every symbol onto a single circle,
    so two *different* symbols separated by a small angle still
    correlate strongly: with the 36-symbol default alphabet ``'A'`` and
    ``'B'`` are only 10° apart and score ``cos(10°) ≈ 0.985``.  That is
    fine for a 4-symbol alphabet at 90° separation (DNA), but it makes
    large-alphabet text search unusable — a total non-match reads as a
    near-perfect hit.

    One-hot encoding removes the angular crosstalk entirely at the cost
    of ``L``x memory and an ``L``-channel correlation.

    Parameters
    ----------
    symbols : sequence
        Ordered symbols to encode.
    alphabet : sequence
        Ordered set defining the symbol→index mapping.
    dtype : numpy dtype, default float64
        Storage dtype for the one-hot matrix.  ``np.float32`` halves
        memory at a negligible accuracy cost for match-fraction scoring.

    Returns
    -------
    np.ndarray
        Real-valued array of shape ``(len(symbols), L)``, one one-hot
        row per symbol.

    Raises
    ------
    ValueError
        If a symbol is not present in ``alphabet``.
    """
    idx = _alphabet_index(alphabet)
    L = len(idx)
    out = np.zeros((len(symbols), L), dtype=dtype)
    for i, sym in enumerate(symbols):
        if sym not in idx:
            raise ValueError(f"symbol {sym!r} not in alphabet")
        out[i, idx[sym]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------

def _binom_sf(k: int, n: int, p: float) -> float:
    """Exact upper tail P(X >= k) for X ~ Binomial(n, p).

    Log-space evaluation with upward recurrence from ``k``; no scipy
    dependency.  Conservative short-circuit: if ``k`` is at or below the
    null mean the tail is large, so 1.0 is returned (the gate will reject
    it as non-significant anyway).
    """
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if k <= n * p:
        return 1.0  # below the null mean: certainly not significant
    logp = math.log(p)
    logq = math.log1p(-p)
    log_term = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * logp + (n - k) * logq
    )
    term = math.exp(log_term)
    total = term
    ratio_pq = p / (1.0 - p)
    for j in range(k + 1, n + 1):
        term *= (n - j + 1) / j * ratio_pq
        total += term
        if term <= total * 1e-15:
            break
    return min(1.0, total)


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
    # Cross-correlation: corr[k] = sum_i conj(a[i]) * b[k+i]
    # FFT identity: ifft(conj(FFT(A)) * FFT(B)) = that sum, computed
    # circularly; zero-padding to >= na+nb-1 makes it linear.
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


def normalized_xcorr_multichannel(
    query: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    r"""Normalized cross-correlation for multi-channel (one-hot) encodings.

    Accepts 2-D arrays of shape ``(n_symbols, n_channels)`` — the output
    of :func:`encode_orthogonal` — and correlates each channel
    independently via FFT, summing the results.  Because one-hot
    channels are mutually orthogonal, the summed correlation at shift
    ``k`` equals the exact **count of matching symbols**, and dividing
    by the query length yields the true match fraction.

    This is the sharp path: distinct symbols contribute exactly 0, so a
    total non-match scores 0.0 no matter how large the alphabet is.

    Complexity is ``O(L · (n+m) log(n+m))`` for an ``L``-symbol
    alphabet.  Only channels actually present in the query are
    transformed, so the practical cost scales with the number of
    *distinct symbols in the query*, not the full alphabet size.

    Parameters
    ----------
    query, reference : np.ndarray
        2-D arrays of shape ``(n_symbols, n_channels)`` with matching
        channel counts.

    Returns
    -------
    np.ndarray
        Scores in ``[0, 1]`` over the valid region — the fraction of
        exactly-matching symbol positions at each shift.
    """
    a = np.asarray(query)
    b = np.asarray(reference)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("multichannel correlation requires 2-D (n, channels) arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"channel mismatch: query has {a.shape[1]}, reference has {b.shape[1]}"
        )
    if a.dtype != b.dtype:
        a = a.astype(np.promote_types(a.dtype, b.dtype), copy=False)
        b = b.astype(np.promote_types(a.dtype, b.dtype), copy=False)
    # Respect the input floating dtype (float32 halves memory); anything
    # non-floating is promoted to float64.
    work = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64
    a = a.astype(work, copy=False)
    b = b.astype(work, copy=False)
    na, nb = a.shape[0], b.shape[0]
    if na == 0 or nb == 0 or na > nb:
        return np.zeros(0, dtype=work)

    n_valid = nb - na + 1
    nfft = 1
    while nfft < na + nb - 1:
        nfft <<= 1

    # Correlate channel-by-channel; skip channels absent from the query
    # (their contribution to the match count is identically zero).
    total = np.zeros(nfft, dtype=work)
    active = np.flatnonzero(a.any(axis=0))
    for ch in active:
        A = np.zeros(nfft, dtype=work)
        B = np.zeros(nfft, dtype=work)
        A[:na] = a[:, ch]
        B[:nb] = b[:, ch]
        prod = np.conj(np.fft.rfft(A)) * np.fft.rfft(B)
        total += np.fft.irfft(prod, n=nfft)

    matches = total[:n_valid].astype(np.float64, copy=False)
    # Each matching position contributes exactly 1.0; normalize by query
    # length to get the match fraction.
    scores = matches / float(na)
    return np.clip(scores, 0.0, 1.0)


def normalized_xcorr_multichannel_batch(
    probes: Sequence[np.ndarray], reference: np.ndarray
) -> List[np.ndarray]:
    r"""Batch multichannel correlation sharing the reference transforms.

    Correlates many probes against one reference, computing the
    reference channel FFTs **once** (at a single FFT size covering the
    longest probe) instead of once per probe per channel.  This is the
    economic shape for repeated probing of a fixed target: encode the
    target once, then scan many probes across it.

    Parameters
    ----------
    probes : sequence of np.ndarray
        2-D ``(m_i, L)`` one-hot encodings (outputs of
        :func:`encode_orthogonal`).  All must share the reference's
        channel count ``L``.
    reference : np.ndarray
        2-D ``(n, L)`` one-hot encoding of the target.

    Returns
    -------
    list of np.ndarray
        One score array per probe, each in ``[0, 1]`` over the valid
        region — identical to calling
        :func:`normalized_xcorr_multichannel` per probe.
    """
    b = np.asarray(reference)
    if b.ndim != 2:
        raise ValueError("batch correlation requires a 2-D (n, channels) reference")
    if not np.issubdtype(b.dtype, np.floating):
        b = b.astype(np.float64)
    nb, L = b.shape
    work = b.dtype

    probes = [np.asarray(p) for p in probes]
    for p in probes:
        if p.ndim != 2 or p.shape[1] != L:
            raise ValueError("every probe must be 2-D with the reference's channel count")
    # longest usable probe drives the shared FFT size
    usable = [p for p in probes if 0 < p.shape[0] <= nb]
    if nb == 0 or not usable:
        return [np.zeros(0, dtype=np.float64) for _ in probes]

    max_na = max(p.shape[0] for p in usable)
    nfft = 1
    while nfft < max_na + nb - 1:
        nfft <<= 1

    # Channels active in ANY probe get one shared reference transform.
    active = np.zeros(L, dtype=bool)
    for p in usable:
        active |= p.any(axis=0)
    rfft_B = {}
    for ch in np.flatnonzero(active):
        B = np.zeros(nfft, dtype=work)
        B[:nb] = b[:, ch]
        rfft_B[ch] = np.fft.rfft(B)

    out: List[np.ndarray] = []
    for p in probes:
        na = p.shape[0]
        if na == 0 or na > nb:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        if not np.issubdtype(p.dtype, np.floating):
            p = p.astype(work, copy=False)
        n_valid = nb - na + 1
        total = np.zeros(nfft, dtype=work)
        for ch in np.flatnonzero(p.any(axis=0)):
            A = np.zeros(nfft, dtype=work)
            A[:na] = p[:, ch]
            total += np.fft.irfft(np.conj(np.fft.rfft(A)) * rfft_B[ch], n=nfft)
        scores = total[:n_valid].astype(np.float64, copy=False) / float(na)
        out.append(np.clip(scores, 0.0, 1.0))
    return out


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
        Must be non-empty and contain no duplicate symbols.
    threshold : float, default 0.5
        Minimum normalized score to report as a match.

    Raises
    ------
    ValueError
        If ``alphabet`` is empty or contains duplicate symbols.
    """

    # Encoding family used by this class; subclasses that switch to
    # one-hot (orthogonal) encoding set this to "orthogonal" so that
    # _significance uses the correct null model.
    _ENCODING = "unit_circle"

    # p-value ceiling applied when threshold="auto": a position is
    # reported only if its match count would occur under the null
    # (independent uniform symbols) with probability <= AUTO_P.
    AUTO_P = 1e-3

    def __init__(
        self,
        alphabet: Optional[Sequence] = None,
        threshold: float = 0.5,
    ) -> None:
        self.alphabet = list(alphabet) if alphabet is not None else list(DEFAULT_ALPHABET)
        _alphabet_index(self.alphabet)  # validates non-empty + unique
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
        threshold: Optional[float | str] = None,
    ) -> List[Match]:
        """Search for ``probe_data`` within a pre-encoded target.

        Parameters
        ----------
        probe_data : sequence
            The query symbols.
        target_encoded : np.ndarray
            Output of :meth:`encode_target` (complex unit-circle codes).
        threshold : float or "auto", optional
            Override the instance threshold.  ``"auto"`` gates on
            statistical significance instead of a fixed score: a
            position is reported only if its z-score against the
            encoding's null model reaches ``AUTO_Z``.

        Returns
        -------
        list of Match
            Matches sorted by score descending.
        """
        thr = self._resolve_threshold(threshold)
        probe = self.encode(probe_data)
        scores = self._score_array(probe, target_encoded)
        return self._collect_matches(scores, thr, len(probe_data))

    def search_direct(
        self,
        probe_data: Sequence,
        target_data: Sequence,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[Match]:
        """One-shot search without pre-encoding the target."""
        target_enc = self.encode_target(target_data)
        return self.search(probe_data, target_enc, threshold=threshold)

    def search_many(
        self,
        probes: Sequence[Sequence],
        target_encoded: np.ndarray,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[List[Match]]:
        """Search many probes within one pre-encoded target.

        Returns one match list per probe, in probe order.  The base
        (unit-circle) implementation simply loops; the text subclass
        shares the target's channel FFTs across all probes, which is
        the economical shape for repeated probing of a fixed target.
        """
        return [self.search(p, target_encoded, threshold=threshold) for p in probes]

    def best_score(self, probe_data: Sequence, target_data: Sequence) -> float:
        """Best normalized match score in ``[0, 1]``, ignoring thresholds.

        Unlike :meth:`search_direct` this applies no threshold filtering —
        a partial match below the reporting threshold is still reflected
        in the return value.  Returns 0.0 for empty input or when the
        probe is longer than the target.
        """
        if len(probe_data) == 0 or len(target_data) == 0:
            return 0.0
        target_enc = self.encode_target(target_data)
        probe_enc = self.encode_probe(probe_data)
        scores = self._score_array(probe_enc, target_enc)
        if scores.size == 0:
            return 0.0
        return float(np.max(scores))

    # -- internals -----------------------------------------------------
    def _resolve_threshold(self, threshold):
        """Resolve an optional threshold override (float or "auto")."""
        thr = threshold if threshold is not None else self.threshold
        if isinstance(thr, str) and thr != "auto":
            raise ValueError(f"invalid threshold {thr!r}: use a float or 'auto'")
        return thr

    def _collect_matches(
        self, scores: np.ndarray, thr, probe_len: int
    ) -> List[Match]:
        """Turn a score array into sorted Matches, applying the threshold.

        ``thr`` is a float (score floor) or the string ``"auto"``
        (exact binomial p-value gate at ``AUTO_P`` — see
        :meth:`_auto_pass`).
        """
        matches: List[Match] = []
        for pos, s in enumerate(scores):
            z = float(self._significance(s, probe_len))
            if thr == "auto":
                if not self._auto_pass(s, probe_len):
                    continue
            elif s < thr:
                continue
            matches.append(Match(position=int(pos), score=float(s), significance=z))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _auto_pass(self, score: float, n: int) -> bool:
        """Exact significance gate for ``threshold="auto"``.

        Computes the exact binomial tail probability of the observed
        match count under the null of independent uniform symbols, and
        reports only if ``p <= AUTO_P``.  Unlike a normal z-approximation
        this stays calibrated for short probes (where a single chance
        symbol match in a 6-char probe is *not* significant).

        For the unit-circle path the exact symbol-match count is not
        recoverable from the summed score, so a conservative lower bound
        is used (mismatches contribute at most ``cos(2π/L)`` each).
        """
        if n <= 0 or self._L <= 1:
            return False
        p_sym = 1.0 / self._L
        if self._ENCODING == "orthogonal":
            count = int(round(score * n))
        else:
            c = math.cos(2.0 * math.pi / self._L)
            count = int(math.ceil((score - c) * n / (1.0 - c) - 1e-9))
        if count <= 0:
            return False
        return _binom_sf(count, n, p_sym) <= self.AUTO_P

    def _score_array(
        self, probe_encoded: np.ndarray, target_encoded: np.ndarray
    ) -> np.ndarray:
        """Correlation scores over all valid shifts (encoding-specific)."""
        return normalized_xcorr(probe_encoded, target_encoded)

    def _significance(self, score: float, n: int) -> float:
        """Heuristic z-score against a uniform-random null model.

        The null model depends on the encoding family:

        * **unit_circle** — for ``L`` equally-likely phasor symbols the
          expected normalized correlation magnitude under the null is
          ``1/sqrt(L)`` with standard error ``1/sqrt(n)``.
        * **orthogonal** (one-hot) — the score is a match *fraction*, so
          the null count of matching positions is ``Binomial(n, 1/L)``;
          the z-score uses that mean (``1/L``) and variance
          (``p(1-p)/n``).  This is the correct null for the text path.
        """
        if n <= 0 or self._L <= 0:
            return 0.0
        if self._ENCODING == "orthogonal":
            p = 1.0 / self._L
            var = p * (1.0 - p) / n
            if var <= 0.0:
                return 0.0
            return float((score - p) / np.sqrt(var))
        expected = 1.0 / np.sqrt(float(self._L))
        se = 1.0 / np.sqrt(n)
        if se == 0:
            return 0.0
        return float((score - expected) / se)
