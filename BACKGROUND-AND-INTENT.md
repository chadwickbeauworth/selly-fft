# Selly-FFT — Background, Intent & Architecture (for independent stress-testing)

> **This document is a background brief, not a test plan.** It exists so a
> separate session can pick up `selly-fft` cold, understand *what it is
> supposed to achieve*, *why it is built the way it is*, and *what the
> working code actually looks like* — then stress-test it without needing
> the author in the loop. No tests are prescribed here; the working files
> are included so the tester can read and probe them directly.

**Library:** `selly-fft` · **Location:** `~/taochadwick/selly-fft`
**Version:** 0.1.1 · **Latest commit:** `bf8d922` (2026-08-04)
**License:** MIT · **Status:** local only, **not yet pushed to GitHub**

---

## 1. What this is, in one paragraph

`selly-fft` is a **classical (NumPy-FFT) fuzzy subsequence search** library.
Given a short *query* ("probe") and a long *reference* ("target") string, it
locates every occurrence of the query in the target and scores each match in
`[0, 1]` using FFT-accelerated **normalized cross-correlation**. It is a
**defensive publication**: it implements the (now-expired) core method of
**US Patent 8,832,139 B2** ("Associative memory and data searching system
and method") and releases it openly under MIT so the technique stays in the
public commons rather than being re-enclosed.

The phrase *"holographic"* in the name is inherited from the patent's
metaphorical language. **There is no quantum computing involved** — it is
plain signal processing. The patent calls the encoding "wavefunction" /
"superposition"; in reality it is complex phasors + FFT.

---

## 2. The problem it solves (and why FFT)

Substring search is normally done with exact algorithms (Boyer–Moore,
Aho–Corasick) or fuzzy ones (edit distance). This library occupies a
different niche: **scoring every possible alignment simultaneously** via
the convolution theorem.

- Naive sliding-window comparison is **O(n·m)**.
- FFT-based correlation computes all alignments at once in
  **O((n+m) log(n+m))**.
- The score is a *normalized correlation* (cosine-similarity-like), so it
  expresses partial/fuzzy matches as a continuous value, not just yes/no.

Use case profile: medium-to-large reference texts where you want a
**soft match score** (how similar is this span?) rather than a binary
hit — e.g. fuzzy keyword spotting, near-duplicate span detection, motif-
find in biological or textual sequences.

---

## 3. The two encodings (this is the crux)

Symbols must become numbers before correlation. There are two encodings,
and the *choice between them is the single most important design decision*
in the library:

### 3a. Unit-circle phasor — `encode_unit_circle`
Each symbol maps to `exp(2j·π·k/L)` where `k` = symbol index, `L` = alphabet
size. All symbols sit on one circle. **Patent-faithful.**

- **Exact for small alphabets.** With 4 symbols (DNA A/T/C/G) placed 90°
  apart, distinct symbols are orthogonal → exact discrimination.
- **Degrades on large alphabets.** In a 36-symbol alphabet, adjacent
  symbols are only 10° apart, so `cos(10°) ≈ 0.985`. Two *different*
  symbols correlate strongly. A total non-match reads as a near-perfect hit.

### 3b. One-hot (orthogonal) — `encode_orthogonal`  ← the path added for text
Each symbol maps to a unit basis vector `e_k` of dimension `L`. Distinct
symbols are **exactly orthogonal** regardless of `L`.

- **Exact for ANY alphabet size.** The normalized cross-correlation becomes
  the true *fraction of matching positions*. A total non-match scores `0.0`;
  an exact match scores `1.0`.
- **Cost:** `L` floats per character (memory scales with alphabet size),
  and correlation is `L` channels (only the channels present in the query
  are actually transformed, so practical cost tracks distinct query symbols).

### Auto-routing
`holographic_match()` uses unit-circle for alphabets **≤ 8 symbols** and
switches to one-hot for **> 8 symbols** (`SHARP_ENCODING_THRESHOLD = 8` in
`func.py`). The `DNAAssociativeMemory` class is hard-wired to unit-circle
(4 symbols, 90°); the `TextAssociativeMemory` class is hard-wired to one-hot.

---

## 4. How the correlation produces a score (math)

For one-hot encoding, define `encoded` sequences `A` (query, `L` channels)
and `B` (reference, `L` channels). For each channel `c`:

```
corr_c[k] = Σ_i  A[i,c] · B[k+i, c]      # via FFT: irfft(conj(rfft(A_c))·rfft(B_c))
total[k]  = Σ_c corr_c[k]                # sum channels
score[k]  = total[k] / len(query)        # clip to [0, 1]
```

Because one-hot channels are mutually orthogonal, `total[k]` equals the
**exact count of matching symbol positions** at shift `k`. Dividing by the
query length yields the match fraction. (See `normalized_xcorr_multichannel`
in `core.py`.)

For unit-circle encoding the analogous formula gives `Re(<query, window>) /
(||query||·||window||)` = `cos(Δθ)` per symbol (`normalized_xcorr`).
The DNA path is exact because its symbols are ≥ 90° apart.

---

## 5. What "sharp" means, and why it mattered here

The library's *raison d'être* for text is **sharpness**: a non-match must
score ≈ 0, not ≈ 1. This was **not true** in the first implementation:

- Old behavior: `"AAAAAAAAAA"` vs `"BBBBB"` → **0.985** (unit-circle, 36-sym)
- Old behavior: lowercase/punctuation → `ValueError` (not in alphabet)
- New behavior (one-hot): `"AAAAAAAAAA"` vs `"BBBBB"` → **0.0**
- New behavior: `"Hello, World!"` / `"World"` → **1.0** at position 7

The user's explicit requirement: **"large-alphabet (text) search as sharp as
the DNA path."** That is the acceptance criterion the stress test must
defend — any non-match scoring > 0.5 is a regression of the guarantee.

---

## 6. The working files (read these directly)

All paths relative to `~/taochadwick/selly-fft`.

| File | Role |
|------|------|
| `src/selly_fft/__init__.py` | Public API exports |
| `src/selly_fft/core.py` | `encode_unit_circle`, `encode_orthogonal`, `linear_correlation`, `normalized_xcorr`, `normalized_xcorr_multichannel`, `SellyAssociativeMemory` base |
| `src/selly_fft/dna.py` | `DNAAssociativeMemory` (unit-circle, 4 bases) |
| `src/selly_fft/text.py` | `TextAssociativeMemory`, `text_match`, `TEXT_ALPHABET` (one-hot) |
| `src/selly_fft/func.py` | `holographic_match` (auto-routing) |
| `tests/test_text.py` | Text-path tests (what currently passes) |
| `README.md` | Usage + benchmarks |

### 6a. `text.py` — the class under test (full)

```python
"""Text / large-alphabet associative memory using orthogonal encoding.

The patents' primary example is DNA (a 4-symbol alphabet), but the most
useful day-to-day application is **text search**: fuzzy matching of words
and phrases across documents, logs, or arbitrary Unicode strings.

The default unit-circle encoding (see :mod:`selly_fft.core`) packs every
symbol onto a single circle, so two *different* symbols close together on
that circle still correlate: in a 36-symbol alphabet ``'A'`` and ``'B'``
are 10° apart and score ``cos(10°) ≈ 0.985``.  On large alphabets that
angular crosstalk makes a total non-match look like a near-perfect hit —
``"AAAAAAAAAA"`` vs ``"BBBBB"`` scored 0.9848.  Unusable for text.

This module uses :func:`selly_fft.core.encode_orthogonal` (one-hot) so
distinct symbols are *exactly* orthogonal.  The normalized
cross-correlation is then the true **fraction of matching positions**,
independently of alphabet size: an exact match scores 1.0 and a total
non-match scores 0.0, just like the DNA path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from selly_fft.core import (
    SellyAssociativeMemory,
    Match,
    encode_orthogonal,
    normalized_xcorr_multichannel,
)


class TextAssociativeMemory(SellyAssociativeMemory):
    """FFT associative memory specialized for **text / large alphabets**.

    Uses one-hot (orthogonal) symbol encoding so discrimination is exact
    regardless of alphabet size — matching the sharpness of the DNA path.

    Arbitrary Unicode is supported.  Out-of-alphabet symbols in the
    *target* are tolerated (they simply never match); symbols in the
    *probe* must be present after normalization or a ``ValueError`` is
    raised.  By default a small built-in alphabet covers ASCII letters,
    digits, space, and common punctuation; ``build_alphabet`` lets you
    derive an alphabet from your data.

    Parameters
    ----------
    case_sensitive : bool, default False
        If False, uppercase-fold probe and target before encoding.
    alphabet : sequence, optional
        Explicit symbol set.  Defaults to :data:`TEXT_ALPHABET`.
    threshold : float, default 0.5
        Minimum normalized score to report as a match.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        alphabet: Optional[Sequence] = None,
        threshold: float = 0.5,
    ) -> None:
        self.case_sensitive = bool(case_sensitive)
        self._explicit_alphabet = list(alphabet) if alphabet is not None else None
        alphabet = self._explicit_alphabet or list(TEXT_ALPHABET)
        super().__init__(alphabet=alphabet, threshold=threshold)

    # -- alphabet handling -------------------------------------------
    def build_alphabet(self, *texts: str) -> "TextAssociativeMemory":
        """Rebuild the alphabet from the symbols present in ``texts``.

        Returns ``self`` (mutated) so it can be chained.  The new
        alphabet preserves first-seen order.  Useful when your data uses
        characters outside :data:`TEXT_ALPHABET` (e.g. accented letters,
        CJK, emoji).
        """
        seen: List[str] = []
        for t in texts:
            for ch in (t if self.case_sensitive else t.upper()):
                if ch not in seen:
                    seen.append(ch)
        self.alphabet = seen
        self._L = len(seen)
        return self

    # -- normalization -----------------------------------------------
    def _norm(self, text: str) -> str:
        return text if self.case_sensitive else text.upper()

    # -- encoding -----------------------------------------------------
    def encode(self, symbols: Sequence) -> np.ndarray:
        """Encode a symbol sequence as one-hot vectors (text alphabet)."""
        return encode_orthogonal(self._norm_sequence(symbols), self.alphabet)

    def encode_target(self, target_data: Sequence) -> np.ndarray:
        """Encode target text (string or sequence) as one-hot.

        Tolerates symbols absent from the alphabet (they simply never
        match).  Accepts either a ``str`` or an iterable of single-char
        symbols, matching :meth:`SellyAssociativeMemory.encode_target`.
        """
        if isinstance(target_data, str):
            chars = list(self._norm(target_data))
        else:
            chars = [self._norm(str(s)) for s in target_data]
        idx = {sym: k for k, sym in enumerate(self.alphabet)}
        L = len(self.alphabet)
        out = np.zeros((len(chars), L), dtype=np.float64)
        for i, ch in enumerate(chars):
            if ch in idx:
                out[i, idx[ch]] = 1.0
        return out

    def encode_probe(self, probe_data: Sequence) -> np.ndarray:
        """Encode probe text (string); unknown symbols raise ``ValueError``."""
        return encode_orthogonal(self._norm_sequence(probe_data), self.alphabet)

    def _norm_sequence(self, symbols: Sequence) -> List[str]:
        return [self._norm(str(s)) for s in symbols]

    # -- search -------------------------------------------------------
    def search(
        self,
        probe_data: Sequence,
        target_encoded: np.ndarray,
        *,
        threshold: Optional[float] = None,
    ) -> List[Match]:
        """Search for ``probe_data`` within a pre-encoded target text."""
        thr = threshold if threshold is not None else self.threshold
        probe = self.encode_probe(probe_data)
        scores = normalized_xcorr_multichannel(probe, target_encoded)
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

    def find_matches(
        self,
        probe: str,
        target: str,
        *,
        threshold: Optional[float] = None,
    ) -> List[Match]:
        """Find all occurrences of ``probe`` in ``target`` (strings)."""
        return self.search_direct(self._norm(probe), self._norm(target), threshold=threshold)


# ASCII letters, digits, space, and common punctuation.  Order is fixed so
# identical text always maps to the same indices.
TEXT_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,;:!?-'\"()[]{}@#&*/+=_<>~^%$|\\"
)


def text_match(probe: str, reference: str, *, threshold: float = 0.5) -> List[Match]:
    """Functional shortcut: fuzzy text match returning scored positions.

    Case-insensitive by default.  An exact substring match scores 1.0; a
    complete non-match scores 0.0.
    """
    mem = TextAssociativeMemory(threshold=threshold)
    return mem.find_matches(probe, reference)
```

### 6b. `func.py` — the auto-routing one-shot (full)

```python
"""Functional convenience API for quick one-shot matching."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from selly_fft.core import (
    SellyAssociativeMemory,
    Match,
)
from selly_fft.text import TextAssociativeMemory, TEXT_ALPHABET

# Beyond this alphabet size the unit-circle phasor encoding (fixed 360°)
# gives adjacent symbols too little angular separation to discriminate
# (e.g. a 36-symbol alphabet separates neighbours by only 10° → cos 10°
# ≈ 0.985).  For alphabets larger than this we switch to exact one-hot
# (orthogonal) encoding so a total non-match scores 0.0 regardless of size.
SHARP_ENCODING_THRESHOLD = 8


def holographic_match(
    reference: Sequence,
    query: Sequence,
    *,
    alphabet: Sequence = TEXT_ALPHABET,
    threshold: float = 0.5,
) -> float:
    """Compute the best normalized holographic match score.

    Parameters
    ----------
    reference : sequence
        The target/database sequence (longer).
    query : sequence
        The probe to search for (shorter).
    alphabet : sequence, optional
        Symbol alphabet.  When it has more than
        ``SHARP_ENCODING_THRESHOLD`` symbols the function uses exact
        one-hot (orthogonal) encoding for sharp discrimination; smaller
        alphabets use the patent-faithful unit-circle phasor encoding,
        which is exact for 4-symbol DNA-style sets at 90° separation.
    threshold : float, optional
        Minimum score to report.  Does not affect the return value of
        this function — it always returns the best score.

    Returns
    -------
    float
        Best normalized cross-correlation score in ``[0, 1]``.
        1.0 = exact match, 0.0 = no match / empty input.
    """
    if len(query) == 0 or len(reference) == 0:
        return 0.0
    if len(alphabet) > SHARP_ENCODING_THRESHOLD:
        mem: SellyAssociativeMemory = TextAssociativeMemory(
            case_sensitive=False, alphabet=list(alphabet), threshold=threshold
        )
    else:
        mem = SellyAssociativeMemory(alphabet=alphabet, threshold=threshold)
    matches = mem.search_direct(query, reference)
    if not matches:
        return 0.0
    return float(max(m.score for m in matches))
```

### 6c. `core.py` — the one-hot correlation primitive (full)

```python
def encode_orthogonal(symbols: Sequence, alphabet: Sequence) -> np.ndarray:
    """Encode symbols as mutually orthogonal (one-hot) vectors.

    Each symbol maps to a unit basis vector ``e_k`` of dimension
    ``L = len(alphabet)``.  Distinct symbols are *exactly* orthogonal
    (inner product 0) regardless of alphabet size, so the normalized
    cross-correlation becomes the true fraction of matching positions.
    """
    idx = _alphabet_index(alphabet)
    L = len(idx)
    out = np.zeros((len(symbols), L), dtype=np.float64)
    for i, sym in enumerate(symbols):
        if sym not in idx:
            raise ValueError(f"symbol {sym!r} not in alphabet")
        out[i, idx[sym]] = 1.0
    return out


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

    Complexity is ``O(L · (n+m) log(n+m))`` for an ``L``-symbol
    alphabet.  Only channels actually present in the query are
    transformed, so the practical cost scales with the number of
    *distinct symbols in the query*, not the full alphabet size.
    """
    a = np.asarray(query, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("multichannel correlation requires 2-D (n, channels) arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"channel mismatch: query has {a.shape[1]}, reference has {b.shape[1]}"
        )
    na, nb = a.shape[0], b.shape[0]
    if na == 0 or nb == 0 or na > nb:
        return np.zeros(0, dtype=np.float64)

    n_valid = nb - na + 1
    nfft = 1
    while nfft < na + nb - 1:
        nfft <<= 1

    total = np.zeros(nfft, dtype=np.float64)
    active = np.flatnonzero(a.any(axis=0))
    for ch in active:
        A = np.zeros(nfft, dtype=np.float64)
        B = np.zeros(nfft, dtype=np.float64)
        A[:na] = a[:, ch]
        B[:nb] = b[:, ch]
        prod = np.conj(np.fft.rfft(A)) * np.fft.rfft(B)
        total += np.fft.irfft(prod, n=nfft)

    matches = total[:n_valid]
    scores = matches / float(na)
    return np.clip(scores, 0.0, 1.0)
```

---

## 7. Known limitations & open questions (for the stress test to probe)

These are **real, documented** weak spots. The stress-test session should
try to break or quantify each:

1. **`Match.significance` is miscalibrated for text.** In `core.py`,
   `_significance` hardcodes `expected = 1/sqrt(36)` — the old 36-symbol
   alphabet. The text alphabet has **94 symbols**, so the null expected
   value should be `1/sqrt(94) ≈ 0.103`, not `0.167`. For Unicode it is
   worse still. *Question:* does this inflate z-scores enough to matter in
   practice?

2. **Inconsistent out-of-alphabet contract.** A symbol absent from the
   alphabet in the **probe** raises `ValueError`; in the **target** it is
   silently zeroed (never matches). *Observed consequence:* `"abc"` searched
   in `"a€bc"` scores **0.666** (the `€` vanishes, `a_c` matches 2/3 of
   `a_bc`). Is that the behavior you want, or should target OOB also error?

3. **One-hot memory scales with alphabet size.** At ~94 floats/char, a 1 MB
   Unicode document encodes to ~80 MB. *Question:* where does that hurt?

4. **Score = match fraction, not edit distance.** Transpositions
   (`"ACTG"` vs `"AGCT"`) score lower than single substitutions, even
   though intuitively "closer". Is fraction-of-matching-positions the right
   similarity measure for your use, or do you want Levenshtein-style?

5. **Unicode normalization.** `build_alphabet` keys on raw code points, so
   `é` (U+00E9) and `e` + combining acute (U+0301) are *different* symbols.
   *Question:* should the library NFC-normalize first?

6. **Auto-routing threshold at 8 symbols.** A custom 9-symbol alphabet
   silently switches from unit-circle to one-hot. *Question:* is 8 the right
   cutoff, and does it surprise users?

---

## 8. How to run it (for the stress-test session)

```bash
REPO=~/taochadwick/selly-fft
PY=/Users/chadwickbeauworth/.hermes/hermes-agent/venv/bin/python

# No install needed — put src on the path:
export PYTHONPATH=$REPO/src
$PY -c "import selly_fft; print(selly_fft.__version__)"   # → 0.1.1

# Smoke test:
$PY -c "from selly_fft import TextAssociativeMemory as T; \
print(T().find_matches('brown','THE QUICK BROWN FOX'))"
```

On a different machine, copy `~/taochadwick/selly-fft` (not yet on GitHub).

---

## 9. The dE/dt framing (why this work exists)

This library is one leaf of a larger research program governed by the
operator's core equation:

> **dE/dt = β(C − D)E**
> Distributed benevolence (E) grows when cooperation (C) exceeds division (D).

The Selly patents, once expired, become a *shared resource*. Implementing
and openly publishing the method **raises C** (the technique is available to
all, cannot be re-enclosed) and **lowers D** (no独占 enclosure of public-
domain knowledge). The text-search sharpness work specifically keeps the
implementation *honest* — a fuzzy matcher that scores non-matches as hits
would itself be a subtle D (misleading output). Sharpness is therefore not
merely a quality metric; it is an integrity requirement under the framing.

---

## 10. Suggested framing for the stress-test session

Hand the session this document and say, roughly:

> *"Read this background brief on the `selly-fft` library. Your job is to
> stress-test the text / large-alphabet path: verify the sharpness claim
> (non-matches score ≈ 0), try to break it, probe the known limitations in
> §7, and measure behavior at scale. Report findings — do not modify the
> code unless I ask."*

The acceptance criterion to defend: **no non-match should score > 0.5.**
