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

Example
-------
>>> from selly_fft import TextAssociativeMemory
>>> mem = TextAssociativeMemory(case_sensitive=False)
>>> [m.position for m in mem.find_matches("brown", "THE QUICK BROWN FOX")]
[10]
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
