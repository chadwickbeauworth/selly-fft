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

import bisect
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from selly_fft.core import (
    SellyAssociativeMemory,
    Match,
    encode_orthogonal,
    normalized_xcorr_multichannel,
    normalized_xcorr_multichannel_batch,
)


@dataclass
class Span:
    """A match mapped back to the caller's original (pre-normalization) text.

    Attributes
    ----------
    position, end : int
        Start and exclusive end in *normalized* coordinates (what the
        correlation engine sees).
    score, significance : float
        As in :class:`selly_fft.core.Match`.
    orig_start, orig_end : int
        Start and exclusive end in the original target string, robust to
        length-changing normalization (``"ß"`` → ``"SS"``, NFC).
    text : str
        ``target[orig_start:orig_end]`` — the matched span as the user
        actually wrote it.
    """

    position: int
    end: int
    score: float
    significance: float
    orig_start: int
    orig_end: int
    text: str


class TextAssociativeMemory(SellyAssociativeMemory):
    """FFT associative memory specialized for **text / large alphabets**.

    Uses one-hot (orthogonal) symbol encoding so discrimination is exact
    regardless of alphabet size — matching the sharpness of the DNA path.

    Arbitrary Unicode is supported.  Input is **NFC-normalized** before
    encoding, so canonically-equivalent text (e.g. composed ``é`` vs
    ``e`` + combining acute) matches.  Out-of-alphabet symbols in the
    *target* are tolerated (they simply never match); symbols in the
    *probe* must be present after normalization or a ``ValueError`` is
    raised.  By default a small built-in alphabet covers ASCII letters,
    digits, space, and common punctuation; ``build_alphabet`` lets you
    derive an alphabet from your data.

    .. note::
       Case folding (``str.upper()``) can change string *length* — e.g.
       ``"ß"`` folds to ``"SS"``.  Match positions are reported in the
       **normalized (folded) string's coordinates**, which are identical
       to the original string's for all text that neither changes length
       under folding nor under NFC normalization.

    Parameters
    ----------
    case_sensitive : bool, default False
        If False, uppercase-fold probe and target before encoding.
    alphabet : sequence, optional
        Explicit symbol set.  Normalized (NFC, plus uppercase-folding
        when ``case_sensitive=False``) and de-duplicated on entry.
        Defaults to :data:`TEXT_ALPHABET`.
    threshold : float, default 0.5
        Minimum normalized score to report as a match.
    dtype : numpy dtype, default float64
        Storage dtype for encoded one-hot matrices.  ``np.float32``
        halves the memory footprint (752 → 376 bytes/char with the
        default 94-symbol alphabet).
    """

    _ENCODING = "orthogonal"

    def __init__(
        self,
        case_sensitive: bool = False,
        alphabet: Optional[Sequence] = None,
        threshold: float = 0.5,
        dtype: "np.typing.DTypeLike" = np.float64,
    ) -> None:
        self.case_sensitive = bool(case_sensitive)
        self.dtype = dtype
        if alphabet is not None:
            # Fold the explicit alphabet through the same normalization
            # applied to probe/target text, so e.g. a lowercase explicit
            # alphabet works with the default case-insensitive matching.
            folded: List[str] = []
            for sym in alphabet:
                s = self._norm(str(sym))
                if s not in folded:
                    folded.append(s)
            self._explicit_alphabet = folded
        else:
            self._explicit_alphabet = None
        alphabet = self._explicit_alphabet or list(TEXT_ALPHABET)
        super().__init__(alphabet=alphabet, threshold=threshold)

    # -- alphabet handling -------------------------------------------
    def build_alphabet(self, *texts: str) -> "TextAssociativeMemory":
        """Rebuild the alphabet from the symbols present in ``texts``.

        Returns ``self`` (mutated) so it can be chained.  The new
        alphabet preserves first-seen order of the *normalized* symbols
        (NFC, plus uppercase-folding when ``case_sensitive=False``).
        Useful when your data uses characters outside :data:`TEXT_ALPHABET`
        (e.g. accented letters, CJK, emoji).
        """
        seen: List[str] = []
        for t in texts:
            for ch in self._norm(t):
                if ch not in seen:
                    seen.append(ch)
        self.alphabet = seen
        self._L = len(seen)
        return self

    # -- normalization -----------------------------------------------
    def _norm(self, text: str) -> str:
        """NFC-normalize, then case-fold unless case_sensitive.

        NFC is applied before *and* after folding because ``str.upper()``
        can introduce decomposed sequences.
        """
        text = unicodedata.normalize("NFC", text)
        if not self.case_sensitive:
            text = text.upper()
        return unicodedata.normalize("NFC", text)

    # -- encoding -----------------------------------------------------
    def encode(self, symbols: Sequence) -> np.ndarray:
        """Encode a symbol sequence as one-hot vectors (text alphabet)."""
        return encode_orthogonal(
            self._norm_sequence(symbols), self.alphabet, dtype=self.dtype
        )

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
        out = np.zeros((len(chars), L), dtype=self.dtype)
        for i, ch in enumerate(chars):
            if ch in idx:
                out[i, idx[ch]] = 1.0
        return out

    def encode_probe(self, probe_data: Sequence) -> np.ndarray:
        """Encode probe text (string); unknown symbols raise ``ValueError``."""
        return encode_orthogonal(
            self._norm_sequence(probe_data), self.alphabet, dtype=self.dtype
        )

    def _norm_sequence(self, symbols: Sequence) -> List[str]:
        return [self._norm(str(s)) for s in symbols]

    # -- scoring ------------------------------------------------------
    def _score_array(
        self, probe_encoded: np.ndarray, target_encoded: np.ndarray
    ) -> np.ndarray:
        return normalized_xcorr_multichannel(probe_encoded, target_encoded)

    # -- search -------------------------------------------------------
    def search(
        self,
        probe_data: Sequence,
        target_encoded: np.ndarray,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[Match]:
        """Search for ``probe_data`` within a pre-encoded target text.

        ``threshold`` is a float score floor or ``"auto"`` (exact
        binomial p-value gate at ``AUTO_P`` — reports significant
        partial matches too, not just near-exact ones).
        """
        thr = self._resolve_threshold(threshold)
        if not isinstance(target_encoded, np.ndarray):
            raise TypeError(
                f"search() expects a pre-encoded target (np.ndarray from "
                f"encode_target), got {type(target_encoded).__name__}. "
                f"For raw probe/target data use search_direct() or "
                f"find_matches() instead."
            )
        probe = self.encode_probe(probe_data)
        scores = normalized_xcorr_multichannel(probe, target_encoded)
        return self._collect_matches(scores, thr, len(probe_data))

    def search_many(
        self,
        probes: Sequence[Sequence],
        target_encoded: np.ndarray,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[List[Match]]:
        """Search many probes within one pre-encoded target text.

        Shares the target's channel FFTs across all probes (one transform
        per channel total, not per probe), so scanning K probes costs
        little more than scanning one.  Returns one match list per probe,
        in probe order.
        """
        thr = self._resolve_threshold(threshold)
        encs = [self.encode_probe(p) for p in probes]
        norm_lens = [
            len(self._norm(p)) if isinstance(p, str) else len(p) for p in probes
        ]
        score_lists = normalized_xcorr_multichannel_batch(encs, target_encoded)
        return [
            self._collect_matches(sc, thr, n)
            for sc, n in zip(score_lists, norm_lens)
        ]

    def find_matches(
        self,
        probe: str,
        target: str,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[Match]:
        """Find all occurrences of ``probe`` in ``target`` (strings)."""
        return self.search_direct(self._norm(probe), self._norm(target), threshold=threshold)

    # -- span mapping ---------------------------------------------------
    def _norm_with_map(self, text: str):
        """Normalize like :meth:`_norm`, tracking original coordinates.

        Returns ``(normalized, index_map, cluster_starts)`` where
        ``index_map[j]`` is the original index of the text cluster that
        produced normalized character ``j``, and ``cluster_starts`` is
        the sorted list of original cluster boundaries.  Segmentation is
        at Unicode starters (combining class 0), which NFC composition
        never crosses, so per-cluster normalization equals whole-string
        normalization.
        """
        parts: List[str] = []
        index_map: List[int] = []
        starts: List[int] = []
        i, n = 0, len(text)
        while i < n:
            j = i + 1
            while j < n and unicodedata.combining(text[j]):
                j += 1
            starts.append(i)
            folded = self._norm(text[i:j])
            parts.append(folded)
            index_map.extend((i,) * len(folded))
            i = j
        return "".join(parts), index_map, starts

    def find_spans(
        self,
        probe: str,
        target: str,
        *,
        threshold: Optional[float | str] = None,
    ) -> List[Span]:
        """Find ``probe`` in ``target`` and return spans in ORIGINAL coordinates.

        Like :meth:`find_matches`, but each result carries the matched
        text and its position in the caller's original string, correctly
        mapped even when normalization changed string length (``"ß"`` →
        ``"SS"``, NFC composition).  This is the tool showing its work:
        every hit can be pointed at in the user's own text.
        """
        norm_target, index_map, starts = self._norm_with_map(target)
        norm_probe = self._norm(probe)
        if not norm_probe or not norm_target:
            return []
        target_enc = self.encode_target(norm_target)
        matches = self.search(norm_probe, target_enc, threshold=threshold)
        m = len(norm_probe)
        spans: List[Span] = []
        for mt in matches:
            first = index_map[mt.position]
            last = index_map[mt.position + m - 1]
            nxt_idx = bisect.bisect_right(starts, last)
            orig_end = starts[nxt_idx] if nxt_idx < len(starts) else len(target)
            spans.append(
                Span(
                    position=mt.position,
                    end=mt.position + m,
                    score=mt.score,
                    significance=mt.significance,
                    orig_start=first,
                    orig_end=orig_end,
                    text=target[first:orig_end],
                )
            )
        return spans


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
