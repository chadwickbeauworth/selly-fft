"""Functional convenience API for quick one-shot matching.

Examples
--------
>>> from selly_fft import holographic_match, dna_match
>>> round(holographic_match("ACGTACGT", "ACGT"), 4)
1.0
>>> matches = dna_match("ATCG", "ATCGATCGATCG")
>>> [m.position for m in matches]
[0, 4, 8]
"""

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
