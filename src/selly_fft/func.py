"""Functional convenience API for quick one-shot matching.

Examples
--------
>>> from selly_fft import holographic_match, dna_match
>>> round(holographic_match("ACGT", "ACGTACGT"), 4)
1.0
>>> matches = dna_match("ATCG", "ATCGATCGATCG")
>>> [m.position for m in matches]
[0, 4, 8]
"""

from __future__ import annotations

from typing import Sequence

from selly_fft.core import (
    SellyAssociativeMemory,
)
from selly_fft.text import TextAssociativeMemory, TEXT_ALPHABET

# Beyond this alphabet size the unit-circle phasor encoding (fixed 360°)
# gives adjacent symbols too little angular separation to discriminate
# sharply: exact discrimination requires >= 90° separation, which only
# alphabets of 4 or fewer symbols achieve (e.g. DNA A/T/C/G at 90°).
# At 6-8 symbols an *adjacent-symbol* total non-match scores
# cos(2π/L) >= 0.5 — a false hit under the default threshold — so for
# alphabets larger than this we switch to exact one-hot (orthogonal)
# encoding, where a total non-match scores 0.0 regardless of size.
SHARP_ENCODING_THRESHOLD = 4


def holographic_match(
    probe: Sequence,
    reference: Sequence,
    *,
    alphabet: Sequence = TEXT_ALPHABET,
    threshold: float = 0.5,
) -> float:
    """Compute the best normalized holographic match score.

    Parameters
    ----------
    probe : sequence
        The query to search for (shorter).
    reference : sequence
        The target/database sequence (longer).
    alphabet : sequence, optional
        Symbol alphabet.  When it has more than
        ``SHARP_ENCODING_THRESHOLD`` symbols the function uses exact
        one-hot (orthogonal) encoding for sharp discrimination; smaller
        alphabets use the patent-faithful unit-circle phasor encoding,
        which is exact for 4-symbol DNA-style sets at 90° separation.
    threshold : float, optional
        Accepted for API compatibility.  Does not affect the return
        value — this function always returns the best (unthresholded)
        score.

    Returns
    -------
    float
        Best normalized cross-correlation score in ``[0, 1]``.
        1.0 = exact match, 0.0 = no match / empty input.

    Notes
    -----
    .. versionchanged:: 0.2.0
        Argument order unified with the rest of the API: probe first,
        reference second (was ``(reference, query)`` in 0.1.x).
    """
    if len(probe) == 0 or len(reference) == 0:
        return 0.0
    if len(alphabet) > SHARP_ENCODING_THRESHOLD:
        mem: SellyAssociativeMemory = TextAssociativeMemory(
            case_sensitive=False, alphabet=list(alphabet), threshold=threshold
        )
    else:
        mem = SellyAssociativeMemory(alphabet=alphabet, threshold=threshold)
    return mem.best_score(probe, reference)
