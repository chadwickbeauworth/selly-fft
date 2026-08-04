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

from selly_fft.core import SellyAssociativeMemory, Match, DEFAULT_ALPHABET


def holographic_match(
    reference: Sequence,
    query: Sequence,
    *,
    alphabet: Sequence = DEFAULT_ALPHABET,
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
        Symbol alphabet for one-hot encoding.
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
    mem = SellyAssociativeMemory(alphabet=alphabet, threshold=threshold)
    matches = mem.search_direct(query, reference)
    if not matches:
        return 0.0
    return float(max(m.score for m in matches))
