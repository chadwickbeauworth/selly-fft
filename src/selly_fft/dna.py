"""DNA/RNA-specialized FFT associative memory.

The patents (US8832139B2, cols 15-17) describe DNA base-pair matching as
the primary motivating application.  This subclass uses a 4-symbol
unit-circle phasor alphabet where A, T, C, G are placed at 90° offsets:

    A ->  1 + 0j   (0°)
    T -> -1 + 0j   (180°)  [antipodal to A]
    C ->  0 + 1j   (90°)
    G ->  0 - 1j   (270°)  [antipodal to C]

Each nucleotide is orthogonal to two others (90°/270°) and antipodal to
one.  With the corrected real-part normalized cross-correlation, an
exact DNA substring match scores 1.0, while a constant non-match (e.g.
``AAAA`` vs ``GGGG``) scores 0.0 (A and G are 90° apart → cos(90°) = 0).
``AAAA`` vs ``TTTT`` also scores 0.0 (antipodal → cos(180°) = −1 → clipped
to 0).

The reference phasor map (A=1, T=−1, C=i, G=−i) is the one described in
US8832139B2; see :func:`encode_dna_phasor`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from selly_fft.core import SellyAssociativeMemory, Match

DNA_ALPHABET = ("A", "T", "C", "G")
RNA_ALPHABET = ("A", "U", "C", "G")

DNA_BASE_MAP: dict = {"A": 1 + 0j, "T": -1 + 0j, "C": 0 + 1j, "G": 0 - 1j}


def encode_dna_phasor(sequence: Sequence[str]) -> np.ndarray:
    """Encode DNA as complex unit-circle phasors (patent-faithful).

    A ->  1+0j,  T -> -1+0j,  C ->  0+1j,  G ->  0-1j.
    """
    out = np.empty(len(sequence), dtype=np.complex128)
    for i, base in enumerate(sequence):
        if base not in DNA_BASE_MAP:
            raise ValueError(f"unexpected nucleotide {base!r} at index {i}")
        out[i] = DNA_BASE_MAP[base]
    return out


class DNAAssociativeMemory(SellyAssociativeMemory):
    """Associative memory specialized for DNA/RNA sequence matching.

    Uses the 4-phased unit-circle encoding (A/T/C/G at 90° separation).
    Distinct nucleotides are either orthogonal (cos = 0) or antipodal
    (cos = −1 → clipped to 0), so a non-match scores ≈ 0 and an exact
    match scores 1.0, with the corrected real-part normalization.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(alphabet=DNA_ALPHABET, threshold=threshold)

    def find_matches(
        self,
        probe_seq: str,
        target_seq: str,
        *,
        threshold: Optional[float] = None,
    ) -> List[Match]:
        """Find all occurrences of ``probe_seq`` in ``target_seq``.

        Convenience wrapper that splits strings into character lists.
        """
        return self.search_direct(list(probe_seq), list(target_seq), threshold=threshold)


def dna_match(probe: str, reference: str, *, threshold: float = 0.5) -> List[Match]:
    """Functional shortcut: DNA associative match.

    Returns matches sorted by score descending.  An exact substring
    match scores 1.0; a complete non-match scores ≈ 0.
    """
    mem = DNAAssociativeMemory(threshold=threshold)
    return mem.find_matches(probe, reference)
