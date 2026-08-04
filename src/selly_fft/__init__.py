"""
selly-fft: FFT-based associative memory search.

A classical (non-quantum) implementation of the associative-memory and
data-searching methodology described in US Patent 8,832,139 B2
("Associative memory and data searching system and method") and related
patents by Roger Selly.

The term "quantum" in the patent literature is used metaphorically
to describe the mathematical properties of the FFT-based correlation
method. this library implements a purely classical signal-processing
algorithm using NumPy's FFT routines.

Public API
----------
- :class:`SellyAssociativeMemory` — core FFT correlate-and-search engine
- :class:`DNAAssociativeMemory` — DNA/RNA sequence-specialized subclass
- :func:`holographic_match` — one-shot functional API
- :class:`Match` — result record
- :func:`encode_unit_circle` — low-level symbol encoding
"""

from selly_fft.core import (
    SellyAssociativeMemory,
    Match,
    encode_unit_circle,
    encode_orthogonal,
    normalized_xcorr,
    normalized_xcorr_multichannel,
    linear_correlation,
    DEFAULT_ALPHABET,
)
from selly_fft.dna import DNAAssociativeMemory, dna_match
from selly_fft.text import TextAssociativeMemory, text_match, TEXT_ALPHABET
from selly_fft.func import holographic_match

__version__ = "0.1.1"

__all__ = [
    "SellyAssociativeMemory",
    "DNAAssociativeMemory",
    "TextAssociativeMemory",
    "Match",
    "holographic_match",
    "dna_match",
    "text_match",
    "encode_unit_circle",
    "encode_orthogonal",
    "normalized_xcorr",
    "normalized_xcorr_multichannel",
    "linear_correlation",
    "DEFAULT_ALPHABET",
    "TEXT_ALPHABET",
]
