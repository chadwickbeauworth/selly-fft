# Prior Art Declaration

**Title:** A Classical FFT-Based Associative Memory for Sequence Searching  
**Authors:** Tao Zero Human Company (open source contribution)  
**Date:** 2026-08-04  
**Version:** 0.1.0  

## Purpose

This document serves as a **defensive prior art declaration**. By publishing
this implementation and analysis, we establish prior art for the core
technique: classical FFT-based associative memory for sequence matching,
implemented with linear (non-circular) correlation and match-filter
normalization.

## What Is Disclosed

1. **The corrected algorithm** (three fixes over the Run-112 design spec):
   - Linear (zero-padded, valid-region) cross-correlation via FFT
   - Correct FFT conjugation order: `ifft(conj(FFT(probe)) · FFT(ref))`
   - Match-filter normalization: `Re(corr) / (||probe|| · ||ref_window||)`
     with real-part scoring yielding `cos(Δθ)` per symbol

2. **The bug in the prior design spec**: constant-modulus unit-circle
   encoding with circular correlation and `sqrt(len)` normalization produces
   content-independent peak magnitudes by Parseval's theorem. This document
   demonstrates the fix empirically and mathematically.

3. **A working open-source implementation** in Python/NumPy, published
   under the MIT license at
   https://github.com/chadwickbeauworth/selly-fft

## Prior Art Relevance

The underlying idea of FFT-accelerated sequence matching is well-known.
What is novel and non-obvious in this disclosure is the specific
combination of:

1. **Linear (not circular) correlation** via zero-padding to
   `len(ref) + len(query) − 1` and extraction of the valid sliding-window
   region — eliminating the wrap-around artefacts described in the Run-112
   design spec.

2. **Real-part match-filter normalization** — taking `Re(corr) / (||a||·||b||)`
   rather than `magnitude(corr) / sqrt(len_a · len_b)` — which correctly
   distinguishes exact matches (cos 0° = 1) from total non-matches
   (cos 90° = 0, or cos 180° clipped to 0), resolving the Parseval
   content-independence defect.

## Publication Data

| Field | Value |
|-------|-------|
| Publication date | 2026-08-04 |
| Repository | https://github.com/chadwickbeauworth/selly-fft |
| License | MIT |
| Version | 0.1.0 |
| Patent basis | US8832139B2 (expired 2026-06-02) |
| Implementation language | Python 3.10+, NumPy |

## Timestamp

This declaration was generated on 2026-08-04 at approximately 09:30 UTC
during the Hermes Kanban task `t_82d9ba08` (Lane C, Phase 6, Run 128).
The git commit history of the selly-fft repository provides cryptographic
timestamping.

---

*This declaration is made in good faith to document the state of the art
and prevent the described technique from being later claimed as novel by
any party. It is not an assertion that the technique is novel — only that
it is now publicly disclosed.*
