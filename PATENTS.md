# Patent Analysis

**Not legal advice.** This document reflects the patent landscape as of
2026-08-04. Patent status can change; consult qualified patent counsel
before any commercial deployment.

## Overview

This library implements the core FFT-based associative memory and data
searching methodology described in US Patent 8,832,139 B2. It deliberately
avoids methods that may fall under active continuation patents.

## Patent Family

| Patent | Title | Filing Date | Grant Date | Adjusted Expiration | Status |
|--------|-------|-------------|------------|---------------------|--------|
| US8832139B2 | Associative memory and data searching system and method | 2006-05-15 | 2014-09-09 | **2026-06-02** | **Expired** |
| US12182662B2 | Programmable quantum computer | 2016-03-29 | 2018-12-25 | **2026-07-12** | **Expired** |
| US10438690B2 | Associative memory and data searching system and method | 2014-09-08 | 2019-10-08 | 2029-09-07 | Active |
| US11561951B2 | Multidimensional associative memory and data searching | 2019-10-07 | 2023-01-24 | 2028-02-01 | Active |

**Assignee:** Panvia Future Technologies Inc. (for US10438690B2 and
US11561951B2).

## Claim Mapping

### What IS implemented (expired claims from US8832139B2)

- **Core FFT/DFT associative memory method** — encode symbols as complex
  phasors on the unit circle, transform into the frequency domain via FFT,
  correlate in the frequency domain, inverse-transform, and detect peaks.
  (Claims 1-10)
- **DNA/RNA sequence matching** — the motivating application where the
  target is a DNA/RNA genome and the probe is a subsequence.
  (Claims 2-3)
- **Superposition representations** — the encoding of stored information
  into complex orthogonal basis functions (wavefunctions).
  (Claim 10)
- **General database searching** — applying the FFT method to any
  database of stored information, not just biological sequences.
  (Claims 1-10)

The specific mathematical technique used is:
1. Encode each symbol to `exp(2j·π·k/L)` on the unit circle
2. Apply FFT to both probe and reference
3. Cross-correlate in the frequency domain (conjugate multiply)
4. Inverse FFT to obtain the correlation in the time domain
5. Detect peaks above a threshold

### What is NOT implemented (active continuation patents)

- **US10438690B2 (Active, expires 2029-09-07):** This continuation-in-part
  adds specific refinement to the modulation/interference method, including
  "each modulation function has a positive integer position index and
  corresponds to a modulation function that has a negative integer position
  index with the same magnitude" (i.e., bidirectional modulation with
  phase-rotated superpositions). This library does not implement
  bidirectional or phase-rotated modulation.
  (Risk: MEDIUM — same core method, narrower claims on implementation details)

- **US11561951B2 (Active, expires 2028-02-01):** This patent claims
  "multidimensional" associative memory — extending the FFT method to
  multi-dimensional data (2D+ arrays, not just 1D sequences). This library
  implements only the 1D method.
  (Risk: LOW-MEDIUM — multidimensional claims do not cover standard 1D FFT)

- **US12182662B2 (Expired 2026-07-12):** This patent adds a "programmable
  quantum computer" framing to the same FFT method. While expired, its
  marketing framing ("quantum computer") is misleading and this library
  explicitly disclaims any quantum functionality.

## Lane A Forensics: Terminal Disclaimers

Lane A is currently investigating whether US10438690B2 and
US11561951B2 carry **terminal disclaimers** that cap their effective
expiration at the parent patent's 2026-06-02 date. Terminal disclaimers
are filed to overcome double-patency rejections and can make a
continuation expire with the parent patent rather than at its own
extended term.

The Run-100 through Run-110 research notes indicate all three
continuations (US10438690B2, US11561951B2, US12182662B2) carry terminal
disclaimers. If confirmed, the entire patent family may be expired as
of 2026-07-12 (US12182662B2's date, which is also expired).

**Until Lane A confirms this, this implementation stays conservative:**
only the core 1D method from US8832139B2 is implemented. No
multidimensional extensions, no bidirectional modulation, no "quantum
computer" stack.

## Disclaimer

This is a classical FFT signal-processing implementation. The term
"quantum" in the patent literature is used metaphorically to describe the
mathematical properties of the algorithm (superposition, interference).
This library does NOT use quantum computing hardware, quantum bits, or
any quantum-mechanical effects.

**This is not legal advice.** Users should consult patent counsel before
commercial deployment, especially regarding US10438690B2 and
US11561951B2.
