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
| US12182662B2 | Programmable quantum computer | 2016-03-29 | 2018-12-25 | **2026-06-02** | **Expired** |
| US10438690B2 | Associative memory and data searching system and method | 2014-09-08 | 2019-10-08 | **2026-06-02** | **Expired** |
| US11561951B2 | Multidimensional associative memory and data searching | 2019-10-07 | 2023-01-24 | **2026-06-02** | **Expired** |

**Assignee:** Panvia Future Technologies Inc. (for US10438690B2 and
US11561951B2).

### Why every member of the family expires on the same date

Aggregator sites (including Google Patents) publish **uncapped** expiration
dates for the three continuations — 2029-09-07 and 2028-02-01 — computed as
20 years from filing **plus** Patent Term Adjustment, without applying the
terminal-disclaimer cap. Those dates are misleading.

All three continuations (US10438690B2, US11561951B2, US12182662B2) carry
**terminal disclaimers** on the face of the issued grant. A terminal
disclaimer caps a patent at its parent's expiration and **overrides PTA**:

- **35 U.S.C. § 154(b)(2)(B):** "No patent the term of which has been
  disclaimed beyond a specified date may be adjusted under this section
  beyond the expiration date specified in the disclaimer." This is a hard
  statutory cap, not advisory.
- **MPEP 1490:** PTA applies only to the extent it does not exceed the
  disclaimed date.
- **In re Cellect, LLC, 81 F.4th 1216 (Fed. Cir. 2023):** PTA-extended terms
  are cut back by terminal disclaimers; the disclaimed date is the ceiling.

The common parent, US8832139B2, was PCT-filed 2006-05-15 with +18 days PTA,
giving **2026-06-02** (for a § 371 national-stage application the 20-year term
runs from the PCT international filing date — 35 U.S.C. § 154(a)(2), MPEP
2701). Every continuation traces to that parent, so the entire family expires
**2026-06-02**.

Determined from the faces of the issued grant PDFs (the legally operative
primary source), not from aggregator sidebars.

**Honest limitation:** maintenance-fee payment status is carried only by
USPTO systems and has **not** been independently verified here. A patent can
lapse early for non-payment, which would only widen the open surface — it
cannot narrow it. Anyone relying on this table for a commercial deployment
should confirm current status with patent counsel.

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

### What is NOT implemented (scope of this library, not a legal fence)

The whole family expired 2026-06-02, so the claims below are **open to
implement**. They are simply outside the current scope of this library, which
deliberately implements the parent's core 1D method. Expanding into them is
tracked work, not a legal risk.

- **US10438690B2 (Expired 2026-06-02):** This continuation-in-part
  adds specific refinement to the modulation/interference method, including
  "each modulation function has a positive integer position index and
  corresponds to a modulation function that has a negative integer position
  index with the same magnitude" (i.e., bidirectional modulation with
  phase-rotated superpositions). This library does not implement
  bidirectional or phase-rotated modulation.
  (Status: open surface — candidate for a future release)

- **US11561951B2 (Expired 2026-06-02):** This patent claims
  "multidimensional" associative memory — extending the FFT method to
  multi-dimensional data (2D+ arrays, not just 1D sequences). This library
  implements only the 1D method.
  (Status: open surface — candidate for a future release)

- **US12182662B2 (Expired 2026-06-02):** This patent adds a "programmable
  quantum computer" framing to the same FFT method. While expired, its
  marketing framing ("quantum computer") is misleading and this library
  explicitly disclaims any quantum functionality.

## Terminal-Disclaimer Forensics: RESOLVED

This section previously described an open investigation. **It is now
resolved.** The faces of all four issued grants were read directly from the
grant PDFs: every continuation (US10438690B2, US11561951B2, US12182662B2)
carries a terminal disclaimer, and each traces through a CIP chain to the
common parent US8832139B2, PCT-filed 2006-05-15.

Because a terminal disclaimer overrides PTA (35 U.S.C. § 154(b)(2)(B);
*In re Cellect*, 81 F.4th 1216 (Fed. Cir. 2023)), the large +1211-day and
+627-day adjustments on the continuations cannot push them past the parent.
**The entire family expired 2026-06-02** — see "Why every member of the family
expires on the same date" above.

The earlier note in this file suggesting a possible 2026-07-12 family date was
based on US12182662B2's uncapped arithmetic and is superseded.

**Current implementation scope:** only the core 1D method from US8832139B2
is implemented — no multidimensional extensions, no bidirectional modulation,
no "quantum computer" stack. This is now a **scope** decision rather than a
legal one: the forensics above confirm the wider claims are open, and
expanding into them is planned work.

## Disclaimer

This is a classical FFT signal-processing implementation. The term
"quantum" in the patent literature is used metaphorically to describe the
mathematical properties of the algorithm (superposition, interference).
This library does NOT use quantum computing hardware, quantum bits, or
any quantum-mechanical effects.

**This is not legal advice.** The expiration analysis here is derived from
the faces of the issued grant PDFs and the controlling statute and case law,
but maintenance-fee status is not independently verified. Users should
consult patent counsel before commercial deployment.
