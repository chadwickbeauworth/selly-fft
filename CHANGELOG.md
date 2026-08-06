# Changelog

All notable changes to selly-fft are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-08-06

### Added

- **`selly_fft.telemetry` module** — convergence telemetry for research
  runs. Instruments the research process itself: extract key claims from
  each run, probe them against the concatenated text of all prior runs
  using FFT-accelerated `search_many`, and produce a convergence curve
  measuring whether the research spiral is converging (C) or fragmenting
  (D). Implements the `dE/dt = β(C−D)E` principle directly as a metric.

  Public API:
  - `run_telemetry(runs, *, max_claims=15, claim_extractor, case_sensitive, alphabet) -> List[RunTelemetry]`
  - `convergence_curve(runs, **kwargs) -> List[float]`
  - `aggregate_telemetry(run_telemetry_list, run_labels) -> TelemetryResult`
  - `load_runs_from_directory(directory, pattern, max_runs, include_date_prefixed) -> (texts, labels)`
  - `default_claim_extractor(max_claims) -> ClaimExtractor`
  - `RunTelemetry` / `TelemetryResult` dataclasses
  - `MAX_CLAIM_WORDS` constant (default 6 — claims longer than 6 words are
    split into overlapping sliding-window key phrases for phrase-granularity
    match-fraction scoring)

  **Scoring convention (explicit):** all telemetry scores use real-part
  normalized cross-correlation with `threshold=0.0` (best unthresholded
  score per claim). 1.0 = exact phrase overlap, 0.0 = no alignment,
  partial = fraction of matching positions.

  Includes 38 tests covering claim extraction, coherence scoring,
  outlier detection, trend classification, and directory loading.

### Changed

- `_BOILERPLATE_PAT` regex in the new telemetry module was initially
  broken (empty alternative matched everything) — fixed before tests
  would pass. This bug only existed in the pre-test development cycle
  and was never released.

### Tests

- 36 new tests in `tests/test_telemetry.py` (total: 162, up from 124).
- Full suite: `162 passed in 1.39s`.

## [0.3.0] — 2026-08-04

### Added

- `dtype=np.uint8` support for `TextAssociativeMemory` (94 bytes/char vs
  752 for float64 with the default 94-symbol alphabet).
- `dtype=np.float32` support (halves memory footprint).
- `threshold="auto"` significance-gated reporting using exact binomial
  null (p ≤ 1e-3), calibrated for short probes.
- `find_spans(probe, target)` returning `Span` objects with original-text
  coordinate mapping (handles Unicode normalization length changes).
- `search_many(probes, target_encoded)` batch search sharing the target's
  channel FFTs across all probes.
- CLI: `selly scan PROBE FILE... [--threshold 0.9|auto] [--case-sensitive]
  [--float32|--uint8] [--build-alphabet] [--context N]` with grep-like
  exit codes (0 = found, 1 = none, 2 = error).

### Fixed

- `SHARP_ENCODING_THRESHOLD` corrected from 8 to 4 (unit-circle encoding
  is only exact for alphabets of ≤ 4 symbols at 90° separation).
- `holographic_match` argument order unified to `(probe, reference)` —
  was `(reference, query)` in 0.1.x.
- NFC normalization applied to explicit alphabets.
- Signficance z-scores use encoding-aware null models (binomial for
  one-hot text, unit-circle for phasor DNA).
- `search()` now explicitly rejects raw string targets with a helpful
  `TypeError` pointing at `search_direct()`.

## [0.2.0] — 2026-08-04

### Added

- One-hot (orthogonal) encoding for large-alphabet text search —
  `TextAssociativeMemory`, `text_match`, `encode_orthogonal`.
- Case-insensitive matching with Unicode NFC normalization.
- `normalized_xcorr_multichannel_batch` for efficient batch search.

### Fixed

- Run-112 Parseval bug: switched from circular correlation with
  `sqrt(len)` normalization to linear correlation with match-filter
  normalization + real-part scoring.

## [0.1.0] — 2026-06-02

Initial release implementing the core 1D FFT associative memory
methodology from US8832139B2 (expired 2026-06-02). Unit-circle phasor
encoding, circular correlation.
