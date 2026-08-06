# selly-fft

A classical, FFT-based associative memory search library implementing the
methodology described in US Patent 8,832,139 B2 ("Associative memory and
data searching system and method") — the core claims of which are now
**expired** (adjusted expiration: 2026-06-02).

> ⚠️ **The term "quantum" in the patents is metaphorical.** This is a
> classical signal-processing algorithm using NumPy's FFT routines. No
> quantum computing hardware or quantum-mechanical effects are involved.

## What It Does

Searches for a short query sequence ("probe") within a longer reference
sequence ("target") using FFT-accelerated normalized cross-correlation.
Each symbol is encoded as a complex phasor on the unit circle
(`exp(2j·π·k/L)`), and the correlation is computed in the frequency domain
via the convolution theorem, giving **O((n+m) log(n+m))** complexity
instead of the naive O(n·m).

Two encodings are available, auto-selected by alphabet size:

* **Unit-circle phasor** (`encode_unit_circle`) — the patent-faithful
  encoding.  Exact for alphabets of 4 or fewer symbols, where symbols
  sit ≥90° apart (e.g. DNA, A/T/C/G at 90°).  For larger alphabets the
  angular separation shrinks below 90° and adjacent symbols correlate
  (at 8 symbols a total non-match scores 0.707!), so the
  library **switches automatically** to the sharp path below.
* **One-hot (orthogonal)** (`encode_orthogonal`) — used for large
  alphabets (text).  Distinct symbols are exactly orthogonal, so a total
  non-match scores **0.0** regardless of alphabet size.  This is the
  path that makes text search as sharp as DNA search.

### Key correction over the published design spec

The Run-112 design spec described a unit-circle encoding with *circular*
FFT correlation and `sqrt(len)` normalization.  This produces
**content-independent peak magnitudes** — by Parseval's theorem, the
circular cross-correlation of two constant-modulus sequences conserves
energy regardless of symbol content, so an exact match and a total
non-match both score 0.707.

This library fixes that bug with three changes (documented in
`Run-126-Selly-FFT-Implementation-Build.md`):

1. **Linear (not circular) correlation** — zero-padded to
   `len(ref) + len(query) - 1`, valid region only.
2. **Correct FFT formula** — `ifft(conj(FFT(probe)) · FFT(ref))`.
3. **Match-filter normalization + real-part scoring** —
   `Re(corr) / (||probe|| · ||ref_window||)` = `cos(Δθ)` per symbol,
   yielding exactly 1.0 for a match, 0.0 for a total non-match.

### Large-alphabet (text) sharpness

The earlier implementation packed every symbol onto a single circle, so
`"AAAAAAAAAA"` vs `"BBBBB"` scored **0.985** under the 36-symbol default
alphabet — a non-match read as a near-perfect hit, and lowercase/punctuation
raised `ValueError`.  The one-hot path fixes this: the normalized
cross-correlation becomes the exact **fraction of matching positions**,
independently of alphabet size.  See `tests/test_text.py`.

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from selly_fft import holographic_match, DNAAssociativeMemory

# Functional API — one-shot match (probe first, reference second)
score = holographic_match("ACGT", "ACGTACGT")
print(f"match score: {score}")  # → 1.0 (exact match)

# DNA subclass — find all occurrences
mem = DNAAssociativeMemory(threshold=0.7)
matches = mem.find_matches("ATCG", "GGGGAATCGGGG")
for m in matches:
    print(f"position {m.position}: score {m.score:.4f}")
# → position 5: score 1.0

# Search with a pre-encoded target (for repeated queries)
target = mem.encode_target(list("ATCGATCGATCG"))
probe = list("ATCG")
results = mem.search(probe, target)
```

### Text / large-alphabet search

The same API works for text. `TextAssociativeMemory` (and the functional
`text_match`) use exact one-hot encoding, so discrimination is independent
of alphabet size — matching the DNA path's sharpness. Lowercase, spaces,
and punctuation are handled; matching is case-insensitive by default.

```python
from selly_fft import TextAssociativeMemory, text_match

mem = TextAssociativeMemory()  # case-insensitive by default
for m in mem.find_matches("brown", "THE QUICK BROWN FOX"):
    print(m.position, m.score)          # → 10 1.0

# One-shot
text_match("world", "Hello, World!")    # → [Match(position=7, score=1.0, ...)]

# Arbitrary Unicode — derive the alphabet from your data
mem.build_alphabet("un café naïve 日本語")
mem.find_matches("café", "UN CAFÉ NAÏVE")  # → [Match(position=3, score=1.0, ...)]
```

`holographic_match` auto-selects the sharp path for alphabets larger than
4 symbols (e.g. `holographic_match("World", "Hello, World!") → 1.0`, and
a total non-match like `"BBBBB"` vs `"AAAAAAAAAA"` → `0.0`).

## DNA/RNA Sequence Matching

The patents' primary application is DNA/RNA sequence matching. The
`DNAAssociativeMemory` class uses a 4-phased encoding where A, T, C, G
are placed 90° apart on the unit circle, ensuring maximal
discriminability:

| Base | Phasor | Angle |
|------|--------|-------|
| A    | `1 + 0j` | 0°    |
| C    | `0 + 1j` | 90°   |
| G    | `-1 + 0j`| 180°  |
| T    | `0 - 1j` | 270°  |

With this encoding:
- Exact match → score 1.0
- Orthogonal mismatch (e.g. A vs C, 90° apart) → score 0.0
- Antipodal mismatch (e.g. A vs G, 180° apart) → score 0.0 (cos(180°) = −1, clipped)

## API Reference

### `holographic_match(probe, reference, *, alphabet, threshold) -> float`
One-shot best match score in [0, 1], unthresholded. **Changed in 0.2.0:**
argument order is now probe-first, consistent with `text_match` /
`dna_match` / `find_matches` (0.1.x was `(reference, query)`).

### `SellyAssociativeMemory(alphabet, threshold)`
Core FFT search engine with `encode()`, `encode_target()`, `encode_probe()`,
`search()`, `search_direct()`, `best_score()`.  Alphabets must be non-empty
and duplicate-free (validated at construction).

### `DNAAssociativeMemory(threshold)`
DNA/RNA-specialized subclass with `find_matches(probe, target)`.

### `TextAssociativeMemory(case_sensitive, alphabet, threshold, dtype)`
Text / large-alphabet subclass using one-hot encoding for exact
discrimination.  Adds `find_matches(probe, target)`, `build_alphabet(*texts)`,
and case-folding.  Functional shortcut: `text_match(probe, reference)`.
Input is NFC-normalized (composed and decomposed Unicode forms match).
`dtype=np.float32` halves the encoded memory footprint (752 → 376
bytes/char with the default 94-symbol alphabet).  Note: case folding can
change string length (`"ß"` → `"SS"`); match positions are reported in the
normalized string's coordinates.

### `Match(position, score, significance)`
Dataclass for match results.

## v0.4.0 — Convergence Telemetry

**New module:** `selly_fft.telemetry` — instruments the research process
itself. Extracts key claims from each run document, probes them against
the concatenated text of all prior runs using `search_many` (shared FFT
acceleration), and produces a **convergence curve** measuring whether the
research spiral is converging (C) or fragmenting (D).

This is a direct application of `dE/dt = β(C − D)E`: rising coherence =
constructive interference = C; scatter = destructive interference = D.

```python
from selly_fft import telemetry

# Load runs from a directory (handles both Run-01-First.md and
# 2026-07-30-Run-01-First.md naming patterns)
texts, labels = telemetry.load_runs_from_directory("/path/to/research-runs")

# Compute per-run coherence against all prior runs
run_results = telemetry.run_telemetry(texts, max_claims=15)

# Aggregate into curve + outlier detection + trend classification
agg = telemetry.aggregate_telemetry(run_results, run_labels=labels)
print(agg.convergence_curve)  # [0.0, 0.153, 0.367, 0.744, ...]
print(agg.trend)              # "rising" / "falling" / "flat" / "insufficient"
print(agg.outliers)           # [index, ...] — runs that drop coherence sharply
print(agg.coherence_mean)     # mean of the curve (excl. run 0)
```

**Scoring convention (explicit):** all scores use real-part normalized
cross-correlation with `threshold=0.0`. 1.0 = exact phrase overlap,
0.0 = no alignment, partial = fraction of matching positions. The
`threshold=0.0` flag ensures partial matches still register — this is
a coherence *curve*, not a significance gate.

**Claim extraction** is deterministic and LLM-independent: Markdown is
stripped, sentences are split into clauses on commas/conjunctions, and
clauses longer than `MAX_CLAIM_WORDS` (6) are broken into overlapping
6-word sliding windows for phrase-granularity scoring.

See `Run-133-Convergence-Telemetry-Module.md` and the docstring on each
public function for details.

## v0.3.0 Features

**`find_spans(probe, target)`** — like `find_matches`, but returns `Span`
objects carrying the matched *text* and coordinates in your **original
string**, correctly mapped even when normalization changed the length
(`"ß"` → `"SS"`, NFC). Every hit can be pointed at in your own text.

**`threshold="auto"`** — significance-gated reporting. Instead of a fixed
score floor, a position is reported only if its match count is
statistically significant under the exact binomial null
(`p ≤ AUTO_P = 1e-3`). Calibrated for short probes: a single chance
symbol in a 6-char probe is correctly *not* reported. Note: on large
alphabets even low-scoring spans can be significant (6/29 matches,
score 0.2, is p≈1e-6 by chance) — **`auto` surfaces significant
*partial* matches, not just near-exact ones.** Combine with a score
floor (`threshold=0.5`) if you only want strong matches.

**`search_many(probes, target_encoded)`** — batch search sharing the
target's channel FFTs across all probes (one transform per channel
total). The economical shape for scanning many probes over one corpus.

**`dtype=np.uint8`** — encodes the one-hot target at **1 byte/char/channel**
(94 bytes/char with the default alphabet, vs 752 for float64).

**CLI** — `selly scan PROBE FILE... [--threshold 0.9|auto]
[--case-sensitive] [--float32|--uint8] [--build-alphabet] [--context N]`.
Grep-like exit codes (0 = found, 1 = none, 2 = error).

```bash
selly scan "brentwood protocol" docs/ --threshold auto --context 40
```

## Benchmarks

FFT vs naive O(n·m) correlation (probe_len=128, genome=10000):

| Method | Time     | Speedup |
|--------|----------|---------|
| FFT    | 9.8 ms   | 3.4×    |
| Naive  | 33.4 ms  | —       |

With probe_len=512: **13.8× speedup**. The crossover point depends on
probe length and genome size; the FFT method is never slower than
O((n+m) log(n+m)).

## Patent Status

See [PATENTS.md](PATENTS.md) for the full claim-mapping analysis. In
short: the **entire Selly patent family expired 2026-06-02**. The
continuations (US10438690B2, US11561951B2, US12182662B2) each carry a
terminal disclaimer running to the common parent US8832139B2, which caps
them regardless of Patent Term Adjustment (35 U.S.C. § 154(b)(2)(B);
*In re Cellect*, 81 F.4th 1216 (Fed. Cir. 2023)). Aggregator sites still
publish uncapped 2029/2028 dates that ignore those disclaimers.

This library currently implements only the parent's core 1D FFT method.
The refinements and multidimensional extensions claimed by the
continuations are **open to implement** — they are outside this library's
present scope, not behind a legal fence.

Maintenance-fee status is carried only by USPTO systems and has not been
independently verified; this is not legal advice.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project is guided by the
principle `dE/dt = β(C − D)E` — maximize cooperation (C), minimize
division (D).
