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

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from selly_fft import holographic_match, DNAAssociativeMemory

# Functional API — one-shot match
score = holographic_match("ACGTACGT", "ACGT")
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

### `holographic_match(reference, query, *, alphabet, threshold) -> float`
One-shot best match score in [0, 1].

### `SellyAssociativeMemory(alphabet, threshold)`
Core FFT search engine with `encode()`, `encode_target()`, `encode_probe()`,
`search()`, `search_direct()`.

### `DNAAssociativeMemory(threshold)`
DNA/RNA-specialized subclass with `find_matches(probe, target)`.

### `Match(position, score, significance)`
Dataclass for match results.

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
short: the core 1D FFT method from US8832139B2 is expired and safe to
implement. Active continuations (US10438690B2, US11561951B2) cover
refinements and multidimensional extensions that this library does
**not** implement.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project is guided by the
principle `dE/dt = β(C − D)E` — maximize cooperation (C), minimize
division (D).
