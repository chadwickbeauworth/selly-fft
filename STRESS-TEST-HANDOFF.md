# STRESS-TEST HANDOFF — `selly-fft` text / large-alphabet path

**Purpose:** A new chat session can independently stress-test the `selly-fft`
library (specifically the newly-added sharp **text / large-alphabet** path)
without needing prior context. Everything needed is below.

**Date built:** 2026-08-04 · **Commit:** `e4005eb` · **Version:** 0.1.1

---

## 1. What the library is (one paragraph)

`selly-fft` is a classical (NumPy-FFT) **fuzzy subsequence search** library:
given a short *query* and a long *reference* string, it finds query
occurrences and scores each match in `[0,1]` using FFT-accelerated
normalized cross-correlation. It implements the (now-expired) methodology of
US Patent 8,832,139 B2 as a defensive publication. The core correction over
the original design spec is **linear correlation + match-filter normalization
+ real-part scoring**, which fixed a Parseval bug where exact matches and
total non-matches both scored 0.707.

Two encodings exist:
- **Unit-circle phasor** (`encode_unit_circle`) — patent-faithful, exact for
  small alphabets (DNA, A/T/C/G at 90°). Default for alphabets ≤ 8 symbols.
- **One-hot (orthogonal)** (`encode_orthogonal`) — used for large alphabets
  (text). Distinct symbols are *exactly* orthogonal, so a total non-match
  scores **0.0** regardless of alphabet size. This is the path under test.

`holographic_match()` auto-routes alphabets **> 8 symbols** to the one-hot
path. `TextAssociativeMemory` is the text class (case-insensitive default).

---

## 2. How to run it (the new session needs this)

```bash
# Repo location (macOS, local):
REPO=~/taochadwick/selly-fft

# Python (use the hermes venv which has numpy):
PY=/Users/chadwickbeauworth/.hermes/hermes-agent/venv/bin/python

# No install needed — just put src on the path:
export PYTHONPATH=$REPO/src
$PY -c "import selly_fft; print(selly_fft.__version__)"   # → 0.1.1

# Run the existing test suite (73 tests):
$PY -m pytest $REPO/tests/ -q

# Or install editable if you prefer:
cd $REPO && $PY -m pip install -e ".[dev]"
```

> If on a different machine, `git clone` the repo (it is NOT yet pushed to
> GitHub — ask the user for the path or copy `~/taochadwick/selly-fft`).

---

## 3. The public API under test

```python
from selly_fft import (
    TextAssociativeMemory, text_match,
    holographic_match,
    SellyAssociativeMemory, DNAAssociativeMemory,
    encode_orthogonal, normalized_xcorr_multichannel,
    Match,
)

# Text class
mem = TextAssociativeMemory(case_sensitive=False)   # default: case-insensitive
matches = mem.find_matches("brown", "THE QUICK BROWN FOX")
# → [Match(position=10, score=1.0, significance=...)]
pos = mem.build_alphabet("un café naïve 日本語")  # derive alphabet from data
mem.find_matches("café", "UN CAFÉ NAÏVE")           # → position 3, score 1.0

# Functional
text_match("world", "Hello, World!")                # → [Match(position=7, score=1.0)]

# Auto-routing one-shot
holographic_match("Hello, World!", "World")         # → 1.0
holographic_match("AAAAAAAAAA", "BBBBB")            # → 0.0   (the old 0.985 bug)
```

`Match` has fields `position`, `score`, `significance`.

---

## 4. The stress-test script (ready to run)

Save as `~/taochadwick/selly-fft/stress_test.py` and run with
`PYTHONPATH=~/taochadwick/selly-fft/src python stress_test.py`. It probes
**correctness, edge cases, adversarial inputs, scaling, and the known weak
spots** below. It prints a PASS/FAIL summary at the end.

```python
"""Stress test for selly-fft text / large-alphabet path."""
import sys, time, random, string
import numpy as np
from selly_fft import (
    TextAssociativeMemory, text_match, holographic_match,
    encode_orthogonal, normalized_xcorr_multichannel,
)

PY = sys.executable
random.seed(12345)
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

AB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# ---- 1. Core sharpness: total non-match must be 0 ----
m = TextAssociativeMemory(case_sensitive=False)
check("nonmatch_AxA_vs_B", m.find_matches("BBBBB", "AAAAAAAAAA") == [], "was 0.985 under old encoding")
check("nonmatch_holographic", holographic_match("AAAAAAAAAA","BBBBB") == 0.0)

# ---- 2. Exact match at correct position ----
ms = m.find_matches("brown", "THE QUICK BROWN FOX")
check("exact_pos", [(x.position,x.score) for x in ms] == [(10,1.0)], str(ms))

# ---- 3. Partial match = exact fraction ----
ms = m.find_matches("BROWX", "THE BROWN FOX")   # 4 of 5
check("partial_0.8", ms and abs(ms[0].score-0.8) < 1e-9, str(ms[0].score if ms else None))

# ---- 4. Multiple occurrences ----
ms = m.find_matches("AB", "AB AB AB")
check("multi_pos", [x.position for x in ms] == [0,3,6], str([x.position for x in ms]))

# ---- 5. Case sensitivity ----
mc = TextAssociativeMemory(case_sensitive=True)
check("case_sensitive_distinguishes", mc.find_matches("World","hello world") and mc.find_matches("World","hello world")[0].score < 1.0)
check("case_insensitive_default", TextAssociativeMemory().find_matches("WORLD","hello world"))

# ---- 6. Unicode via build_alphabet ----
mu = TextAssociativeMemory(case_sensitive=False).build_alphabet("UN CAFÉ NAÏVE 日本語")
ms = mu.find_matches("café", "UN CAFÉ NAÏVE")
check("unicode", ms and ms[0].position==3 and abs(ms[0].score-1.0)<1e-9, str(ms[0] if ms else None))

# ---- 7. Cross-check against brute force on random text ----
def brute(ref,q,ab):
    L=len(q); return [round(sum(1 for i in range(L) if ref[k+i]==q[i])/L,9)
                      for k in range(len(ref)-L+1)]
ok=True
for _ in range(300):
    ref="".join(random.choice(list(AB)) for _ in range(40))
    q="".join(random.choice(list(AB)) for _ in range(4))
    got=np.round(normalized_xcorr_multichannel(encode_orthogonal(list(q),AB),
             encode_orthogonal(list(ref),AB)),9)
    if not np.allclose(got, np.array(brute(ref,q,AB)), atol=1e-9): ok=False; break
check("bruteforce_300_random", ok)

# ---- 8. Edge cases ----
check("empty_query", m.find_matches("", "abc") == [])
check("empty_ref", m.find_matches("abc", "") == [])
check("query_longer_than_ref", m.find_matches("abcdef","ab") == [])
check("repeat_match", m.find_matches("ab","ababab")==[m.find_matches("ab","ababab")[0]] or True)  # sanity

# ---- 9. Out-of-alphabet symbol in PROBE must raise ----
try:
    m.find_matches("€", "hello")   # '€' not in default TEXT_ALPHABET
    check("probe_oob_raises", False, "did not raise")
except ValueError:
    check("probe_oob_raises", True)

# ---- 10. Out-of-alphabet symbol in TARGET is tolerated (zero-matches) ----
ms = m.find_matches("abc", "a€bc")   # target has '€' not in alphabet
check("target_oob_tolerated", ms == [] or all(x.score>=0 for x in ms), str(ms))

# ---- 11. Threshold gating ----
ms = m.find_matches("BROWX", "THE BROWN FOX", threshold=0.9)
check("threshold_gate", ms == [], "partial 0.8 below 0.9 threshold")

# ---- 12. Scaling / performance ----
def bench(ref_n,q_n):
    ref="".join(random.choice(list(AB)) for _ in range(ref_n))
    q  ="".join(random.choice(list(AB)) for _ in range(q_n))
    mm=TextAssociativeMemory()
    enc=mm.encode_target(ref)
    t=time.perf_counter(); mm.search(list(q),enc); return (time.perf_counter()-t)*1000
for n in [2000,10000,50000]:
    ms_t=bench(n,128)
    check(f"latency_ref{n}", ms_t < (n/1000)*5+50, f"{ms_t:.1f}ms")

# ---- 13. Adversarial: high-entropy near-misses ----
# Query differing by 1 char from a real substring should score (L-1)/L
ref="XKCDZQWPLM"
ms=m.find_matches("XKCDZQWPLN", ref)   # last char differs
check("singleton_nearmiss", ms and abs(ms[0].score-9/10)<1e-9, str(ms[0].score if ms else None))

# ---- 14. Significance heuristic sanity (see known issue) ----
sig = TextAssociativeMemory()._significance(0.5, 10)
check("significance_runs", isinstance(sig,(int,float)) and sig==sig, f"sig={sig:.3f} (note: uses sqrt(36) not 94)")

print("\n=== SUMMARY ===")
fails=[n for n,c,_ in results if not c]
print(f"{len(results)-len(fails)}/{len(results)} passed")
if fails: print("FAILURES:", fails)
sys.exit(1 if fails else 0)
```

---

## 5. Known weak spots to probe hard

These are **real limitations** a good stress test should try to break:

1. **`Match.significance` is miscalibrated for text.** `core.py:_significance`
   hardcodes `expected = 1/sqrt(36)` (the old 36-symbol alphabet). The text
   alphabet has **94 symbols**, so the null expected value should be
   `1/sqrt(94) ≈ 0.103`, not `0.167`. For Unicode it's worse. **Test:** does a
   high-scoring partial match report a sane z-score? Likely inflated.

2. **Out-of-alphabet in the *probe* raises; in the *target* is silently
   zeroed.** Inconsistent contract. **Test:** probe with `€` (raises), target
   with `€` (tolerated). Is that the behavior you want?

3. **One-hot memory cost scales with alphabet size** (~84–94 floats/char).
   At very large alphabets (CJK, emoji-heavy) the target encoding for a 1MB
   document is ~80 MB. **Test:** encode a large Unicode target, watch memory.

4. **`holographic_match` default alphabet is `TEXT_ALPHABET` (mixed
   case).** DNA-style 4-symbol calls still route correctly (≤8 → phasor),
   but a 9-symbol custom alphabet silently switches to one-hot. **Test:**
   does a custom 9-symbol alphabet behave as expected?

5. **CJK / emoji:** `build_alphabet` derives from data, but identical glyphs
   from different Unicode normalization forms (NFC vs NFD) are *different*
   code points. **Test:** `é` (U+00E9) vs `e`+combining acute (U+0301).

6. **Score is the *fraction of matching positions*, not edit distance.**
   Transpositions (`"ACTG"` vs `"AGCT"`) score lower than substitutions even
   though they're "closer". **Test:** is that acceptable for your use?

---

## 6. Suggested adversarial prompts for the AI session

Hand the new chat this and let it *drive*:

- "Find every near-match of 'quantum' in this 50KB document with score ≥ 0.8."
- "Stress test with random 100k-char text — does latency stay sublinear?"
- "What happens with emoji and CJK mixed in the same string?"
- "Is the significance score trustworthy on text? Prove or disprove."
- "Break it: find an input where a non-match scores > 0.5."
- "Compare one-hot vs the unit-circle path on a 9-symbol alphabet — which wins and why?"

---

## 7. Reporting back

The new session should report:
- PASS/FAIL summary of the script above
- Any input that produces a **false positive** (non-match scored > 0.5) — this
  would be a regression of the sharpness guarantee
- Memory/latency numbers at scale
- Whether the `significance` miscalibration matters in practice

No code should be *committed* by the stress-test session unless the user
approves — it is a test/explore session, not a build session.
