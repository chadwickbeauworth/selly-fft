"""Stress test for selly-fft text / large-alphabet path."""
import sys, time, random
import numpy as np
from selly_fft import (
    TextAssociativeMemory, text_match, holographic_match,
    encode_orthogonal, normalized_xcorr_multichannel,
)

random.seed(12345)
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

AB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# ---- 1. Core sharpness: total non-match must be 0 ----
m = TextAssociativeMemory(case_sensitive=False)
check("nonmatch_AxA_vs_B", m.find_matches("BBBBB", "AAAAAAAAAA") == [], "was 0.985 under old encoding")
check("nonmatch_holographic", holographic_match("AAAAAAAAAA", "BBBBB") == 0.0)

# ---- 2. Exact match at correct position ----
ms = m.find_matches("brown", "THE QUICK BROWN FOX")
check("exact_pos", [(x.position, x.score) for x in ms] == [(10, 1.0)], str(ms))

# ---- 3. Partial match = exact fraction ----
ms = m.find_matches("BROWX", "THE BROWN FOX")   # 4 of 5
check("partial_0.8", ms and abs(ms[0].score - 0.8) < 1e-9, str(ms[0].score if ms else None))

# ---- 4. Multiple occurrences ----
ms = m.find_matches("AB", "AB AB AB")
check("multi_pos", [x.position for x in ms] == [0, 3, 6], str([x.position for x in ms]))

# ---- 5. Case sensitivity ----
mc = TextAssociativeMemory(case_sensitive=True)
check("case_sensitive_distinguishes", mc.find_matches("World", "hello world") and mc.find_matches("World", "hello world")[0].score < 1.0)
check("case_insensitive_default", TextAssociativeMemory().find_matches("WORLD", "hello world"))

# ---- 6. Unicode via build_alphabet ----
mu = TextAssociativeMemory(case_sensitive=False).build_alphabet("UN CAFE NAIVE 日本語".replace("AFE", "AFÉ").replace("AIVE", "AÏVE"))
ms = mu.find_matches("café", "UN CAFÉ NAÏVE")
check("unicode", ms and ms[0].position == 3 and abs(ms[0].score - 1.0) < 1e-9, str(ms[0] if ms else None))

# ---- 7. Cross-check against brute force on random text ----
def brute(ref, q, ab):
    L = len(q)
    return [round(sum(1 for i in range(L) if ref[k + i] == q[i]) / L, 9)
            for k in range(len(ref) - L + 1)]
ok = True
for _ in range(300):
    ref = "".join(random.choice(list(AB)) for _ in range(40))
    q = "".join(random.choice(list(AB)) for _ in range(4))
    got = np.round(normalized_xcorr_multichannel(encode_orthogonal(list(q), AB),
             encode_orthogonal(list(ref), AB)), 9)
    if not np.allclose(got, np.array(brute(ref, q, AB)), atol=1e-9):
        ok = False
        break
check("bruteforce_300_random", ok)

# ---- 8. Edge cases ----
check("empty_query", m.find_matches("", "abc") == [])
check("empty_ref", m.find_matches("abc", "") == [])
check("query_longer_than_ref", m.find_matches("abcdef", "ab") == [])

# ---- 9. Out-of-alphabet symbol in PROBE must raise ----
try:
    m.find_matches("€", "hello")   # '€' not in default TEXT_ALPHABET
    check("probe_oob_raises", False, "did not raise")
except ValueError:
    check("probe_oob_raises", True)

# ---- 10. Out-of-alphabet symbol in TARGET is tolerated (zero-matches) ----
ms = m.find_matches("abc", "a€bc")   # target has '€' not in alphabet
check("target_oob_tolerated", ms == [] or all(x.score >= 0 for x in ms), str(ms))

# ---- 11. Threshold gating ----
ms = m.find_matches("BROWX", "THE BROWN FOX", threshold=0.9)
check("threshold_gate", ms == [], "partial 0.8 below 0.9 threshold")

# ---- 12. Scaling / performance ----
def bench(ref_n, q_n):
    ref = "".join(random.choice(list(AB)) for _ in range(ref_n))
    q = "".join(random.choice(list(AB)) for _ in range(q_n))
    mm = TextAssociativeMemory()
    enc = mm.encode_target(ref)
    t = time.perf_counter()
    mm.search(list(q), enc)
    return (time.perf_counter() - t) * 1000
for n in [2000, 10000, 50000]:
    ms_t = bench(n, 128)
    check(f"latency_ref{n}", ms_t < (n / 1000) * 5 + 50, f"{ms_t:.1f}ms")

# ---- 13. Adversarial: high-entropy near-misses ----
ref = "XKCDZQWPLM"
ms = m.find_matches("XKCDZQWPLN", ref)   # last char differs
check("singleton_nearmiss", ms and abs(ms[0].score - 9 / 10) < 1e-9, str(ms[0].score if ms else None))

# ---- 14. Significance heuristic sanity (see known issue) ----
sig = TextAssociativeMemory()._significance(0.5, 10)
check("significance_runs", isinstance(sig, (int, float)) and sig == sig, f"sig={sig:.3f} (note: uses sqrt(36) not 94)")

print("\n=== SUMMARY ===")
fails = [n for n, c, _ in results if not c]
print(f"{len(results) - len(fails)}/{len(results)} passed")
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
sys.exit(0)
