# selly-fft Flows — A Usage Playbook

How to plug selly-fft into research runs, kanban lanes, cron jobs, and the
knowledge-graph work. Every recipe here is tested behavior, not aspiration.

**The one-sentence mental model:** selly-fft slides a probe across a target
and scores *every* position in [0,1] — exact match 1.0, total non-match 0.0,
partial = fraction of matching positions. It is a **fuzzy span-detector with
a sharpness guarantee**, not a search engine.

---

## 0. Choosing the right entry point

| Shape of the job | Use |
|---|---|
| One number: "how well does Q match T?" | `holographic_match(probe, reference)` |
| All hits in one string, scores only | `mem.find_matches(probe, target)` |
| All hits **with the actual text + original positions** | `mem.find_spans(probe, target)` |
| Many probes × one corpus | `mem.search_many(probes, target_enc)` |
| Shell / cron / kanban lane, no Python | `selly scan PROBE FILES...` |
| DNA/RNA motifs | `DNAAssociativeMemory` / `dna_match` |

**Tuning knobs that matter:**

| Knob | When |
|---|---|
| `threshold=0.7–0.8` | fuzzy phrase spotting (1–3 typos in a 15–25 char probe) |
| `threshold="auto"` | autonomous/unattended runs — statistics picks the cutoff (exact binomial, p ≤ 1e-3). **Default choice for lanes and cron.** |
| `dtype=np.uint8` | corpus > ~500 KB (94 bytes/char vs 752) |
| `dtype=np.float32` | middle ground, slight accuracy margin |
| `build_alphabet(*texts)` | any non-ASCII corpus (accents, CJK, emoji) |
| `case_sensitive=True` | identifiers, code, DNA — never prose |

---

## 1. Kanban research lanes — the fleet pre-filter

**Problem:** lanes burn LLM tokens reading corpus text that is 95%
irrelevant. **Recipe:** scan locally, send only hit spans + context to the
fleet.

```python
from selly_fft import TextAssociativeMemory
import numpy as np

mem = TextAssociativeMemory(dtype=np.uint8)          # big corpora
target_enc = mem.encode_target(corpus_text)          # pay once

probes = ["holographic associative memory", "wavefunction encoding",
          "superposition representation", "correlation fingerprint"]
hits_per_probe = mem.search_many(probes, target_enc, threshold="auto")

for probe, hits in zip(probes, hits_per_probe):
    for h in hits:
        lo, hi = max(0, h.position - 200), h.position + len(probe) + 200
        send_to_fleet(probe, h.score, corpus_text[lo:hi])   # LLM sees only this
```

**Why it works:** `search_many` shares the target FFTs across all probes, so
20 probes cost little more than one. `auto` threshold means the lane never
drowns in chance partials. **Verified cost:** ~0.4 s per 1 MB per probe
(float64; less in uint8 batch).

## 2. Iterative research runs — convergence telemetry

**Problem:** "is run N converging or wandering?" was a judgment call.
**Recipe:** probe the run's key claims against all prior run outputs; track
mean best-score over time.

```python
claims = extract_key_claims(run_n_output)            # 5-10 short phrases
prior = "\n".join(all_prior_run_outputs)

mem = TextAssociativeMemory()
enc = mem.encode_target(prior)
scores = [m[0].score if m else 0.0
          for m in mem.search_many(claims, enc, threshold=0.0)]
convergence = sum(scores) / len(scores)
# log per run; rising curve = spiral converging, flat = wandering
```

Use the best-score-per-claim (`search_many` then max) rather than a fixed
threshold — you want the *curve*, not a hit count.

## 3. Checkpoint re-anchoring (session continuity)

**Problem:** checkpoint files preserve context across lane reclaims, but
exact strings drift between runs. **Recipe:** fuzzy-relocate a checkpoint's
key phrases in earlier outputs.

```python
mem = TextAssociativeMemory()
spans = mem.find_spans(checkpoint_key_phrase, earlier_run_output,
                       threshold=0.8)
if spans:
    anchor = spans[0].orig_start          # position in the ORIGINAL text
```

`find_spans` is the right tool here (not `find_matches`) because positions
are in the original string even if the text had Unicode that folded.

## 4. Patent / knowledge-graph runs

**a) Claim-phrase scanning** — harvest patent full texts, scan for claim
language across the collection:

```bash
selly scan "associative memory" patents/*.txt --threshold auto --context 80
```

**b) Near-duplicate claim detection** — probe each patent's independent
claims against every other patent's text. Boilerplate *should* score high;
the `significance` z-score separates meaningful echo from chance.
Claims that reappear across patents with score ≥ 0.9 are your
prior-art resonance candidates.

**c) Motif census** — build a fixed probe list once (the "motif
vocabulary" of the KG), then `search_many` every new document as it
arrives. This is the scalable shape: N docs × K probes, each doc encoded
once (uint8), K probes batched.

## 5. Obsidian vault weave

**Near-duplicate / link discovery:**

```python
mem = TextAssociativeMemory()
for note_a, note_b in candidate_pairs(vault):        # cheap pre-filter by length
    score = holographic_match(longest_paragraph(note_a), note_b)
    if score > 0.8:
        suggest_link_or_merge(note_a, note_b)
```

Paragraphs that keep reappearing unplanned are usually load-bearing ideas —
candidate axioms for the graph.

## 6. Cron / watchdog — incoming document monitor

The CLI is the whole integration. Example: scan new arXiv abstracts daily
for KG motifs, silent when nothing significant (watchdog pattern):

```bash
selly scan "topological data storage" ~/feeds/arxiv-new/*.txt \
    --threshold auto --build-alphabet --max-per-file 5
```

Exit code 1 = nothing found = silence. Exit 0 = hits on stdout = the
notification. Wrap in a `no_agent` cron script and it only speaks when
there is signal.

## 7. DNA / motif work

```python
from selly_fft import dna_match
hits = dna_match("GATTACA", genome, threshold=0.85)   # allows 1 mismatch in 7
```

The sharpest path (orthogonal/antipodal bases). `threshold="auto"` works
here too, with honest statistics: a 7-mer exact match in random DNA is
chance-plausible; a 12-mer is not — the gate knows the difference.

---

## 8. Capacity planner (verified numbers)

| Corpus size | dtype | Encoded memory | ~Scan time/probe |
|---|---|---|---|
| 100 KB | float64 | 75 MB | ~40 ms |
| 1 MB | float32 | 376 MB | ~0.4 s |
| 1 MB | **uint8** | **94 MB** | ~0.4 s |
| 10 MB | uint8 | 940 MB | ~4 s |

- Many probes on one corpus → always `search_many` (shared FFTs).
- Probe count doesn't change memory; corpus size does.
- `auto` threshold cost: zero extra compute (gate is arithmetic).

## 9. Honest limits — don't reach for it when

- **Exact substring** → `str.find` (~1500× faster).
- **Patterns** (emails, IDs) → regex.
- **Transpositions/swaps matter** → `rapidfuzz` (selly-fft scores a swap
  at 0.25 by design — match-fraction, not edit distance).
- **Deletions/insertions** → penalized heavily (misalignment cascades);
  use rapidfuzz to adjudicate selly-fft's hits when indels are expected.
- **Meaning** ("car" ≈ "automobile") → embeddings; selly-fft is the
  symbol-level layer *beneath* semantic search, not a substitute.
- **One-off tiny searches** → anything; FFT pays off at scale or in batch.

**The hybrid pattern that covers most real jobs:**
selly-fft finds candidate spans cheaply at scale → rapidfuzz/LLM
adjudicates the shortlist. Cheap signal first, expensive judgment second.
