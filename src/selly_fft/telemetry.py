"""Convergence telemetry for selly-fft research runs.

**Scoring convention (read this first):**

Every score in this module uses the library's **real-part normalized
cross-correlation** (:meth:`TextAssociativeMemory.search` / `search_many`)
with ``threshold=0.0`` — i.e. the *best* unthresholded score per probe,
on the unit-circle / one-hot encoding depending on alphabet size.

- **1.0** = exact phrase overlap at some position in the corpus
- **0.0** = no symbol-level alignment
- partial = fraction of positions that match exactly (text path)

``threshold=0.0`` is used deliberately: the telemetry measures the
*curve* of coherence over time, so partial matches must still register.
A fixed ``"auto"`` significance gate would zero-out partials and flatten
the curve — defeating the purpose of measuring *how much* the spiral
converges, not just *whether* it converges at all.

**What this module is and is not:**

This module instruments the research process itself.  Given a sequence
of "run" documents (e.g. the Markdown files in a research-runs directory),
it:

1. Extracts a deterministic set of **key claims** (prominent content-bearing
   phrases) from each run.
2. Probes each run's claims against the **concatenated text of all prior
   runs** using ``search_many`` (which shares target FFTs across probes,
   so K probes over N prior runs cost ~ the same as one probe).
3. Reports the **mean best-score** per run — the convergence curve.

A rising curve means later runs increasingly echo earlier ones (constructive
interference = C in ``dE/dt = β(C−D)E``). A flat or falling curve means the
spiral is fragmenting (destructive interference = D). Both are real,
honest results — do not manufacture a rising curve.

**Claim extraction** is deterministic and LLM-independent: it takes the
first ``max_claims`` content-bearing lines (after stripping Markdown
boilerplate).  This makes the telemetry reproducible and testable.  For
richer claim selection, callers can inject a custom ``claim_extractor``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from selly_fft.text import TextAssociativeMemory

# Maximum number of words per extracted claim.  Claims longer than this are
# broken into overlapping sliding-window key phrases so that match-fraction
# scoring works at phrase granularity.
MAX_CLAIM_WORDS = 6

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Markdown boilerplate lines that are not "claims" — headings, separators,
# list bullets, YAML front matter, etc.  Content lines are prose that
# carries substantive meaning.
_BOILERPLATE_PAT = re.compile(
    r"^\s*(?:#{1,6}\s*|-{3,}|\*{3,}|>{1,}|!\[|\s*[-*]\s*|^\s*\d+\.\s*|`{3}|<)"
)

# Headings like "# Some Title" — still content-bearing at the sentence level
# but we skip pure structural lines for claims.
_HEADING_PAT = re.compile(r"^\s*#{1,6}\s+")


def _strip_markdown_structure(text: str) -> str:
    """Strip Markdown syntax that would fragment sentence-level claims
    into meaningless fragments.  We keep paragraph breaks so that
    consecutive short lines are treated as one claim.
    """
    # Remove YAML front matter
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)
    # Remove code blocks
    text = re.sub(r"`{3}[^\n]*\n.*?\n`{3}", "", text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "``", text)
    # Remove image/link syntax but keep link text
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    return text


def _split_clauses(sentence: str) -> List[str]:
    """Break a sentence into shorter content-bearing clauses/phrases.

    Splits on commas, semicolons, and conjunctions (that, which, and, but,
    so, because) to produce shorter, more discriminating claims.  Each
    clause is returned stripped; clauses shorter than 10 chars are dropped.

    If a clause is still very long (> ``MAX_CLAIM_WORDS`` words), it is
    further broken into overlapping sliding-window key phrases so that
    the match-fraction scoring works at phrase granularity rather than
    sentence granularity (an 8-word phrase that doesn't match exactly
    scores 0.0, but a 4-word sub-phrase within it can score 1.0).
    """
    parts = re.split(
        r",|;|\s+(?:that|which|and|but|so|because|while)\s+", sentence, flags=re.IGNORECASE
    )
    result: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) >= 10:
            words = part.split()
            if len(words) > MAX_CLAIM_WORDS:
                # Generate overlapping windows of MAX_CLAIM_WORDS words
                step = max(1, MAX_CLAIM_WORDS // 2)
                for start in range(0, len(words) - MAX_CLAIM_WORDS + 1, step):
                    window = " ".join(words[start:start + MAX_CLAIM_WORDS])
                    if len(window) >= 10:
                        result.append(window)
            else:
                result.append(part)
    if not result:
        return [sentence.strip()] if len(sentence.strip()) >= 10 else []
    return result


def _extract_claims_default(text: str, max_claims: int = 15) -> List[str]:
    """Extract up to ``max_claims`` content-bearing phrases from run text.

    Strategy:
    1. Strip Markdown structure (code blocks, links, bold) so we measure
       semantic content, not formatting.
    2. Split into paragraphs on blank lines.
    3. Within each paragraph, split into sentences (naive split on
       period/newline that ends a clause).
    4. Break each sentence into shorter clauses via ``_split_clauses``,
       which splits on commas, semicolons, and conjunctions (that,
       which, and, but, so, because).  This yields shorter, more
       discriminating key phrases — a full sentence may be 12 words, but
       a 4-word key phrase like "holographic memory methods" matches far
       more sensitively than the whole sentence.
    5. Keep clauses that are substantive (> 10 chars, not boilerplate).
    6. Take the first ``max_claims`` such clauses across all paragraphs.

    This is deterministic (no LLM, no randomness) so results are
    reproducible and testable.

    **Scoring note:** each claim is probed against the prior corpus using
    ``search_many`` with ``threshold=0.0`` (best unthresholded score).  Shorter
    claims produce higher match-fraction scores, so splitting into clauses
    rather than using full sentences gives a more granular coherence signal.
    """
    cleaned = _strip_markdown_structure(text)
    paragraphs = re.split(r"\n\s*\n", cleaned)
    claims: List[str] = []
    for para in paragraphs:
        if _is_boilerplate(para):
            continue
        # Split into sentences
        sentences = re.split(r"[.!?]\s+", para.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10 or _is_boilerplate(sent):
                continue
            # Break into shorter, more discriminating clauses
            clauses = _split_clauses(sent)
            for clause in clauses:
                if len(clause) > 10 and not _is_boilerplate(clause):
                    claims.append(clause)
                    if len(claims) >= max_claims:
                        return claims
        # Fallback: if no clauses were extracted from this paragraph,
        # treat the paragraph itself as a claim
        if not sentences or all(len(s.strip()) < 10 for s in sentences):
            if len(para.strip()) > 10 and not _is_boilerplate(para):
                claims.append(para.strip())
                if len(claims) >= max_claims:
                    return claims
    return claims


def _is_boilerplate(line: str) -> bool:
    """Return True if a line is structural Markdown, not content."""
    if not line or not line.strip():
        return True
    stripped = line.strip()
    if _HEADING_PAT.match(stripped):
        return True
    if _BOILERPLATE_PAT.match(stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# Core telemetry
# ---------------------------------------------------------------------------

@dataclass
class TelemetryResult:
    """Aggregated result of running telemetry across a corpus of runs."""

    n_runs: int
    max_claims: int
    # mean best-score per run (run i probed against runs 0..i-1)
    convergence_curve: List[float] = field(default_factory=list)
    # per-run: list of the best score for each claim
    per_run_best_scores: List[List[float]] = field(default_factory=list)
    # per-run: list of the claim texts extracted
    per_run_claims: List[List[str]] = field(default_factory=list)
    # cumulative corpus length (chars) used as reference for each run
    per_run_corpus_chars: List[int] = field(default_factory=list)
    # outlier run indices (mean coherence < outlier_sigma_std of the curve)
    outliers: List[int] = field(default_factory=list)

    @property
    def coherence_mean(self) -> float:
        """Mean of the convergence curve (excluding the first run, which has
        no prior corpus to be coherent against)."""
        meaningful = self.convergence_curve[1:] if len(self.convergence_curve) > 1 else []
        return float(np.mean(meaningful)) if meaningful else 0.0

    @property
    def trend(self) -> str:
        """Classify the overall trend of coherence across the run sequence."""
        if len(self.convergence_curve) < 3:
            return "insufficient"
        # Linear regression slope on the meaningful (non-first) part
        y = list(self.convergence_curve[1:])
        x = list(range(1, len(y) + 1))
        if len(y) < 2:
            return "insufficient"
        slope = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
        if slope > 0.02:
            return "rising"
        elif slope < -0.02:
            return "falling"
        else:
            return "flat"


@dataclass
class RunTelemetry:
    """Telemetry for a single run."""

    run_index: int
    run_label: str
    mean_best_score: float
    best_scores: List[float]
    claims: List[str]
    corpus_chars: int


# Type alias for a callable claim extractor: text -> list[str] of claims
ClaimExtractor = Callable[[str], List[str]]


def default_claim_extractor(max_claims: int = 15) -> ClaimExtractor:
    """Return a claim extractor bound to ``max_claims``."""

    def _extract(text: str) -> List[str]:
        return _extract_claims_default(text, max_claims=max_claims)

    return _extract


def _build_mem(case_sensitive: bool = False) -> TextAssociativeMemory:
    """Build a TextAssociativeMemory, building its alphabet from text lazily."""
    mem = TextAssociativeMemory(case_sensitive=case_sensitive)
    return mem


def _encode_corpus(mem: TextAssociativeMemory, corpus_text: str) -> np.ndarray:
    """Encode the full corpus text, building the alphabet from the text."""
    mem.build_alphabet(corpus_text)
    return mem.encode_target(corpus_text)


def run_telemetry(
    runs: Sequence[str],
    *,
    max_claims: int = 15,
    claim_extractor: Optional[ClaimExtractor] = None,
    case_sensitive: bool = False,
    alphabet: Optional[Sequence] = None,
) -> List[RunTelemetry]:
    """Compute per-run coherence telemetry across a sequence of run texts.

    Parameters
    ----------
    runs : sequence of str
        The text content of each research run, in chronological order
        (run 0 is the earliest, run N-1 is the latest).
    max_claims : int, default 15
        Maximum number of key claims to extract per run.  The first N
        content-bearing phrases are taken deterministically.
    claim_extractor : callable, optional
        Custom claim extractor.  If None, the default deterministic
        extractor (``_extract_claims_default``) is used.  This is the
        hook for richer claim selection (e.g. LLM-based) if desired,
        but the default is sufficient and reproducible.
    case_sensitive : bool, default False
        Passed to ``TextAssociativeMemory``.  False folds case so that
        "Quantum" and "quantum" are the same symbol — appropriate for
        prose coherence measurement.
    alphabet : sequence, optional
        Explicit alphabet.  If None, the alphabet is built from the
        corpus text (recommended for large corpora with diverse vocab).

    Returns
    -------
    list of RunTelemetry
        One entry per run.  The first run (index 0) has no prior corpus
        to be coherent against, so its ``mean_best_score`` is 0.0 and
        ``best_scores`` is empty — this is correct, not a bug.

    Notes
    -----
    **Scoring convention:** each claim is probed against the *concatenated
    text of all prior runs* (runs 0..i-1 for run i) using
    ``search_many`` with ``threshold=0.0``, returning the best unthresholded
    score per probe.  The mean of those best scores is the run's coherence
    value.  A score of 1.0 means at least one claim appears verbatim in the
    prior corpus; 0.0 means no overlap at all.
    """
    if claim_extractor is None:
        claim_extractor = default_claim_extractor(max_claims)

    results: List[RunTelemetry] = []
    corpus_chars = 0

    for i, run_text in enumerate(runs):
        claims = claim_extractor(run_text)

        if i == 0:
            # No prior corpus — first run is the origin, coherence against
            # nothing is undefined.  Report 0.0 (no prior to be coherent with).
            results.append(RunTelemetry(
                run_index=0,
                run_label=f"run_{i}",
                mean_best_score=0.0,
                best_scores=[],
                claims=claims,
                corpus_chars=0,
            ))
            # Start building the corpus with run 0's text
            corpus_chars = len(run_text)
            continue

        # Corpus = concatenation of all prior runs (0..i-1)
        prior_corpus = "\n\n".join(runs[:i])

        if not prior_corpus or not claims:
            results.append(RunTelemetry(
                run_index=i,
                run_label=f"run_{i}",
                mean_best_score=0.0,
                best_scores=[],
                claims=claims,
                corpus_chars=len(prior_corpus),
            ))
            continue

        # Build the search engine over the prior corpus.
        # The alphabet must include symbols from BOTH the prior corpus and
        # the current run's claims — a claim may contain a word not yet
        # seen in prior runs.  We build from the concatenation so the
        # orthogonal one-hot encoding covers every probe symbol.
        mem = _build_mem(case_sensitive=case_sensitive)
        if alphabet is not None:
            mem.alphabet = list(alphabet)
            mem._L = len(mem.alphabet)
        else:
            mem.build_alphabet(prior_corpus + "\n" + " ".join(claims))

        target_enc = mem.encode_target(prior_corpus)

        # Probe claims against prior corpus with threshold=0.0 (unthresholded)
        # so we get the best score per claim, including partials.
        # Use search_many to share the corpus FFTs across all claims.
        results_per_probe = mem.search_many(claims, target_enc, threshold=0.0)

        best_scores = []
        for probe_matches in results_per_probe:
            if probe_matches:
                best_score = max(m.score for m in probe_matches)
            else:
                best_score = 0.0
            best_scores.append(best_score)

        mean_score = float(np.mean(best_scores)) if best_scores else 0.0
        results.append(RunTelemetry(
            run_index=i,
            run_label=f"run_{i}",
            mean_best_score=mean_score,
            best_scores=best_scores,
            claims=claims,
            corpus_chars=len(prior_corpus),
        ))

        corpus_chars = len(prior_corpus) + len(run_text)

    return results


def convergence_curve(runs: Sequence[str], **kwargs) -> List[float]:
    """Convenience: just the mean-best-score per run.

    See :func:`run_telemetry` for parameter documentation.
    """
    telemetry = run_telemetry(runs, **kwargs)
    return [rt.mean_best_score for rt in telemetry]


def aggregate_telemetry(
    run_telemetry_list: Sequence[RunTelemetry],
    run_labels: Optional[Sequence[str]] = None,
) -> TelemetryResult:
    """Aggregate per-run telemetry into a summary result with outlier detection.

    Parameters
    ----------
    run_telemetry_list : sequence of RunTelemetry
        Output of :func:`run_telemetry`.
    run_labels : sequence of str, optional
        Human-readable labels for each run (e.g. "Run-01-Initial-Scan").
        If None, ``"run_0"``, ``"run_1"``, ... are used.

    Returns
    -------
    TelemetryResult
    """
    n = len(run_telemetry_list)
    curve = [rt.mean_best_score for rt in run_telemetry_list]
    best_scores = [rt.best_scores for rt in run_telemetry_list]
    claims = [rt.claims for rt in run_telemetry_list]
    corpus_chars = [rt.corpus_chars for rt in run_telemetry_list]

    # Outlier detection: runs whose mean coherence is substantially lower
    # than the rest of the curve.  We use a robust criterion: a run is an
    # outlier if its coherence is below (mean - k*std) of the *meaningful*
    # part of the curve (excluding run 0, which is always 0.0 by definition —
    # it has no prior corpus to be coherent against).
    #
    # For small samples (n < 8) the z-score threshold is relaxed so that
    # genuine topic-drift runs are still caught.  With only a handful of
    # data points, a -2.0 std gate is too strict (the outlier pulls the
    # mean down and inflates the std), so we use a fraction-of-the-median
    # rule instead: a run is an outlier if its score is less than 50% of
    # the median coherence of all non-zero runs.
    outliers: List[int] = []
    if n > 3:
        meaningful = curve[1:]  # exclude run 0
        meaningful_nonzero = [s for s in meaningful if s > 0]
        if meaningful_nonzero:
            median_coh = float(np.median(meaningful_nonzero))
            mean_coh = float(np.mean(meaningful_nonzero))
            std_coh = float(np.std(meaningful_nonzero))
            for i, score in enumerate(curve):
                if i == 0:
                    continue  # run 0 is structurally zero, not an outlier
                # Two criteria, either triggers:
                # (a) z-score < -1.5 (relaxed for small samples)
                # (b) score < 50% of the median (robust to small samples)
                z = (score - mean_coh) / std_coh if std_coh > 0 else 0.0
                if z < -1.5 or (median_coh > 0 and score < 0.5 * median_coh):
                    outliers.append(i)

    return TelemetryResult(
        n_runs=n,
        max_claims=max((len(c) for c in claims), default=0),
        convergence_curve=curve,
        per_run_best_scores=best_scores,
        per_run_claims=claims,
        per_run_corpus_chars=corpus_chars,
        outliers=outliers,
    )


def load_runs_from_directory(
    directory: str,
    pattern: str = r"Run-\d+",
    max_runs: Optional[int] = None,
    include_date_prefixed: bool = True,
) -> tuple[List[str], List[str]]:
    """Load run files from a directory, sorted by run number.

    Parameters
    ----------
    directory : str
        Path to the directory containing run Markdown files.
    pattern : str
        Regex to match run filenames.  By default matches any filename
        containing ``Run-NNN`` (e.g. ``Run-01-First.md`` or
        ``2026-07-30-Run-01-First.md``).  The run number is always
        extracted from the ``Run-(\\d+)`` portion for sorting.
    max_runs : int, optional
        Cap the number of runs loaded (oldest first).
    include_date_prefixed : bool, default True
        If False, only files matching ``^Run-\\d+`` (not date-prefixed)
        are loaded.  This is ignored when a custom ``pattern`` is given.

    Returns
    -------
    (texts, labels) : tuple of (list[str], list[str])
        ``texts`` is the file contents in chronological order (sorted by
        run number); ``labels`` is the filename stems (without extension).
    """
    import re

    name_regex = re.compile(pattern)
    num_regex = re.compile(r"Run-(\d+)")

    if not include_date_prefixed and pattern == r"Run-\d+":
        name_regex = re.compile(r"^Run-\d+")

    matched_files: List[tuple[int, str]] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md"):
            continue
        if not name_regex.search(fname):
            continue
        num_match = num_regex.search(fname)
        if num_match:
            matched_files.append((int(num_match.group(1)), fname))

    matched_files.sort(key=lambda x: x[0])
    if max_runs is not None:
        matched_files = matched_files[:max_runs]

    texts: List[str] = []
    labels: List[str] = []
    for _, fname in matched_files:
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
        labels.append(os.path.splitext(fname)[0])

    return texts, labels
