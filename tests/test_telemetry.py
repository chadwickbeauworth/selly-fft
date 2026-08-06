"""Tests for the v0.4.0 convergence telemetry module.

These tests verify:
1. Claim extraction is deterministic and filters boilerplate.
2. run_telemetry computes correct coherence scores.
3. The convergence curve reflects ground-truth coherence (echoing runs
   score higher than unrelated runs).
4. Outlier detection flags low-coherence runs.
5. load_runs_from_directory correctly sorts and loads files.
6. Empty edge cases are handled gracefully.
7. Custom claim_extractors are respected.

**Scoring convention:** all scores use real-part normalized cross-correlation
(threshold=0.0, best unthresholded score per claim).  Exact phrase overlap = 1.0,
no overlap = ~0.0, partial = fraction of matching positions (text one-hot path).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from selly_fft import telemetry
from selly_fft.telemetry import (
    MAX_CLAIM_WORDS,
    RunTelemetry,
    TelemetryResult,
    _extract_claims_default,
    _is_boilerplate,
    _split_clauses,
    _strip_markdown_structure,
    aggregate_telemetry,
    convergence_curve,
    default_claim_extractor,
    load_runs_from_directory,
    run_telemetry,
)


# ---------------------------------------------------------------------------
# API ergonomics + performance regression (selly-fft v0.4.1)
# ---------------------------------------------------------------------------

class TestApiErgonomicsAndScale:
    def test_tuple_from_loader_unwraps(self, tmp_path):
        """load_runs_from_directory returns (texts, labels); passing that
        tuple straight into convergence_curve / run_telemetry must work
        (regression: it used to raise TypeError, and len() returned 2 —
        the tuple arity, not the file count)."""
        for i in range(1, 6):
            (tmp_path / f"Run-{i:02d}-T.md").write_text(f"Content {i} shared term here.")
        loaded = load_runs_from_directory(str(tmp_path))
        # Naive composition must NOT raise.
        curve = convergence_curve(loaded, max_claims=5)
        assert len(curve) == 5
        # run_telemetry form too.
        rt = run_telemetry(loaded, max_claims=5)
        assert len(rt) == 5
        # The first run is always 0.0 (no prior corpus); later runs echo 'shared term'.
        assert rt[0].mean_best_score == 0.0
        assert all(r.mean_best_score > 0.0 for r in rt[1:])

    def test_len_of_loader_tuple_is_two_not_filecount(self, tmp_path):
        """Document the trap: len(loader_return) is the tuple arity (2),
        not the file count.  Callers must unpack or rely on unwrap."""
        for i in range(1, 4):
            (tmp_path / f"Run-{i:02d}-T.md").write_text(f"Content {i}.")
        loaded = load_runs_from_directory(str(tmp_path))
        assert len(loaded) == 2  # (texts, labels)
        assert len(loaded[0]) == 3  # real file count

    def test_large_corpus_completes_quickly(self, tmp_path):
        """The coherence pass must FINISH on a large corpus (the original
        per-prefix loop was O(R·C log C) and hung on a 143-run corpus).

        This single-pass design correlates each DISTINCT claim against the
        whole corpus once (shared reference FFT, duplicate claims computed
        once) then takes a running prefix-max per run.  On a 120-file corpus
        it completes in a bounded time (a few minutes) — it must not hang.
        We assert a generous bound so CI catches a genuine regression to the
        old O(R·C log C) behavior, while tolerating the intrinsic FFT cost of
        correlating hundreds of distinct claims against a ~1 MB corpus."""
        import time

        rng = np.random.default_rng(0)
        words = ["quantum", "holographic", "memory", "coherence", "methods",
                 "tensor", "photon", "code", "error", "correction"]
        n = 120
        runs = []
        for i in range(n):
            sent = " ".join(rng.choice(words) for _ in range(200))
            runs.append(f"Run {i}: {sent}")
        for i, r in enumerate(runs, 1):
            (tmp_path / f"Run-{i:03d}-X.md").write_text(r)

        t0 = time.time()
        rt = run_telemetry(runs, max_claims=10)
        dt = time.time() - t0
        assert len(rt) == n
        # Generous bound: catches a regression to O(R·C log C) hang while
        # allowing the intrinsic FFT cost on a ~1 MB corpus.
        assert dt < 300.0, f"large-corpus telemetry took {dt:.1f}s (>300s)"

    def test_scores_match_reference_prefix_semantics(self):
        """Run i's coherence must equal the best score against the PREFIX
        (runs 0..i-1), not the full corpus.  Construct a case where a claim
        appears ONLY in a later run, so a naive full-corpus search would
        wrongly credit it to an earlier run."""
        runs = [
            "alpha beta gamma delta epsilon",          # run 0 origin
            "alpha beta gamma delta epsilon zeta",      # echoes run 0
            "unique later phrase only here omega",      # introduces a new claim
            "alpha beta gamma delta epsilon unique later phrase only here",  # echoes both
        ]
        rt = run_telemetry(runs, max_claims=20)
        # Run 2's only novel claim must NOT score near 1.0 against runs 0..1
        # (it is not in the prefix). A small partial-match score is expected
        # from incidental word overlap; what matters is it is far below the
        # score once the claim IS in the prefix (run 3).
        assert rt[2].mean_best_score < 0.5  # novel claim not yet in prefix
        assert rt[3].mean_best_score > rt[2].mean_best_score  # now in prefix



# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

class TestClaimExtraction:
    def test_extracts_content_claims(self):
        text = (
            "The quantum holographic memory methods explore coherence in depth. "
            "This is the second sentence with enough words here."
        )
        claims = _extract_claims_default(text, max_claims=10)
        assert len(claims) >= 2
        assert all(len(c) > 10 for c in claims)

    def test_strips_markdown_boilerplate(self):
        text = (
            "# Heading\n\n"
            "---\n\n"
            "```python\ncode\n```\n\n"
            "This is real content sentence here.\n\n"
            "1. numbered list item with text.\n"
        )
        claims = _extract_claims_default(text, max_claims=10)
        assert all(not _is_boilerplate(c) for c in claims)
        assert any("real content" in c for c in claims)

    def test_max_claims_respected(self):
        text = "Sentence one with enough words here. " * 10
        claims = _extract_claims_default(text, max_claims=3)
        assert len(claims) <= 3

    def test_clauses_shorten_sentences(self):
        """Long sentences should be split into shorter clauses (max 6 words)."""
        text = "The quantum holographic memory methods explore coherence in depth."
        claims = _extract_claims_default(text, max_claims=10)
        assert len(claims) > 0
        # Clauses should be at most MAX_CLAIM_WORDS
        for c in claims:
            assert len(c.split()) <= MAX_CLAIM_WORDS

    def test_clauses_split_on_commas(self):
        sent = "Quantum coherence, holographic memory, and FFT correlation analysis."
        clauses = _split_clauses(sent)
        assert len(clauses) >= 2
        assert any("Quantum coherence" in c for c in clauses)

    def test_clauses_split_on_conjunctions(self):
        sent = "Quantum memory methods that explore coherence in depth"
        clauses = _split_clauses(sent)
        assert any("Quantum memory methods" in c for c in clauses)

    def test_deterministic_output(self):
        """Same input always produces same claims (no randomness)."""
        text = "Quantum holographic coherence methods explore memory in depth across dimensions."
        c1 = _extract_claims_default(text, max_claims=5)
        c2 = _extract_claims_default(text, max_claims=5)
        assert c1 == c2

    def test_empty_text_returns_empty(self):
        assert _extract_claims_default("", max_claims=5) == []

    def test_short_text_returns_empty(self):
        assert _extract_claims_default("hi", max_claims=5) == []

    def test_strip_markdown_structure_removes_code(self):
        text = "```python\nimport numpy\n```\nReal text here with enough words."
        stripped = _strip_markdown_structure(text)
        assert "import numpy" not in stripped
        assert "Real text here" in stripped


# ---------------------------------------------------------------------------
# run_telemetry
# ---------------------------------------------------------------------------

class TestRunTelemetry:
    def test_first_run_is_zero(self):
        """Run 0 has no prior corpus — coherence is 0.0 by definition."""
        runs = ["First run establishes the baseline quantum holographic methods."]
        result = run_telemetry(runs, max_claims=3)
        assert len(result) == 1
        assert result[0].mean_best_score == 0.0
        assert result[0].best_scores == []

    def test_echoing_run_scores_higher(self):
        """A run that echoes prior claims should score higher than one
        that introduces a totally different topic."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further now.",
            "Gardening tomatoes requires specific soil pH for optimization.",
        ]
        result = run_telemetry(runs, max_claims=5)
        # Run 1 echoes Run 0 → high coherence
        assert result[1].mean_best_score > 0.3
        # Run 2 is about gardening → lower coherence
        assert result[2].mean_best_score < result[1].mean_best_score

    def test_exact_phrase_overlap_scores_high(self):
        """When a claim appears verbatim in the prior corpus, it scores 1.0."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence in depth further.",
        ]
        result = run_telemetry(runs, max_claims=5)
        # The 6-word phrase "quantum holographic memory methods explore coherence"
        # appears in both runs → should get a 1.0 best score
        assert max(result[1].best_scores) > 0.9

    def test_unrelated_run_scores_low(self):
        """A run with a totally different vocabulary scores low."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Zyg qzxjkv wplofr nitm eqptbuh jvoh upi nbm.",  # gibberish
        ]
        result = run_telemetry(runs, max_claims=3)
        assert result[1].mean_best_score < 0.2

    def test_returns_run_telemetry_objects(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods continue the exploration.",
        ]
        result = run_telemetry(runs, max_claims=3)
        assert len(result) == 2
        assert isinstance(result[0], RunTelemetry)
        assert result[0].run_index == 0
        assert result[1].run_index == 1
        assert result[0].claims  # has extracted claims
        assert result[1].claims

    def test_empty_runs_list(self):
        result = run_telemetry([], max_claims=3)
        assert result == []

    def test_empty_run_text_handled(self):
        runs = ["Some content here with enough words to extract.", ""]
        result = run_telemetry(runs, max_claims=3)
        assert len(result) == 2
        assert result[1].mean_best_score == 0.0  # no claims to probe

    def test_custom_claim_extractor(self):
        """A custom claim_extractor is used instead of the default."""
        custom_claims = ["custom claim phrase", "another custom phrase"]
        def extract(text):
            return custom_claims[:3]

        runs = [
            "The quantum holographic memory methods explore coherence.",
            "Quantum holographic memory methods continue the exploration.",
        ]
        result = run_telemetry(runs, max_claims=3, claim_extractor=extract)
        assert result[1].claims == custom_claims[:3]

    def test_convergence_curve_helper(self):
        """convergence_curve() returns just the scores."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further.",
            "Gardening tomatoes requires soil pH for growth.",
        ]
        curve = convergence_curve(runs, max_claims=3)
        assert len(curve) == 3
        assert curve[0] == 0.0
        assert all(0.0 <= s <= 1.0 for s in curve[1:])

    def test_best_scores_per_claim_reflect_individual_alignment(self):
        """best_scores should have one entry per claim."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods continue the exploration.",
        ]
        rt = run_telemetry(runs, max_claims=5)
        assert len(rt[1].best_scores) == len(rt[1].claims)
        assert all(0.0 <= s <= 1.0 for s in rt[1].best_scores)

    def test_corpus_chars_positive_for_nonzero_run(self):
        """corpus_chars should reflect the size of the prior corpus."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods continue the exploration further.",
            "Quantum holographic memory methods expand the coherence analysis.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        # Run 0: 0 (no prior)
        assert rt[0].corpus_chars == 0
        # Run 1: prior = run 0's text
        assert rt[1].corpus_chars == len(runs[0])
        # Run 2: prior = runs 0 and 1 joined by "\n\n"
        assert rt[2].corpus_chars == len(runs[0]) + 2 + len(runs[1])

    def test_dtype_uint8_produces_same_results(self):
        """The dtype parameter should not change the coherence scores."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods continue the exploration.",
            "Gardening tomatoes requires soil pH for growth.",
        ]
        rt_f64 = run_telemetry(runs, max_claims=3, dtype=np.float64)
        rt_u8 = run_telemetry(runs, max_claims=3, dtype=np.uint8)
        for a, b in zip(rt_f64, rt_u8):
            assert a.mean_best_score == pytest.approx(b.mean_best_score, abs=1e-6)
            assert a.best_scores == pytest.approx(b.best_scores, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregate_telemetry
# ---------------------------------------------------------------------------

class TestAggregateTelemetry:
    def test_basic_aggregation(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further.",
            "Gardening tomatoes requires soil pH for growth.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        assert agg.n_runs == 3
        assert len(agg.convergence_curve) == 3
        assert len(agg.per_run_claims) == 3
        assert len(agg.per_run_best_scores) == 3

    def test_outlier_detection_flags_low_coherence(self):
        """A run that drops coherence sharply should be flagged as outlier."""
        # Runs 0-3: all echoing each other (high coherence)
        # Run 4: totally different topic (low coherence → outlier)
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods build on prior coherence methods.",
            "Quantum holographic memory methods expand coherence analysis broadly.",
            "Quantum holographic memory methods deepen coherence investigation now.",
            "Zyg qzxjkv wplofr nitm eqptbuh jvoh upi nbm.",  # gibberish
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        # Run 4 should be an outlier (low coherence vs high prior runs)
        assert 4 in agg.outliers

    def test_trend_classification_rising(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods build on prior coherence methods.",
            "Quantum holographic memory methods deepen coherence exploration.",
            "Quantum holographic memory methods further expand coherence methods.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        assert agg.trend == "rising"

    def test_trend_classification_falling(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further.",
            "Zyg qzxjkv wplofr nitm eqptbuh jvoh upi nbm.",  # gibberish
            "Qwerty uiop asdf zxcv jklm nopq rstu vwxyz abcd.",  # gibberish
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        assert agg.trend == "falling"

    def test_trend_insufficient_for_short_sequence(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods build on prior methods.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        assert agg.trend == "insufficient"

    def test_coherence_mean_excludes_run_zero(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further.",
            "Gardening tomatoes requires soil pH for growth.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        # Run 0 is excluded (structurally 0.0)
        meaningful = agg.convergence_curve[1:]
        expected = float(np.mean(meaningful))
        assert agg.coherence_mean == pytest.approx(expected)

    def test_run_labels_in_aggregation(self):
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence further.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        labels = ["Run-01", "Run-02"]
        agg = aggregate_telemetry(rt, run_labels=labels)
        assert len(agg.per_run_claims) == 2


# ---------------------------------------------------------------------------
# load_runs_from_directory
# ---------------------------------------------------------------------------

class TestLoadRunsFromDirectory:
    def test_loads_and_sorts_by_run_number(self, tmp_path):
        (tmp_path / "Run-02-Second.md").write_text("Second run content here.")
        (tmp_path / "Run-01-First.md").write_text("First run content here.")
        (tmp_path / "Run-03-Third.md").write_text("Third run content here.")
        texts, labels = telemetry.load_runs_from_directory(str(tmp_path))
        assert len(texts) == 3
        assert labels == ["Run-01-First", "Run-02-Second", "Run-03-Third"]
        assert "First" in texts[0]
        assert "Second" in texts[1]

    def test_loads_date_prefixed_runs(self, tmp_path):
        """Date-prefixed runs (2026-07-30-Run-01-...) should also load."""
        (tmp_path / "2026-07-30-Run-02-Second.md").write_text("Second run content here.")
        (tmp_path / "2026-07-30-Run-01-First.md").write_text("First run content here.")
        (tmp_path / "Run-03-Third.md").write_text("Third run content here.")
        texts, labels = telemetry.load_runs_from_directory(str(tmp_path))
        assert len(texts) == 3
        assert labels == ["2026-07-30-Run-01-First", "2026-07-30-Run-02-Second", "Run-03-Third"]
        assert "First" in texts[0]
        assert "Second" in texts[1]

    def test_exclude_date_prefixed(self, tmp_path):
        """include_date_prefixed=False should skip date-prefixed files."""
        (tmp_path / "2026-07-30-Run-02-Second.md").write_text("Second run.")
        (tmp_path / "Run-01-First.md").write_text("First run.")
        texts, labels = telemetry.load_runs_from_directory(
            str(tmp_path), include_date_prefixed=False
        )
        assert len(texts) == 1
        assert labels == ["Run-01-First"]

    def test_pattern_filter(self, tmp_path):
        (tmp_path / "Run-01-First.md").write_text("First content here.")
        (tmp_path / "NOTE-02-random.md").write_text("Random note.")
        (tmp_path / "template.md").write_text("Template.")
        texts, labels = telemetry.load_runs_from_directory(str(tmp_path))
        assert len(texts) == 1
        assert labels == ["Run-01-First"]

    def test_max_runs(self, tmp_path):
        for i in range(1, 6):
            (tmp_path / f"Run-{i:02d}-Test.md").write_text(f"Content {i} here.")
        texts, labels = telemetry.load_runs_from_directory(str(tmp_path), max_runs=3)
        assert len(texts) == 3
        assert labels == ["Run-01-Test", "Run-02-Test", "Run-03-Test"]

    def test_empty_directory(self, tmp_path):
        texts, labels = telemetry.load_runs_from_directory(str(tmp_path))
        assert texts == []
        assert labels == []


# ---------------------------------------------------------------------------
# Integration: end-to-end telemetry on synthetic runs
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_coherent_sequence_rising(self):
        """A sequence of runs all echoing the same core concepts should
        produce a rising or high coherence curve."""
        core = "quantum holographic memory methods"
        runs = [
            f"The {core} explore coherence in depth.",
            f"Building on {core}, we explore coherence deeply now.",
            f"The {core} and coherence are central to our approach.",
            f"Our {core} demonstrates coherence across all experiments.",
        ]
        rt = run_telemetry(runs, max_claims=4)
        agg = aggregate_telemetry(rt)
        # All runs except 0 should have non-trivial coherence
        for i, score in enumerate(agg.convergence_curve[1:], 1):
            assert score > 0.1
        # The trend should be rising or flat (not falling)
        assert agg.trend in ("rising", "flat")

    def test_fragmenting_sequence_has_low_coherence(self):
        """A sequence that introduces new unrelated topics should show
        low coherence on the later runs."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods build on prior coherence methods.",
            "Gardening tomatoes requires soil pH for optimization purposes.",
            "Stock trading strategies use technical analysis indicators.",
        ]
        rt = run_telemetry(runs, max_claims=3)
        agg = aggregate_telemetry(rt)
        # Run 3 (stock trading) should have low coherence
        assert agg.convergence_curve[3] < 0.20

    def test_scoring_convention_is_real_part(self):
        """The telemetry uses real-part normalized xcorr (not magnitude),
        so orthogonal symbols score 0, not 1.  Verify by checking that
        a totally unrelated short claim still scores near 0."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Zyg qzxjkv wplofr nitm eqptbuh jvoh upi nbm.",  # gibberish
        ]
        rt = run_telemetry(runs, max_claims=2)
        assert rt[1].mean_best_score < 0.2

    def test_best_scores_reflect_phrase_granularity(self):
        """Shorter claims (6 words) should produce more granular scores."""
        runs = [
            "Quantum holographic memory methods explore coherence in depth.",
            "Quantum holographic memory methods explore coherence far further.",
        ]
        rt = run_telemetry(runs, max_claims=5)
        # The 6-word phrase "quantum holographic memory methods explore coherence"
        # appears in Run 0 and is partially in Run 1 → at least one claim
        # should score very high (exact substring)
        assert any(s > 0.9 for s in rt[1].best_scores)
