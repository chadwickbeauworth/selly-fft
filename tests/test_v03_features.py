"""Tests for v0.3.0 features: batch search, auto threshold, find_spans, uint8, CLI."""

from __future__ import annotations

import subprocess
import sys
import unicodedata

import numpy as np
import pytest

from selly_fft import (
    TextAssociativeMemory,
    SellyAssociativeMemory,
    encode_orthogonal,
    normalized_xcorr_multichannel,
    normalized_xcorr_multichannel_batch,
    text_match,
)


# ---------------------------------------------------------------------------
# Batch search (shared target FFTs)
# ---------------------------------------------------------------------------
def test_batch_matches_individual_results():
    mem = TextAssociativeMemory(threshold=0.0)
    target = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    enc = mem.encode_target(target)
    probes = ["BROWN", "LAZY", "QQQQ", "FOX JUMPS"]
    batch = mem.search_many(probes, enc, threshold=0.0)
    assert len(batch) == len(probes)
    for probe, batch_matches in zip(probes, batch):
        solo = mem.search(probe, enc, threshold=0.0)
        assert [(m.position, m.score) for m in batch_matches] == pytest.approx(
            [(m.position, m.score) for m in solo]
        )


def test_batch_primitive_matches_single():
    ref = encode_orthogonal(list("ABRACADABRA"), "ABRCD")
    probes = [encode_orthogonal(list("ABRA"), "ABRCD"),
              encode_orthogonal(list("CAD"), "ABRCD")]
    batch = normalized_xcorr_multichannel_batch(probes, ref)
    for p, scores in zip(probes, batch):
        solo = normalized_xcorr_multichannel(p, ref)
        assert np.allclose(scores, solo)


def test_batch_handles_empty_and_oversized_probes():
    ref = encode_orthogonal(list("ABCABC"), "ABC")
    probes = [encode_orthogonal([], "ABC"),
              encode_orthogonal(list("ABCABCAB"), "ABC"),  # longer than ref
              encode_orthogonal(list("AB"), "ABC")]
    out = normalized_xcorr_multichannel_batch(probes, ref)
    assert len(out[0]) == 0 and len(out[1]) == 0
    assert out[2][0] == pytest.approx(1.0)


def test_batch_is_faster_than_loop_at_scale():
    import time
    mem = TextAssociativeMemory()
    target = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. " * 2000
    enc = mem.encode_target(target)
    probes = ["QUICK", "BROWN", "JUMPS", "LAZY", "DOG"] * 4  # 20 probes
    t0 = time.perf_counter()
    for p in probes:
        mem.search(p, enc)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    mem.search_many(probes, enc)
    t_batch = time.perf_counter() - t0
    assert t_batch < t_loop  # shared target FFTs should always win at scale


# ---------------------------------------------------------------------------
# Auto threshold (significance gate)
# ---------------------------------------------------------------------------
def test_auto_threshold_reports_strong_match():
    mem = TextAssociativeMemory()
    matches = mem.find_matches("BROWN", "THE QUICK BROWN FOX", threshold="auto")
    assert matches and matches[0].score == pytest.approx(1.0)


def test_auto_threshold_rejects_chance_partials():
    mem = TextAssociativeMemory()
    # a 1-char probe scores 1.0 at some position but one symbol of
    # evidence is never significant under the exact binomial null
    matches = mem.find_matches("E", "THE QUICK BROWN FOX", threshold="auto")
    assert matches == []


def test_auto_threshold_quiets_random_noise():
    rng = np.random.default_rng(7)
    from selly_fft import TEXT_ALPHABET
    alpha = list(TEXT_ALPHABET)
    mem = TextAssociativeMemory()
    false_hits = 0
    for _ in range(50):
        t = "".join(rng.choice(alpha, size=300))
        q = "".join(rng.choice(alpha, size=6))
        false_hits += len(mem.find_matches(q, t, threshold="auto"))
    assert false_hits <= 2  # binomial null at z>=3 should be near-silent


def test_invalid_threshold_string_raises():
    mem = TextAssociativeMemory()
    with pytest.raises(ValueError, match="auto"):
        mem.find_matches("A", "AAA", threshold="bogus")


def test_auto_threshold_on_unit_circle_path():
    mem = SellyAssociativeMemory(alphabet="ATCG")
    # an exact 8-mer match: p = (1/4)^8 ~ 1.5e-5 < AUTO_P -> reported
    matches = mem.search_direct(list("ACGTACGT"), list("TTACGTACGTT"), threshold="auto")
    assert matches and matches[0].score == pytest.approx(1.0)
    # an exact 4-mer in a small target: p = (1/4)^4 ~ 3.9e-3 > AUTO_P.
    # Honest statistics: that IS chance-plausible, so the gate rejects it.
    short = mem.search_direct(list("ACGT"), list("TTACGTT"), threshold="auto")
    assert short == []


# ---------------------------------------------------------------------------
# find_spans (original-coordinate mapping)
# ---------------------------------------------------------------------------
def test_spans_plain_ascii():
    mem = TextAssociativeMemory()
    spans = mem.find_spans("brown", "THE QUICK BROWN FOX")
    assert len(spans) == 1
    sp = spans[0]
    assert (sp.orig_start, sp.orig_end) == (10, 15)
    assert sp.text == "BROWN"
    assert sp.score == pytest.approx(1.0)


def test_spans_map_through_length_changing_fold():
    mem = TextAssociativeMemory()
    target = "straßeX"  # folds to STRASSEX (7 -> 8 chars)
    spans = mem.find_spans("X", target)
    sp = [s for s in spans if s.score == pytest.approx(1.0)][0]
    assert sp.orig_start == 6 and sp.orig_end == 7 and sp.text == "X"


def test_spans_eszett_fold():
    mem = TextAssociativeMemory().build_alphabet("STRASE")
    spans = mem.find_spans("STRASSE", "die STRASSE ist")
    assert spans and spans[0].text == "STRASSE"
    assert spans[0].orig_start == 4


def test_spans_nfc_composition():
    mem = TextAssociativeMemory().build_alphabet("café" + unicodedata.normalize("NFD", "café"))
    target = "un " + unicodedata.normalize("NFD", "café") + " noir"
    spans = mem.find_spans("café", target)  # NFC probe, NFD target
    assert spans and spans[0].text == unicodedata.normalize("NFD", "café")
    assert spans[0].orig_start == 3


def test_spans_case_sensitive_identity_mapping():
    mem = TextAssociativeMemory(case_sensitive=True)
    spans = mem.find_spans("World", "Hello, World!")
    assert spans[0].orig_start == 7 and spans[0].text == "World"


# ---------------------------------------------------------------------------
# uint8 encoding
# ---------------------------------------------------------------------------
def test_uint8_dtype_memory_and_accuracy():
    mem = TextAssociativeMemory(dtype=np.uint8)
    target = "THE QUICK BROWN FOX " * 500
    enc = mem.encode_target(target)
    assert enc.dtype == np.uint8
    assert enc.nbytes == len(target) * len(mem.alphabet)  # 94 bytes/char
    matches = mem.find_matches("QUICK", target)
    assert matches and abs(matches[0].score - 1.0) < 1e-9
    # non-match stays silent
    assert mem.find_matches("QQQQQ", target) == []


def test_uint8_batch_path():
    mem = TextAssociativeMemory(dtype=np.uint8)
    target = "ABRACADABRA " * 100
    enc = mem.encode_target(target)
    out = mem.search_many(["ABRA", "CAD"], enc)
    assert out[0] and out[0][0].score == pytest.approx(1.0)
    assert out[1] and out[1][0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_cli(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "selly_fft.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_cli_scan_finds_match(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("THE QUICK BROWN FOX\nsecond line brown again\n")
    r = _run_cli(["scan", "brown", str(f)], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "doc.txt:1:10: 1.000 'BROWN'" in r.stdout
    assert "doc.txt:2:32: 1.000 'brown'" in r.stdout


def test_cli_scan_no_match_exit_1(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("nothing here\n")
    r = _run_cli(["scan", "zebra", str(f)], cwd=tmp_path)
    assert r.returncode == 1


def test_cli_scan_fuzzy_threshold(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("the brentw0od protocol applies\n")  # one substitution, same length
    r = _run_cli(["scan", "brentwood protocol", str(f), "--threshold", "0.9"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "0.944" in r.stdout  # 17/18 positions match


def test_cli_scan_auto_threshold_and_context(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("xx THE QUICK BROWN FOX yy")
    r = _run_cli(["scan", "quick brown", str(f), "--threshold", "auto", "--context", "4"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "QUICK BROWN" in r.stdout


def test_cli_unicode_hint(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("un café noir")
    r = _run_cli(["scan", "café", str(f)], cwd=tmp_path)
    assert r.returncode == 2
    assert "--build-alphabet" in r.stderr
    r2 = _run_cli(["scan", "café", str(f), "--build-alphabet"], cwd=tmp_path)
    assert r2.returncode == 0, r2.stderr
