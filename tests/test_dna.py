"""Tests for the DNA/RNA specialized associative memory (Run-127)."""

import numpy as np
import pytest

from selly_fft import DNAAssociativeMemory, dna_match, Match
from selly_fft.dna import DNA_ALPHABET, DNA_BASE_MAP, encode_dna_phasor


class TestDNAAssociativeMemory:

    def test_exact_match_at_known_position(self):
        mem = DNAAssociativeMemory(threshold=0.7)
        matches = mem.find_matches("ATCG", "GGGGAATCGGGG")
        assert len(matches) >= 1
        assert matches[0].position == 5
        assert matches[0].score > 0.99

    def test_multiple_exact_matches(self):
        mem = DNAAssociativeMemory(threshold=0.7)
        matches = mem.find_matches("ATCG", "ATCGATCGATCG")
        positions = sorted(m.position for m in matches)
        assert positions == [0, 4, 8]

    def test_dna_nonmatch_scores_zero(self):
        """AAAA vs GGGG: A=0°, G=270° → orthogonal → score 0."""
        from selly_fft import holographic_match
        s = holographic_match("AAAAAAAA", "GGGG", alphabet=DNA_ALPHABET)
        assert s < 0.01

    def test_dna_antipodal_scores_zero(self):
        """AAAA vs TTTT: A=0°, T=180° → antipodal → cos=-1 → clipped 0."""
        from selly_fft import holographic_match
        s = holographic_match("AAAAAAAA", "TTTT", alphabet=DNA_ALPHABET)
        assert s < 0.01

    def test_dna_self_match(self):
        from selly_fft import holographic_match
        assert holographic_match("ATCGATCG", "ATCG", alphabet=DNA_ALPHABET) > 0.99

    def test_partial_dna_match(self):
        """3/4 DNA symbols matching → score ≈ 0.75."""
        mem = DNAAssociativeMemory(threshold=0.0)
        target = "ATCGATCG"
        m = mem.search_direct(list("ATCA"), list(target), threshold=0.0)
        assert abs(m[0].score - 0.75) < 0.01


class TestDNACodonExample:
    """DNA/RNA sequence-matching example (the patents' motivating use case)."""

    def test_find_codon_in_genome(self):
        """Find the start codon ATG in a synthetic genome."""
        genome = "GGGATGCGGTAACGATCGATGGCATG"
        # ATG appears at positions 3, 18, 23
        mem = DNAAssociativeMemory(threshold=0.9)
        matches = mem.find_matches("ATG", genome)
        positions = sorted(m.position for m in matches)
        assert 3 in positions
        assert 18 in positions
        assert 23 in positions

    def test_find_start_codon_with_threshold(self):
        """Search with a high threshold to avoid false positives."""
        genome = "AAAATGCGCGCG"
        mem = DNAAssociativeMemory(threshold=0.9)
        matches = mem.find_matches("ATCG", genome)
        # ATCG does not appear here — should have no high-confidence matches
        assert len(matches) == 0

    def test_no_false_positive_codon(self):
        """A codon not present in the genome should not match."""
        mem = DNAAssociativeMemory(threshold=0.9)
        matches = mem.find_matches("TTTT", "AAAACGCGCGCG")
        assert len(matches) == 0

    def test_rna_uppercase_normalization(self):
        """RNA uses U instead of T — verify it's handled."""
        mem = DNAAssociativeMemory(threshold=0.9)
        # Default DNA alphabet doesn't include U; this should raise
        with pytest.raises(ValueError):
            mem.find_matches("AUCG", "AAAAAUCGUUU")

    def test_codon_scan_all_frames(self):
        """Scan a sequence in all 3 reading frames for codons."""
        seq = "ATGGCTAGCTAGGCTAA"
        # ATG at position 0, TAA is a stop codon at the end
        mem = DNAAssociativeMemory(threshold=0.95)
        atg_matches = mem.find_matches("ATG", seq)
        assert atg_matches[0].position == 0
        taa_matches = mem.find_matches("TAA", seq)
        assert taa_matches[0].position == len(seq) - 3


class TestDNASearchAPI:

    def test_dna_match_functional(self):
        matches = dna_match("ATCG", "GGGGAATCGGGG", threshold=0.8)
        assert len(matches) >= 1
        assert matches[0].position == 5
        assert matches[0].score > 0.99

    def test_dna_match_returns_empty_for_no_match(self):
        matches = dna_match("TTTT", "AAAAAAAA", threshold=0.5)
        assert len(matches) == 0

    def test_pre_encoded_target(self):
        """Test the two-phase API: encode target once, search many probes."""
        mem = DNAAssociativeMemory(threshold=0.8)
        target = "ATCGATCGATCGATCG"
        encoded = mem.encode_target(list(target))
        m1 = mem.search(list("ATCG"), encoded)
        m2 = mem.search(list("TCGA"), encoded)
        assert m1[0].score > 0.99
        assert m2[0].score > 0.99
        assert m1[0].position == 0
        assert m2[0].position == 1
