# fmt: off
from pathlib import Path
import pytest
import numpy as np

import aligners
from aligners import (SequenceRegistry, count_fasta, populate_registry_from_fasta,
                      parse_mmseqs_tsv, batch_parse_mmseqs_tsv)


def test_sequence_registry_basic_and_frozen():
    reg = SequenceRegistry()
    assert len(reg) == 0
    id1 = reg.get_id("seqA")
    id2 = reg.get_id("seqB")
    assert id1 == 0
    assert id2 == 1
    assert reg.get_id("seqA") == 0
    assert reg.get_id(0) == 0
    assert reg.get_str(0) == "seqA"
    assert reg.get_str("seqA") == "seqA"
    assert "seqA" in reg
    assert "seqC" not in reg

    reg.freeze()
    assert reg.frozen
    # Existing IDs query fine
    assert reg.get_id("seqA") == 0
    # New ID triggers explicit error
    with pytest.raises(KeyError) as exc_info:
        reg.get_id("seqC")
    assert "🛑" in str(exc_info.value)
    assert "not found in pre-populated SequenceRegistry" in str(exc_info.value)


def test_count_fasta_and_populate_registry(tmp_path):
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">seq1 Description 1\nACGT\n>seq2\nTGCA\n>seq3 other\nGGCC\n")

    # Count without registry
    assert count_fasta(fasta_file) == 3

    # Populate registry
    reg = populate_registry_from_fasta(fasta_file)
    assert len(reg) == 3
    assert reg.frozen
    assert reg.get_id("seq1") == 0
    assert reg.get_id("seq2") == 1
    assert reg.get_id("seq3") == 2

    # Non-existent fasta
    non_existent = tmp_path / "non_existent.fasta"
    assert count_fasta(non_existent) == 0


def test_parse_mmseqs_tsv_sorting_and_shapes(tmp_path):
    tsv_content = (
        "seq1\tseq2\t100.0\t100\t0\t0\t50\t100\t10\t60\t1e-10\t200\n"
        "seq1\tseq3\t100.0\t100\t0\t0\t10\t80\t5\t75\t1e-10\t200\n"
        "seq1\tseq4\t100.0\t100\t0\t0\t10\t50\t20\t60\t1e-10\t200\n"
        "seq2\tseq3\t95.0\t90\t0\t0\t15\t90\t15\t90\t1e-15\t180\n"
    )
    tsv_file = tmp_path / "alignments.tsv"
    tsv_file.write_text(tsv_content)

    fasta_file = tmp_path / "seqs.fasta"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n>seq3\nACGT\n>seq4\nACGT\n")

    reg = populate_registry_from_fasta(fasta_file)
    blocks = list(parse_mmseqs_tsv(tsv_file, reg))
    assert len(blocks) == 2

    # Block 1: seq1
    q1_id, q1_spans, t1_spans = blocks[0]
    assert reg.get_str(q1_id) == "seq1"
    assert q1_spans.shape == (3, 2)
    assert t1_spans.shape == (3, 4)

    # Deterministic sorting check: primary start, tiebreaker end
    # For seq1, hits had spans: (50, 100), (10, 80), (10, 50)
    # Sorted order should be: (10, 50) -> (10, 80) -> (50, 100)
    assert np.array_equal(q1_spans[0], [10, 50])
    assert np.array_equal(q1_spans[1], [10, 80])
    assert np.array_equal(q1_spans[2], [50, 100])

    # Check that target_spans maintained identical row alignment
    assert t1_spans[0, 0] == reg.get_id("seq4")
    assert t1_spans[1, 0] == reg.get_id("seq3")
    assert t1_spans[2, 0] == reg.get_id("seq2")

    # Block 2: seq2
    q2_id, q2_spans, t2_spans = blocks[1]
    assert reg.get_str(q2_id) == "seq2"
    assert q2_spans.shape == (1, 2)
    assert t2_spans.shape == (1, 4)


def test_strand_detection_and_coordinate_normalization(tmp_path):
    tsv_content = (
        "seq1\tseq2\t100.0\t100\t0\t0\t100\t10\t10\t100\t1e-10\t200\n"  # inverted query
        "seq1\tseq3\t100.0\t100\t0\t0\t20\t80\t80\t20\t1e-10\t200\n"   # inverted target
        # both inverted (same relative strand)
        "seq1\tseq4\t100.0\t100\t0\t0\t90\t30\t90\t30\t1e-10\t200\n"
    )
    tsv_file = tmp_path / "strands.tsv"
    tsv_file.write_text(tsv_content)

    fasta_file = tmp_path / "seqs.fasta"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n>seq3\nACGT\n>seq4\nACGT\n")

    reg = populate_registry_from_fasta(fasta_file)
    blocks = list(parse_mmseqs_tsv(tsv_file, reg))
    assert len(blocks) == 1
    q_id, q_spans, t_spans = blocks[0]

    # All spans must have start <= end
    for s, e in q_spans:
        assert s <= e
    for row in t_spans:
        _tid, s, e, _strand = row
        assert s <= e

    # Find rows for seq2, seq3, seq4
    seq2_row = t_spans[t_spans[:, 0] == reg.get_id("seq2")][0]
    seq3_row = t_spans[t_spans[:, 0] == reg.get_id("seq3")][0]
    seq4_row = t_spans[t_spans[:, 0] == reg.get_id("seq4")][0]

    assert seq2_row[3] == -1  # query was inverted, target wasn't -> strand -1
    assert seq3_row[3] == -1  # query wasn't inverted, target was -> strand -1
    assert seq4_row[3] == 1   # both inverted -> strand +1


def test_batch_parse_mmseqs_tsv(tmp_path):
    tsv_content = (
        "seq1\tseq2\t100.0\t100\t0\t0\t10\t50\t10\t50\t1e-10\t200\n"
        "seq2\tseq3\t100.0\t100\t0\t0\t10\t50\t10\t50\t1e-10\t200\n"
        "seq3\tseq4\t100.0\t100\t0\t0\t10\t50\t10\t50\t1e-10\t200\n"
    )
    tsv_file = tmp_path / "batch.tsv"
    tsv_file.write_text(tsv_content)

    reg = SequenceRegistry()
    batches = list(batch_parse_mmseqs_tsv(tsv_file, n=2, registry=reg))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_empty_tsv_raises_error(tmp_path):
    empty_tsv = tmp_path / "empty.tsv"
    empty_tsv.write_text("")
    reg = SequenceRegistry()
    with pytest.raises(ValueError) as exc_info:
        list(parse_mmseqs_tsv(empty_tsv, reg))
    assert "🛑" in str(exc_info.value)
    assert "empty or contains only excluded self-hits" in str(exc_info.value)


def test_self_hits_only_raises_error(tmp_path):
    self_tsv = tmp_path / "self_only.tsv"
    self_tsv.write_text("seq1\tseq1\t100.0\t100\t0\t0\t1\t100\t1\t100\t0.0\t200\n")
    reg = SequenceRegistry()
    with pytest.raises(ValueError) as exc_info:
        list(parse_mmseqs_tsv(self_tsv, reg))
    assert "🛑" in str(exc_info.value)
    assert "empty or contains only excluded self-hits" in str(exc_info.value)


def test_non_contiguous_query_blocks_raises_error(tmp_path):
    non_contig_tsv = tmp_path / "non_contig.tsv"
    non_contig_tsv.write_text(
        "seq1\tseq2\t100.0\t100\t0\t0\t1\t50\t1\t50\t1e-5\t100\n"
        "seq2\tseq3\t100.0\t100\t0\t0\t1\t50\t1\t50\t1e-5\t100\n"
        "seq1\tseq4\t100.0\t100\t0\t0\t1\t50\t1\t50\t1e-5\t100\n"
    )
    reg = SequenceRegistry()
    with pytest.raises(ValueError) as exc_info:
        list(parse_mmseqs_tsv(non_contig_tsv, reg))
    assert "🛑" in str(exc_info.value)
    assert "not grouped contiguously" in str(exc_info.value)


def test_timeout_file_raises_error(tmp_path):
    timeout_tsv = tmp_path / "timeout.tsv"
    timeout_tsv.write_text("_TimeoutError_\n")
    reg = SequenceRegistry()
    with pytest.raises(TimeoutError) as exc_info:
        list(parse_mmseqs_tsv(timeout_tsv, reg))
    assert "🛑" in str(exc_info.value)
    assert "timed out" in str(exc_info.value)


def test_missing_tsv_raises_error(tmp_path):
    missing_tsv = tmp_path / "does_not_exist.tsv"
    reg = SequenceRegistry()
    with pytest.raises(FileNotFoundError) as exc_info:
        list(parse_mmseqs_tsv(missing_tsv, reg))
    assert "🛑" in str(exc_info.value)
# fmt: on
