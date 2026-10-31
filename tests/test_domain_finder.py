# fmt: off
from pathlib import Path
import pytest
import numpy as np

import domain_finder
from aligners import SequenceRegistry


def test_determine_strand():
    assert domain_finder.determine_strand(np.array([-1, -1, 1])) == "-"
    assert domain_finder.determine_strand(np.array([1, 1, -1])) == "+"
    assert domain_finder.determine_strand(np.array([1, -1])) == "+"
    assert domain_finder.determine_strand(np.array([], dtype=np.int32)) == "+"


def test_merge_overlapping_spans():
    empty_spans = np.empty((0, 2), dtype=np.int32)
    assert np.array_equal(domain_finder.merge_overlapping_spans(empty_spans), empty_spans)

    # Disjoint spans
    disjoint = np.array([[10, 50], [60, 100]], dtype=np.int32)
    merged = domain_finder.merge_overlapping_spans(disjoint)
    assert np.array_equal(merged, disjoint)

    # Overlapping spans
    overlapping = np.array([[10, 50], [30, 80], [100, 150]], dtype=np.int32)
    expected = np.array([[10, 80], [100, 150]], dtype=np.int32)
    assert np.array_equal(domain_finder.merge_overlapping_spans(overlapping), expected)

    # Embedded spans
    embedded = np.array([[10, 100], [20, 50], [120, 150]], dtype=np.int32)
    expected_emb = np.array([[10, 100], [120, 150]], dtype=np.int32)
    assert np.array_equal(domain_finder.merge_overlapping_spans(embedded), expected_emb)


def test_domain_instances_on_query_basic():
    # 2 query spans clustered together, 2 target hits
    sorted_q_spans = np.array([[10, 50], [12, 52]], dtype=np.int32)
    target_spans = np.array([[2, 100, 140, -1], [3, 200, 240, -1]], dtype=np.int32)
    local_doms, all_targets = domain_finder.domain_instances_on_query(
        query_id=1,
        sorted_query_spans=sorted_q_spans,
        target_spans=target_spans,
        min_dom_sim=0.3,
        min_containment=0.5,
        reassign_targets=True,
        halo_stringency=1.0,
        min_dom_dens=0.0,
        halo_method=None
    )

    assert len(local_doms) == 1
    q_id, start, end, strand = local_doms[0]
    assert q_id == 1
    assert strand == "-"  # majority -1
    # Check that self-target is appended
    assert all_targets.shape[0] == 3  # 2 original targets + 1 self target
    assert (all_targets[:, 0] == 1).any()  # self target exists


def test_target_batch_processor():
    # Test batch processing of target sequences
    # Target sequence 1 has hits from dom 0 and dom 1 that overlap
    seq1 = (1, np.array([[10, 50], [30, 70], [20, 60]], dtype=np.int32), np.array([0, 0, 1], dtype=np.int32))
    rows, cols, inters, dom_lens = domain_finder.target_batch_processor([seq1])
    assert len(rows) == 1
    assert rows[0] == 0
    assert cols[0] == 1
    assert inters[0] == 40  # dom 0 merged to [10, 70], dom 1 is [20, 60], overlap is 40
    assert dom_lens[0] == 60
    assert dom_lens[1] == 40


def test_family_finder_and_evaluate_clusters(tmp_path):
    # End-to-end test with synthetic mini dataset
    fasta_file = tmp_path / "test.fna"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGT\n>seq3\nACGT\n")

    tsv_file = tmp_path / "test.tsv"
    tsv_content = (
        "seq1\tseq2\t95.0\t50\t0\t0\t10\t60\t20\t70\t1e-10\t200.0\n"
        "seq1\tseq3\t95.0\t50\t0\t0\t15\t65\t30\t80\t1e-10\t200.0\n"
        "seq2\tseq3\t95.0\t50\t0\t0\t25\t75\t35\t85\t1e-10\t200.0\n"
    )
    tsv_file.write_text(tsv_content)

    out_gff = tmp_path / "out.gff"
    attrs = {
        "aligner": "mmseqs2",
        "tsv": str(tsv_file),
        "min_dom_sim": 0.2,
        "dom_halo": 1.0,
        "min_dom_dens": 0.0,
        "min_fam_sim": 0.1,
        "fam_halo": 1.0,
        "min_fam_dens": 0.0,
        "halo_method": None,
        "reassign_targets": True,
        "min_containment": 0.5,
    }

    res_path = domain_finder.find_domains(
        fasta_path=fasta_file,
        attributes=attrs,
        output_gff=out_gff,
        cpu=1
    )

    assert res_path.exists()
    lines = [line.strip() for line in out_gff.open() if not line.startswith("#") and line.strip()]
    assert len(lines) > 0
    # Verify GFF format
    parts = lines[0].split("\t")
    assert len(parts) == 9
    assert parts[1] == "mmseqs2"
    assert parts[2] == "Domain"
    assert parts[6] in ("+", "-")
    assert parts[8].startswith("ID=")
# fmt: on
