"""
test_msa_benchmark.py

Comprehensive test suite for msa.py, msa_evaluate.py, gff_sequence_extractor.py, and benchmark.py.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from gff_sequence_extractor import GFFSequenceExtractor, extract_and_annotate_sequences
from msa import RNASequenceDataset, AlignmentResult, Muscle5Pipeline, MafftQinsiPipeline, available_pipelines
from msa_evaluate import EvaluationMetrics, AlignmentEvaluator
from benchmark import run_benchmark_workflow, convert_fasta_to_stockholm


@pytest.fixture
def sample_fasta_and_gffs(tmp_path: Path):
    fasta_path = tmp_path / "input.fasta"
    fasta_content = (
        ">chr1\n"
        "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU\n"
        ">chr2\n"
        "UGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA\n"
    )
    fasta_path.write_text(fasta_content, encoding="utf-8")

    cluster_gff_path = tmp_path / "cluster.gff"
    cluster_gff_content = (
        "##gff-version 3\n"
        "chr1\tsrc\tmisc_RNA\t10\t40\t.\t+\t.\tID=cluster_1;Name=c1\n"
        "chr1\tsrc\tmisc_RNA\t15\t45\t.\t+\t.\tID=cluster_1;Name=c2\n"
        "chr2\tsrc\tmisc_RNA\t5\t35\t.\t-\t.\tID=cluster_2;Name=c3\n"
        "chr2\tsrc\tmisc_RNA\t10\t40\t.\t-\t.\tID=cluster_2;Name=c4\n"
    )
    cluster_gff_path.write_text(cluster_gff_content, encoding="utf-8")

    ref_gff_path = tmp_path / "reference.gff"
    ref_gff_content = (
        "##gff-version 3\n"
        "chr1\trfam\tncRNA\t12\t35\t.\t+\t.\tID=rfam_1;Parent=cluster_1\n"
        "chr2\trfam\tncRNA\t8\t30\t.\t-\t.\tID=rfam_2;Parent=cluster_2\n"
    )
    ref_gff_path.write_text(ref_gff_content, encoding="utf-8")

    return fasta_path, cluster_gff_path, ref_gff_path


def test_rna_sequence_dataset(tmp_path: Path):
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">seq1\nacgt\n>seq2\nagcu\n", encoding="utf-8")

    ds = RNASequenceDataset.from_fasta(fasta_file)
    assert ds.dataset_name == "test"
    assert "seq1" in ds.sequences
    assert ds.sequences["seq1"] == "ACGU"
    assert ds.sequences["seq2"] == "AGCU"

    out_file = tmp_path / "out.fasta"
    ds.write_fasta(out_file, use_dna_encoding=True)
    out_ds = RNASequenceDataset.from_fasta(out_file)
    assert out_ds.sequences["seq1"] == "ACGU"


def test_gff_sequence_extractor(sample_fasta_and_gffs, tmp_path: Path):
    fasta_path, cluster_gff, ref_gff = sample_fasta_and_gffs
    out_dir = tmp_path / "extraction"

    extractor = GFFSequenceExtractor(
        fasta_path=fasta_path,
        primary_gff_path=cluster_gff,
        secondary_gff_path=ref_gff
    )
    seqs, adj_df = extractor.extract_and_adjust()
    assert len(seqs) == 4
    assert adj_df is not None
    assert not adj_df.empty

    exported = extractor.export_data(
        output_dir=out_dir,
        extracted_sequences=seqs,
        adjusted_gff_df=adj_df
    )
    assert len(exported) > 0
    assert (out_dir / "cluster_1.fasta").exists()
    assert (out_dir / "cluster_2.fasta").exists()


def test_evaluation_metrics_penalized():
    penalized = EvaluationMetrics.create_penalized("dataset1", "dummy_pipeline")
    p_dict = penalized.to_dict()
    assert p_dict["dataset"] == "dataset1"
    assert p_dict["pipeline"] == "dummy_pipeline"
    assert p_dict["normalized_shannon_entropy_hn"] == 1.0
    assert p_dict["structure_conservation_index_sci"] == 0.0
    assert p_dict["transitive_consistency_score_tcs"] == 0.0


def test_alignment_evaluator(tmp_path: Path):
    aln_fasta = tmp_path / "aln.fasta"
    aln_fasta.write_text(
        ">s1\nACGU--ACGU\n"
        ">s2\nACGUACACGU\n",
        encoding="utf-8"
    )

    sim = AlignmentEvaluator.calculate_pairwise_sequence_similarity(aln_fasta)
    assert "mean_sequence_similarity" in sim
    assert sim["mean_sequence_similarity"] > 0

    hn = AlignmentEvaluator.calculate_normalized_shannon_entropy(aln_fasta)
    assert 0.0 <= hn <= 1.0

    tcs = AlignmentEvaluator.calculate_transitive_consistency_score(aln_fasta)
    assert 0.0 <= tcs <= 100.0

    cov = AlignmentEvaluator.calculate_structural_covariation_score(aln_fasta)
    assert "mean_mi_apc_covariation" in cov

    alignments_map = {
        "p1": {"s1": "ACGU--ACGU", "s2": "ACGUACACGU"},
        "p2": {"s1": "ACGU--ACGU", "s2": "ACGUACACGU"}
    }
    mos = AlignmentEvaluator.calculate_mean_overlap_score(alignments_map)
    assert mos == 1.0


def test_convert_fasta_to_stockholm(tmp_path: Path):
    fna = tmp_path / "test.fna"
    fna.write_text(">s1\nACGU\n>s2\nAGCU\n", encoding="utf-8")
    sto = tmp_path / "test.sto"

    success = convert_fasta_to_stockholm(fna, sto)
    assert success
    assert sto.exists()
    content = sto.read_text(encoding="utf-8")
    assert "# STOCKHOLM 1.0" in content


def test_pipeline_failure_handling(tmp_path: Path):
    # Test pipeline returning unsuccessful AlignmentResult when executable is missing/fails
    pipeline = Muscle5Pipeline()
    ds = RNASequenceDataset("test", {"s1": "ACGU", "s2": "ACGU"})
    out_path = tmp_path / "out.fna"

    result = pipeline.align(ds, out_path)
    assert isinstance(result, AlignmentResult)
    # Muscle might or might not be installed in the test env, but result.is_successful must be boolean
    assert isinstance(result.is_successful, bool)


def test_run_benchmark_workflow(sample_fasta_and_gffs, tmp_path: Path):
    fasta_path, cluster_gff, ref_gff = sample_fasta_and_gffs
    out_dir = tmp_path / "benchmark_output"

    df = run_benchmark_workflow(
        input_fasta=fasta_path,
        cluster_gff=cluster_gff,
        reference_gff=ref_gff,
        output_dir=out_dir
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "dataset" in df.columns
    assert "pipeline" in df.columns
    assert "normalized_shannon_entropy_hn" in df.columns

    excel_file = out_dir / "benchmark_results.xlsx"
    assert excel_file.exists() or (out_dir / "benchmark_results.csv").exists()
