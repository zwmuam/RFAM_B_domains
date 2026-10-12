# fmt: off
"""
test_msa_benchmark.py

Comprehensive test suite for msa.py, msa_evaluate.py, gff_sequence_extractor.py, and
benchmark.py. Verifies preprocessing rules, IUPAC density filtering, soft-masking preservation,
orientation handling, trimming strategies, explicit pipeline error handling without silent
fallbacks, and pre/post-trim benchmark evaluation.
"""

# built-ins
import sys
from pathlib import Path

# Ensure parent directory is in sys.path when running tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# standard libraries
import pandas as pd
import pytest

# internal repository/package imports
from benchmark import convert_fasta_to_stockholm, run_benchmark_workflow
from gff_sequence_extractor import (GFFSequenceExtractor,
                                     calculate_iupac_density,
                                     reverse_complement,
                                     standardize_rna_sequence)
from msa import (AlignmentResult,
                 Muscle5Pipeline,
                 ProfileCovarianceModelPipeline,
                 RNASequenceDataset,
                 available_pipelines,
                 trim_cialign_crop_from_ends,
                 trim_consensus_structure_masking,
                 trim_trimal_gappyout)
from msa_evaluate import AlignmentEvaluator, EvaluationMetrics


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


def test_preprocessing_and_softmasking():
    # T -> U conversion preserving soft-masking
    seq_dna = "acgtACGT"
    seq_rna = standardize_rna_sequence(seq_dna, convert_to_rna=True, preserve_soft_masking=True)
    assert seq_rna == "acguACGU"

    # Reverse complement preserving soft-masking and converting to RNA
    rc_seq = standardize_rna_sequence(reverse_complement(seq_rna, preserve_case=True),
                                       convert_to_rna=True,
                                       preserve_soft_masking=True)
    assert rc_seq == "ACGUacgu"


def test_rna_sequence_dataset_iupac_filtering(tmp_path: Path):
    fasta_file = tmp_path / "test.fasta"
    # seq1: low IUPAC density, seq2: high IUPAC density (> 5%)
    fasta_file.write_text(">seq1\nACGUACGUACGUACGU\n"
                         ">seq2\nACGUNRRRYSWKMBDH\n",
                         encoding="utf-8")

    ds = RNASequenceDataset.from_fasta(fasta_file, max_iupac_density=0.05)
    assert "seq1" in ds.sequences
    assert "seq2" not in ds.sequences  # Filtered due to high IUPAC density > 5%


def test_gff_sequence_extractor(sample_fasta_and_gffs, tmp_path: Path):
    fasta_path, cluster_gff, ref_gff = sample_fasta_and_gffs
    out_dir = tmp_path / "extraction"

    extractor = GFFSequenceExtractor(fasta_path=fasta_path,
                                     primary_gff_path=cluster_gff,
                                     secondary_gff_path=ref_gff)
    seqs, adj_df = extractor.extract_and_adjust()
    assert len(seqs) == 4
    assert adj_df is not None
    assert not adj_df.empty

    exported = extractor.export_data(output_dir=out_dir,
                                     extracted_sequences=seqs,
                                     adjusted_gff_df=adj_df)
    assert len(exported) > 0
    assert (out_dir / "cluster_1.fasta").exists()
    assert (out_dir / "cluster_2.fasta").exists()


def test_trimming_strategies(tmp_path: Path):
    aln_fasta = tmp_path / "aln.fasta"
    aln_fasta.write_text(">s1\n-----ACGUACGU-----\n"
                         ">s2\n-----ACGUAGCU-----\n",
                         encoding="utf-8")

    # Test CIAlign crop from ends
    crop_out = tmp_path / "crop.fasta"
    success = trim_cialign_crop_from_ends(aln_fasta, crop_out)
    assert success
    assert crop_out.exists()
    crop_ds = RNASequenceDataset.from_fasta(crop_out)
    assert crop_ds.sequences["s1"] == "ACGUACGU"

    # Test trimAl gappyout
    trimal_out = tmp_path / "trimal.fasta"
    success = trim_trimal_gappyout(aln_fasta, trimal_out)
    assert success
    assert trimal_out.exists()

    # Test consensus structure masking
    mask_out = tmp_path / "mask.fasta"
    success = trim_consensus_structure_masking(aln_fasta, mask_out)
    assert success
    assert mask_out.exists()


def test_evaluation_metrics_penalized():
    penalized = EvaluationMetrics.create_penalized("dataset1",
                                                  "dummy_pipeline",
                                                  trimming_stage="pre-trim",
                                                  execution_time_seconds=2.5)
    p_dict = penalized.to_dict()
    assert p_dict["dataset"] == "dataset1"
    assert p_dict["pipeline"] == "dummy_pipeline"
    assert p_dict["trimming_stage"] == "pre-trim"
    assert p_dict["execution_time_seconds"] == 2.5
    assert p_dict["normalized_shannon_entropy_hn"] == 1.0
    assert p_dict["structure_conservation_index_sci"] == 0.0
    assert p_dict["transitive_consistency_score_tcs"] == 0.0


def test_alignment_evaluator(tmp_path: Path):
    aln_fasta = tmp_path / "aln.fasta"
    aln_fasta.write_text(">s1\nACGU--ACGU\n"
                         ">s2\nACGUACACGU\n",
                         encoding="utf-8")

    sim = AlignmentEvaluator.calculate_pairwise_sequence_similarity(aln_fasta)
    assert "mean_sequence_similarity" in sim
    assert sim["mean_sequence_similarity"] > 0

    hn = AlignmentEvaluator.calculate_normalized_shannon_entropy(aln_fasta)
    assert 0.0 <= hn <= 1.0

    tcs = AlignmentEvaluator.calculate_transitive_consistency_score(aln_fasta)
    assert 0.0 <= tcs <= 100.0

    cov = AlignmentEvaluator.calculate_structural_covariation_score(aln_fasta)
    assert "mean_mi_apc_covariation" in cov

    alignments_map = {"p1": {"s1": "ACGU--ACGU", "s2": "ACGUACACGU"},
                      "p2": {"s1": "ACGU--ACGU", "s2": "ACGUACACGU"}}
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


def test_pipeline_failure_handling_no_silent_fallback(tmp_path: Path):
    pipeline = ProfileCovarianceModelPipeline()
    ds = RNASequenceDataset("test", {"s1": "ACGUACGU", "s2": "ACGUAGCU"})
    out_path = tmp_path / "out.fna"

    result = pipeline.align(ds, out_path)
    assert isinstance(result, AlignmentResult)
    if not result.is_successful:
        assert ("🛑" in result.error_message or
                "Missing required executables" in result.error_message or
                "not found" in result.error_message)


def test_run_benchmark_workflow(sample_fasta_and_gffs, tmp_path: Path):
    fasta_path, cluster_gff, ref_gff = sample_fasta_and_gffs
    out_dir = tmp_path / "benchmark_output"

    df = run_benchmark_workflow(input_fasta=fasta_path,
                                cluster_gff=cluster_gff,
                                reference_gff=ref_gff,
                                output_dir=out_dir)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "dataset" in df.columns
    assert "pipeline" in df.columns
    assert "trimming_stage" in df.columns
    assert "execution_time_seconds" in df.columns
    assert "normalized_shannon_entropy_hn" in df.columns

    excel_file = out_dir / "benchmark_results.xlsx"
    assert excel_file.exists() or (out_dir / "benchmark_results.csv").exists()
# fmt: on
