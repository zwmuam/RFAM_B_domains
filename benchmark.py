"""
benchmark.py

Benchmarking framework for non-coding RNA Multiple Sequence Alignment (MSA) algorithms.

Workflow:
1. Extract annotations specific to a certain ID number (GFF "ID" attribute) from cluster.gff + input.fasta pair
   and create coordinate-shifted extraction of reference annotations (extracted FASTA, shifted/filtered reference GFF).
2. Align every FASTA with every aligner in msa.py and output alignments in aligned FASTA (.fna) and Stockholm (.sto) format for manual inspection.
3. Evaluate all alignments with every metric to produce per-alignment metric DataFrame. On aligner failure or missing output,
   penalize as worst possible metric value (normalized_shannon_entropy_hn=1.0, all other metrics=0.0).
4. Export the metric DataFrame to Excel (.xlsx) format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gff_sequence_extractor import GFFSequenceExtractor
from msa import AlignmentResult, RNASequenceDataset, available_pipelines
from msa_evaluate import AlignmentEvaluator, EvaluationMetrics


def convert_fasta_to_stockholm(fasta_path: Path, sto_path: Path) -> bool:
    """
    Converts an aligned FASTA file (.fna) to Stockholm (.sto) format using BioPython.

    :param fasta_path: Path to the aligned FASTA file.
    :param sto_path: Path where the Stockholm file should be saved.
    :return: True if conversion was successful, False otherwise.
    """
    try:
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            return False
        sto_records = []
        for rec in records:
            sto_records.append(SeqRecord(Seq(str(rec.seq)), id=rec.id, description=""))
        SeqIO.write(sto_records, str(sto_path), "stockholm")
        return sto_path.exists() and sto_path.stat().st_size > 0
    except Exception:
        return False


def run_benchmark_workflow(
    input_fasta: Path,
    cluster_gff: Path,
    reference_gff: Path,
    output_dir: Path,
    flank_length: int = 0
) -> pd.DataFrame:
    """
    Executes the 4-step ncRNA MSA benchmark workflow:
      Step 1: Extract ID-specific sequences and shifted reference GFFs.
      Step 2: Align extracted FASTAs across all pipelines and export .fna and .sto files.
      Step 3: Evaluate all alignments across quality metrics with failure penalization.
      Step 4: Export full metrics DataFrame to Excel (.xlsx) workbook.

    :param input_fasta: Path to input FASTA file containing target sequences.
    :param cluster_gff: Path to primary cluster GFF file.
    :param reference_gff: Path to secondary/reference annotation GFF file.
    :param output_dir: Destination directory path for benchmark outputs.
    :param flank_length: Flanking nucleotides length around features.
    :return: Summary DataFrame of evaluated alignment metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extraction
    extraction_dir = out_dir / "extracted_data"
    extractor = GFFSequenceExtractor(
        fasta_path=input_fasta,
        primary_gff_path=cluster_gff,
        secondary_gff_path=reference_gff,
        flank_length=flank_length
    )
    seqs, adj_df = extractor.extract_and_adjust()
    exported_files = extractor.export_data(
        output_dir=extraction_dir,
        extracted_sequences=seqs,
        adjusted_gff_df=adj_df
    )

    extracted_fasta_files: List[Path] = sorted(
        [f for f in exported_files if f.suffix.lower() in (".fasta", ".fa", ".fna")]
    )

    alignments_dir = out_dir / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)

    metrics_list: List[EvaluationMetrics] = []
    evaluator = AlignmentEvaluator()

    # Step 2 & 3: Align and Evaluate
    for fasta_file in extracted_fasta_files:
        dataset: RNASequenceDataset = RNASequenceDataset.from_fasta(fasta_file)
        dataset_id = dataset.dataset_name

        if not dataset.sequences or len(dataset.sequences) < 2:
            continue

        cluster_alignments_dir = alignments_dir / dataset_id
        cluster_alignments_dir.mkdir(parents=True, exist_ok=True)

        alignment_results: Dict[str, AlignmentResult] = {}
        alignments_map: Dict[str, Dict[str, str]] = {}

        for pipeline in available_pipelines:
            fna_path = cluster_alignments_dir / f"{dataset_id}_{pipeline.name}.fna"
            sto_path = cluster_alignments_dir / f"{dataset_id}_{pipeline.name}.sto"

            result: AlignmentResult = pipeline.align(dataset, fna_path)
            alignment_results[pipeline.name] = result

            if result.is_successful and result.aligned_sequences and fna_path.exists():
                alignments_map[pipeline.name] = result.aligned_sequences
                convert_fasta_to_stockholm(fna_path, sto_path)

        mos_score: float = evaluator.calculate_mean_overlap_score(alignments_map)

        for pipeline in available_pipelines:
            pipe_name = pipeline.name
            result = alignment_results.get(pipe_name)
            exec_time = result.execution_time_seconds if result else 0.0
            peak_mb = result.memory_peak_mb if result else 0.0

            if not result or not result.is_successful or not result.aligned_fasta_path or not result.aligned_fasta_path.exists():
                penalized_metrics = EvaluationMetrics.create_penalized(
                    dataset_name=dataset_id,
                    pipeline_name=pipe_name,
                    execution_time_seconds=exec_time,
                    memory_peak_mb=peak_mb
                )
                metrics_list.append(penalized_metrics)
                continue

            aln_path: Path = result.aligned_fasta_path

            try:
                sci_val: float = evaluator.calculate_structure_conservation_index(aln_path)
                tcs_val: float = evaluator.calculate_transitive_consistency_score(aln_path)
                cov_dict: Dict[str, float] = evaluator.calculate_structural_covariation_score(aln_path)
                hn_val: float = evaluator.calculate_normalized_shannon_entropy(aln_path)
                sim_dict: Dict[str, float] = evaluator.calculate_pairwise_sequence_similarity(aln_path)

                metrics = EvaluationMetrics(
                    dataset_name=dataset_id,
                    pipeline_name=pipe_name,
                    execution_time_seconds=exec_time,
                    memory_peak_mb=peak_mb,
                    structure_conservation_index_sci=sci_val,
                    transitive_consistency_score_tcs=tcs_val,
                    mean_mi_apc_covariation=cov_dict["mean_mi_apc_covariation"],
                    consensus_bp_covariation_score=cov_dict["consensus_bp_covariation_score"],
                    compensatory_mutation_count=cov_dict["compensatory_mutation_count"],
                    normalized_shannon_entropy_hn=hn_val,
                    mean_overlap_score_mos=mos_score,
                    mean_sequence_similarity=sim_dict["mean_sequence_similarity"],
                    median_sequence_similarity=sim_dict["median_sequence_similarity"],
                    min_sequence_similarity=sim_dict["min_sequence_similarity"],
                    max_sequence_similarity=sim_dict["max_sequence_similarity"],
                )
                metrics_list.append(metrics)
            except Exception:
                metrics_list.append(EvaluationMetrics.create_penalized(
                    dataset_id, pipe_name, execution_time_seconds=exec_time, memory_peak_mb=peak_mb
                ))

    # Step 4: Export to Excel
    records: List[Dict[str, Union[str, float]]] = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame(records)

    excel_path = out_dir / "benchmark_results.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Alignment Metrics", index=False)
            if not df.empty:
                summary_df = df.groupby("pipeline").mean(numeric_only=True).reset_index()
                summary_df.to_excel(writer, sheet_name="Pipeline Means Summary", index=False)
    except Exception:
        csv_path = excel_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    features_to_extract = Path("cluster.gff")
    reference_annotations = Path("reference.gff")
    fasta_file = Path("input.fasta")
    output_dir = Path("benchmark_results")

    if fasta_file.exists() and features_to_extract.exists():
        run_benchmark_workflow(
            input_fasta=fasta_file,
            cluster_gff=features_to_extract,
            reference_gff=reference_annotations if reference_annotations.exists() else None,
            output_dir=output_dir
        )
