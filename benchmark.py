"""
Benchmarking the performance of the different ALNS algorithms
1. Use gff_sequence_extractor to extract predicted clusters from fasta/gff files with additional gff providing reference RFAM annotations
2. Use align_and_evaluate to align the predicted clusters with the reference annotations using different alignment methods
2a. Gather runtime and memory usage for each alignment method
3. Evaluate the alignment results using different evaluation metrics and generate comprehensive Excel report that gathers all the results and metrics for each cluster and alignment method
4. Generate plots and visualizations to compare the performance of the different ALNS algorithms based on the evaluation metrics
"""

from pathlib import Path

import psutil

from gff_sequence_extractor import GFFSequenceExtractor
from align_and_evaluate import AlignmentEvaluator, AlignmentResult, available_pipelines, RNASequenceDataset

if __name__ == '__main__':
    features_to_extract = Path('XXX.gff')
    reference_annotations = Path('XXX.gff')
    fasta_file = Path('XXX.fasta')
    output_dir = Path('benchmark_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = GFFSequenceExtractor(fasta_path=fasta_file,
                                     primary_gff_path=features_to_extract,
                                     secondary_gff_path=reference_annotations,
                                     flank_length=0)
    seqs, adj_df = extractor.extract_and_adjust()

    exported_files = extractor.export_data(output_dir=output_dir,
                                           extracted_sequences=seqs,
                                           adjusted_gff_df=adj_df)

    for pipeline in available_pipelines:
        output_subdir = output_dir.joinpath(pipeline.__name__)
        output_subdir.mkdir(parents=True, exist_ok=True)
        for file in exported_files:

            dataset: RNASequenceDataset = RNASequenceDataset.from_fasta(fasta_file)
            if not dataset.sequences or len(dataset.sequences) < 2:
                continue
            out_path = output_subdir.joinpath(f"{file.stem}_alignment_result.pkl")
            result: AlignmentResult = pipeline.align(dataset, out_path)

