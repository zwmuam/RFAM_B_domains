# fmt: off
"""
generate_mock_results.py

Generates a realistic mock_results.xlsx file simulating 100-cluster alignment evaluation output
across all 6 alignment pipelines and 15 metrics (including execution_time_seconds and
memory_peak_mb).
"""

# built-ins
import sys
from pathlib import Path

# Ensure parent directory is in sys.path when running from tests directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# standard libraries
import numpy as np
import pandas as pd

# internal repository/package imports
from msa import available_pipelines
from msa_evaluate import EvaluationMetrics


def generate_mock_excel(output_path: Path) -> None:
    np.random.seed(42)  # Reproducibility

    pipelines = [p.name for p in available_pipelines]
    num_clusters = 100
    records = []

    for c in range(1, num_clusters + 1):
        cluster_id = f"cluster_{c:03d}"

        # Determine if one pipeline fails for this cluster (~5% probability)
        failing_pipe_idx = (
            np.random.choice(len(pipelines)) if np.random.rand() < 0.05 else None
        )

        base_sim = float(np.random.uniform(45.0, 95.0))

        for idx, pipe_name in enumerate(pipelines):
            # Realistic execution time and peak memory footprint based on tool complexity
            if pipe_name in ("mafft_qinsi", "mafft_xinsi"):
                exec_time = float(np.random.uniform(3.5, 12.0))
                peak_mem = float(np.random.uniform(180.0, 450.0))
            elif pipe_name == "rcoffee":
                exec_time = float(np.random.uniform(2.0, 8.0))
                peak_mem = float(np.random.uniform(120.0, 320.0))
            elif pipe_name == "muscle5":
                exec_time = float(np.random.uniform(0.3, 1.8))
                peak_mem = float(np.random.uniform(60.0, 150.0))
            else:
                exec_time = float(np.random.uniform(0.2, 1.5))
                peak_mem = float(np.random.uniform(40.0, 110.0))

            if failing_pipe_idx is not None and idx == failing_pipe_idx:
                penalized = EvaluationMetrics.create_penalized(dataset_name=cluster_id,
                                                               pipeline_name=pipe_name,
                                                               execution_time_seconds=exec_time,
                                                               memory_peak_mb=peak_mem)
                records.append(penalized.to_dict())
                continue

            is_structural = pipe_name in ("mafft_qinsi", "mafft_xinsi", "rcoffee",
                                         "structural_encoding")

            sci = float(np.clip(np.random.normal(0.85 if is_structural else 0.65, 0.1),
                                0.0, 1.3))
            tcs = float(np.clip(np.random.normal(82.0 if is_structural else 70.0, 8.0),
                                0.0, 100.0))
            mi_apc = float(np.clip(np.random.normal(0.12 if is_structural else 0.05, 0.03),
                                   0.0, 0.5))
            bp_cov = float(np.clip(mi_apc * np.random.uniform(1.2, 1.8), 0.0, 0.8))
            comp_mut = float(np.random.poisson(lam=4 if is_structural else 1))
            hn = float(np.clip(np.random.normal(0.25, 0.05), 0.05, 0.8))
            mos = float(np.clip(np.random.normal(0.88, 0.05), 0.4, 1.0))

            mean_sim = float(np.clip(np.random.normal(base_sim, 3.0), 20.0, 100.0))
            med_sim = float(np.clip(mean_sim + np.random.uniform(-1.0, 1.0), 20.0, 100.0))
            min_sim = float(np.clip(mean_sim - np.random.uniform(5.0, 15.0), 10.0, 100.0))
            max_sim = float(np.clip(mean_sim + np.random.uniform(5.0, 15.0), 20.0, 100.0))

            metrics = EvaluationMetrics(dataset_name=cluster_id,
                                        pipeline_name=pipe_name,
                                        execution_time_seconds=exec_time,
                                        memory_peak_mb=peak_mem,
                                        structure_conservation_index_sci=sci,
                                        transitive_consistency_score_tcs=tcs,
                                        mean_mi_apc_covariation=mi_apc,
                                        consensus_bp_covariation_score=bp_cov,
                                        compensatory_mutation_count=comp_mut,
                                        normalized_shannon_entropy_hn=hn,
                                        mean_overlap_score_mos=mos,
                                        mean_sequence_similarity=mean_sim,
                                        median_sequence_similarity=med_sim,
                                        min_sequence_similarity=min_sim,
                                        max_sequence_similarity=max_sim)
            records.append(metrics.to_dict())

    df = pd.DataFrame(records)

    excel_path = Path(output_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Alignment Metrics", index=False)
        summary_df = df.groupby("pipeline").mean(numeric_only=True).reset_index()
        summary_df.to_excel(writer, sheet_name="Pipeline Means Summary", index=False)


if __name__ == "__main__":
    generate_mock_excel(Path("tests/mock_results.xlsx"))
# fmt: on
