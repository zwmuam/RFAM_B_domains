"""
msa_evaluate.py

Evaluation module providing scientific quality metrics for non-coding RNA multiple sequence alignments.
Calculates Structure Conservation Index (SCI), Transitive Consistency Score (TCS),
mutual information with APC covariation (MI-APC), consensus base-pair covariation scores,
normalized Shannon entropy (H_N), Mean Overlap Score (MOS), and pairwise sequence identity statistics.
"""

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from msa import RNASequenceDataset


@dataclass
class EvaluationMetrics:
    """
    Data structure aggregating scientific quality evaluation metrics for a single alignment pipeline output.
    """
    dataset_name: str
    pipeline_name: str
    structure_conservation_index_sci: float
    transitive_consistency_score_tcs: float
    mean_mi_apc_covariation: float
    consensus_bp_covariation_score: float
    compensatory_mutation_count: float
    normalized_shannon_entropy_hn: float
    mean_overlap_score_mos: float
    mean_sequence_similarity: float
    median_sequence_similarity: float
    min_sequence_similarity: float
    max_sequence_similarity: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """
        Converts EvaluationMetrics dataclass attributes into a flat dictionary suitable for DataFrame creation.

        :return: Dictionary mapping metric names to evaluated floating point or string values.
        """
        return {
            "dataset": self.dataset_name,
            "pipeline": self.pipeline_name,
            "structure_conservation_index_sci": self.structure_conservation_index_sci,
            "transitive_consistency_score_tcs": self.transitive_consistency_score_tcs,
            "mean_mi_apc_covariation": self.mean_mi_apc_covariation,
            "consensus_bp_covariation_score": self.consensus_bp_covariation_score,
            "compensatory_mutation_count": self.compensatory_mutation_count,
            "normalized_shannon_entropy_hn": self.normalized_shannon_entropy_hn,
            "mean_overlap_score_mos": self.mean_overlap_score_mos,
            "mean_sequence_similarity": self.mean_sequence_similarity,
            "median_sequence_similarity": self.median_sequence_similarity,
            "min_sequence_similarity": self.min_sequence_similarity,
            "max_sequence_similarity": self.max_sequence_similarity,
        }

    @classmethod
    def create_penalized(cls, dataset_name: str, pipeline_name: str) -> "EvaluationMetrics":
        """
        Creates an EvaluationMetrics instance with penalized worst-case metric values
        representing aligner or evaluation failure.

        :param dataset_name: Name of the sequence dataset.
        :param pipeline_name: Name of the alignment pipeline.
        :return: EvaluationMetrics instance with penalized metric values.
        """
        return cls(
            dataset_name=dataset_name,
            pipeline_name=pipeline_name,
            structure_conservation_index_sci=0.0,
            transitive_consistency_score_tcs=0.0,
            mean_mi_apc_covariation=0.0,
            consensus_bp_covariation_score=0.0,
            compensatory_mutation_count=0.0,
            normalized_shannon_entropy_hn=1.0,  # Maximum disorder/entropy
            mean_overlap_score_mos=0.0,
            mean_sequence_similarity=0.0,
            median_sequence_similarity=0.0,
            min_sequence_similarity=0.0,
            max_sequence_similarity=0.0,
        )


class AlignmentEvaluator:
    """
    Evaluator class providing methods for assessing biological alignment quality metrics,
    including structure conservation (SCI), covariation (MI-APC), consistency (TCS),
    entropy (H_N), MOS, and pairwise sequence similarity.
    """

    @staticmethod
    def calculate_pairwise_sequence_similarity(alignment_fasta_path: Path) -> Dict[str, float]:
        """
        Computes pairwise sequence similarity / identity statistics (mean, median, min, max)
        across all sequence pairs in a multiple sequence alignment.
        """
        aligned_dataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        seq_list = list(aligned_dataset.sequences.values())
        if len(seq_list) < 2:
            return {
                "mean_sequence_similarity": 100.0,
                "median_sequence_similarity": 100.0,
                "min_sequence_similarity": 100.0,
                "max_sequence_similarity": 100.0,
            }

        char_matrix = np.array([list(s) for s in seq_list])
        num_seqs, align_len = char_matrix.shape

        pair_identities: List[float] = []

        for i in range(num_seqs):
            seq1 = char_matrix[i]
            for j in range(i + 1, num_seqs):
                seq2 = char_matrix[j]
                valid_mask = ((seq1 != "-") & (seq1 != ".")) | ((seq2 != "-") & (seq2 != "."))
                total_cols = np.sum(valid_mask)
                if total_cols == 0:
                    continue
                matches = np.sum((seq1 == seq2) & valid_mask)
                identity_pct = float((matches / total_cols) * 100.0)
                pair_identities.append(identity_pct)

        if not pair_identities:
            return {
                "mean_sequence_similarity": 0.0,
                "median_sequence_similarity": 0.0,
                "min_sequence_similarity": 0.0,
                "max_sequence_similarity": 0.0,
            }

        arr = np.array(pair_identities)
        return {
            "mean_sequence_similarity": float(np.mean(arr)),
            "median_sequence_similarity": float(np.median(arr)),
            "min_sequence_similarity": float(np.min(arr)),
            "max_sequence_similarity": float(np.max(arr)),
        }

    @staticmethod
    def calculate_structure_conservation_index(alignment_fasta_path: Path) -> float:
        """
        Calculates the Structure Conservation Index (SCI) by comparing RNAalifold consensus minimum free energy (E_consensus)
        to the mean individual sequence MFEs (E_mean) from RNAfold.
        """
        aligned_dataset: RNASequenceDataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        aligned_sequences: Dict[str, str] = aligned_dataset.sequences
        if not aligned_sequences:
            return 0.0

        rnafold_exec: Optional[str] = shutil.which("RNAfold")
        alifold_exec: Optional[str] = shutil.which("RNAalifold")

        ungapped_items: List[Tuple[str, str]] = []
        for header_name, aligned_seq in aligned_sequences.items():
            ungapped_seq: str = aligned_seq.replace("-", "").replace(".", "")
            if ungapped_seq:
                ungapped_items.append((header_name, ungapped_seq))

        if not ungapped_items:
            return 0.0

        if rnafold_exec and alifold_exec:
            single_mfe_list: List[float] = []
            with tempfile.TemporaryDirectory() as temporary_directory:
                temp_dir_path: Path = Path(temporary_directory)
                batch_input: str = "\n".join([f">{h}\n{s}" for h, s in ungapped_items]) + "\n"
                try:
                    rnafold_proc = subprocess.run(
                        [rnafold_exec, "--noPS"], input=batch_input, capture_output=True, text=True, check=True,
                        cwd=temp_dir_path, timeout=60
                    )
                    for line in rnafold_proc.stdout.splitlines():
                        mfe_match = re.search(r"\(\s*(-?\d+\.\d+)\)", line)
                        if mfe_match:
                            single_mfe_list.append(float(mfe_match.group(1)))

                    if single_mfe_list:
                        mean_single_mfe: float = float(np.mean(single_mfe_list))
                        alifold_proc = subprocess.run(
                            [alifold_exec, "--noLP", "--noPS", str(alignment_fasta_path.resolve())],
                            capture_output=True, text=True, check=True, cwd=temp_dir_path, timeout=60
                        )
                        consensus_match = re.search(r"\(\s*(-?\d+\.\d+)\s*=\s*", alifold_proc.stdout) or re.search(
                            r"\(\s*(-?\d+\.\d+)\)", alifold_proc.stdout)
                        if consensus_match:
                            consensus_mfe = float(consensus_match.group(1))
                            if abs(mean_single_mfe) < 1e-6 or mean_single_mfe >= 0.0:
                                sci_val = 0.0 if abs(consensus_mfe) < 1e-6 or consensus_mfe >= 0.0 else 1.0
                            else:
                                sci_val = consensus_mfe / mean_single_mfe
                            if sci_val == 0.0 or abs(sci_val) < 1e-6:
                                return 0.0
                            return float(sci_val)
                except Exception:
                    return 0.0

        return 0.0

    @staticmethod
    def calculate_transitive_consistency_score(alignment_fasta_path: Path) -> float:
        """
        Calculates Transitive Consistency Score (TCS) measuring column residue consistency across alignment paths
        using T-Coffee evaluation or vectorized matrix computation.
        """
        tcoffee_exec: Optional[str] = shutil.which("t_coffee")
        if tcoffee_exec:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temp_dir_path: Path = Path(temporary_directory)
                score_file: Path = temp_dir_path / "tcs.score"
                eval_cmd = [tcoffee_exec, "-evaluate", "-aln", str(alignment_fasta_path.resolve()), "-output",
                            "score_ascii", "-outfile", str(score_file.resolve())]

                try:
                    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, check=False, cwd=temp_dir_path,
                                               timeout=60)
                    if eval_proc.returncode == 0 and score_file.exists():
                        file_text = score_file.read_text(encoding="utf-8")
                        match = re.search(r"SCORE\s*=\s*(\d+)", file_text)
                        if match:
                            return float(match.group(1))
                except Exception:
                    pass

        # Vectorized consistency score
        aligned_dataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        seq_list = list(aligned_dataset.sequences.values())
        if len(seq_list) < 2:
            return 100.0

        char_matrix = np.array([list(s) for s in seq_list])
        _, align_len = char_matrix.shape

        consistent_pairs = 0
        total_pairs = 0

        for col in range(align_len):
            column_chars = char_matrix[:, col]
            non_gap_mask = (column_chars != "-") & (column_chars != ".")
            valid_chars = column_chars[non_gap_mask]
            n_valid = len(valid_chars)
            if n_valid > 1:
                pairs_in_col = (n_valid * (n_valid - 1)) // 2
                total_pairs += pairs_in_col
                _, counts = np.unique(valid_chars, return_counts=True)
                consistent_pairs += int(np.sum((counts * (counts - 1)) // 2))

        if total_pairs == 0:
            return 0.0

        return float((consistent_pairs / total_pairs) * 100.0)

    @staticmethod
    def calculate_mean_overlap_score(alignments_map: Dict[str, Dict[str, str]]) -> float:
        """
        Computes the Mean Overlap Score (MOS) measuring Jaccard consensus index across aligned residue pair positions
        produced by different pipelines.
        """
        pipeline_names = list(alignments_map.keys())
        if len(pipeline_names) < 2:
            return 1.0

        pipeline_pair_sets = {}

        for name in pipeline_names:
            seq_dict = alignments_map[name]
            aligned_pairs = set()

            for seq_id, aligned_seq in seq_dict.items():
                res_index = 0
                for col_idx, char in enumerate(aligned_seq):
                    if char not in ("-", "."):
                        aligned_pairs.add((seq_id, res_index, col_idx))
                        res_index += 1

            pipeline_pair_sets[name] = aligned_pairs

        jaccard_scores = []
        n_pipes = len(pipeline_names)

        for i in range(n_pipes):
            for j in range(i + 1, n_pipes):
                set1 = pipeline_pair_sets[pipeline_names[i]]
                set2 = pipeline_pair_sets[pipeline_names[j]]

                if not set1 or not set2:
                    continue

                intersection_len = len(set1.intersection(set2))
                union_len = len(set1.union(set2))

                if union_len > 0:
                    jaccard_scores.append(intersection_len / union_len)

        if not jaccard_scores:
            return 0.0

        return float(np.mean(jaccard_scores))

    @staticmethod
    def calculate_structural_covariation_score(alignment_fasta_path: Path) -> Dict[str, float]:
        """
        Calculates mutual information (MI) with Average Product Correction (APC) covariation metrics and compensatory mutation counts.
        """
        aligned_dataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        seq_list = list(aligned_dataset.sequences.values())
        if not seq_list:
            return {
                "mean_mi_apc_covariation": 0.0,
                "consensus_bp_covariation_score": 0.0,
                "compensatory_mutation_count": 0.0,
            }

        alifold_exec = shutil.which("RNAalifold")
        consensus_dbn = ""
        if alifold_exec:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    proc = subprocess.run(
                        [alifold_exec, "--noLP", "--noPS", str(alignment_fasta_path.resolve())],
                        capture_output=True, text=True, check=True, cwd=Path(temp_dir), timeout=60
                    )
                    lines = proc.stdout.splitlines()
                    if len(lines) >= 2:
                        consensus_dbn = lines[1].split()[0]
                except Exception:
                    pass

        char_matrix = np.array([list(s) for s in seq_list])
        _, align_len = char_matrix.shape

        consensus_base_pairs = []
        if consensus_dbn:
            stack = []
            for idx, char in enumerate(consensus_dbn):
                if char in "({<[":
                    stack.append(idx)
                elif char in ")}>]" and stack:
                    start_idx = stack.pop()
                    consensus_base_pairs.append((start_idx, idx))

        nuc_map = np.full((256,), 4, dtype=np.int32)
        nuc_map[ord("A")] = 0
        nuc_map[ord("C")] = 1
        nuc_map[ord("G")] = 2
        nuc_map[ord("U")] = 3

        int_matrix = np.vectorize(lambda c: nuc_map[ord(c)] if ord(c) < 256 else 4)(char_matrix)
        mi_matrix = np.zeros((align_len, align_len), dtype=float)

        for i in range(align_len):
            col_i = int_matrix[:, i]
            valid_i_mask = col_i < 4
            for j in range(i + 1, align_len):
                col_j = int_matrix[:, j]
                valid_mask = valid_i_mask & (col_j < 4)
                n_valid = int(np.sum(valid_mask))
                if n_valid < 3:
                    continue

                v_i = col_i[valid_mask]
                v_j = col_j[valid_mask]
                pair_codes = v_i * 4 + v_j

                counts_2d = np.bincount(pair_codes, minlength=16).reshape((4, 4))
                p_ij = counts_2d / n_valid
                p_i = np.sum(p_ij, axis=1)
                p_j = np.sum(p_ij, axis=0)

                nonzero_mask = p_ij > 0
                if not np.any(nonzero_mask):
                    continue

                outer_prod = np.outer(p_i, p_j)
                valid_mi_cells = nonzero_mask & (outer_prod > 0)
                mi_val = float(
                    np.sum(p_ij[valid_mi_cells] * np.log2(p_ij[valid_mi_cells] / outer_prod[valid_mi_cells])))

                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val

        mean_mi_rows = np.mean(mi_matrix, axis=1)
        overall_mean_mi = float(np.mean(mi_matrix))
        mi_apc_matrix = np.zeros_like(mi_matrix)

        if overall_mean_mi > 0:
            apc_outer = np.outer(mean_mi_rows, mean_mi_rows) / overall_mean_mi
            mi_apc_matrix = np.maximum(0.0, mi_matrix - apc_outer)
            np.fill_diagonal(mi_apc_matrix, 0.0)

        upper_tri_indices = np.triu_indices(align_len, k=1)
        mean_apc_cov = float(np.mean(mi_apc_matrix[upper_tri_indices])) if len(upper_tri_indices[0]) > 0 else 0.0

        bp_cov_scores = []
        compensatory_count = 0
        canonical_pairs = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}

        for i, j in consensus_base_pairs:
            if i < align_len and j < align_len:
                bp_apc_score = float(mi_apc_matrix[i, j])
                bp_cov_scores.append(bp_apc_score)

                col_i_str = char_matrix[:, i]
                col_j_str = char_matrix[:, j]
                observed_pairs = set(zip(col_i_str, col_j_str))
                valid_canonical_observed = {p for p in observed_pairs if p in canonical_pairs}

                if len(valid_canonical_observed) >= 2 and bp_apc_score > 0.01:
                    compensatory_count += 1

        consensus_bp_cov = float(np.mean(bp_cov_scores)) if bp_cov_scores else mean_apc_cov * 1.5

        return {
            "mean_mi_apc_covariation": float(mean_apc_cov),
            "consensus_bp_covariation_score": float(consensus_bp_cov),
            "compensatory_mutation_count": float(compensatory_count),
        }

    @staticmethod
    def calculate_normalized_shannon_entropy(alignment_fasta_path: Path) -> float:
        """
        Computes column-wise Normalized Shannon Entropy (H_N) across non-gap sequence positions in the alignment.
        Preserves IUPAC ambiguous/degenerate nucleotide codes rather than filtering them out.
        """
        aligned_dataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        seq_list = list(aligned_dataset.sequences.values())
        if not seq_list:
            return 1.0  # Max entropy for empty alignment

        char_matrix = np.array([list(s) for s in seq_list])
        _, align_len = char_matrix.shape

        column_entropies = []
        valid_iupac = {"A", "C", "G", "U", "N", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"}

        for col in range(align_len):
            col_chars = char_matrix[:, col]
            valid_chars = np.array([c for c in col_chars if c in valid_iupac])
            n_valid = len(valid_chars)
            if n_valid == 0:
                continue

            _, counts = np.unique(valid_chars, return_counts=True)
            probs = counts / n_valid
            entropy = float(-np.sum(probs * np.log2(probs)))
            max_entropy = math.log2(len(valid_iupac)) if len(valid_iupac) > 1 else math.log2(4)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            column_entropies.append(normalized_entropy)

        if not column_entropies:
            return 1.0

        return float(np.mean(column_entropies))
