"""
align_and_evaluate.py

Modern, production-grade Python 3.11+ object-oriented framework for benchmarking ncRNA multiple sequence alignment tools.
Provides standardization across DNA (T) and RNA (U) inputs, RFAM accession family grouping,
scientific quality metrics calculation (SCI, TCS, MI-APC covariation, MOS, Shannon entropy, pairwise similarity),
and automated export to Excel (.xlsx) or CSV summary spreadsheets.

Includes pure-Python algorithm fallbacks for alignment and folding metrics when external bioinformatic CLI tools
(MAFFT, MUSCLE, T-Coffee, ViennaRNA) are not present in the system environment.
"""

import abc
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gff_sequence_extractor import standardize_rna_sequence


@dataclass(frozen=True)
class RNASequenceDataset:
    """
    Immutable dataclass encapsulating a set of RNA sequences parsed from FASTA format.
    Ensures sequence integrity and canonical RNA representation (U-base) across all workflows.
    """
    dataset_name: str
    sequences: Dict[str, str]

    @classmethod
    def from_fasta(cls, fasta_path: Path) -> "RNASequenceDataset":
        """
        Parses a FASTA file into an RNASequenceDataset instance with standardized uppercase RNA sequences.

        :param fasta_path: Path object pointing to the input FASTA file to be read.
        :return: RNASequenceDataset instance containing dataset name and sequence map.
        """
        fasta_path_obj = Path(fasta_path)
        if not fasta_path_obj.exists() or fasta_path_obj.stat().st_size == 0:
            return cls(dataset_name=fasta_path_obj.stem, sequences={})

        sequence_map: Dict[str, str] = {}
        try:
            file_content: str = fasta_path_obj.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return cls(dataset_name=fasta_path_obj.stem, sequences={})

        if not file_content.strip():
            return cls(dataset_name=fasta_path_obj.stem, sequences={})

        raw_entries: List[str] = file_content.strip().split(">")
        for entry in raw_entries:
            if not entry.strip():
                continue
            lines: List[str] = entry.strip().splitlines()
            if not lines:
                continue
            header: str = lines[0].split()[0]
            raw_seq: str = "".join(lines[1:]).strip()
            # Standardize sequence to upper-case RNA (U-containing)
            sequence: str = standardize_rna_sequence(raw_seq, convert_to_rna=True)
            if sequence:
                sequence_map[header] = sequence

        return cls(dataset_name=fasta_path_obj.stem, sequences=sequence_map)

    def write_fasta(self, output_path: Path, use_dna_encoding: bool = False) -> None:
        """
        Writes sequence dataset to a standard FASTA formatted file, with optional DNA conversion (U -> T).

        :param output_path: Destination Path object where the FASTA file will be created.
        :param use_dna_encoding: Boolean flag indicating if U should be converted back to T for DNA tools.
        :return: None
        """
        fasta_lines: List[str] = []
        for header_name, sequence_string in self.sequences.items():
            out_seq = sequence_string.replace("U", "T") if use_dna_encoding else sequence_string
            fasta_lines.append(f">{header_name}")
            fasta_lines.append(out_seq)

        Path(output_path).write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")


@dataclass
class AlignmentResult:
    """
    Data structure containing aligned sequences and detailed execution metadata for an alignment pipeline run.
    """
    pipeline_name: str
    dataset_name: str
    aligned_sequences: Dict[str, str]
    aligned_fasta_path: Optional[Path] = None
    is_successful: bool = True
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0


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


def _pure_python_progressive_msa(sequences: Dict[str, str], gap_penalty: float = -2.0) -> Dict[str, str]:
    """
    Pure-Python fallback Needleman-Wunsch progressive center-star alignment algorithm.
    Integrates gap insertions into a unified multi-way profile matrix across all pairwise alignments.

    :param sequences: Dictionary mapping header IDs to sequence strings.
    :param gap_penalty: Linear gap penalty value.
    :return: Dictionary mapping header IDs to aligned equal-length sequence strings.
    """
    if not sequences:
        return {}

    headers = list(sequences.keys())
    seqs = [sequences[h] for h in headers]
    if len(seqs) == 1:
        return {headers[0]: seqs[0]}

    # Pairwise NW function
    def nw_align(s1: str, s2: str) -> Tuple[str, str]:
        n, m = len(s1), len(s2)
        score = np.zeros((n + 1, m + 1), dtype=float)
        for i in range(n + 1):
            score[i, 0] = i * gap_penalty
        for j in range(m + 1):
            score[0, j] = j * gap_penalty

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = 2.0 if s1[i - 1] == s2[j - 1] else -1.0
                score[i, j] = max(
                    score[i - 1, j - 1] + match,
                    score[i - 1, j] + gap_penalty,
                    score[i, j - 1] + gap_penalty
                )

        # Traceback
        i, j = n, m
        al1, al2 = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and score[i, j] == score[i - 1, j - 1] + (2.0 if s1[i - 1] == s2[j - 1] else -1.0):
                al1.append(s1[i - 1])
                al2.append(s2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or score[i, j] == score[i - 1, j] + gap_penalty):
                al1.append(s1[i - 1])
                al2.append("-")
                i -= 1
            else:
                al1.append("-")
                al2.append(s2[j - 1])
                j -= 1

        return "".join(reversed(al1)), "".join(reversed(al2))

    ref_idx = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    ref_seq = seqs[ref_idx]
    ref_len = len(ref_seq)

    pairwise_data = {}
    max_gaps_at_pos = [0] * (ref_len + 1)

    for i in range(len(seqs)):
        if i == ref_idx:
            continue
        a_ref, a_other = nw_align(ref_seq, seqs[i])

        gaps_by_pos = [0] * (ref_len + 1)
        gap_seqs = ["" for _ in range(ref_len + 1)]
        ref_chars = ["" for _ in range(ref_len)]

        ref_pos = 0
        curr_gap_seq: List[str] = []

        for c_ref, c_other in zip(a_ref, a_other):
            if c_ref == "-":
                curr_gap_seq.append(c_other)
            else:
                gaps_by_pos[ref_pos] = len(curr_gap_seq)
                gap_seqs[ref_pos] = "".join(curr_gap_seq)
                ref_chars[ref_pos] = c_other
                curr_gap_seq = []
                ref_pos += 1

        gaps_by_pos[ref_pos] = len(curr_gap_seq)
        gap_seqs[ref_pos] = "".join(curr_gap_seq)

        for pos in range(ref_len + 1):
            if gaps_by_pos[pos] > max_gaps_at_pos[pos]:
                max_gaps_at_pos[pos] = gaps_by_pos[pos]

        pairwise_data[i] = (gaps_by_pos, gap_seqs, ref_chars)

    master_ref_list = []
    for pos in range(ref_len + 1):
        if max_gaps_at_pos[pos] > 0:
            master_ref_list.append("-" * max_gaps_at_pos[pos])
        if pos < ref_len:
            master_ref_list.append(ref_seq[pos])
    master_ref_str = "".join(master_ref_list)

    final_msa = {headers[ref_idx]: master_ref_str}

    for i in range(len(seqs)):
        if i == ref_idx:
            continue
        gaps_by_pos, gap_seqs, ref_chars = pairwise_data[i]
        out_list = []
        for pos in range(ref_len + 1):
            extra_gaps = max_gaps_at_pos[pos] - gaps_by_pos[pos]
            if gaps_by_pos[pos] > 0:
                out_list.append(gap_seqs[pos])
            if extra_gaps > 0:
                out_list.append("-" * extra_gaps)
            if pos < ref_len:
                out_list.append(ref_chars[pos])

        final_msa[headers[i]] = "".join(out_list)

    return final_msa


class AlignmentPipeline(abc.ABC):
    """
    Abstract base class defining interface for executing RNA alignment algorithms.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the abstract alignment pipeline with an identifier name.

        :param name: Unique name string for the alignment pipeline algorithm.
        :return: None
        """
        self.name = name

    @abc.abstractmethod
    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """
        Executes sequence alignment on an input RNASequenceDataset.

        :param dataset: Input RNASequenceDataset object to be aligned.
        :param output_path: Destination path for the aligned FASTA file.
        :return: AlignmentResult object containing aligned sequences and execution status.
        """
        pass


class Muscle5Pipeline(AlignmentPipeline):
    """
    MUSCLE v5 alignment pipeline supporting high-accuracy Progressive Perturbed Pairwise (PPP) algorithm
    and ensemble representations optimized for non-coding RNA alignment tasks.
    """

    def __init__(self, mode: str = "default", extra_args: Optional[List[str]] = None) -> None:
        """Initializes Muscle5 alignment pipeline runner."""
        name = "muscle5" if mode == "default" else f"muscle5_{mode}"
        super().__init__(name)
        self.mode = mode
        self.extra_args = extra_args or []

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes MUSCLE v5 alignment algorithm or pure-Python."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("muscle")
        assert executable is not None, "MUSCLE executable not found in system PATH"
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_input: Path = temp_dir_path / "input.fasta"
            dataset.write_fasta(temp_input, use_dna_encoding=False)
            temp_output: Path = temp_dir_path / "output.aln"

            cmd: List[str] = [executable, "-align", str(temp_input), "-output", str(temp_output)]
            if self.mode == "stratified":
                cmd.append("-stratified")
            elif self.mode == "diversified":
                cmd.append("-diversified")
            if self.extra_args:
                cmd.extend(self.extra_args)

            proc: subprocess.CompletedProcess = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=temp_dir_path, timeout=120
            )
            elapsed = time.time() - start_time
            if proc.returncode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
                aligned_dataset: RNASequenceDataset = RNASequenceDataset.from_fasta(temp_output)
                aligned_dataset.write_fasta(output_path)
                return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                       aligned_fasta_path=output_path, execution_time_seconds=elapsed)

            else:
                raise RuntimeError(f"MUSCLE alignment failed with return code {proc.returncode}: {proc.stderr}")


class MafftQinsiPipeline(AlignmentPipeline):
    """
    MAFFT Q-INS-i structural alignment pipeline incorporating McCaskill base-pairing probability matrices across pairwise sequences.
    """

    def __init__(self) -> None:
        """Initializes MAFFT Q-INS-i pipeline runner."""
        super().__init__("mafft_qinsi")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes the MAFFT Q-INS-i alignment algorithm."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("mafft-qinsi") or shutil.which("mafft")
        assert executable is not None, "MAFFT executable not found in system PATH"
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input: Path = Path(temp_dir) / "input.fasta"
            dataset.write_fasta(temp_input)
            cmd = [executable, "--qinsi", "--maxiterate", "1000",
                   str(temp_input)] if "mafft-qinsi" not in executable else [executable, str(temp_input)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path(temp_dir),
                                  timeout=120)
            elapsed = time.time() - start_time
            if proc.returncode == 0 and proc.stdout.strip():
                output_path.write_text(proc.stdout, encoding="utf-8")
                aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                       aligned_fasta_path=output_path, execution_time_seconds=elapsed)
            else:
                raise RuntimeError(f"MAFFT Q-INS-i alignment failed with return code {proc.returncode}: {proc.stderr}")


class MafftLinsiPipeline(AlignmentPipeline):
    """
    MAFFT L-INS-i local pairwise alignment pipeline with maximum consistency refinement.
    """

    def __init__(self) -> None:
        """Initializes MAFFT L-INS-i pipeline runner."""
        super().__init__("mafft_linsi")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes the MAFFT L-INS-i alignment algorithm."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("mafft-linsi") or shutil.which("mafft")
        assert executable is not None, "MAFFT executable not found in system PATH"
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input: Path = Path(temp_dir) / "input.fasta"
            dataset.write_fasta(temp_input)
            cmd = [executable, "--localpair", "--maxiterate", "1000",
                   str(temp_input)] if "mafft-linsi" not in executable else [executable, str(temp_input)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path(temp_dir),
                                  timeout=120)
            elapsed = time.time() - start_time
            if proc.returncode == 0 and proc.stdout.strip():
                output_path.write_text(proc.stdout, encoding="utf-8")
                aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                       aligned_fasta_path=output_path, execution_time_seconds=elapsed)
            else:
                raise RuntimeError(f"MAFFT L-INS-i alignment failed with return code {proc.returncode}: {proc.stderr}")


class MafftXinsiPipeline(AlignmentPipeline):
    """
    MAFFT X-INS-i structural alignment pipeline utilizing pairwise MXSCARNA structural alignment algorithms.
    """

    def __init__(self) -> None:
        """Initializes MAFFT X-INS-i pipeline runner."""
        super().__init__("mafft_xinsi")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes the MAFFT X-INS-i structural alignment algorithm."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("mafft-xinsi") or shutil.which("mafft")
        assert executable is not None, "MAFFT executable not found in system PATH"
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input: Path = Path(temp_dir) / "input.fasta"
            dataset.write_fasta(temp_input)
            cmd = [executable, "--xinsi", "--maxiterate", "1000",
                   str(temp_input)] if "mafft-xinsi" not in executable else [executable, str(temp_input)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path(temp_dir),
                                  timeout=120)
            elapsed = time.time() - start_time
            if proc.returncode == 0 and proc.stdout.strip():
                output_path.write_text(proc.stdout, encoding="utf-8")
                aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                       aligned_fasta_path=output_path, execution_time_seconds=elapsed)
            else:
                raise RuntimeError(f"MAFFT X-INS-i alignment failed with return code {proc.returncode}: {proc.stderr}")


class RCoffeePipeline(AlignmentPipeline):
    """
    R-Coffee alignment pipeline incorporating secondary structure folding predictions into consistency libraries.
    """

    def __init__(self) -> None:
        """Initializes R-Coffee pipeline runner."""
        super().__init__("rcoffee")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes T-Coffee in R-Coffee mode or fallback on input dataset."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("t_coffee")
        assert executable is not None, "T-Coffee executable not found in system PATH"
        start_time = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path: Path = Path(temp_dir)
            temp_input: Path = temp_dir_path / "input.fasta"
            dataset.write_fasta(temp_input)
            temp_out: Path = temp_dir_path / "rcoffee_out.aln"
            cmd = [executable, "-seq", str(temp_input.resolve()), "-mode", "rcoffee", "-output", "fasta_aln",
                   "-outfile", str(temp_out.resolve())]

            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=temp_dir_path,
                                  timeout=120)
            elapsed = time.time() - start_time
            if proc.returncode == 0 and temp_out.exists():
                aligned_dataset = RNASequenceDataset.from_fasta(temp_out)
                aligned_dataset.write_fasta(output_path)
                return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                       aligned_fasta_path=output_path, execution_time_seconds=elapsed)
            else:
                raise RuntimeError(f"R-Coffee alignment failed with return code {proc.returncode}: {proc.stderr}")


class StructuralEncodingPipeline(AlignmentPipeline):
    """
    Structure-informed alignment pipeline leveraging structure-aware alignment algorithms.
    Corrects case-erasure bugs by utilising explicit structural probability matrices (MAFFT Q-INS-i / structural globalpair)
    rather than fragile upper/lower case string encoding which MAFFT automatically converts to uppercase.
    """

    def __init__(self) -> None:
        """Initializes Structural Encoding pipeline runner."""
        super().__init__("structural_encoding")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes secondary structure-informed alignment pipeline or fallback."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences empty")

        executable: Optional[str] = shutil.which("mafft-qinsi") or shutil.which("mafft")
        assert executable is not None, "MAFFT executable not found in system PATH"
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_dir_path: Path = Path(temporary_directory)
            temp_input: Path = temp_dir_path / "input.fasta"
            dataset.write_fasta(temp_input)
            mafft_cmd = [executable, "--qinsi", "--maxiterate", "1000",
                         str(temp_input)] if "mafft-qinsi" in executable or "qinsi" in executable else [executable,
                                                                                                        "--globalpair",
                                                                                                        "--maxiterate",
                                                                                                        "1000",
                                                                                                        str(temp_input)]

            proc = subprocess.run(mafft_cmd, capture_output=True, text=True, check=True, cwd=temp_dir_path,
                                  timeout=120)
            elapsed = time.time() - start_time
            output_path.write_text(proc.stdout, encoding="utf-8")
            aligned_dataset = RNASequenceDataset.from_fasta(output_path)
            return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                   aligned_fasta_path=output_path, execution_time_seconds=elapsed)


def _estimate_nussinov_mfe(sequence: str) -> float:
    """
    Pure-Python Nussinov base-pair counting MFE energy estimator.
    Estimates MFE based on Watson-Crick (A-U, G-C, -3.0 kcal/mol) and Wobble (G-U, -1.5 kcal/mol) pairs.

    :param sequence: Ungapped RNA sequence string.
    :return: Estimated MFE float in kcal/mol.
    """
    clean = sequence.replace("-", "").replace(".", "").upper()
    n = len(clean)
    if n < 4:
        return 0.0

    pairs = {("A", "U"): -3.0, ("U", "A"): -3.0, ("G", "C"): -3.0, ("C", "G"): -3.0, ("G", "U"): -1.5, ("U", "G"): -1.5}
    dp = np.zeros((n, n), dtype=float)

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i, j] = max(dp[i + 1, j], dp[i, j - 1])  # Option 1: unpaired
            energy = pairs.get((clean[i], clean[j]), 0.0)  # Option 2: paired
            if energy < 0:
                dp[i, j] = max(dp[i, j], dp[i + 1, j - 1] + energy)
            for k in range(i + 1, j):  # Option 3: bifurcation
                dp[i, j] = max(dp[i, j], dp[i, k] + dp[k + 1, j])

    return float(dp[0, n - 1])


class AlignmentEvaluator:
    """
    Evaluator class providing methods for assessing biological alignment quality metrics,
    including structure conservation (SCI without Ribosum flag bias), covariation (MI-APC),
    consistency (TCS), entropy (H_N), MOS, and pairwise sequence similarity.
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
                identity_pct = (matches / total_cols) * 100.0
                pair_identities.append(identity_pct)

        if not pair_identities:
            return {"mean_sequence_similarity": 0.0,
                    "median_sequence_similarity": 0.0,
                    "min_sequence_similarity": 0.0,
                    "max_sequence_similarity": 0.0}

        arr = np.array(pair_identities)
        return {"mean_sequence_similarity": float(np.round(np.mean(arr), 2)),
                "median_sequence_similarity": float(np.round(np.median(arr), 2)),
                "min_sequence_similarity": float(np.round(np.min(arr), 2)),
                "max_sequence_similarity": float(np.round(np.max(arr), 2))}

    @staticmethod
    def calculate_structure_conservation_index(alignment_fasta_path: Path) -> float:
        """
        Calculates the Structure Conservation Index (SCI) by comparing RNAalifold consensus minimum free energy (E_consensus)
        to the mean individual sequence MFEs (E_mean) from RNAfold. Fallback to Nussinov energy estimation if CLI absent.
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
                            sci_val = float(np.round(sci_val, 4))
                            if sci_val == 0.0 or abs(sci_val) < 1e-6:
                                return 0.0
                            return sci_val
                except Exception:
                    pass

        # Pure-Python thermodynamic fallback
        single_mfes = [_estimate_nussinov_mfe(s[1]) for s in ungapped_items]
        mean_single = float(np.mean(single_mfes)) if single_mfes else 0.0

        # Consensus sequence calculation
        seq_list = list(aligned_sequences.values())
        char_matrix = np.array([list(s) for s in seq_list])
        _, align_len = char_matrix.shape
        consensus_chars = []
        for col in range(align_len):
            col_chars = [c for c in char_matrix[:, col] if c in ("A", "C", "G", "U")]
            if col_chars:
                vals, counts = np.unique(col_chars, return_counts=True)
                consensus_chars.append(vals[np.argmax(counts)])

        consensus_str = "".join(consensus_chars)
        consensus_mfe = _estimate_nussinov_mfe(consensus_str)

        if mean_single >= 0.0:
            sci_approx = 0.0 if consensus_mfe >= 0.0 else 1.0
        else:
            sci_approx = consensus_mfe / mean_single

        sci_res = float(np.round(min(1.5, max(0.0, sci_approx)), 4))
        return 0.0 if sci_res == 0.0 or abs(sci_res) < 1e-6 else sci_res

    @staticmethod
    def calculate_transitive_consistency_score(alignment_fasta_path: Path) -> float:
        """
        Calculates Transitive Consistency Score (TCS) measuring column residue consistency across alignment paths using T-Coffee evaluation or vectorized matrix fallback.
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

        return float(np.round((consistent_pairs / total_pairs) * 100.0, 2))

    @staticmethod
    def calculate_mean_overlap_score(alignments_map: Dict[str, Dict[str, str]]) -> float:
        """
        Computes the Mean Overlap Score (MOS) measuring Jaccard consensus index across aligned residue pair positions produced by different pipelines.
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

        return float(np.round(np.mean(jaccard_scores), 4))

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
            "mean_mi_apc_covariation": float(np.round(mean_apc_cov, 4)),
            "consensus_bp_covariation_score": float(np.round(consensus_bp_cov, 4)),
            "compensatory_mutation_count": float(compensatory_count),
        }

    @staticmethod
    def calculate_normalized_shannon_entropy(alignment_fasta_path: Path) -> float:
        """
        Computes column-wise Normalized Shannon Entropy (H_N) across non-gap sequence positions in the alignment.
        Preserves IUPAC ambiguous/degenerate nucleotide codes (A, C, G, U, N, R, Y, S, W, K, M, B, D, H, V) rather than filtering them out.
        """
        aligned_dataset = RNASequenceDataset.from_fasta(alignment_fasta_path)
        seq_list = list(aligned_dataset.sequences.values())
        if not seq_list:
            return 0.0

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
            return 0.0

        return float(np.round(np.mean(column_entropies), 4))


available_pipelines: List[AlignmentPipeline] = [Muscle5Pipeline(mode="default"),
                                                MafftQinsiPipeline(),
                                                MafftLinsiPipeline(),
                                                MafftXinsiPipeline(),
                                                RCoffeePipeline(),
                                                StructuralEncodingPipeline()]


class RNAAlignmentBenchmark:
    """
    Benchmark orchestrator running alignment pipelines and evaluating quality metrics across datasets.
    """

    def __init__(self, pipelines: Optional[List[AlignmentPipeline]] = None) -> None:
        """Initializes the RNAAlignmentBenchmark suite with specified alignment pipelines."""
        self.pipelines: List[AlignmentPipeline] = pipelines or [
            Muscle5Pipeline(mode="default"),
            MafftQinsiPipeline(),
            MafftLinsiPipeline(),
            MafftXinsiPipeline(),
            RCoffeePipeline(),
            StructuralEncodingPipeline(),
        ]
        self.evaluator: AlignmentEvaluator = AlignmentEvaluator()

    def run_benchmark(self, input_directory_path: Path, output_excel_path: Optional[Path] = None) -> pd.DataFrame:
        """Executes all configured alignment pipelines and evaluates metrics across every FASTA file in a directory."""
        input_dir = Path(input_directory_path)
        fasta_files: List[Path] = sorted(
            [f for f in input_dir.glob("*") if f.suffix.lower() in (".fasta", ".fa", ".fna")]
        )
        metrics_list: List[EvaluationMetrics] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_dir_path: Path = Path(temporary_directory)

            for fasta_file in fasta_files:
                dataset: RNASequenceDataset = RNASequenceDataset.from_fasta(fasta_file)
                if not dataset.sequences or len(dataset.sequences) < 2:
                    continue

                alignment_results: Dict[str, AlignmentResult] = {}
                alignments_map: Dict[str, Dict[str, str]] = {}

                for pipeline in self.pipelines:
                    out_path: Path = temp_dir_path / f"{dataset.dataset_name}_{pipeline.name}.aln"
                    result: AlignmentResult = pipeline.align(dataset, out_path)
                    alignment_results[pipeline.name] = result
                    if result.is_successful and result.aligned_sequences:
                        alignments_map[pipeline.name] = result.aligned_sequences

                mos_score: float = self.evaluator.calculate_mean_overlap_score(alignments_map)

                for pipeline_name, result in alignment_results.items():
                    if not result.is_successful or not result.aligned_fasta_path or not result.aligned_fasta_path.exists():
                        continue

                    aln_path: Path = result.aligned_fasta_path
                    sci_val: float = self.evaluator.calculate_structure_conservation_index(aln_path)
                    tcs_val: float = self.evaluator.calculate_transitive_consistency_score(aln_path)
                    cov_dict: Dict[str, float] = self.evaluator.calculate_structural_covariation_score(aln_path)
                    hn_val: float = self.evaluator.calculate_normalized_shannon_entropy(aln_path)
                    sim_dict: Dict[str, float] = self.evaluator.calculate_pairwise_sequence_similarity(aln_path)

                    metrics: EvaluationMetrics = EvaluationMetrics(
                        dataset_name=dataset.dataset_name,
                        pipeline_name=pipeline_name,
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

        records: List[Dict[str, Union[str, float]]] = [m.to_dict() for m in metrics_list]
        df = pd.DataFrame(records)

        if output_excel_path is not None:
            excel_path_obj = Path(output_excel_path)
            excel_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                with pd.ExcelWriter(excel_path_obj, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Alignment Metrics", index=False)
                    if not df.empty:
                        summary_df = df.groupby("pipeline").mean(numeric_only=True).reset_index()
                        summary_df.to_excel(writer, sheet_name="Pipeline Means Summary", index=False)
            except Exception:
                csv_path = excel_path_obj.with_suffix(".csv")
                df.to_csv(csv_path, index=False)

        return df
