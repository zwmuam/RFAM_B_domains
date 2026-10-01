"""
msa.py

Object-oriented framework for running Multiple Sequence Alignment (MSA) tools on ncRNA sequences.
Defines RNASequenceDataset, AlignmentResult dataclass, abstract AlignmentPipeline interface,
and implementations for MUSCLE v5, MAFFT (Q-INS-i, L-INS-i, X-INS-i), R-Coffee, and Structural Encoding.
"""

import abc
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
        """Executes MUSCLE v5 alignment algorithm."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("muscle")
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="MUSCLE executable not found in system PATH")

        start_time = time.time()
        try:
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
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


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
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="MAFFT executable not found in system PATH")

        start_time = time.time()
        try:
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
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


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
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="MAFFT executable not found in system PATH")

        start_time = time.time()
        try:
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
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


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
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="MAFFT executable not found in system PATH")

        start_time = time.time()
        try:
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
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


class RCoffeePipeline(AlignmentPipeline):
    """
    R-Coffee alignment pipeline incorporating secondary structure folding predictions into consistency libraries.
    """

    def __init__(self) -> None:
        """Initializes R-Coffee pipeline runner."""
        super().__init__("rcoffee")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes T-Coffee in R-Coffee mode on input dataset."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        executable: Optional[str] = shutil.which("t_coffee")
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="T-Coffee executable not found in system PATH")

        start_time = time.time()
        try:
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
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


class StructuralEncodingPipeline(AlignmentPipeline):
    """
    Structure-informed alignment pipeline leveraging structure-aware alignment algorithms.
    Utilises explicit structural probability matrices (MAFFT Q-INS-i / structural globalpair).
    """

    def __init__(self) -> None:
        """Initializes Structural Encoding pipeline runner."""
        super().__init__("structural_encoding")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes secondary structure-informed alignment pipeline."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences empty")

        executable: Optional[str] = shutil.which("mafft-qinsi") or shutil.which("mafft")
        if not executable:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="MAFFT executable not found in system PATH")

        start_time = time.time()
        try:
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

                proc = subprocess.run(mafft_cmd, capture_output=True, text=True, check=False, cwd=temp_dir_path,
                                      timeout=120)
                elapsed = time.time() - start_time
                if proc.returncode == 0 and proc.stdout.strip():
                    output_path.write_text(proc.stdout, encoding="utf-8")
                    aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name, aligned_dataset.sequences,
                                           aligned_fasta_path=output_path, execution_time_seconds=elapsed)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                           error_message=proc.stderr or f"Exit code {proc.returncode}",
                                           execution_time_seconds=elapsed)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err), execution_time_seconds=time.time() - start_time)


available_pipelines: List[AlignmentPipeline] = [
    Muscle5Pipeline(mode="default"),
    MafftQinsiPipeline(),
    MafftLinsiPipeline(),
    MafftXinsiPipeline(),
    RCoffeePipeline(),
    StructuralEncodingPipeline(),
]
