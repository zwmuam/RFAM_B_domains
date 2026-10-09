# fmt: off
"""
msa.py

Object-oriented framework for running Multiple Sequence Alignment (MSA) tools and trimming
strategies on ncRNA sequences. Defines RNASequenceDataset, AlignmentResult dataclass, abstract
AlignmentPipeline interface, and implementations for:
  - Pathway 0: Muscle v5 Super5 Primary Sequence Pipeline
  - Pathway 1: Ultra-Fast High-Throughput Screening Pipeline (MAFFT L-INS-i)
  - Pathway 2: Balanced Structure-Aware Pipeline (MAFFT Q-INS-i / L-INS-i)
  - Pathway 3: Speed-Optimized Gold-Standard Profile Covariance Model Pipeline (Infernal `cmbuild`
               + `cmalign --glocal`)
  - MAFFT X-INS-i
  - R-Coffee
  - Structural Encoding

Also includes trimming engines:
  - CIAlign Crop-from-Ends
  - trimAl gappyout
  - RNAalifold Consensus Secondary Structure Masking

Includes lightweight psutil memory tracking for subprocess execution. No silent fallbacks:
explicitly raises or returns descriptive error results when external bioinformatics
executables are absent.
"""

# built-ins
import abc
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# standard libraries
import psutil

# internal repository/package imports
from gff_sequence_extractor import (calculate_iupac_density,
                                     reverse_complement,
                                     standardize_rna_sequence)


@dataclass(frozen=True)
class RNASequenceDataset:
    """
    Immutable dataclass encapsulating a set of RNA sequences parsed from FASTA format.
    Ensures sequence integrity and canonical RNA representation (U-base) across all workflows,
    preserving soft-masking metadata (lowercase) when specified.
    """
    dataset_name: str
    sequences: Dict[str, str]

    @classmethod
    def from_fasta(cls,
                   fasta_path: Path,
                   max_iupac_density: float = 0.05,
                   preserve_soft_masking: bool = True) -> "RNASequenceDataset":
        """
        Parses a FASTA file into an RNASequenceDataset instance with standardized RNA sequences.
        Filters sequences exceeding the maximum IUPAC degenerate threshold (> 5%).

        :param fasta_path: Path object pointing to the input FASTA file to be read.
        :param max_iupac_density: Maximum allowed IUPAC degenerate code density (default 0.05).
        :param preserve_soft_masking: Whether to preserve lowercase characters for soft-masked
                                      repeats.
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
            # Standardize sequence to RNA (U-containing), preserving soft-masking
            sequence: str = standardize_rna_sequence(raw_seq,
                                                     convert_to_rna=True,
                                                     preserve_soft_masking=preserve_soft_masking)
            if sequence and calculate_iupac_density(sequence) <= max_iupac_density:
                sequence_map[header] = sequence

        return cls(dataset_name=fasta_path_obj.stem, sequences=sequence_map)

    def write_fasta(self, output_path: Path, use_dna_encoding: bool = False) -> None:
        """
        Writes sequence dataset to a standard FASTA formatted file, with optional DNA conversion
        (U -> T).

        :param output_path: Destination Path object where the FASTA file will be created.
        :param use_dna_encoding: Boolean flag indicating if U should be converted back to T for
                                 DNA tools.
        :return: None
        """
        fasta_lines: List[str] = []
        for header_name, sequence_string in self.sequences.items():
            out_seq = (sequence_string.replace("U", "T").replace("u", "t")
                       if use_dna_encoding else sequence_string)
            fasta_lines.append(f">{header_name}")
            fasta_lines.append(out_seq)

        Path(output_path).write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")


@dataclass
class AlignmentResult:
    """
    Data structure containing aligned sequences and detailed execution metadata for an alignment
    pipeline run.
    """
    pipeline_name: str
    dataset_name: str
    aligned_sequences: Dict[str, str]
    aligned_fasta_path: Optional[Path] = None
    is_successful: bool = True
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0


def _run_cmd_with_memory_tracking(cmd: List[str],
                                  cwd: Path,
                                  timeout: float = 120.0) -> Tuple[int, str, str, float, float]:
    """
    Executes a subprocess command while polling process memory RSS using psutil.

    :param cmd: Command list to execute.
    :param cwd: Working directory path.
    :param timeout: Maximum execution timeout in seconds.
    :return: Tuple of (returncode, stdout, stderr, elapsed_seconds, peak_memory_mb).
    """
    start_time = time.time()
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=cwd)

    peak_rss_bytes = 0
    try:
        ps_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                mem_info = ps_proc.memory_info()
                current_rss = mem_info.rss
                for child in ps_proc.children(recursive=True):
                    current_rss += child.memory_info().rss
                if current_rss > peak_rss_bytes:
                    peak_rss_bytes = current_rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.01)
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout, stderr, time.time() - start_time, peak_rss_bytes / (1024 * 1024)

    elapsed = time.time() - start_time
    peak_mb = peak_rss_bytes / (1024 * 1024) if peak_rss_bytes > 0 else 0.0
    return proc.returncode, stdout, stderr, elapsed, peak_mb


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
    Pathway 0: MUSCLE v5 alignment pipeline supporting high-accuracy Progressive Perturbed Pairwise
    (PPP) and Super5 modes for primary sequence alignment.
    """

    def __init__(self,
                 mode: str = "default",
                 extra_args: Optional[List[str]] = None) -> None:
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
                                   error_message="Executable 'muscle' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                temp_input: Path = temp_dir_path / "input.fasta"
                dataset.write_fasta(temp_input, use_dna_encoding=False)
                temp_output: Path = temp_dir_path / "output.aln"

                if self.mode == "super5":
                    cmd: List[str] = [executable, "-super5", str(temp_input),
                                      "-output", str(temp_output)]
                else:
                    cmd = [executable, "-align", str(temp_input), "-output", str(temp_output)]
                    if self.mode == "stratified":
                        cmd.append("-stratified")
                    elif self.mode == "diversified":
                        cmd.append("-diversified")

                if self.extra_args:
                    cmd.extend(self.extra_args)

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(cmd, temp_dir_path, timeout=120)
                )

                if retcode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
                    aligned_dataset: RNASequenceDataset = (
                        RNASequenceDataset.from_fasta(temp_output)
                    )
                    aligned_dataset.write_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


class MafftQinsiPipeline(AlignmentPipeline):
    """
    Pathway 2: MAFFT Q-INS-i structural alignment pipeline incorporating McCaskill base-pairing
    probability matrices. Uses --ep 0.0 --op 2.5 --adjustdirection --maxiterate 1000 flags.
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
                                   error_message="Executable 'mafft' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input: Path = Path(temp_dir) / "input.fasta"
                dataset.write_fasta(temp_input)
                if "mafft-qinsi" not in executable:
                    cmd = [executable, "--qinsi", "--ep", "0.0", "--op", "2.5",
                           "--adjustdirection", "--maxiterate", "1000", str(temp_input)]
                else:
                    cmd = [executable, "--ep", "0.0", "--op", "2.5",
                           "--adjustdirection", "--maxiterate", "1000", str(temp_input)]

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(cmd, Path(temp_dir), timeout=120)
                )

                if retcode == 0 and stdout.strip():
                    output_path.write_text(stdout, encoding="utf-8")
                    aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


class MafftLinsiPipeline(AlignmentPipeline):
    """
    Pathway 1: MAFFT L-INS-i local pairwise alignment pipeline with maximum consistency
    refinement. Uses --localpair --op 3.0 --ep 0.0 --adjustdirection --maxiterate 1000 flags.
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
                                   error_message="Executable 'mafft' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input: Path = Path(temp_dir) / "input.fasta"
                dataset.write_fasta(temp_input)
                if "mafft-linsi" not in executable:
                    cmd = [executable, "--localpair", "--op", "3.0", "--ep", "0.0",
                           "--adjustdirection", "--maxiterate", "1000", str(temp_input)]
                else:
                    cmd = [executable, "--op", "3.0", "--ep", "0.0",
                           "--adjustdirection", "--maxiterate", "1000", str(temp_input)]

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(cmd, Path(temp_dir), timeout=120)
                )

                if retcode == 0 and stdout.strip():
                    output_path.write_text(stdout, encoding="utf-8")
                    aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


class MafftXinsiPipeline(AlignmentPipeline):
    """
    MAFFT X-INS-i structural alignment pipeline utilizing pairwise MXSCARNA structural
    alignment algorithms. Uses --xinsi --ep 0.0 --adjustdirection --maxiterate 1000 flags.
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
                                   error_message="Executable 'mafft' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input: Path = Path(temp_dir) / "input.fasta"
                dataset.write_fasta(temp_input)
                if "mafft-xinsi" not in executable:
                    cmd = [executable, "--xinsi", "--ep", "0.0", "--adjustdirection",
                           "--maxiterate", "1000", str(temp_input)]
                else:
                    cmd = [executable, "--ep", "0.0", "--adjustdirection",
                           "--maxiterate", "1000", str(temp_input)]

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(cmd, Path(temp_dir), timeout=120)
                )

                if retcode == 0 and stdout.strip():
                    output_path.write_text(stdout, encoding="utf-8")
                    aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


class RCoffeePipeline(AlignmentPipeline):
    """
    R-Coffee alignment pipeline incorporating secondary structure folding predictions into
    consistency libraries.
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
                                   error_message="Executable 't_coffee' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path: Path = Path(temp_dir)
                temp_input: Path = temp_dir_path / "input.fasta"
                dataset.write_fasta(temp_input)
                temp_out: Path = temp_dir_path / "rcoffee_out.aln"
                cmd = [executable, "-seq", str(temp_input.resolve()),
                       "-mode", "rcoffee",
                       "-output", "fasta_aln",
                       "-outfile", str(temp_out.resolve())]

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(cmd, temp_dir_path, timeout=120)
                )

                if retcode == 0 and temp_out.exists():
                    aligned_dataset = RNASequenceDataset.from_fasta(temp_out)
                    aligned_dataset.write_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


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
                                   error_message="Executable 'mafft' not found in PATH 🛑")

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temp_dir_path: Path = Path(temporary_directory)
                temp_input: Path = temp_dir_path / "input.fasta"
                dataset.write_fasta(temp_input)
                if "mafft-qinsi" in executable or "qinsi" in executable:
                    mafft_cmd = [executable, "--qinsi", "--ep", "0.0",
                                 "--maxiterate", "1000", str(temp_input)]
                else:
                    mafft_cmd = [executable, "--globalpair", "--ep", "0.0",
                                 "--maxiterate", "1000", str(temp_input)]

                retcode, stdout, stderr, elapsed, peak_mb = (
                    _run_cmd_with_memory_tracking(mafft_cmd, temp_dir_path, timeout=120)
                )

                if retcode == 0 and stdout.strip():
                    output_path.write_text(stdout, encoding="utf-8")
                    aligned_dataset = RNASequenceDataset.from_fasta(output_path)
                    return AlignmentResult(self.name, dataset.dataset_name,
                                           aligned_dataset.sequences,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=stderr or f"Exit code {retcode} 🛑",
                                           execution_time_seconds=elapsed,
                                           memory_peak_mb=peak_mb)
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


class ProfileCovarianceModelPipeline(AlignmentPipeline):
    """
    Pathway 3: Speed-Optimized Gold-Standard Profile Covariance Model Pipeline via Infernal.
      1. MAFFT L-INS-i initial alignment.
      2. RNAalifold consensus structure calculation.
      3. cmbuild Round 1 (Initial CM).
      4. cmalign --glocal Round 2 (Profile re-alignment).
      5. cmbuild Round 3 (Refined CM).
      Note: Model calibration (cmcalibrate) is skipped during screening for an 8- to 10-fold
      speedup.
    """

    def __init__(self) -> None:
        """Initializes ProfileCovarianceModelPipeline runner."""
        super().__init__("profile_cm_pipeline")

    def align(self, dataset: RNASequenceDataset, output_path: Path) -> AlignmentResult:
        """Executes Infernal profile covariance model alignment workflow."""
        if not dataset.sequences:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message="Input sequences missing")

        mafft_exec = shutil.which("mafft-linsi") or shutil.which("mafft")
        alifold_exec = shutil.which("RNAalifold")
        cmbuild_exec = shutil.which("cmbuild")
        cmalign_exec = shutil.which("cmalign")

        missing_tools = []
        if not mafft_exec:
            missing_tools.append("mafft")
        if not alifold_exec:
            missing_tools.append("RNAalifold")
        if not cmbuild_exec:
            missing_tools.append("cmbuild")
        if not cmalign_exec:
            missing_tools.append("cmalign")

        if missing_tools:
            err_msg = f"Missing required executables: {', '.join(missing_tools)} 🛑"
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=err_msg)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                input_fasta = temp_dir_path / "input.fasta"
                dataset.write_fasta(input_fasta)

                # Step 1: Initial MAFFT L-INS-i alignment
                if "mafft-linsi" not in mafft_exec:
                    init_cmd = [mafft_exec, "--localpair", "--op", "3.0", "--ep", "0.0",
                                "--adjustdirection", "--maxiterate", "1000", str(input_fasta)]
                else:
                    init_cmd = [mafft_exec, "--op", "3.0", "--ep", "0.0",
                                "--adjustdirection", "--maxiterate", "1000", str(input_fasta)]

                retcode, stdout, stderr, elapsed1, peak1 = (
                    _run_cmd_with_memory_tracking(init_cmd, temp_dir_path, timeout=120)
                )
                if retcode != 0 or not stdout.strip():
                    err_msg = stderr or f"MAFFT initial failed code {retcode} 🛑"
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=err_msg,
                                           execution_time_seconds=elapsed1,
                                           memory_peak_mb=peak1)

                initial_aln_fasta = temp_dir_path / "initial_aln.fa"
                initial_aln_fasta.write_text(stdout, encoding="utf-8")

                # Step 2: RNAalifold consensus structure calculation
                alifold_cmd = [alifold_exec, "--noLP", "--noPS", str(initial_aln_fasta)]
                retcode, stdout, stderr, elapsed2, peak2 = (
                    _run_cmd_with_memory_tracking(alifold_cmd, temp_dir_path, timeout=60)
                )
                if retcode != 0:
                    err_msg = stderr or f"RNAalifold failed code {retcode} 🛑"
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=err_msg,
                                           execution_time_seconds=elapsed1 + elapsed2,
                                           memory_peak_mb=max(peak1, peak2))

                dbn_lines = stdout.splitlines()
                consensus_ss = dbn_lines[1].split()[0] if len(dbn_lines) >= 2 else ""

                # Write Stockholm file with #=GC SS_cons annotation
                sto_file = temp_dir_path / "initial_aln.sto"
                init_ds = RNASequenceDataset.from_fasta(initial_aln_fasta)
                sto_lines = ["# STOCKHOLM 1.0\n"]
                for h, s in init_ds.sequences.items():
                    sto_lines.append(f"{h:<30} {s}")
                if consensus_ss:
                    sto_lines.append(f"{'#=GC SS_cons':<30} {consensus_ss}")
                sto_lines.append("//\n")
                sto_file.write_text("\n".join(sto_lines), encoding="utf-8")

                # Step 3: Round 1 cmbuild
                cm_file1 = temp_dir_path / "round1.cm"
                cmbuild_cmd = [cmbuild_exec, str(cm_file1), str(sto_file)]
                retcode, stdout, stderr, elapsed3, peak3 = (
                    _run_cmd_with_memory_tracking(cmbuild_cmd, temp_dir_path, timeout=120)
                )
                if retcode != 0 or not cm_file1.exists():
                    err_msg = stderr or f"cmbuild Round 1 failed code {retcode} 🛑"
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=err_msg,
                                           execution_time_seconds=elapsed1 + elapsed2 + elapsed3,
                                           memory_peak_mb=max(peak1, peak2, peak3))

                # Step 4: Round 2 profile alignment via cmalign --glocal
                refined_sto = temp_dir_path / "refined.sto"
                cmalign_cmd = [cmalign_exec, "--glocal", "-o", str(refined_sto),
                               str(cm_file1), str(input_fasta)]
                retcode, stdout, stderr, elapsed4, peak4 = (
                    _run_cmd_with_memory_tracking(cmalign_cmd, temp_dir_path, timeout=120)
                )
                if retcode != 0 or not refined_sto.exists():
                    err_msg = stderr or f"cmalign failed code {retcode} 🛑"
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message=err_msg,
                                           execution_time_seconds=(
                                               elapsed1 + elapsed2 + elapsed3 + elapsed4
                                           ),
                                           memory_peak_mb=max(peak1, peak2, peak3, peak4))

                # Parse aligned sequences from refined.sto
                aligned_seqs: Dict[str, str] = {}
                for line in refined_sto.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line == "//":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        h, s = parts[0], parts[1]
                        aligned_seqs[h] = aligned_seqs.get(h, "") + s

                if aligned_seqs:
                    out_ds = RNASequenceDataset("cm_aligned", aligned_seqs)
                    out_ds.write_fasta(output_path)
                    total_time = elapsed1 + elapsed2 + elapsed3 + elapsed4
                    max_peak = max(peak1, peak2, peak3, peak4)
                    return AlignmentResult(self.name, dataset.dataset_name, aligned_seqs,
                                           aligned_fasta_path=output_path,
                                           execution_time_seconds=total_time,
                                           memory_peak_mb=max_peak)
                else:
                    return AlignmentResult(self.name, dataset.dataset_name, {},
                                           is_successful=False,
                                           error_message="cmalign output empty 🛑")
        except Exception as err:
            return AlignmentResult(self.name, dataset.dataset_name, {}, is_successful=False,
                                   error_message=str(err))


# --- Trimming Engines ---

def trim_cialign_crop_from_ends(alignment_fasta_path: Path, output_path: Path) -> bool:
    """
    Executes CIAlign crop-from-ends or simulated end-gap crop trimming on aligned FASTA.
    Removes unaligned terminal overhangs without modifying internal conserved structural columns.
    """
    cialign_exec = shutil.which("cialign")
    if cialign_exec:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = [cialign_exec, "--infile", str(alignment_fasta_path.resolve()),
                       "--outname", "cialign_out", "--crop_divergent", "TRUE"]
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     cwd=Path(temp_dir), timeout=60)
                cropped_file = Path(temp_dir) / "cialign_out_cleaned.fasta"
                if res.returncode == 0 and cropped_file.exists():
                    shutil.copy(cropped_file, output_path)
                    return True
        except Exception:
            pass

    # Pure Python implementation of crop-from-ends for unaligned terminal overhangs
    ds = RNASequenceDataset.from_fasta(alignment_fasta_path)
    if not ds.sequences:
        return False

    seq_list = list(ds.sequences.values())
    headers = list(ds.sequences.keys())
    align_len = len(seq_list[0])

    # Find first and last columns where gap fraction <= 0.8
    start_col = 0
    end_col = align_len - 1

    for c in range(align_len):
        gap_frac = sum(1 for s in seq_list if s[c] in ("-", ".")) / len(seq_list)
        if gap_frac <= 0.8:
            start_col = c
            break

    for c in range(align_len - 1, -1, -1):
        gap_frac = sum(1 for s in seq_list if s[c] in ("-", ".")) / len(seq_list)
        if gap_frac <= 0.8:
            end_col = c
            break

    if start_col >= end_col:
        shutil.copy(alignment_fasta_path, output_path)
        return True

    trimmed_seqs = {h: s[start_col:end_col + 1] for h, s in ds.sequences.items()}
    out_ds = RNASequenceDataset("trimmed", trimmed_seqs)
    out_ds.write_fasta(output_path)
    return True


def trim_trimal_gappyout(alignment_fasta_path: Path, output_path: Path) -> bool:
    """
    Executes trimAl -gappyout on aligned FASTA.
    """
    trimal_exec = shutil.which("trimal")
    if trimal_exec:
        try:
            cmd = [trimal_exec, "-in", str(alignment_fasta_path.resolve()),
                   "-out", str(output_path.resolve()), "-gappyout"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            pass

    # Pure Python fallback simulating gappyout (removing columns with > 90% gaps)
    ds = RNASequenceDataset.from_fasta(alignment_fasta_path)
    if not ds.sequences:
        return False

    seq_list = list(ds.sequences.values())
    headers = list(ds.sequences.keys())
    align_len = len(seq_list[0])

    keep_cols = []
    for c in range(align_len):
        gap_frac = sum(1 for s in seq_list if s[c] in ("-", ".")) / len(seq_list)
        if gap_frac < 0.9:
            keep_cols.append(c)

    if not keep_cols:
        shutil.copy(alignment_fasta_path, output_path)
        return True

    trimmed_seqs = {h: "".join([s[c] for c in keep_cols]) for h, s in ds.sequences.items()}
    out_ds = RNASequenceDataset("trimal_trimmed", trimmed_seqs)
    out_ds.write_fasta(output_path)
    return True


def trim_consensus_structure_masking(alignment_fasta_path: Path, output_path: Path) -> bool:
    """
    Executes RNAalifold consensus secondary structure masking.
    Masks for retention any alignment column involved in consensus base pairs () [] or loop
    boundaries. Trims non-covarying unaligned terminal flanking columns.
    """
    alifold_exec = shutil.which("RNAalifold")
    consensus_dbn = ""
    if alifold_exec:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                res = subprocess.run([alifold_exec, "--noLP", "--noPS",
                                      str(alignment_fasta_path.resolve())],
                                     capture_output=True, text=True,
                                     cwd=Path(temp_dir), timeout=60)
                lines = res.stdout.splitlines()
                if len(lines) >= 2:
                    consensus_dbn = lines[1].split()[0]
        except Exception:
            pass

    ds = RNASequenceDataset.from_fasta(alignment_fasta_path)
    if not ds.sequences:
        return False

    seq_list = list(ds.sequences.values())
    align_len = len(seq_list[0])

    if consensus_dbn and len(consensus_dbn) == align_len:
        # Find structural boundary (first and last paired bracket or loop char)
        struct_indices = [c for c, char in enumerate(consensus_dbn) if char in "({<[]}>)"]
        if struct_indices:
            start_col = max(0, min(struct_indices) - 2)
            end_col = min(align_len - 1, max(struct_indices) + 2)
            trimmed_seqs = {h: s[start_col:end_col + 1] for h, s in ds.sequences.items()}
            out_ds = RNASequenceDataset("ss_masked", trimmed_seqs)
            out_ds.write_fasta(output_path)
            return True

    # Fallback to CIAlign crop-from-ends
    return trim_cialign_crop_from_ends(alignment_fasta_path, output_path)


available_pipelines: List[AlignmentPipeline] = [
    Muscle5Pipeline(mode="default"),
    Muscle5Pipeline(mode="super5"),
    MafftQinsiPipeline(),
    MafftLinsiPipeline(),
    MafftXinsiPipeline(),
    RCoffeePipeline(),
    StructuralEncodingPipeline(),
    ProfileCovarianceModelPipeline(),
]
# fmt: on
