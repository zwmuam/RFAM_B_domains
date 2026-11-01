# fmt: off
from pathlib import Path
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from tweaks import shell, logger, default_threads


class SequenceRegistry:
    """
    Bidirectional string-to-int32 sequence ID registry using sys.intern.
    Reduces memory overhead by eliminating redundant string allocations across billions of hits.
    Can be frozen to enforce a closed sequence ID set derived from input FASTA sequences.
    """
    __slots__ = ("str_to_id", "id_to_str", "frozen")

    def __init__(self, frozen: bool = False) -> None:
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: List[str] = []
        self.frozen: bool = frozen

    def freeze(self) -> None:
        """Freeze the registry to prohibit adding new sequence IDs."""
        self.frozen = True

    def get_id(self, seq_id: Union[str, int]) -> int:
        if isinstance(seq_id, int):
            return seq_id
        val = self.str_to_id.get(seq_id)
        if val is None:
            if self.frozen:
                raise KeyError(
                    f"🛑 Sequence ID '{seq_id}' not found in pre-populated SequenceRegistry."
                )
            s = sys.intern(seq_id)
            val = len(self.id_to_str)
            self.str_to_id[s] = val
            self.id_to_str.append(s)
        return val

    def get_str(self, seq_int: Union[int, str]) -> str:
        if isinstance(seq_int, str):
            return seq_int
        if isinstance(seq_int, int) and 0 <= seq_int < len(self.id_to_str):
            return self.id_to_str[seq_int]
        return str(seq_int)

    def __len__(self) -> int:
        return len(self.id_to_str)

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self.str_to_id


def count_fasta(fasta_path: Path,
                registry: Optional[SequenceRegistry] = None) -> int:
    """
    Count sequence headers (starting with '>') and optionally populate registry.

    :param fasta_path: Path to the FASTA sequence file.
    :param registry: Optional SequenceRegistry to populate with FASTA query IDs.
    :return: Integer count of query sequence headers.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        return 0
    header_count = 0
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as fasta_file:
        for line in fasta_file:
            if line.startswith(">"):
                header_count += 1
                if registry is not None:
                    seq_id = line[1:].strip().split()[0]
                    registry.get_id(seq_id)
    if registry is not None:
        registry.freeze()
    return header_count


def populate_registry_from_fasta(fasta_path: Path,
                                 registry: Optional[SequenceRegistry] = None) -> SequenceRegistry:
    """
    Populate and freeze a SequenceRegistry in a single pass over a FASTA file.

    :param fasta_path: Path to the FASTA sequence file.
    :param registry: Optional SequenceRegistry instance to populate (created if None).
    :return: Frozen SequenceRegistry containing all sequence IDs from the FASTA file.
    """
    if registry is None:
        registry = SequenceRegistry()
    count_fasta(fasta_path=fasta_path, registry=registry)
    return registry


def parse_mmseqs_tsv(tsv_file: Path,
                     registry: SequenceRegistry) \
        -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """
    Single-pass streaming parser for MMseqs2 BLAST-TAB TSV output files.
    Yields (q_seq_id, sorted_query_spans, target_spans) query block by query block.

    :param tsv_file: Path to the MMseqs2 BLAST-TAB TSV output file.
    :param registry: SequenceRegistry instance to use for sequence ID lookups.
    :return: Generator yielding tuples of (query_seq_int, query_spans, target_spans).
    """
    tsv_file = Path(tsv_file)
    if not tsv_file.exists():
        raise FileNotFoundError(f"🛑 TSV file not found: {tsv_file}")

    with tsv_file.open("r") as handle:
        first_line = handle.readline().strip()
        if first_line == "_TimeoutError_":
            raise TimeoutError(f"🛑 MMseqs2 search timed out. Result file: {tsv_file}")

    def format_block(query_int: int,
                     query_hits: List[Tuple[int, int]],
                     target_hits: List[Tuple[int, int, int, int]]) \
            -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Format hits into sorted NumPy arrays (primary: start, tiebreaker: end).
        """
        query_spans = np.array(query_hits, dtype=np.int32)
        target_spans = np.array(target_hits, dtype=np.int32)

        # Deterministic sorting: primary on start (col 0), tie-breaker on end (col 1)
        sort_order = np.lexsort((query_spans[:, 1], query_spans[:, 0]))
        return query_int, query_spans[sort_order], target_spans[sort_order]

    seen_queries = set()
    current_q_int = None
    current_query_hits: List[Tuple[int, int]] = []
    current_target_hits: List[Tuple[int, int, int, int]] = []
    total_valid_hits = 0

    logger.info(f"streaming alignment results from {tsv_file}")

    with tsv_file.open("r") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                raise ValueError(
                    f"🛑 Malformed BLAST-TAB line with fewer than 10 fields: {line}"
                )

            q_seq_str = fields[0]
            t_seq_str = fields[1]

            if q_seq_str == t_seq_str:
                continue

            raw_qs, raw_qe = int(fields[6]), int(fields[7])
            raw_ts, raw_te = int(fields[8]), int(fields[9])

            # Determine relative alignment strand and convert 1-based closed to 0-based half-open
            t_strand = -1 if (raw_qs > raw_qe) ^ (raw_ts > raw_te) else 1
            q_start = min(raw_qs, raw_qe) - 1
            q_end = max(raw_qs, raw_qe)
            t_start = min(raw_ts, raw_te) - 1
            t_end = max(raw_ts, raw_te)

            if q_start < 0 or q_end <= q_start or t_start < 0 or t_end <= t_start:
                raise ValueError(
                    f"🛑 Invalid alignment hit coordinates: q=[{q_start}, {q_end}), "
                    f"t=[{t_start}, {t_end}) in line: {line}"
                )

            q_seq_int = registry.get_id(q_seq_str)
            t_seq_int = registry.get_id(t_seq_str)

            if current_q_int is None:
                current_q_int = q_seq_int
                seen_queries.add(q_seq_int)
            elif current_q_int != q_seq_int:
                if q_seq_int in seen_queries:
                    raise ValueError(
                        f"🛑 Input TSV {tsv_file} is not grouped contiguously by query ID."
                    )
                if current_query_hits:
                    yield format_block(current_q_int, current_query_hits, current_target_hits)
                    current_query_hits.clear()
                    current_target_hits.clear()
                current_q_int = q_seq_int
                seen_queries.add(q_seq_int)

            current_query_hits.append((q_start, q_end))
            current_target_hits.append((t_seq_int, t_start, t_end, t_strand))
            total_valid_hits += 1

    if current_q_int is not None and current_query_hits:
        yield format_block(current_q_int, current_query_hits, current_target_hits)

    if total_valid_hits == 0:
        raise ValueError(
            f"🛑 Input TSV {tsv_file} is empty or contains only excluded self-hits."
        )


def batch_parse_mmseqs_tsv(tsv_file: Path,
                           n: int = 1000,
                           registry: Optional[SequenceRegistry] = None) \
        -> Generator[List[Tuple[int, np.ndarray, np.ndarray]], None, None]:
    """
    Collect n query blocks from a TSV file and yield them as a batch.

    :param tsv_file: Path to the MMseqs2 BLAST-TAB TSV output file.
    :param n: Number of query blocks to collect before yielding.
    :param registry: SequenceRegistry instance to use for sequence ID lookups.
    :return: Generator yielding lists of query blocks.
    """
    if registry is None:
        registry = SequenceRegistry()
    batch = []
    for query_block in parse_mmseqs_tsv(tsv_file=tsv_file, registry=registry):
        batch.append(query_block)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def riboseek_easy_search(fasta_file: Path,
                         tmp_dir: Path,
                         result_dir: Path,
                         num_iterations: int,
                         s: float,
                         gap_open: float,
                         min_seq_id: float,
                         timeout: Optional[int] = None,
                         cpu: int = default_threads - 1,
                         registry: Optional[SequenceRegistry] = None,
                         batch_size: int = 1000) \
        -> Generator[List[Tuple[int, np.ndarray, np.ndarray]], None, None]:
    """
    Runs a Riboseek easy-search with the given parameters and parses the results.

    =============================================================================
    THE 10 CRITICAL OPTIMIZATION PARAMETERS (Selected for Bayesian Optimization):
    =============================================================================
    1. s (Sensitivity) [-s]:
       - Role: Controls list length of similar k-mers. Higher is slower but more sensitive.
       - Optimization Range: [1.0, 7.5] (or up to 8.5 for BLAST-like sensitivity). Default: 5.7.
    2. gap_open (Gap Opening Cost) [--gap-open]:
       - Role: Cost to open a gap in the Smith-Waterman-Gotoh gapped alignment.
       - Optimization Range: [10, 30] (Riboseek paper's grid search optimized this to 23 for RNA).
    3. gap_extend (Gap Extension Cost) [--gap-extend]:
       - Role: Cost to extend a gap by one residue.
       - Optimization Range: [1, 5] (Riboseek paper's grid search optimized this to 1 for RNA).
    4. pca (Pseudocount Admixture Strength) [--pca]:
       - Role: Blends substitution matrix probabilities with query sequence counts for PSSMs.
       - Optimization Range: [0.5, 2.5] (Paper optimized this to 1.1).
    5. pcb (Pseudocount Neff threshold) [--pcb]:
       - Role: Effective sequence count at which max admixture is halved.
       - Optimization Range: [1.0, 3.5] (Paper optimized this to 1.8).
    6. e_profile (Profile Inclusion E-value) [--e-profile]:
       - Role: Maximum E-value for a hit to be included in the profile of subsequent search rounds.
       - Optimization Range: [1e-10, 1e-1] (Logarithmic scale). Default: 1e-3.
    7. min_ungapped_score (Ungapped Prefilter Threshold) [--min-ungapped-score]:
       - Role: Threshold score a prefilter hit must exceed to be forwarded to gapped SW alignment.
       - Optimization Range: [10, 40]. Default: 15.
    8. max_seqs (Max results passing prefilter) [--max-seqs]:
       - Role: Maximum hits per query passing prefiltering to avoid aligning random decoys.
       - Optimization Range: [100, 5000]. Default: 300.
    9. num_iterations (Iterative Profile Search Rounds) [--num-iterations]:
       - Role: Rounds of PSI-BLAST-like profile search. Dynamically builds deep PSSMs.
       - Optimization Range: [1, 5] (discrete). Default: 1.
    10. k (k-mer Length) [-k]:
       - Role: Size of seed word. Longer reduces chance hits but needs larger RAM.
       - Optimization Range: [0, 14] (0 indicates auto-determination based on DB size). Default: 0.

    =============================================================================
    STATIONARY/COMMENTED PARAMETERS (Fixed or kept constant):
    =============================================================================
    # --spaced-kmer-mode INT       : 0: consecutive, 1: spaced k-mers [1]
    # --diag-score BOOL            : Use ungapped diagonal scoring [1]
    # --comp-bias-corr INT         : Local compositional bias correction [1]
    # --mask INT                   : Low complexity masking using TANTAN [0]
    # --mask-profile INT           : Mask profile using TANTAN [1]
    # --search-type INT            : Set to 3 for pure nucleotide searches [3]
    # --cov-mode INT               : Coverage calculation mode [0: bidirectional]
    # -c FLOAT                     : Aligned residue coverage fraction [0.0]
    # --min-seq-id FLOAT           : Minimum sequence identity threshold [0.0]

    =============================================================================
    ITERATIVE RNA PROFILE OPTIONS & MMSEQS2 DIFFERENCES:
    =============================================================================
    Riboseek introduces crucial departures from MMseqs2 to handle structural RNA:

    1. Di-mer vs Mononucleotide/AA Alphabet:
       Riboseek operates on a 24-state overlapping di-mer alphabet (16 canonical,
       8 wildcards) with a stride of 1 nucleotide. This preserves base-pairing signals.
       Standard MMseqs2 assumes independent mononucleotides or amino acid states.
    2. learned Di-mer Substitution Matrix:
       The default substitution matrix is 'nucl:dinuc.out' (learned from Rfam seed alignments)
       instead of BLOSUM or simple match/mismatch matrices.
    3. Strand-Aware Complementary Profiles:
       Unlike proteins, nucleotide queries must scan both strands. Riboseek builds separate
       forward and reverse-complement profiles. In the reverse profile, substitution scores
       for each di-mer are swapped based on base-pairing rules (e.g., AA->AC becomes AA->GU).
    4. Calibrated E-value Math:
       Because overlapping di-mers introduce local dependency, Riboseek scales the raw Gumbel
       E-values post hoc: Riboseek E = 0.038 * (raw E)^0.828.
    5. Integrated Covariance Models (Infernal Bridge):
       Riboseek integrates Infernal directly. Users can build Covariance Models (CMs) from
       Riboseek MSAs via `cmbuild` and realign hits with the CYK algorithm using `cmsearch`
       for structure-aware alignments, which standard MMseqs2 cannot do.
    """
    fasta_file = Path(fasta_file)
    result_dir = Path(result_dir)
    tmp_dir = Path(tmp_dir)

    if registry is None:
        registry = populate_registry_from_fasta(fasta_file)

    name_string = f"{fasta_file.stem}_s{s}_Iter{num_iterations}_go{gap_open}_msi{min_seq_id}"
    result_file = result_dir.joinpath(name_string)

    if not result_file.exists():
        run_tmp_dir = tmp_dir.joinpath(f"{fasta_file.stem}.riboseek.tmp")
        run_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_subdir = run_tmp_dir.joinpath("riboseek_tmp")
        tmp_subdir.mkdir(parents=True, exist_ok=True)

        command = [
            "riboseek", "easy-search",
            str(fasta_file.as_posix()),
            str(fasta_file.as_posix()),
            str(result_file.as_posix()),
            str(tmp_subdir.as_posix()),
            "--threads", str(cpu),
            "-s", f"{s}",
            "--mask", "1",
            "--max-seqs", "9999",
            "--num-iterations", str(num_iterations),
            "--gap-open", str(gap_open),
            "--min-seq-id", str(min_seq_id),
            "--search-type", "3",
            "--max-rejected", "100",
            "--qsc", "0.0",
            "-c", "0.0"
        ]
        try:
            shell(command=command, timeout=timeout)
        except TimeoutError:
            with open(result_file, "w") as f:
                f.write("_TimeoutError_\n")

    return batch_parse_mmseqs_tsv(tsv_file=result_file, n=batch_size, registry=registry)


def mmseq_easy_search(fasta_file: Path,
                      tmp_dir: Path,
                      result_dir: Path,
                      k: int,
                      s: float,
                      gap_extend: float,
                      gap_open: float,
                      min_seq_id: float,
                      timeout: int,
                      cpu: int = default_threads,
                      registry: Optional[SequenceRegistry] = None,
                      batch_size: int = 1000) \
        -> Generator[List[Tuple[int, np.ndarray, np.ndarray]], None, None]:
    """
    1. Run MMseqs2 search on input FASTA file against itself
    2. Stream-parse results into batches of query sequence blocks
    3. Yield batches of (query_int, query_spans, target_spans)

    :param fasta_file: Path to the input FASTA file.
    :param tmp_dir: Path to the temporary directory for MMseqs2 output.
    :param result_dir: Path to the directory for alignment TSV output.
    :param k: k-mer size.
    :param s: Sensitivity.
    :param gap_extend: Gap extension penalty.
    :param gap_open: Gap opening penalty.
    :param min_seq_id: Minimum sequence identity.
    :param timeout: Timeout for search (in seconds).
    :param cpu: Number of CPU threads to use.
    :param registry: Pre-populated SequenceRegistry (created from FASTA if None).
    :param batch_size: Number of query blocks to yield per batch.
    :return: Generator yielding lists of query blocks.
    """
    fasta_file = Path(fasta_file)
    result_dir = Path(result_dir)
    tmp_dir = Path(tmp_dir)

    if registry is None:
        registry = populate_registry_from_fasta(fasta_file)

    name_string = f"{fasta_file.stem}_k{k}_s{s}_gap{gap_open}_{gap_extend}_minid_{min_seq_id}.tsv"
    result_file = result_dir.joinpath(name_string)

    if not result_file.exists():
        run_tmp_dir = tmp_dir.joinpath(f"{fasta_file.stem}.riboseek.tmp")
        run_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_subdir = tmp_dir.joinpath("mmseqs_tmp")
        tmp_subdir.mkdir(parents=True, exist_ok=True)

        command = [
            "mmseqs", "easy-search",
            str(fasta_file.as_posix()),
            str(fasta_file.as_posix()),
            str(result_file.as_posix()),
            str(tmp_subdir.as_posix()),
            "--threads", str(cpu),
            "-k", f"{k}",
            "--spaced-kmer-mode", "1",
            "--diag-score", "1",
            "--gap-extend", f"{gap_extend}",
            "--gap-open", f"{gap_open}",
            "-s", f"{s}",
            "--min-seq-id", f"{min_seq_id}",
            "-c", "0",
            "--max-seqs", "9999",
            "--search-type", "3"
        ]
        try:
            shell(command=command, timeout=timeout)
        except TimeoutError:
            with open(result_file, "w") as f:
                f.write("_TimeoutError_\n")

    return batch_parse_mmseqs_tsv(tsv_file=result_file, n=batch_size, registry=registry)
# fmt: on
