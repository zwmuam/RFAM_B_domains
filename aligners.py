from pathlib import Path
from typing import Dict, Optional, Literal

from tweaks import shell, logger

# Mock or imported default thread settings
default_threads = 4


class Span:
    __slots__ = ('seq_id', 'start', 'end', 'length', 'partners', 'strand', 'domain')

    def __init__(self,
                 seq_id: str,
                 start: int,
                 end: int,
                 strand: int,
                 partners: list = None):
        self.seq_id = seq_id
        self.start = start
        self.end = end
        self.length = self.end - self.start + 1
        self.strand = strand
        self.partners = partners if partners is not None else []
        self.domain = None

    def n_partners(self) -> int:
        return len(self.partners)


def parse_mmseqs_tsv(tsv_file: Path) -> dict:
    """
    Parse a MMseqs2 BLAST-TAB TSV file into a dictionary of query sequences and their corresponding hits.
    :param tsv_file: path to the MMseqs2 BLAST-TAB TSV file
    :return: dictionary of query sequences and their hits
    """

    # check if the file is not result of a timeout error
    with open(tsv_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line == "_TimeoutError_":
            raise TimeoutError(f'MMseqs2 search timed out. Result file: {tsv_file}')

    n_alns, n_non_self, n_unique_queries = 0, 0, 0

    unsorted_result = {}
    with tsv_file.open() as handle:
        logger.info(f'reading results from {tsv_file}')
        for line in handle:
            if line.startswith('#') or not line.strip():
                continue
            else:
                n_alns += 1
                line = line.rstrip('\n')
                fields = line.split('\t')

                q_seq_id, query_start, query_end = fields[1], int(fields[8]), int(fields[9])
                # TODO: debugging: check if the query strand is always + for MMseqs2 results
                inverted = query_start > query_end
                assert not inverted, f'Query inversion is not expected for MMseqs2 results: {line}'
                local_key = (query_start, query_end)

                t_seq_id, target_start, target_end = fields[0], int(fields[6]), int(fields[7])
                if not q_seq_id == t_seq_id:  # exclude self-alignments
                    n_non_self += 1
                    inverted = target_start > target_end
                    if inverted:
                        t_strand = -1
                        target_start, target_end = target_end, target_start
                    else:
                        t_strand = 1
                    if q_seq_id not in unsorted_result:
                        unsorted_result[q_seq_id] = {}
                    if local_key not in unsorted_result[q_seq_id]:
                        unsorted_result[q_seq_id][local_key] = []
                    unsorted_result[q_seq_id][local_key].append((t_seq_id, target_start, target_end, t_strand))

    sorted_queries = {seq_id: [] for seq_id in unsorted_result.keys()}
    for seq_id, coordinate_dict in unsorted_result.items():
        # sort the coordinates by start position
        sorted_coordinates = sorted([q for q in coordinate_dict.keys()], key=lambda x: x[0])
        for local_key in sorted_coordinates:
            span = Span(seq_id=seq_id,
                        start=local_key[0],
                        end=local_key[1],
                        strand=1)
            partners = [Span(seq_id=t_seq_id, start=target_start, end=target_end, strand=t_strand) for
                        t_seq_id, target_start, target_end, t_strand in coordinate_dict[local_key]]
            span.partners = partners
            sorted_queries[seq_id].append(span)

    logger.info(
        f'found {n_alns} alignments, {n_non_self} non-self alignments, and {len(sorted_queries)} unique queries')

    return sorted_queries


def riboseek_easy_search(fasta_file: Path,
                         tmp_dir: Path,
                         result_dir: Path,
                         num_iterations: int,
                         s: float,
                         gap_open: float,
                         min_seq_id: float,
                         timeout: Optional[int] = None,
                         cpu: int = default_threads - 1) -> Dict[str, list]:
    """
    Runs a Riboseek easy-search with the given parameters and parses the results.
    This wrapper function is structured for downstream Bayesian optimization 
    of nucleotide/RFAM homology search parameters.

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

    # generate a result file path based on the input fasta file name riboseek flags
    name_string = f'{fasta_file.stem}_s{s}_Iter{num_iterations}_go{gap_open}_msi{min_seq_id}'
    result_file = result_dir.joinpath(name_string)
    # if file already exists (from a previous run), skip the search and parse the existing results
    try:
        return parse_mmseqs_tsv(result_file)
    # if file does not exist, run Riboseek easy-search and parse the results
    except FileNotFoundError:
        # Establish directory structure
        fasta_file = Path(fasta_file)
        run_tmp_dir = Path(tmp_dir).joinpath(f"{fasta_file.stem}.riboseek.tmp")
        run_tmp_dir.mkdir(parents=True, exist_ok=True)


        tmp_subdir = run_tmp_dir.joinpath("riboseek_tmp")
        tmp_subdir.mkdir(parents=True, exist_ok=True)

        # Construct command list
        command = ["riboseek", "easy-search",
                   str(fasta_file.as_posix()),  # Query Fasta (Self-search for RFAM identification)
                   str(fasta_file.as_posix()),  # Target Fasta
                   str(result_file.as_posix()),  # Output BLAST-TAB TSV
                   str(tmp_subdir.as_posix()),  # Temporary scratch subdirectory

                   # Performance/System Parameters
                   "--threads", str(cpu),

                   # 1. Sensitivity
                   "-s", f"{s}",
                   "--mask", "1",  # Enable TANTAN low-complexity masking for RNA to avoid repeat-driven hit explosion

                   # 7. Ungapped Prefilter Threshold
                   "--max-seqs", '9999',  # Set to 2000 to capture more potential hits for RNA

                   # 9. Iterations
                   "--num-iterations", str(num_iterations),
                   "--gap-open", str(gap_open),
                   "--min-seq-id", str(min_seq_id),  # Minimum sequence identity threshold
                   "--search-type", "3",  # Explicitly force nucleotide search
                   "--max-rejected", "100",  # Limit rejected candidates
                   "--qsc", "0.0", # perform rudimentary candidate filtering to avoid profile pollution
                   "-c", "0.0",  # Set coverage threshold to 0 to capture short domains
                   ]
        try:
            shell(command=command, timeout=timeout)
        except TimeoutError:
            # make a fake output file that contains only "_TimeoutError_" to indicate that the search timed out
            with open(result_file, 'w') as f:
                f.write("_TimeoutError_")

        return parse_mmseqs_tsv(result_file)


def mmseq_easy_search(fasta_file: Path,
                      tmp_dir: Path,
                      result_dir: Path,
                      k: int,
                      s: float,
                      gap_extend: float,
                      gap_open: float,
                      min_seq_id: float,
                      timeout: int,
                      cpu: int = default_threads) -> dict:
    """
    1. Run MMseqs2 search on the input fasta file against itself
    2. Parse the results into a dictionary of query sequences and their corresponding hits
    3. Return the dictionary of query sequences and their hits
    :param fasta_file: path to the input fasta file
    :param tmp_dir: path to the temporary directory for MMseqs2 output
    :param k: k-mer size
    :param s: sensitivity
    :param gap_extend: gap extension penalty
    :param gap_open: gap opening penalty
    :param min_seq_id: minimum sequence identity
    :param timeout: timeout for the search (in seconds)
    :param cpu: number of CPU threads to use (default: all but one)
    :return:
    """
    name_string = f"{fasta_file.stem}_k{k}_s{s}_gap{gap_open}_{gap_extend}_minid_{min_seq_id}.tsv"
    result_file = result_dir.joinpath(name_string)
    try:
        return parse_mmseqs_tsv(result_file)
    except FileNotFoundError:
        fasta_file = Path(fasta_file)
        run_tmp_dir = Path(tmp_dir).joinpath(f"{fasta_file.stem}.riboseek.tmp")
        run_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_subdir = tmp_dir.joinpath('mmseqs_tmp')
        tmp_subdir.mkdir(parents=True, exist_ok=True)
        command = ['mmseqs',
                   'easy-search',
                   str(fasta_file.as_posix()),
                   str(fasta_file.as_posix()),
                   str(result_file.as_posix()),
                   str(tmp_subdir.as_posix()),
                   '--threads', str(cpu),
                   '-k', f'{k}',
                   '--spaced-kmer-mode', '1',
                   '--diag-score', '1',
                   '--gap-extend', f'{gap_extend}',
                   '--gap-open', f'{gap_open}',
                   '-s', f'{s}',
                   '--min-seq-id', f'{min_seq_id}',
                   '-c', '0',
                   '--max-seqs', '9999',
                   '--search-type', '3']

        try:
            shell(command=command, timeout=timeout)
        except TimeoutError:
            # make a fake output file that contains only "_TimeoutError_" to indicate that the search timed out
            with open(result_file, 'w') as f:
                f.write("_TimeoutError_")

        return parse_mmseqs_tsv(result_file)
