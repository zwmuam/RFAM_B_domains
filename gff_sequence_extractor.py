"""
gff_sequence_extractor.py

Modern, production-grade Python 3.11+ object-oriented framework for genomic feature extraction,
IUPAC-aware coordinate mapping, and non-coding RNA sequence standardization.
Extracts genomic intervals specified by primary GFF annotations from target FASTA sequences,
adjusts overlapping secondary GFF feature coordinates to extracted sequence frames,
and exports ID-sorted FASTA and GFF file pairs into structured output directories.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


# IUPAC-compliant nucleotide translation table for DNA/RNA reverse complementation
_REVERSE_COMPLEMENT_TRANS: Dict[int, int] = str.maketrans(
    "ACGTUNRYSWKMBDHVacgtunryswkmbdhv",
    "TGCAANYRSWMKVHDBtgcaanyrswmkvhdb"
)

DEGENERATE_IUPAC_CODES = set("RYSWKMBDHVNryswkmbdhvn")


def reverse_complement(sequence: str, preserve_case: bool = True) -> str:
    """
    Computes the reverse complement of a nucleotide sequence string using optimized translation tables.
    Handles standard (A, C, G, T, U) and IUPAC degenerate nucleotide bases (N, R, Y, S, W, K, M, B, D, H, V).
    Preserves character case (soft-masking metadata) when preserve_case is True.

    Mathematical/Biological bias:
        Maintains strict 5'-to-3' orientation conversion across Watson-Crick and wobble pairing options.

    :param sequence: Input DNA or RNA sequence string.
    :param preserve_case: Whether to preserve original character case (soft-masking).
    :return: Reverse complemented nucleotide sequence string.
    """
    if not sequence:
        return ""
    rc = sequence.translate(_REVERSE_COMPLEMENT_TRANS)[::-1]
    return rc if preserve_case else rc.upper()


def calculate_iupac_density(sequence: str) -> float:
    """
    Calculates the proportion of ambiguous/degenerate IUPAC nucleotide codes in a sequence string.

    :param sequence: Input nucleotide sequence.
    :return: Float ratio (0.0 to 1.0) of degenerate IUPAC bases.
    """
    if not sequence:
        return 0.0
    ungapped = [c for c in sequence if c not in ("-", ".")]
    if not ungapped:
        return 0.0
    degenerate_count = sum(1 for c in ungapped if c in DEGENERATE_IUPAC_CODES)
    return float(degenerate_count / len(ungapped))


def standardize_rna_sequence(
    sequence: str,
    convert_to_rna: bool = True,
    preserve_soft_masking: bool = True
) -> str:
    """
    Standardizes nucleotide sequences into canonical RNA (U-containing) or DNA (T-containing) representations.
    Removes whitespace and line returns while preserving IUPAC ambiguous character codes and soft-masking.

    :param sequence: Input nucleotide sequence string.
    :param convert_to_rna: Boolean flag indicating conversion to RNA (U) if True, or DNA (T) if False.
    :param preserve_soft_masking: If True, preserves lowercase character soft-masking metadata.
    :return: Standardized sequence string.
    """
    if not sequence:
        return ""
    clean_seq: str = sequence.strip().replace(" ", "").replace("\r", "").replace("\n", "")
    if not preserve_soft_masking:
        clean_seq = clean_seq.upper()

    if convert_to_rna:
        clean_seq = clean_seq.replace("T", "U").replace("t", "u")
    else:
        clean_seq = clean_seq.replace("U", "T").replace("u", "t")

    return clean_seq


class GFFSequenceExtractor:
    """
    Object-oriented sequence extractor and annotation coordinate mapper for genomic features and non-coding RNAs.
    Provides robust parsing of multi-record FASTA files and GFF3 coordinate transformations.
    """

    def __init__(
        self,
        fasta_path: Path,
        primary_gff_path: Path,
        secondary_gff_path: Optional[Path] = None,
        flank_length: int = 0,
        max_iupac_density: float = 0.05,
        preserve_soft_masking: bool = True
    ) -> None:
        """
        Initializes GFFSequenceExtractor with input file paths and flanking/filtering configurations.

        :param fasta_path: Path object specifying the input FASTA sequence file.
        :param primary_gff_path: Path object specifying primary GFF annotation features.
        :param secondary_gff_path: Optional Path object specifying secondary GFF features.
        :param flank_length: Non-negative integer indicating flanking nucleotides to extend boundaries.
        :param max_iupac_density: Maximum allowed fraction of IUPAC degenerate bases (default 0.05).
        :param preserve_soft_masking: Whether to preserve lowercase characters for soft-masked repeats.
        :return: None
        """
        self.fasta_path: Path = Path(fasta_path)
        self.primary_gff_path: Path = Path(primary_gff_path)
        self.secondary_gff_path: Optional[Path] = Path(secondary_gff_path) if secondary_gff_path else None
        self.flank_length: int = max(0, flank_length)
        self.max_iupac_density: float = max_iupac_density
        self.preserve_soft_masking: bool = preserve_soft_masking

        self.fasta_dict: Dict[str, str] = {}
        self.primary_gff_df: pd.DataFrame = pd.DataFrame()
        self.secondary_gff_df: Optional[pd.DataFrame] = None

    def parse_fasta(self) -> Dict[str, str]:
        """
        Parses multi-record FASTA files into a dictionary mapping sequence identifiers to clean sequence strings.

        :return: Dictionary mapping fasta record headers to continuous sequence strings.
        """
        if not self.fasta_path.exists() or self.fasta_path.stat().st_size == 0:
            self.fasta_dict = {}
            return self.fasta_dict

        sequences: Dict[str, str] = {}
        current_id: Optional[str] = None
        seq_chunks: List[str] = []

        with open(self.fasta_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line_str: str = line.strip()
                if not line_str:
                    continue
                if line_str.startswith(">"):
                    if current_id is not None:
                        seq_str = "".join(seq_chunks)
                        sequences[current_id] = seq_str if self.preserve_soft_masking else seq_str.upper()
                    current_id = line_str[1:].split()[0]
                    seq_chunks = []
                else:
                    seq_chunks.append(line_str)
            if current_id is not None:
                seq_str = "".join(seq_chunks)
                sequences[current_id] = seq_str if self.preserve_soft_masking else seq_str.upper()

        self.fasta_dict = sequences
        return self.fasta_dict

    def parse_gff(self, gff_path: Path) -> pd.DataFrame:
        """
        Parses GFF3 annotation files into a structured pandas DataFrame.
        Extracts feature 'ID' attributes via regular expressions with fallback column generation.

        :param gff_path: Path object identifying the GFF annotation file to parse.
        :return: DataFrame containing 9 standard GFF columns along with extracted ID column.
        """
        gff_path_obj: Path = Path(gff_path)
        empty_cols: List[str] = [
            "seqid", "source", "type", "start", "end",
            "score", "strand", "phase", "attributes", "ID"
        ]
        if not gff_path_obj.exists() or gff_path_obj.stat().st_size == 0:
            return pd.DataFrame(columns=empty_cols)

        try:
            df: pd.DataFrame = pd.read_csv(
                gff_path_obj,
                sep="\t",
                comment="#",
                header=None,
                names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"],
                dtype={
                    "seqid": str, "source": str, "type": str,
                    "start": int, "end": int, "score": str,
                    "strand": str, "phase": str, "attributes": str
                },
                engine="c"
            )
            df["ID"] = df["attributes"].str.extract(r'(?:^|;)ID=([^;]+)', expand=False).fillna("")
            return df
        except Exception:
            return pd.DataFrame(columns=empty_cols)

    def extract_and_adjust(self) -> Tuple[Dict[str, str], Optional[pd.DataFrame]]:
        """
        Extracts genomic sub-sequences bounded by primary GFF features (plus optional flanks)
        and adjusts coordinates of overlapping secondary GFF features onto extracted sequence frames.
        Filters sequences exceeding the maximum IUPAC density threshold (default > 5%).

        :return: Tuple containing extracted sequence dictionary and adjusted secondary GFF DataFrame.
        """
        if not self.fasta_dict:
            self.parse_fasta()

        if self.primary_gff_df.empty:
            self.primary_gff_df = self.parse_gff(self.primary_gff_path)

        if self.secondary_gff_path and (self.secondary_gff_df is None or self.secondary_gff_df.empty):
            self.secondary_gff_df = self.parse_gff(self.secondary_gff_path)

        extracted_sequences: Dict[str, str] = {}
        adjusted_records: List[Dict[str, Union[str, int]]] = []

        sec_grouped: Dict[str, List[Tuple[int, int, str, str, str, str, str, str, str]]] = {}
        if self.secondary_gff_df is not None and not self.secondary_gff_df.empty:
            for s_row in self.secondary_gff_df.itertuples(index=False):
                sec_grouped.setdefault(str(s_row.seqid), []).append((
                    int(s_row.start),
                    int(s_row.end),
                    str(s_row.source),
                    str(s_row.type),
                    str(s_row.score),
                    str(s_row.strand),
                    str(s_row.phase),
                    str(s_row.attributes),
                    str(s_row.ID)
                ))

        for p_row in self.primary_gff_df.itertuples(index=True):
            p_seqid: str = str(p_row.seqid)
            if p_seqid not in self.fasta_dict:
                continue

            parent_seq: str = self.fasta_dict[p_seqid]
            p_seq_len: int = len(parent_seq)

            p_fid: str = str(p_row.ID) if str(p_row.ID) else f"feature_{p_row.Index}"
            p_start: int = int(p_row.start)
            p_end: int = int(p_row.end)
            p_strand: str = str(p_row.strand)

            ext_start: int = max(1, p_start - self.flank_length)
            ext_end: int = min(p_seq_len, p_end + self.flank_length)

            subseq: str = parent_seq[ext_start - 1: ext_end]

            is_neg_strand: bool = (p_strand == "-")
            if is_neg_strand:
                subseq = reverse_complement(subseq, preserve_case=self.preserve_soft_masking)

            # Check IUPAC degenerate density threshold
            if calculate_iupac_density(subseq) > self.max_iupac_density:
                # Exclude sequence exceeding high IUPAC density threshold (>5%)
                continue

            seq_header: str = f"{p_fid}::{p_seqid}::{ext_start}_{ext_end}_flank{self.flank_length}"
            extracted_sequences[seq_header] = subseq

            if self.secondary_gff_df is not None and not self.secondary_gff_df.empty:
                sec_sub = sec_grouped.get(p_seqid)
                if sec_sub:
                    for s_st, s_en, s_so, s_ty, s_sc, s_stnd, s_ph, s_at, s_id in sec_sub:
                        if s_st <= ext_end and s_en >= ext_start:
                            c_start: int = max(s_st, ext_start)
                            c_end: int = min(s_en, ext_end)

                            subseq_len: int = len(subseq)
                            if is_neg_strand:
                                raw_start: int = ext_end - c_end + 1
                                raw_end: int = ext_end - c_start + 1
                                adj_start: int = max(1, min(raw_start, subseq_len))
                                adj_end: int = max(1, min(raw_end, subseq_len))
                                adj_strand: str = "-" if s_stnd == "+" else ("+" if s_stnd == "-" else s_stnd)
                            else:
                                raw_start = c_start - ext_start + 1
                                raw_end = c_end - ext_start + 1
                                adj_start = max(1, min(raw_start, subseq_len))
                                adj_end = max(1, min(raw_end, subseq_len))
                                adj_strand = s_stnd

                            attr_str: str = f"{s_at};PrimaryFeatureID={p_fid}" if s_at else f"PrimaryFeatureID={p_fid}"

                            adjusted_records.append({
                                "seqid": seq_header,
                                "source": s_so,
                                "type": s_ty,
                                "start": adj_start,
                                "end": adj_end,
                                "score": s_sc,
                                "strand": adj_strand,
                                "phase": s_ph,
                                "attributes": attr_str,
                                "ID": s_id,
                                "parent_feature_id": p_fid
                            })

        df_adj: Optional[pd.DataFrame] = None
        if self.secondary_gff_df is not None:
            if adjusted_records:
                df_adj = pd.DataFrame(adjusted_records)
            else:
                df_adj = pd.DataFrame(columns=[
                    "seqid", "source", "type", "start", "end", "score",
                    "strand", "phase", "attributes", "ID", "parent_feature_id"
                ])

        return extracted_sequences, df_adj

    def export_data(
        self,
        output_dir: Path,
        extracted_sequences: Dict[str, str],
        adjusted_gff_df: Optional[pd.DataFrame] = None
    ) -> List[Path]:
        """
        Exports extracted sequence FASTA files and adjusted GFF files grouped into feature ID-sorted pairs.

        :param output_dir: Path object specifying destination directory for exported files.
        :param extracted_sequences: Dictionary mapping headers to sequence strings.
        :param adjusted_gff_df: Optional DataFrame of adjusted secondary GFF annotations.
        :return: List of Path objects corresponding to all generated output FASTA and GFF files.
        """
        output_path: Path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []

        feature_seqs: Dict[str, List[Tuple[str, str]]] = {}
        for header, seq in extracted_sequences.items():
            if "::" in header:
                feature_id: str = header.split("::")[0]
            else:
                feature_id = header.split("_")[0]
            feature_seqs.setdefault(feature_id, []).append((header, seq))

        for feature_id in sorted(feature_seqs.keys()):
            seq_list = feature_seqs[feature_id]
            fasta_file: Path = output_path / f"{feature_id}.fasta"
            with open(fasta_file, "w", encoding="utf-8") as handle:
                for header, seq in seq_list:
                    handle.write(f">{header}\n{seq}\n")
            created_files.append(fasta_file)

        if adjusted_gff_df is not None and not adjusted_gff_df.empty:
            if "parent_feature_id" in adjusted_gff_df.columns:
                grouped_gff = adjusted_gff_df.groupby("parent_feature_id")
            else:
                grouped_gff = adjusted_gff_df.groupby("ID")

            for feat_id, group in sorted(grouped_gff, key=lambda x: str(x[0])):
                gff_file: Path = output_path / f"{feat_id}.gff"
                with open(gff_file, "w", encoding="utf-8") as handle:
                    handle.write("##gff-version 3\n")
                    for row in group.itertuples(index=False):
                        line: str = f"{row.seqid}\t{row.source}\t{row.type}\t{row.start}\t{row.end}\t{row.score}\t{row.strand}\t{row.phase}\t{row.attributes}\n"
                        handle.write(line)
                created_files.append(gff_file)

        return created_files


def extract_and_annotate_sequences(
    fasta_path: Path,
    primary_gff_path: Path,
    secondary_gff_path: Optional[Path] = None,
    flank_length: int = 0,
    output_dir: Optional[Path] = None,
    max_iupac_density: float = 0.05,
    preserve_soft_masking: bool = True
) -> Tuple[Dict[str, str], Optional[pd.DataFrame], Optional[List[Path]]]:
    """
    High-level functional API wrapper coordinating FASTA sequence parsing, primary/secondary GFF coordinate mapping,
    and optional exporting of ID-sorted FASTA/GFF file pairs into an output folder.

    :param fasta_path: Path object to input FASTA file.
    :param primary_gff_path: Path object to primary GFF file.
    :param secondary_gff_path: Optional Path object to secondary GFF annotation file.
    :param flank_length: Non-negative integer specifying flanking region extension.
    :param output_dir: Optional Path object specifying output directory.
    :param max_iupac_density: Maximum allowed fraction of IUPAC degenerate bases.
    :param preserve_soft_masking: Whether to preserve lowercase characters for soft-masked repeats.
    :return: Tuple containing extracted sequence map, adjusted GFF DataFrame, and exported file paths list.
    """
    extractor = GFFSequenceExtractor(
        fasta_path=fasta_path,
        primary_gff_path=primary_gff_path,
        secondary_gff_path=secondary_gff_path,
        flank_length=flank_length,
        max_iupac_density=max_iupac_density,
        preserve_soft_masking=preserve_soft_masking
    )
    seqs, adj_df = extractor.extract_and_adjust()

    exported_files: Optional[List[Path]] = None
    if output_dir is not None:
        exported_files = extractor.export_data(
            output_dir=output_dir,
            extracted_sequences=seqs,
            adjusted_gff_df=adj_df
        )

    return seqs, adj_df, exported_files
