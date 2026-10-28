# fmt: off
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Domain Finder: Consolidated MMseqs2/RiboSeek & DPC Domain Family Extraction
===============================================================================

Merges sequence alignment search results with Density Peak Clustering (DPC) to
identify conserved domain instances and group them into domain families.
Exports resulting domain families to GFF3 format.
"""

__author__ = "Jakub Barylski"
__maintainer__ = "Jakub Barylski"
__license__ = "GNU GENERAL PUBLIC LICENSE"
__email__ = "jakub.barylski@gmail.com"

# built-ins
from pathlib import Path
from typing import Any, Dict, Optional

# standard libraries
from tqdm import tqdm

# internal repository/package imports
from aligners import SequenceRegistry, parse_mmseqs_tsv_stream
from domain_finder import DomainFinder, DomainInstance, count_fasta_queries
from tweaks import logger


def find_domains(fasta_path: Path,
                 aligner_name: str,
                 tsv_val: Path,
                 min_dom_sim: float,
                 dom_halo: float,
                 min_dom_dens: float,
                 min_fam_sim: float,
                 fam_halo: float,
                 min_fam_dens: float,
                 halo_method: Optional[str],
                 reassign_targets: bool,
                 min_containment: float,
                 output_gff: Path,
                 tmp_dir: Optional[Path] = None,
                 result_dir: Optional[Path] = None) -> Path:
    """
    Find domain families from a FASTA sequence file given parameter attributes.

    Runs MMseqs2 or RiboSeek (or reads pre-computed TSV), filters spurious domain
    instances, merges target spans, constructs domain overlap graphs, and clusters
    domain instances into domain families written out to GFF format.

    :param fasta_path: Path to input FASTA file containing query sequences.
    :param aligner_name: Name of the alignment tool used (e.g., "mmseqs2" or "riboseek").
    :param tsv_val: Path to pre-computed alignment TSV file (if available).
    :param min_dom_sim: Minimum similarity threshold for domain instance detection.
    :param dom_halo: Halo stringency parameter for domain instance detection.
    :param min_dom_dens: Minimum density threshold for domain instance detection.
    :param min_fam_sim: Minimum similarity threshold for domain family clustering.
    :param fam_halo: Halo stringency parameter for domain family clustering.
    :param min_fam_dens: Minimum density threshold for domain family clustering.
    :param halo_method: Halo method to use for clustering ("traditional", "knn", etc.) or None.
    :param reassign_targets: Whether to reassign targets of removed spurious domains to their strongest neighbor.
    :param min_containment: Minimum containment threshold for filtering spurious domains.
    :param output_gff: Path where the resulting GFF3 file will be written.
    :param tmp_dir: Optional directory path for temporary alignment scratch files.
    :param result_dir: Optional directory path for alignment TSV output files.
    :return: Path to the written output GFF3 file.
    """
    fasta_path = Path(fasta_path)
    output_gff = Path(output_gff)

    tmp_dir = Path(tmp_dir) if tmp_dir else fasta_path.parent / "tmp"
    result_dir = Path(result_dir) if result_dir else fasta_path.parent / "results_tsvs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Sequence registry for efficient string management
    registry = SequenceRegistry()

    total_queries = count_fasta_queries(fasta_path)
    logger.info(f"Reading pre-computed alignments from {tsv_val}")
    logger.info(f"Estimated work size: {total_queries} query sequences in {fasta_path.name}")
    search_stream = parse_mmseqs_tsv_stream(Path(tsv_val), registry=registry)

    # Perform domain finding and clustering
    finder = DomainFinder(search=search_stream,
                          source=aligner_name,
                          registry=registry,
                          fasta_path=fasta_path,
                          total_queries=total_queries)
    finder.find_domain_instances(min_dom_sim=min_dom_sim,
                                 halo_stringency=dom_halo,
                                 min_dom_dens=min_dom_dens,
                                 halo_method=halo_method)
    finder.filter_spurious_domains(min_containment=min_containment,
                                   reassign_targets=reassign_targets)
    finder.merge_targets()
    finder.calculate_target_overlaps()

    families_list = list(finder.families(min_fam_sim=min_fam_sim,
                                         halo_stringency=fam_halo,
                                         min_fam_dens=min_fam_dens,
                                         halo_method=halo_method))

    output_gff.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Estimated work size for GFF3 export: {len(families_list)} domain families (to {output_gff.name})")
    with output_gff.open("w") as gff_writer:
        for fam_id, fam_domains in tqdm(families_list,
                                       desc="Exporting domain families to GFF3",
                                       unit="families"):
            for domain in fam_domains:
                gff_writer.write(domain.to_gff(cluster_id=fam_id, registry=registry) + "\n")

    logger.info(f"Successfully exported domain families to {output_gff}")
    return output_gff


if __name__ == "__main__":
    FASTA_PATH = Path("/media/solificatus/10 TB/RFAMB/DeRep_pedro/recluster_c05_result_rep_seq.fasta")
    MM_TSV = Path("/media/solificatus/10 TB/RFAMB/DeRep_pedro/rep_seqs.default_c0_ms9999_st05.tsv")
    OUTPUT_GFF = Path("/media/solificatus/10 TB/RFAMB/DeRep_pedro/rep_seqs.default_c0_ms9999_st05.gff")
    TMP_DIR = Path("/media/solificatus/10 TB/RFAMB/DeRep_pedro/output/tmp")
    RESULT_DIR = Path("/media/solificatus/10 TB/RFAMB/DeRep_pedro/output/results_tsvs")

    if FASTA_PATH.exists() and MM_TSV.exists():
        find_domains(fasta_path=FASTA_PATH,
                     aligner_name="mmseqs2",
                     min_dom_sim=0.5,
                     dom_halo=0.5,
                     min_dom_dens=0.1,
                     min_fam_sim=0.5,
                     fam_halo=0.5,
                     min_fam_dens=0.1,
                     halo_method="traditional",
                     reassign_targets=True,
                     min_containment=0.5,
                     tsv_val=MM_TSV,
                     output_gff=OUTPUT_GFF,
                     tmp_dir=TMP_DIR,
                     result_dir=RESULT_DIR)
# fmt: on
