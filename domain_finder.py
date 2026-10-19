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
from collections import defaultdict
from itertools import count
from pathlib import Path
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

# standard libraries
import numpy as np
from scipy.sparse import coo_matrix, csgraph
from tqdm import tqdm

# internal repository/package imports
from aligners import (Span, mmseq_easy_search, parse_mmseqs_tsv,
                      riboseek_easy_search)
from clustering import (compare_sorted_spans, dpc, jaccard_index,
                        max_containment)
from tweaks import default_threads, logger


def _normalize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize key names in the attributes dictionary.

    Strips leading non-alphanumeric characters (such as dashes) and replaces
    non-alphanumeric characters with underscores.

    :param attributes: Input dictionary with search and clustering parameters.
    :return: Normalized attributes dictionary with cleaned keys.
    """
    normalized: Dict[str, Any] = {}
    for key, val in attributes.items():
        clean_key = key.lstrip("-")
        clean_key = re.sub(r"[^a-zA-Z0-9_]", "_", clean_key)
        normalized[clean_key] = val
    return normalized


class DomainInstance(Span):
    """
    Represents a candidate domain instance mapped to a sequence region.

    Extends Span to represent genomic domain coordinates, partner target
    alignments, and the originating aligner tool name.
    """

    instance_counter = count()

    __slots__ = ("seq_id", "start", "end", "length", "number", "strand",
                 "targets", "source")

    def __init__(self,
                 seq_id: str,
                 start: int,
                 summary_strand: int,
                 end: int,
                 targets: List[Span],
                 source: str = "mmseqs2") -> None:
        """
        Initialize a DomainInstance.

        :param seq_id: Target sequence identifier.
        :param start: 1-based start position.
        :param summary_strand: Net directionality sum across partners.
        :param end: 1-based end position.
        :param targets: List of partner alignment spans.
        :param source: Name of the alignment tool (e.g. mmseqs2 or riboseek).
        :return: None
        """
        super().__init__(seq_id=seq_id,
                         start=start,
                         end=end,
                         strand=(summary_strand > 0) - (summary_strand < 0),
                         partners=targets)
        self.source = source

    def to_gff(self,
               cluster_id: str) -> str:
        """
        Format domain instance as a standard GFF3 string.

        :param cluster_id: Family/cluster identifier for the GFF attributes.
        :return: Tab-delimited GFF3 record.
        """
        strand_str = "+" if self.strand > 0 else "-" if self.strand < 0 else "."
        return (f"{self.seq_id}\t{self.source}\tDomain\t{self.start}\t"
                f"{self.end}\t.\t{strand_str}\t.\tID={cluster_id}")

    def total_target_length(self) -> int:
        """
        Calculate total sequence length of partner targets.

        :return: Sum of length of partner target spans.
        """
        return sum(target.length for target in self.partners)

    def coverage(self) -> float:
        """
        Return the number of partner targets as alignment coverage proxy.

        :return: Count of partner target spans.
        """
        return len(self.partners)


class DomainFinder:
    """
    Selects representative conserved domain instances from alignment spans and
    clusters them into domain families using Density Peak Clustering.
    """

    def __init__(self,
                 search: Dict[str, List[Span]],
                 source: str = "mmseqs2") -> None:
        """
        Initialize DomainFinder.

        :param search: Dictionary mapping query IDs to lists of alignment spans.
        :param source: Aligner source name used for GFF generation.
        :return: None
        """
        self.sorted_queries = search
        self.source = source
        self.domains_per_seq: Dict[str, List[DomainInstance]] = {}
        self.n_domains = 0
        self.domain_index: Dict[int, DomainInstance] = {}
        self.domain_total_lengths: Dict[int, int] = {}
        self.targets_per_seq: Dict[str, List[Span]] = {}
        self.target_intersections: Dict[Tuple[int, int], int] = defaultdict(int)
        self.n_halo_alns = 0
        self.n_halo_domains = 0

    def find_domain_instances(self,
                              min_dom_sim: float,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: Optional[str]) -> None:
        """
        Identify candidate domain instances across query sequences via DPC density peaks.

        :param min_dom_sim: Minimum domain similarity threshold.
        :param halo_stringency: Halo stringency parameter.
        :param min_dom_dens: Minimum domain density threshold.
        :param halo_method: Halo method ('traditional', 'knn', etc.) or None.
        :return: None
        """
        with tqdm(total=len(self.sorted_queries),
                  desc="Finding domain instances") as pbar:
            for sorted_query_spans in self.sorted_queries.values():
                n_spans = len(sorted_query_spans)
                if n_spans == 0:
                    pbar.update(1)
                    continue
                elif n_spans == 1:
                    peak_span, = sorted_query_spans
                    self.create_new_domain(peak_span=peak_span,
                                           targets=peak_span.partners)
                else:
                    rows, cols, intersections = compare_sorted_spans(sorted_query_spans)
                    if rows.size == 0:
                        peak_to_cluster = {p: np.array([p]) for p in range(n_spans)}
                    else:
                        span_lengths = np.array([s.length for s in sorted_query_spans])
                        weights = np.array([s.n_partners() for s in sorted_query_spans])

                        similarities = jaccard_index(intersections=intersections,
                                                     row_lengths=span_lengths[rows],
                                                     column_lengths=span_lengths[cols])

                        peak_to_cluster, n_noise = dpc(rows=rows,
                                                       cols=cols,
                                                       sims=similarities,
                                                       n=n_spans,
                                                       min_sim=min_dom_sim,
                                                       weights=weights,
                                                       backend="numpy",
                                                       filter_transparency=halo_stringency,
                                                       halo_method=halo_method,
                                                       min_dens=min_dom_dens)
                        self.n_halo_alns += n_noise

                    for peak_id, member_ids in peak_to_cluster.items():
                        peak_span = sorted_query_spans[peak_id]
                        member_targets = []
                        for m_id in member_ids:
                            member_targets.extend(sorted_query_spans[int(m_id)].partners)
                        self.create_new_domain(peak_span=peak_span,
                                               targets=member_targets)
                pbar.update(1)

        logger.info(f"Found {self.n_domains} domain instances across "
                    f"{len(self.sorted_queries)} query sequences")

    def create_new_domain(self,
                          peak_span: Span,
                          targets: List[Span]) -> None:
        """
        Create a new DomainInstance based on a density peak span and target partners.

        :param peak_span: Span representing the density peak.
        :param targets: Target spans aligned to the cluster members.
        :return: None
        """
        strands = [target.strand for target in targets]
        supplemented_targets = targets + [peak_span]
        new_domain = DomainInstance(seq_id=peak_span.seq_id,
                                    start=peak_span.start,
                                    summary_strand=sum(strands),
                                    end=peak_span.end,
                                    targets=supplemented_targets,
                                    source=self.source)
        if new_domain.seq_id not in self.domains_per_seq:
            self.domains_per_seq[new_domain.seq_id] = []
        self.domains_per_seq[new_domain.seq_id].append(new_domain)
        self.n_domains += 1

    def filter_spurious_domains(self,
                                min_containment: float,
                                reassign_targets: bool) -> Tuple[int, int]:
        """
        Filter spurious domain instances formed by random overlap of adjacent domains.

        :param min_containment: Jaccard containment threshold.
        :param reassign_targets: Whether to reassign targets of removed domains.
        :return: Tuple of (number of discarded domains, number of kept domains).
        """
        dc = count()
        n_discarded = 0
        n_kept = 0
        for seq_id, domain_list in self.domains_per_seq.items():
            sorted_domains = sorted(domain_list, key=lambda d: d.start)
            domain_lengths = np.array([s.length for s in sorted_domains])
            rows, columns, intersections = compare_sorted_spans(sorted_domains)
            if rows.size == 0:
                connected_mask = np.array([], dtype=bool)
            else:
                containment = max_containment(intersections=intersections,
                                              row_lengths=domain_lengths[rows],
                                              column_lengths=domain_lengths[columns])
                connected_mask = containment >= min_containment

            if not np.any(connected_mask):
                kept_domains = sorted_domains
                n_kept += len(kept_domains)
            else:
                sources, targets = rows[connected_mask], columns[connected_mask]
                remaining_edges = [{s, t} for s, t in zip(sources, targets)]
                indices_to_remove = set()
                while remaining_edges:
                    connected_dom_indices = set()
                    for edge in remaining_edges:
                        connected_dom_indices.update(edge)
                    connected_domains = {i: sorted_domains[i]
                                         for i in connected_dom_indices}
                    scoring_dict = {i: (d.coverage(), -d.length)
                                    for i, d in connected_domains.items()}
                    weak_i = min(connected_domains,
                                 key=lambda i: scoring_dict[i])
                    indices_to_remove.add(weak_i)
                    if reassign_targets:
                        removed_domain = sorted_domains[weak_i]
                        removed_edges = [e for e in remaining_edges if weak_i in e]
                        connected_to_removed = set()
                        for edge in removed_edges:
                            connected_to_removed.update(edge)
                        connected_to_removed.discard(weak_i)
                        if connected_to_removed:
                            heir_domain = connected_domains[
                                max(connected_to_removed,
                                    key=lambda i: scoring_dict[i])
                            ]
                            heir_domain.partners.extend(removed_domain.partners)
                    remaining_edges = [e for e in remaining_edges if weak_i not in e]

                kept_domains = [d for i, d in enumerate(sorted_domains)
                                if i not in indices_to_remove]
                n_discarded += len(indices_to_remove)
                n_kept += len(kept_domains)

            for d in kept_domains:
                self.domain_index[next(dc)] = d

        return n_discarded, n_kept

    def merge_targets(self) -> None:
        """
        Merge overlapping target spans for each domain to prepare for overlap matrix generation.

        :return: None
        """
        logger.info(f"Merging overlapping targets for {len(self.domain_index)} domains")
        n_merged = 0
        n_distinct = 0
        n_singletons = 0
        with tqdm(total=len(self.domain_index),
                  desc="Merging overlapping targets") as pbar:
            for d_id, domain in self.domain_index.items():
                self.domain_total_lengths[d_id] = 0
                domain_targets_by_seq = defaultdict(list)
                for target in domain.partners:
                    domain_targets_by_seq[target.seq_id].append(target)

                for t_seq_id, targets in domain_targets_by_seq.items():
                    if t_seq_id not in self.targets_per_seq:
                        self.targets_per_seq[t_seq_id] = []
                    n_targets = len(targets)
                    if n_targets > 1:
                        sorted_group = sorted(targets, key=lambda s: s.start)
                        rows, cols, intersections = compare_sorted_spans(sorted_group)
                        similarity_matrix = coo_matrix(
                            (intersections, (rows, cols)),
                            shape=(n_targets, n_targets)
                        ).tocsr()
                        components, labels = csgraph.connected_components(
                            similarity_matrix,
                            directed=False,
                            return_labels=True
                        )
                        for component_id in range(components):
                            component_index = {
                                i: s for i, s in enumerate(sorted_group)
                                if labels[i] == component_id
                            }
                            if len(component_index) == 1:
                                distinct_target, = component_index.values()
                                distinct_target.domain = d_id
                                self.targets_per_seq[t_seq_id].append(distinct_target)
                                self.domain_total_lengths[d_id] += distinct_target.length
                                n_distinct += 1
                            else:
                                merged_start = min(s.start for s in component_index.values())
                                merged_end = max(s.end for s in component_index.values())
                                merged_target = Span(seq_id=t_seq_id,
                                                     start=merged_start,
                                                     end=merged_end,
                                                     strand=0)
                                merged_target.domain = d_id
                                self.targets_per_seq[t_seq_id].append(merged_target)
                                self.domain_total_lengths[d_id] += merged_target.length
                                n_merged += len(component_index) - 1
                    else:
                        singleton_target, = targets
                        singleton_target.domain = d_id
                        self.targets_per_seq[t_seq_id].append(singleton_target)
                        self.domain_total_lengths[d_id] += singleton_target.length
                        n_singletons += 1
                pbar.update(1)

        logger.info(f"Merged: {n_merged}; Distinct: {n_distinct}; "
                    f"Singletons: {n_singletons}")

    def calculate_target_overlaps(self) -> None:
        """
        Calculate pairwise target overlaps across domains to construct overlap graph.

        :return: None
        """
        with tqdm(total=len(self.targets_per_seq),
                  desc="Calculating target overlaps") as pbar:
            for seq_id, targets in self.targets_per_seq.items():
                n_targets = len(targets)
                if n_targets > 1:
                    local_rows, local_cols, intersections = compare_sorted_spans(targets)
                    if local_rows.size > 0:
                        domain_ids = np.array([t.domain for t in targets])
                        global_rows = domain_ids[local_rows]
                        global_cols = domain_ids[local_cols]
                        for r, c, intersect in zip(global_rows, global_cols, intersections):
                            canonical_key = tuple(sorted([r, c]))
                            assert r != c, (f"Unexpected self-overlap for domain {r} "
                                            f"in sequence {seq_id}")
                            self.target_intersections[canonical_key] += intersect
                pbar.update(1)

    def families(self,
                 min_fam_sim: float,
                 min_fam_dens: float,
                 halo_stringency: float,
                 halo_method: Optional[str]) -> Generator[Tuple[str, List[DomainInstance]],
                                                         None, None]:
        """
        Cluster domains into domain families using DPC on target overlap similarities.

        :param min_fam_sim: Minimum family similarity threshold.
        :param min_fam_dens: Minimum family density threshold.
        :param halo_stringency: Halo stringency parameter.
        :param halo_method: Halo method ('traditional', 'knn', etc.) or None.
        :return: Generator yielding tuples of (family_id, list of DomainInstances).
        """
        n_domains = len(self.domain_index)
        if n_domains == 0:
            return

        if len(self.target_intersections) == 0:
            logger.warning("No edges in the overlap graph. "
                           "Yielding single-domain families.")
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            similarities = np.array([], dtype=np.float64)
            n_intersections = 0
        else:
            n_intersections = len(self.target_intersections)
            logger.info(f"Calculating Jaccard similarities for {n_domains} domains")
            rows = np.zeros(n_intersections, dtype=np.int32)
            cols = np.zeros(n_intersections, dtype=np.int32)
            row_domain_lengths = np.zeros(n_intersections, dtype=np.int32)
            col_domain_lengths = np.zeros(n_intersections, dtype=np.int32)
            intersections = np.zeros(n_intersections, dtype=np.int32)

            for i, ((row, col), intersection) in enumerate(self.target_intersections.items()):
                rows[i] = row
                cols[i] = col
                intersections[i] = intersection
                row_domain_lengths[i] = self.domain_total_lengths[row]
                col_domain_lengths[i] = self.domain_total_lengths[col]
                if row_domain_lengths[i] <= 0 or col_domain_lengths[i] <= 0:
                    raise ValueError(f"Unexpected non-positive domain length for "
                                     f"domains {row} and {col}.")

            similarities = jaccard_index(intersections=intersections,
                                         row_lengths=row_domain_lengths,
                                         column_lengths=col_domain_lengths)

        logger.info(f"Clustering {n_domains} domains into families using DPC "
                    f"with min_fam_sim={min_fam_sim}, halo_stringency={halo_stringency}, "
                    f"min_fam_dens={min_fam_dens}, halo_method={halo_method}")

        peak_to_cluster, n_halo = dpc(rows=rows,
                                      cols=cols,
                                      sims=similarities,
                                      n=n_domains,
                                      min_sim=min_fam_sim,
                                      backend="scipy",
                                      filter_transparency=halo_stringency,
                                      halo_method=halo_method,
                                      min_dens=min_fam_dens)
        self.n_halo_domains = n_halo
        n_clusters = len(peak_to_cluster)

        with tqdm(total=n_clusters, desc="Evaluating clusters") as pbar:
            for peak_id, member_ids in peak_to_cluster.items():
                cluster_domains_per_seq = defaultdict(list)
                for d_id in member_ids:
                    d = self.domain_index[int(d_id)]
                    cluster_domains_per_seq[d.seq_id].append(d)

                component_domains = []
                for seq_id, domains in cluster_domains_per_seq.items():
                    sorted_domains = sorted(domains, key=lambda d: d.start)
                    c_rows, c_cols, c_intersections = compare_sorted_spans(sorted_domains)
                    if c_rows.size == 0:
                        component_domains.extend(sorted_domains)
                    else:
                        similarity_matrix = coo_matrix(
                            (c_intersections, (c_rows, c_cols)),
                            shape=(len(sorted_domains), len(sorted_domains))
                        ).tocsr()
                        components, labels = csgraph.connected_components(
                            similarity_matrix,
                            directed=False,
                            return_labels=True
                        )
                        for component_id in range(components):
                            component_index = {
                                i: s for i, s in enumerate(sorted_domains)
                                if labels[i] == component_id
                            }
                            if len(component_index) == 1:
                                distinct_domain, = component_index.values()
                                component_domains.append(distinct_domain)
                            else:
                                merged_start = min(s.start for s in component_index.values())
                                merged_end = max(s.end for s in component_index.values())
                                merged_domain = DomainInstance(
                                    seq_id=seq_id,
                                    start=merged_start,
                                    summary_strand=0,
                                    end=merged_end,
                                    targets=[self.domain_index[int(i)]
                                             for i in component_index],
                                    source=self.source
                                )
                                component_domains.append(merged_domain)
                pbar.update(1)
                yield f"{peak_id}", component_domains


def find_domains(fasta_path: Path,
                 attributes: Dict[str, Any],
                 output_gff: Path,
                 tmp_dir: Optional[Path] = None,
                 result_dir: Optional[Path] = None) -> Path:
    """
    Find domain families from a FASTA sequence file given parameter attributes.

    Runs MMseqs2 or RiboSeek (or reads pre-computed TSV), filters spurious domain
    instances, merges target spans, constructs domain overlap graphs, and clusters
    domain instances into domain families written out to GFF format.

    :param fasta_path: Path to input FASTA file containing query sequences.
    :param attributes: Dictionary of aligner and DPC parameter settings.
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

    norm_attrs = _normalize_attributes(attributes)

    # Determine aligner and search strategy
    tsv_val = norm_attrs.get("tsv") or norm_attrs.get("tsv_path") or norm_attrs.get("read")
    aligner_setting = str(norm_attrs.get("aligner", "")).lower()

    if tsv_val and Path(tsv_val).exists():
        aligner_name = aligner_setting if aligner_setting and aligner_setting != "read" else "mmseqs2"
        logger.info(f"Reading pre-computed alignments from {tsv_val}")
        search_hits = parse_mmseqs_tsv(Path(tsv_val))
    elif aligner_setting == "riboseek":
        aligner_name = "riboseek"
        logger.info("Executing RiboSeek search")
        search_hits = riboseek_easy_search(
            fasta_file=fasta_path,
            tmp_dir=tmp_dir,
            result_dir=result_dir,
            num_iterations=int(norm_attrs.get("num_iterations", 1)),
            s=float(norm_attrs.get("s", 5.7)),
            gap_open=float(norm_attrs.get("gap_open", 23.0)),
            min_seq_id=float(norm_attrs.get("min_seq_id", 0.0)),
            timeout=norm_attrs.get("timeout"),
            cpu=int(norm_attrs.get("cpu", max(default_threads - 1, 1)))
        )
    else:
        aligner_name = "mmseqs2"
        logger.info("Executing MMseqs2 search")
        search_hits = mmseq_easy_search(
            fasta_file=fasta_path,
            tmp_dir=tmp_dir,
            result_dir=result_dir,
            k=int(norm_attrs.get("k", 7)),
            s=float(norm_attrs.get("s", 5.7)),
            gap_extend=float(norm_attrs.get("gap_extend", 1.0)),
            gap_open=float(norm_attrs.get("gap_open", 11.0)),
            min_seq_id=float(norm_attrs.get("min_seq_id", 0.0)),
            timeout=int(norm_attrs.get("timeout", 1800)),
            cpu=int(norm_attrs.get("cpu", default_threads))
        )

    # Extract DPC parameters
    min_dom_sim = float(norm_attrs.get("min_dom_sim", 0.35))
    dom_halo = float(norm_attrs.get("dom_halo", 0.5))
    min_dom_dens = float(norm_attrs.get("min_dom_dens", 1.0))
    min_fam_sim = float(norm_attrs.get("min_fam_sim", 0.25))
    fam_halo = float(norm_attrs.get("fam_halo", 0.5))
    min_fam_dens = float(norm_attrs.get("min_fam_dens", 1.0))
    halo_method = norm_attrs.get("halo_method", "knn")
    if halo_method and str(halo_method).lower() == "none":
        halo_method = None
    reassign_targets = bool(norm_attrs.get("reassign_targets", True))
    min_containment = float(norm_attrs.get("min_containment", 0.6))

    # Perform domain finding and clustering
    finder = DomainFinder(search=search_hits, source=aligner_name)
    finder.find_domain_instances(min_dom_sim=min_dom_sim,
                                 halo_stringency=dom_halo,
                                 min_dom_dens=min_dom_dens,
                                 halo_method=halo_method)
    finder.filter_spurious_domains(min_containment=min_containment,
                                   reassign_targets=reassign_targets)
    finder.merge_targets()
    finder.calculate_target_overlaps()

    output_gff.parent.mkdir(parents=True, exist_ok=True)
    with output_gff.open("w") as gff_writer:
        for fam_id, fam_domains in finder.families(min_fam_sim=min_fam_sim,
                                                   halo_stringency=fam_halo,
                                                   min_fam_dens=min_fam_dens,
                                                   halo_method=halo_method):
            for domain in fam_domains:
                gff_writer.write(domain.to_gff(cluster_id=fam_id) + "\n")

    logger.info(f"Successfully exported domain families to {output_gff}")
    return output_gff


if __name__ == "__main__":
    FASTA_PATH = Path("original_datasets/sequences.fna")
    OUTPUT_GFF = Path("output/domain_families.gff")
    TMP_DIR = Path("output/tmp")
    RESULT_DIR = Path("output/results_tsvs")

    attributes = {
        "aligner": "mmseqs2",
        "k": 7,
        "s": 5.7,
        "gap_open": 11.0,
        "gap_extend": 1.0,
        "min_seq_id": 0.0,
        "num_iterations": 1,
        "min_dom_sim": 0.35,
        "dom_halo": 0.5,
        "min_dom_dens": 1.0,
        "min_fam_sim": 0.25,
        "fam_halo": 0.5,
        "min_fam_dens": 1.0,
        "halo_method": "knn",
        "reassign_targets": True,
        "min_containment": 0.6,
    }

    if FASTA_PATH.exists():
        find_domains(fasta_path=FASTA_PATH,
                     attributes=attributes,
                     output_gff=OUTPUT_GFF,
                     tmp_dir=TMP_DIR,
                     result_dir=RESULT_DIR)
# fmt: on
