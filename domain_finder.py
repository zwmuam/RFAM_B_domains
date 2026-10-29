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
import math
from pathlib import Path
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

# standard libraries
import numpy as np
from scipy.sparse import coo_matrix, csgraph
from tqdm import tqdm

# internal repository/package imports
from aligners import (SequenceRegistry, Span, mmseq_easy_search, parse_mmseqs_tsv,
                      parse_mmseqs_tsv_stream, riboseek_easy_search)
from clustering import (compare_sorted_spans, dpc, jaccard_index,
                        max_containment)
from executor import Parallel, get_default_cpu_threads
from tweaks import default_threads, logger


def _process_query_batch(batch_items: Any,
                         min_dom_sim: float,
                         halo_stringency: float,
                         min_dom_dens: float,
                         halo_method: Optional[str],
                         source: str) -> Tuple[List[Tuple[Any, int, int, int, List[Span]]], int]:
    """
    Worker function to process a batch chunk of query sequences for domain discovery.
    Returns tuple of (batch_results, batch_halo_alns).
    """
    if isinstance(batch_items, tuple) and len(batch_items) == 2 and not isinstance(batch_items[0], (tuple, list)):
        batch_items = [batch_items]

    batch_results = []
    batch_halo_alns = 0
    for q_seq_id, sorted_query_spans in batch_items:
        n_spans = len(sorted_query_spans)
        if n_spans == 0:
            continue
        elif n_spans == 1:
            peak_span, = sorted_query_spans
            targets = peak_span.partners + [peak_span]
            strands = [t.strand for t in peak_span.partners]
            batch_results.append((q_seq_id, peak_span.start, sum(strands), peak_span.end, targets))
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
                batch_halo_alns += n_noise

            for peak_id, member_ids in peak_to_cluster.items():
                peak_span = sorted_query_spans[peak_id]
                member_targets = []
                for m_id in member_ids:
                    member_targets.extend(sorted_query_spans[int(m_id)].partners)
                targets = member_targets + [peak_span]
                strands = [t.strand for t in member_targets]
                batch_results.append((q_seq_id, peak_span.start, sum(strands), peak_span.end, targets))
    return batch_results, batch_halo_alns


def _process_filter_spurious_batch(batch_items: Any,
                                   min_containment: float,
                                   reassign_targets: bool) -> Tuple[List[Tuple[Any, List[Span]]], int, int]:
    """
    Worker function to process a batch of (seq_id, domain_list) for spurious domain filtering.
    """
    if isinstance(batch_items, tuple) and len(batch_items) == 2 and not isinstance(batch_items[0], (tuple, list)):
        batch_items = [batch_items]

    batch_kept_domains = []
    batch_discarded = 0
    batch_kept_count = 0

    for seq_id, domain_list in batch_items:
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
            batch_kept_count += len(kept_domains)
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
            batch_discarded += len(indices_to_remove)
            batch_kept_count += len(kept_domains)

        batch_kept_domains.append((seq_id, kept_domains))

    return batch_kept_domains, batch_discarded, batch_kept_count


def _process_merge_targets_batch(batch_items: Any) -> Tuple[List[Tuple[Any, Span]], List[Tuple[int, int]], int, int, int]:
    """
    Worker function to merge overlapping target spans for a batch chunk of domains.
    """
    if isinstance(batch_items, tuple) and len(batch_items) == 2 and not isinstance(batch_items[0], (tuple, list)):
        batch_items = [batch_items]

    batch_targets = []
    batch_domain_lengths = []
    n_merged = 0
    n_distinct = 0
    n_singletons = 0

    for d_id, domain in batch_items:
        domain_total_length = 0
        domain_targets_by_seq = defaultdict(list)
        for target in domain.partners:
            domain_targets_by_seq[target.seq_id].append(target)

        for t_seq_id, targets in domain_targets_by_seq.items():
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
                        batch_targets.append((t_seq_id, distinct_target))
                        domain_total_length += distinct_target.length
                        n_distinct += 1
                    else:
                        merged_start = min(s.start for s in component_index.values())
                        merged_end = max(s.end for s in component_index.values())
                        merged_target = Span(seq_id=t_seq_id,
                                             start=merged_start,
                                             end=merged_end,
                                             strand=0)
                        merged_target.domain = d_id
                        batch_targets.append((t_seq_id, merged_target))
                        domain_total_length += merged_target.length
                        n_merged += len(component_index) - 1
            else:
                singleton_target, = targets
                singleton_target.domain = d_id
                batch_targets.append((t_seq_id, singleton_target))
                domain_total_length += singleton_target.length
                n_singletons += 1

        batch_domain_lengths.append((d_id, domain_total_length))

    return batch_targets, batch_domain_lengths, n_merged, n_distinct, n_singletons


def _process_overlap_batch(batch_targets: Any,
                           n_domains: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function to process target sequence batches and return deduplicated sparse COO triplets.
    """
    if isinstance(batch_targets, tuple) and len(batch_targets) == 2 and not isinstance(batch_targets[0], (tuple, list)):
        batch_targets = [batch_targets]

    batch_rows, batch_cols, batch_data = [], [], []
    for seq_id, targets in batch_targets:
        n_targets = len(targets)
        if n_targets > 1:
            sorted_targets = sorted(targets, key=lambda s: s.start)
            local_rows, local_cols, intersections = compare_sorted_spans(sorted_targets)
            if local_rows.size > 0:
                domain_ids = np.array([t.domain for t in sorted_targets], dtype=np.int32)
                global_rows = domain_ids[local_rows]
                global_cols = domain_ids[local_cols]
                mask = global_rows != global_cols
                if np.any(mask):
                    g_r = np.minimum(global_rows[mask], global_cols[mask])
                    g_c = np.maximum(global_rows[mask], global_cols[mask])
                    coo = coo_matrix((intersections[mask], (g_r, g_c)), shape=(n_domains, n_domains))
                    csr = coo.tocsr()
                    csr.sum_duplicates()
                    coo_dedup = csr.tocoo()
                    batch_rows.append(coo_dedup.row)
                    batch_cols.append(coo_dedup.col)
                    batch_data.append(coo_dedup.data)

    if not batch_rows:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    cat_r = np.concatenate(batch_rows)
    cat_c = np.concatenate(batch_cols)
    cat_d = np.concatenate(batch_data)
    coo_batch = coo_matrix((cat_d, (cat_r, cat_c)), shape=(n_domains, n_domains))
    csr_batch = coo_batch.tocsr()
    csr_batch.sum_duplicates()
    coo_out = csr_batch.tocoo()
    return coo_out.row, coo_out.col, coo_out.data


def _process_evaluate_cluster_batch(batch_items: Any,
                                    source: str) -> List[Tuple[str, List[Any]]]:
    """
    Worker function to evaluate domain family clusters in parallel.
    """
    if isinstance(batch_items, tuple) and len(batch_items) == 2 and not isinstance(batch_items[0], (tuple, list)):
        batch_items = [batch_items]

    batch_results = []
    for peak_id, cluster_domains in batch_items:
        cluster_domains_per_seq = defaultdict(list)
        for d in cluster_domains:
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
                            targets=list(component_index.values()),
                            source=source
                        )
                        component_domains.append(merged_domain)
        batch_results.append((f"{peak_id}", component_domains))
    return batch_results


def count_fasta_queries(fasta_path: Path) -> int:
    """
    Count the number of sequence header lines starting with '>' in a FASTA sequence file.

    :param fasta_path: Path to the FASTA sequence file.
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
    return header_count


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

    __slots__ = ("source", "number", "targets")

    def __init__(self,
                 seq_id: Any,
                 start: int,
                 summary_strand: int,
                 end: int,
                 targets: List[Span],
                 source: str = "mmseqs2") -> None:
        """
        Initialize a DomainInstance.

        :param seq_id: Target sequence identifier (string or integer index).
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
               cluster_id: str,
               registry: Optional[SequenceRegistry] = None) -> str:
        """
        Format domain instance as a standard GFF3 string.

        :param cluster_id: Family/cluster identifier for the GFF attributes.
        :param registry: Optional SequenceRegistry to decode integer sequence ID.
        :return: Tab-delimited GFF3 record.
        """
        seq_str = registry.get_str(self.seq_id) if registry is not None else str(self.seq_id)
        strand_str = "+" if self.strand > 0 else "-" if self.strand < 0 else "."
        return (f"{seq_str}\t{self.source}\tDomain\t{self.start}\t"
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
                 search: Any,
                 source: str = "mmseqs2",
                 registry: Optional[SequenceRegistry] = None,
                 fasta_path: Optional[Path] = None,
                 total_queries: Optional[int] = None) -> None:
        """
        Initialize DomainFinder.

        :param search: Dictionary or Generator mapping query IDs to lists of alignment spans.
        :param source: Aligner source name used for GFF generation.
        :param registry: Optional SequenceRegistry instance.
        :param fasta_path: Optional FASTA file path used for initial size estimations.
        :param total_queries: Optional integer count of query sequences.
        :return: None
        """
        self.search = search
        self.source = source
        self.registry = registry
        self.fasta_path = Path(fasta_path) if fasta_path else None
        if total_queries is not None:
            self.total_queries = total_queries
        elif self.fasta_path:
            self.total_queries = count_fasta_queries(self.fasta_path)
        elif isinstance(search, dict):
            self.total_queries = len(search)
        else:
            self.total_queries = None

        self.domains_per_seq: Dict[Any, List[DomainInstance]] = {}
        self.n_domains = 0
        self.domain_index: Dict[int, DomainInstance] = {}
        self.domain_total_lengths: Dict[int, int] = {}
        self.targets_per_seq: Dict[Any, List[Span]] = {}
        self._overlap_rows: List[np.ndarray] = []
        self._overlap_cols: List[np.ndarray] = []
        self._overlap_data: List[np.ndarray] = []
        self.n_halo_alns = 0
        self.n_halo_domains = 0
        self.n_clusters = 0

    def find_domain_instances(self,
                              min_dom_sim: float,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: Optional[str]) -> None:
        """
        Identify candidate domain instances across query sequences via DPC density peaks using parallel task batching.

        :param min_dom_sim: Minimum domain similarity threshold.
        :param halo_stringency: Halo stringency parameter.
        :param min_dom_dens: Minimum domain density threshold.
        :param halo_method: Halo method ('traditional', 'knn', etc.) or None.
        :return: None
        """
        if isinstance(self.search, dict):
            query_iterable = list(self.search.items())
            total_queries = len(query_iterable)
        else:
            query_iterable = self.search
            total_queries = self.total_queries

        file_info = f" (from {self.fasta_path.name})" if self.fasta_path else ""
        if total_queries is not None:
            logger.info(f"Estimated work size for domain instance discovery: {total_queries} query sequences{file_info}")
        else:
            logger.info("Starting domain instance discovery")

        def _batch_generator(iterable: Any, batch_size: int) -> Generator[List[Any], None, None]:
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        n_jobs = get_default_cpu_threads()
        if total_queries is not None and total_queries > 0:
            batch_size = min(1000, max(1, total_queries // (n_jobs * 4)))
            total_batches = math.ceil(total_queries / batch_size)
        elif isinstance(query_iterable, (list, tuple)):
            total_queries = len(query_iterable)
            batch_size = min(1000, max(1, total_queries // (n_jobs * 4))) if total_queries > 0 else 1000
            total_batches = math.ceil(total_queries / batch_size) if total_queries > 0 else None
        else:
            batch_size = 1000
            total_batches = None

        query_batches = _batch_generator(query_iterable, batch_size)

        kwargs = {
            "min_dom_sim": min_dom_sim,
            "halo_stringency": halo_stringency,
            "min_dom_dens": min_dom_dens,
            "halo_method": halo_method,
            "source": self.source
        }

        p_runner = Parallel(parallelized_function=_process_query_batch,
                            input_collection=query_batches,
                            kwargs=kwargs,
                            description="Finding domain instances",
                            bar=True,
                            unit="batches",
                            total=total_batches)

        for batch_res, batch_halo in p_runner.result:
            self.n_halo_alns += batch_halo
            for q_seq_id, start, summary_strand, end, targets in batch_res:
                new_domain = DomainInstance(seq_id=q_seq_id,
                                            start=start,
                                            summary_strand=summary_strand,
                                            end=end,
                                            targets=targets,
                                            source=self.source)
                if new_domain.seq_id not in self.domains_per_seq:
                    self.domains_per_seq[new_domain.seq_id] = []
                self.domains_per_seq[new_domain.seq_id].append(new_domain)
                self.n_domains += 1

        logger.info(f"Found {self.n_domains} domain instances across "
                    f"{len(self.domains_per_seq)} query sequences")

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
        if not self.domains_per_seq:
            return 0, 0

        domain_items = list(self.domains_per_seq.items())
        total_items = len(domain_items)
        file_info = f" (from {self.fasta_path.name})" if self.fasta_path else ""
        logger.info(f"Estimated work size for spurious domain filtering: {total_items} sequence domain sets across {self.n_domains} candidate domains{file_info}")

        n_jobs = get_default_cpu_threads()
        batch_size = min(1000, max(1, total_items // (n_jobs * 4))) if total_items > 0 else 1000
        item_batches = [
            domain_items[i:i + batch_size]
            for i in range(0, total_items, batch_size)
        ]

        kwargs = {
            "min_containment": min_containment,
            "reassign_targets": reassign_targets
        }

        p_runner = Parallel(parallelized_function=_process_filter_spurious_batch,
                            input_collection=item_batches,
                            kwargs=kwargs,
                            description="Filtering spurious domains",
                            bar=True,
                            unit="batches",
                            total=len(item_batches))

        dc = count()
        n_discarded = 0
        n_kept = 0

        for batch_kept_domains, b_discarded, b_kept in p_runner.result:
            n_discarded += b_discarded
            n_kept += b_kept
            for seq_id, kept_domains in batch_kept_domains:
                for d in kept_domains:
                    self.domain_index[next(dc)] = d

        return n_discarded, n_kept

    def merge_targets(self) -> None:
        """
        Merge overlapping target spans for each domain to prepare for overlap matrix generation.

        :return: None
        """
        n_domains = len(self.domain_index)
        file_info = f" (from {self.fasta_path.name})" if self.fasta_path else ""
        logger.info(f"Estimated work size for merging target spans: {n_domains} domain instances{file_info}")
        if n_domains == 0:
            return

        domain_items = list(self.domain_index.items())
        n_jobs = get_default_cpu_threads()
        batch_size = min(1000, max(1, n_domains // (n_jobs * 4))) if n_domains > 0 else 1000
        domain_batches = [
            domain_items[i:i + batch_size]
            for i in range(0, n_domains, batch_size)
        ]

        p_runner = Parallel(parallelized_function=_process_merge_targets_batch,
                            input_collection=domain_batches,
                            description="Merging overlapping targets",
                            bar=True,
                            unit="batches",
                            total=len(domain_batches))

        tot_merged = 0
        tot_distinct = 0
        tot_singletons = 0

        for b_targets, b_lengths, b_merged, b_distinct, b_singletons in p_runner.result:
            tot_merged += b_merged
            tot_distinct += b_distinct
            tot_singletons += b_singletons
            for d_id, d_len in b_lengths:
                self.domain_total_lengths[d_id] = d_len
            for t_seq_id, target in b_targets:
                if t_seq_id not in self.targets_per_seq:
                    self.targets_per_seq[t_seq_id] = []
                self.targets_per_seq[t_seq_id].append(target)

        logger.info(f"Merged: {tot_merged}; Distinct: {tot_distinct}; "
                    f"Singletons: {tot_singletons}")

    def calculate_target_overlaps(self) -> None:
        """
        Calculate pairwise target overlaps across domains to construct overlap graph using parallel batching.

        :return: None
        """
        n_domains = len(self.domain_index)
        if n_domains == 0 or not self.targets_per_seq:
            return

        target_items = list(self.targets_per_seq.items())
        total_targets = len(target_items)
        file_info = f" (from {self.fasta_path.name})" if self.fasta_path else ""
        logger.info(f"Estimated work size for target overlap calculation: {total_targets} target sequence groups{file_info}")

        n_jobs = get_default_cpu_threads()
        batch_size = min(1000, max(1, total_targets // (n_jobs * 4))) if total_targets > 0 else 1000
        target_batches = [
            target_items[i:i + batch_size]
            for i in range(0, total_targets, batch_size)
        ]

        kwargs = {"n_domains": n_domains}
        p_runner = Parallel(parallelized_function=_process_overlap_batch,
                            input_collection=target_batches,
                            kwargs=kwargs,
                            description="Calculating target overlaps",
                            bar=True,
                            unit="batches",
                            total=len(target_batches))

        for b_rows, b_cols, b_data in p_runner.result:
            if b_rows.size > 0:
                self._overlap_rows.append(b_rows)
                self._overlap_cols.append(b_cols)
                self._overlap_data.append(b_data)

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

        if not self._overlap_rows:
            logger.warning("No edges in the overlap graph. "
                           "Yielding single-domain families.")
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            similarities = np.array([], dtype=np.float64)
        else:
            cat_rows = np.concatenate(self._overlap_rows)
            cat_cols = np.concatenate(self._overlap_cols)
            cat_data = np.concatenate(self._overlap_data)

            # Deduplicate and sum duplicate edges using SciPy sparse matrix aggregation
            coo = coo_matrix((cat_data, (cat_rows, cat_cols)), shape=(n_domains, n_domains))
            csr = coo.tocsr()
            csr.sum_duplicates()
            coo_dedup = csr.tocoo()

            rows = coo_dedup.row
            cols = coo_dedup.col
            intersections = coo_dedup.data

            row_domain_lengths = np.array([self.domain_total_lengths[r] for r in rows], dtype=np.int32)
            col_domain_lengths = np.array([self.domain_total_lengths[c] for c in cols], dtype=np.int32)

            if np.any(row_domain_lengths <= 0) or np.any(col_domain_lengths <= 0):
                raise ValueError("Unexpected non-positive domain length encountered.")

            similarities = jaccard_index(intersections=intersections,
                                         row_lengths=row_domain_lengths,
                                         column_lengths=col_domain_lengths)

        file_info = f" (from {self.fasta_path.name})" if self.fasta_path else ""
        logger.info(f"Estimated work size for family clustering: {n_domains} domain instances{file_info}")

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
        self.n_clusters = n_clusters

        logger.info(f"Estimated work size for cluster evaluation: {n_clusters} domain family clusters{file_info}")

        cluster_items = [
            (peak_id, [self.domain_index[int(d_id)] for d_id in member_ids])
            for peak_id, member_ids in peak_to_cluster.items()
        ]
        n_jobs = get_default_cpu_threads()
        batch_size = min(1000, max(1, n_clusters // (n_jobs * 4))) if n_clusters > 0 else 1000
        cluster_batches = [
            cluster_items[i:i + batch_size]
            for i in range(0, n_clusters, batch_size)
        ]

        kwargs = {"source": self.source}
        p_runner = Parallel(parallelized_function=_process_evaluate_cluster_batch,
                            input_collection=cluster_batches,
                            kwargs=kwargs,
                            description="Evaluating clusters",
                            bar=True,
                            unit="batches",
                            total=len(cluster_batches))

        for batch_res in p_runner.result:
            for fam_id, component_domains in batch_res:
                yield fam_id, component_domains


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

    # Sequence registry for efficient string management
    registry = SequenceRegistry()

    # Determine aligner and search strategy
    tsv_val = norm_attrs.get("tsv") or norm_attrs.get("tsv_path") or norm_attrs.get("read")
    aligner_setting = str(norm_attrs.get("aligner", "")).lower()

    if tsv_val and Path(tsv_val).exists():
        aligner_name = aligner_setting if aligner_setting and aligner_setting != "read" else "mmseqs2"
        logger.info(f"Reading pre-computed alignments from {tsv_val}")
        search_stream = parse_mmseqs_tsv_stream(Path(tsv_val), registry=registry)
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
            cpu=int(norm_attrs.get("cpu", max(default_threads - 1, 1))),
            registry=registry
        )
        search_stream = search_hits
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
            cpu=int(norm_attrs.get("cpu", default_threads)),
            registry=registry
        )
        search_stream = search_hits

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

    # Count FASTA sequence headers for initial size estimation
    total_queries = count_fasta_queries(fasta_path)
    logger.info(f"Estimated work size: {total_queries} query sequences in {fasta_path.name}")

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

# fmt: on
