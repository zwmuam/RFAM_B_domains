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
from aligners import (SequenceRegistry, Span, mmseq_easy_search,
                      parse_mmseqs_tsv_stream, riboseek_easy_search)
from clustering import (compare_sorted_spans, dpc, jaccard_index,
                        max_containment)
from executor import Parallel, get_default_cpu_threads
from tweaks import default_threads, logger

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


# =============================================================================
# Stage 1: Domain Discovery
# =============================================================================


def process_query(batch_items: Any,
                  min_dom_sim: float,
                  halo_stringency: float,
                  min_dom_dens: float,
                  halo_method: Optional[str],
                  source: str) -> Tuple[List[Tuple[Any, int, int, int, List[Span]]], int]:
    """
    Worker function to process a batch chunk of query sequences for domain discovery.
    Returns tuple of (batch_results, batch_halo_alns).
    """

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


# =============================================================================
# Stage 2: Filtering Spurious Domains
# =============================================================================


def filter_spurious(batch_items: Any,
                    min_containment: float,
                    reassign_targets: bool) -> Tuple[List[Tuple[Any, List[Span]]], int, int]:
    """
    Worker function to process a batch of (seq_id, domain_list) for spurious domain filtering.
    """

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


# =============================================================================
# Stage 3: Target Merging
# =============================================================================


def merge_targets(batch_items: Any) -> Tuple[List[Tuple[Any, Span]], List[Tuple[int, int]], int, int, int]:
    """
    Worker function to merge overlapping target spans for a batch chunk of domains.
    """

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


# =============================================================================
# Stage 4: Target Overlap Calculation
# =============================================================================


def process_overlap(batch_targets: Any,
                    n_domains: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function to process target sequence batches and return deduplicated sparse COO triplets.
    """

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


# =============================================================================
# Stage 5: Family Finding & Cluster Evaluation
# =============================================================================


def evaluate_cluster(batch_items: Any,
                     source: str) -> List[Tuple[str, List[Any]]]:
    """
    Worker function to evaluate domain family clusters in parallel.
    """

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

# fmt: on
