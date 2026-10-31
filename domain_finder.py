# fmt: off
# !/usr/bin/env python3
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
from typing import Any, List, Optional, Tuple, Generator
from pathlib import Path

# standard libraries
import numpy as np
from scipy.sparse import coo_matrix, csgraph

# internal repository/package imports
from tweaks import Parallel
from aligners import batch_parse_mmseqs_tsv, SequenceRegistry
from clustering import (compare_sorted_spans,
                        jaccard_index, max_containment,
                        dpc)


def domain_instances_on_query(query_id: int,
                              sorted_query_spans: np.ndarray,
                              target_spans: np.ndarray,
                              min_dom_sim: float,
                              min_containment: float,
                              reassign_targets: bool,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: Optional[str]):
    """
    Process a query and identify conserved domain instances.
    :param query_id: The ID of the query sequence.
    :param sorted_query_spans: A numpy array of sorted query spans.
    :param target_spans: A numpy array of target spans.
    :param min_dom_sim: The minimum similarity for domain identification.
    :param min_containment: The minimum containment for domain identification.
    :param reassign_targets: Whether to reassign targets.
    :param halo_stringency: The stringency for halo filtering.
    :param min_dom_dens: The minimum density for domain identification.
    :param halo_method: The method for halo filtering.

    Returns:
        A list of tuples containing the peak ID and the corresponding component domains.
    """

    unique_query_spans, cluster_map, weights = np.unique(sorted_query_spans,
                                                         axis=0,
                                                         return_inverse=True,
                                                         return_counts=True)
    n_spans = unique_query_spans.shape[0]
    if n_spans == 1:
        # all spans are identical or there is only one span - trivial case, no clustering needed
        cluster_map[:] = 0
    else:
        rows, cols, intersections = compare_sorted_spans(sorted_query_spans)
        if rows.size == 0:
            # all spans are disjoint cluster_map unchanged (every unique span is its own cluster)
            pass
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

            for peak_id, member_ids in peak_to_cluster.items():
                cluster_map[np.isin(cluster_map, member_ids)] = peak_id

            if len(peak_to_cluster) > 1:
                unique_domain_indices = np.array(list(peak_to_cluster.keys()))
                peak_mask = np.isin(rows, unique_domain_indices) & np.isin(cols, unique_domain_indices)
                if peak_mask.sum() > 0:
                    cluster_map = filter_spurious_domains(cluster_map=cluster_map,
                                                          suspect_rows=rows[peak_mask],
                                                          suspect_cols=cols[peak_mask],
                                                          suspect_intersections=intersections[peak_mask],
                                                          suspect_lengths=span_lengths[peak_mask],
                                                          min_containment=min_containment,
                                                          reassign_targets=reassign_targets)
    unique_domain_indices = np.unique(cluster_map)

    # add domain pointers to the target spans
    targets_with_domains = np.column_stack((target_spans, cluster_map))
    # now each span is defined as: t_seq_int, t_start, t_end, t_strand, t_domain_id

    # TODO the domain themselves MUST be appended to the target spans - otherwise the target_batch_processor will underestimate the overlaps.

    raise NotImplementedError("This function is not yet implemented.")

    # return


def filter_spurious_domains(cluster_map: np.ndarray,
                            suspect_rows: np.ndarray,
                            suspect_cols: np.ndarray,
                            suspect_intersections: np.ndarray,
                            suspect_lengths: np.ndarray,
                            min_containment: float,
                            reassign_targets: bool) -> np.ndarray:
    """
    Filter out low-scoring overlapping domains.
    :param cluster_map: cluster map to filter.
    :param suspect_rows: indices of the suspect overlaps (in the unique_spans array).
    :param suspect_cols: indices of the suspect overlaps (in the unique_spans array).
    :param suspect_intersections: intersection lengths of the suspect domain instances.
    :param suspect_lengths: The lengths of the suspect domain instances.
    :param min_containment: The minimum containment threshold for filtering.
    :param reassign_targets: should the targets be reassigned to high-scoring domains after filtering
    :return: refined cluster map with no significant domain overlaps.
    """

    containment = max_containment(intersections=suspect_intersections,
                                  row_lengths=suspect_lengths[suspect_rows],
                                  column_lengths=suspect_lengths[suspect_cols])
    connected_mask = containment >= min_containment

    source_nodes = suspect_rows[connected_mask]
    target_nodes = suspect_cols[connected_mask]

    node_weights = {node: weight for node, weight in enumerate(suspect_lengths)}

    neighbors = defaultdict(set)
    for source, target in zip(np.asarray(source_nodes).tolist(), np.asarray(target_nodes).tolist()):
        if source != target:  # guard against self-loops
            neighbors[source].add(target)
            neighbors[target].add(source)

    removed_nodes = []

    for node in sorted(neighbors, key=lambda n: (node_weights[n], n)):
        live_neighbors = neighbors.pop(node)
        if live_neighbors:  # node is still part of a connected component
            removed_nodes.append(node)
            if reassign_targets:
                # Reassign targets to the neighbor with the highest weight
                best_neighbor = max(live_neighbors, key=lambda n: node_weights[n])
                cluster_map[cluster_map == node] = best_neighbor
            else:
                # Remove the node from the cluster map
                cluster_map[cluster_map == node] = -1  # Mark as removed
            for neighbor in live_neighbors:
                neighbors[neighbor].remove(node)

    return cluster_map


def target_cleanup(target_spans: np.ndarray):
    """
    Sort spans by target sequence and domain pointers.
    Foe every target-sequence-domain pair, merge overlapping spans into a single span.
    :param target_spans: numpy array of target spans (start, end) on a single sequence.
    :return: numpy array of cleaned target spans.
    """
    raise NotImplementedError("This function is not yet implemented.")


def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
    """
    Merge overlapping target spans into non-overlapping intervals.
    :param targets_on_sequence: numpy array of target spans (start, end) on a single sequence
    :return: numpy array of merged target spans
    """
    if len(targets_on_sequence) == 0:
        return targets_on_sequence.copy()
    s = np.ascontiguousarray(targets_on_sequence[:, 0])  # copies, so the caller's array is untouched
    e = np.ascontiguousarray(targets_on_sequence[:, 1])
    s.sort()
    e.sort()
    brk = np.flatnonzero(s[1:] > e[:-1])  # gaps between merged groups
    out = np.empty((len(brk) + 1, 2), dtype=targets_on_sequence.dtype)
    out[0, 0] = s[0]
    out[1:, 0] = s[brk + 1]
    out[:-1, 1] = e[brk]
    out[-1, 1] = e[-1]
    return out


def batch_query_processor(batch: List[Tuple[int, np.ndarray, np.ndarray]],
                          min_dom_sim: float,
                          min_containment: float,
                          reassign_targets: bool,
                          halo_stringency: float,
                          min_dom_dens: float,
                          halo_method: Optional[str]) -> List[Tuple[str, List[Any]]]:
    """
    Process a batch of queries and prepare it for efficient accumulation for subsequent per-target processing.
    (this function is designed to be run in parallel across multiple processes to generate domain instances and
    per-target
    :param q_batch_generator: A generator yielding batches of query IDs and their corresponding sorted query spans and target spans.
    :param min_dom_sim: The minimum similarity for domain identification.
    :param min_containment: The minimum containment for domain identification.
    :param reassign_targets: Whether to reassign targets.
    :param halo_stringency: The stringency for halo filtering.
    :param min_dom_dens: The minimum density for domain identification.
    :param halo_method: The method for halo filtering.
    :return: A list of tuples containing the peak ID and the corresponding component domains.
    """
    for query_id, sorted_query_spans, target_spans in batch:
        XXX = domain_instances_on_query(query_id=query_id,
                                        sorted_query_spans=sorted_query_spans,
                                        target_spans=target_spans,
                                        min_dom_sim=min_dom_sim,
                                        min_containment=min_containment,
                                        reassign_targets=reassign_targets,
                                        halo_stringency=halo_stringency,
                                        min_dom_dens=min_dom_dens,
                                        halo_method=halo_method)
        raise NotImplementedError("This function is not yet implemented.")


# =============================================================================
# Stage 4: Target Overlap Calculation
# =============================================================================

def overlaps_on_target(sorted_target_spans: np.ndarray,
                       aligned_domain_pointers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate overlaps between target spans and aligned domain pointers.
    """

    n_targets = sorted_target_spans.shape[0]
    if n_targets > 1:
        local_rows, local_cols, intersections = compare_sorted_spans(sorted_target_spans)
        if local_rows.size > 0:
            global_rows = aligned_domain_pointers[local_rows]
            global_cols = aligned_domain_pointers[local_cols]
            return global_rows, global_cols, intersections
    return None, None, None


def target_batch_processor(batch_targets: Any,
                           n_domains: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a batch of target sequences and return deduplicated sparse COO triplets.
    """
    batch_rows, batch_cols, batch_data = [], [], []
    for target_id, target_spans, domain_pointers in batch_targets:
        sorted_target_spans = target_spans[np.argsort(target_spans[:, 0])]
        aligned_domain_pointers = domain_pointers[np.argsort(target_spans[:, 0])]
        rows, cols, intersections = overlaps_on_target(sorted_target_spans, aligned_domain_pointers)
        if rows is not None:
            batch_rows.append(rows)
            batch_cols.append(cols)
            batch_data.append(intersections)


def agregate_sparse_batches(q_batch_generator: Generator[List[int, np.ndarray, np.ndarray], None, None]):
    """
    Aggregate sparse COO triplets from multiple batches into a single deduplicated sparse COO matrix.
    """
    # the fastest most memory-efficient way to aggregate sparse canonical COO
    raise NotImplementedError("This function is not yet implemented.")


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
                similarity_matrix = coo_matrix((c_intersections,
                                                (c_rows, c_cols)),
                                               shape=(len(sorted_domains),
                                                      len(sorted_domains))).tocsr()
                components, labels = csgraph.connected_components(similarity_matrix,
                                                                  directed=False,
                                                                  return_labels=True)
                for component_id in range(components):
                    component_index = {i: s for i, s in enumerate(sorted_domains)
                                       if labels[i] == component_id}
                    if len(component_index) == 1:
                        distinct_domain, = component_index.values()
                        component_domains.append(distinct_domain)
                    else:
                        merged_start = min(s.start for s in component_index.values())
                        merged_end = max(s.end for s in component_index.values())
    return batch_results


def find_conserved_domains(mmseqs_tsv: Path,
                           min_dom_sim: float,
                           min_containment: float,
                           reassign_targets: bool,
                           halo_transparency: float,
                           min_dom_dens: float,
                           halo_method: Optional[str],
                           cpu: int) -> List[Tuple[str, List[Any]]]:
    """
    Central function of the module, orchestrating the identification of
    conserved domain instances and families from query and target spans.
    :param mmseqs_tsv: Path to the MMseqs2 BLAST-TAB TSV output file.
    :param min_dom_sim: The minimum similarity for domain identification.
    :param min_containment: The minimum containment for domain identification.
    :param reassign_targets: Whether to reassign targets.
    :param halo_transparency: The transparency for halo filtering.
    :param min_dom_dens: The minimum density for domain identification.
    :param halo_method: The method for halo filtering. Either 'XXX' or 'YYY'.
    :param cpu: Number of CPU threads to use for parallel processing.
    :return: A list of tuples containing the peak ID and the corresponding component domains.
    """
    sequence_registry = SequenceRegistry()

    # run the batch_query_processor in parallel to get domain instances, per-target span batches, and aligned domain pointers
    workers = Parallel(XXX, cpu=cpu)

    # aggregate the results from the workers and prepare for target overlap calculation

    # run the target_batch_processor in parallel to get target opverlaps between domain instances


