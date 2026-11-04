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
from itertools import count
from pathlib import Path
import tempfile
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# standard libraries
import numpy as np
from tqdm import tqdm

# internal repository/package imports
from aligners import (SequenceRegistry,
                      populate_registry_from_fasta,
                      batch_parse_mmseqs_tsv)
from clustering import (compare_sorted_spans,
                        jaccard_index, max_containment,
                        dpc)
from mpire import WorkerPool
from executor import Parallel, default_threads, logger, _safe_ctx_name


def determine_strand(strands: np.ndarray) -> str:
    """
    Determine the consensus strand of a domain instance based on its target hits.

    Minimizes opposite strand targets: if majority of targets are -1, then the domain
    itself is on the '-' strand; otherwise '+'.

    :param strands: 1D NumPy array of alignment strands (-1 or 1).
    :return: Consensus strand string ('+' or '-').
    """
    if strands.size == 0:
        return "+"
    neg_count = np.sum(strands == -1)
    pos_count = np.sum(strands == 1)
    return "-" if neg_count > pos_count else "+"


def filter_spurious_domains(cluster_map: np.ndarray,
                            suspect_rows: np.ndarray,
                            suspect_cols: np.ndarray,
                            suspect_intersections: np.ndarray,
                            suspect_lengths: np.ndarray,
                            min_containment: float,
                            reassign_targets: bool,
                            domain_ranks: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Filter overlapping candidates using pre-computed biological priority ranking.
    Lower rank index indicates higher biological priority:
    1. Primary: Alignment hit support N_targets (highest support wins).
    2. Secondary: Span compactness (shorter length wins on equal support).
    3. Deterministic tie-breaker: Coordinate index (ascending).

    :param cluster_map: Cluster map to filter.
    :param suspect_rows: Indices of the suspect overlaps (in the unique_spans array).
    :param suspect_cols: Indices of the suspect overlaps (in the unique_spans array).
    :param suspect_intersections: Intersection lengths of suspect domain instances.
    :param suspect_lengths: Lengths of unique domain instances.
    :param min_containment: Minimum containment threshold for filtering.
    :param reassign_targets: Whether targets should be reassigned to strongest neighbors.
    :param domain_ranks: 1D array of priority ranks (0 = best candidate).
    :return: Refined cluster map with spurious domains removed or reassigned.
    """
    containment = max_containment(intersections=suspect_intersections,
                                  row_lengths=suspect_lengths[suspect_rows],
                                  column_lengths=suspect_lengths[suspect_cols])
    connected_mask = containment >= min_containment
    if not np.any(connected_mask):
        return cluster_map

    source_nodes = suspect_rows[connected_mask]
    target_nodes = suspect_cols[connected_mask]

    n_spans = suspect_lengths.shape[0]
    if domain_ranks is None:
        valid_mask = cluster_map >= 0
        support_counts = np.bincount(cluster_map[valid_mask], minlength=n_spans)
        sort_order = np.lexsort((np.arange(n_spans), suspect_lengths, -support_counts))
        domain_ranks = np.empty(n_spans, dtype=np.int32)
        domain_ranks[sort_order] = np.arange(n_spans, dtype=np.int32)

    neighbors = defaultdict(set)
    for source, target in zip(np.asarray(source_nodes).tolist(), np.asarray(target_nodes).tolist()):
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)

    # Prune worst candidate first (highest rank index) using pre-computed integer ranking
    for node in sorted(neighbors, key=domain_ranks.__getitem__, reverse=True):
        live_neighbors = neighbors.pop(node, None)
        if live_neighbors:
            if reassign_targets:
                # O(k) selection of strongest neighbor (lowest rank index)
                best_neighbor = min(live_neighbors, key=domain_ranks.__getitem__)
                cluster_map[cluster_map == node] = best_neighbor
            else:
                cluster_map[cluster_map == node] = -1
            for neighbor in live_neighbors:
                neighbors[neighbor].discard(node)

    return cluster_map


def domain_instances_on_query(query_id: int,
                              sorted_query_spans: np.ndarray,
                              target_spans: np.ndarray,
                              min_dom_sim: float,
                              min_containment: float,
                              reassign_targets: bool,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: Optional[str],
                              point_specific: bool = False) \
        -> Tuple[List[Tuple[int, int, int, str]], np.ndarray]:
    """
    Process a query sequence and identify conserved domain instances using DPC.

    :param query_id: Integer sequence ID of the query.
    :param sorted_query_spans: Sorted 2D NumPy array of query alignment spans (start, end).
    :param target_spans: 2D NumPy array of target hits (t_seq_int, t_start, t_end, t_strand).
    :param min_dom_sim: Minimum similarity for domain identification.
    :param min_containment: Minimum containment for domain filtering.
    :param reassign_targets: Whether to reassign targets of removed spurious domains.
    :param halo_stringency: Stringency for halo filtering.
    :param min_dom_dens: Minimum density for domain identification.
    :param halo_method: Halo filtering algorithm.
    :param point_specific: Whether to evaluate halo border filtering individually per point.
    :return: Tuple of:
             - local_domains: List of (query_id, peak_start, peak_end, consensus_strand)
             - all_targets: 2D NumPy array of (t_seq_int, t_start, t_end, local_domain_id)
    """
    unique_query_spans, cluster_map, weights = np.unique(sorted_query_spans,
                                                         axis=0,
                                                         return_inverse=True,
                                                         return_counts=True)
    n_spans = unique_query_spans.shape[0]
    span_lengths = unique_query_spans[:, 1] - unique_query_spans[:, 0]

    if n_spans == 1:
        if weights[0] >= min_dom_dens:
            cluster_map[:] = 0
            peak_indices = np.array([0], dtype=np.int32)
        else:
            cluster_map[:] = -1
            peak_indices = np.empty(0, dtype=np.int32)
    else:
        rows, cols, intersections = compare_sorted_spans(unique_query_spans)
        if rows.size == 0:
            qualifying = np.flatnonzero(weights >= min_dom_dens).astype(np.int32)
            assigned_peak_map = np.full(n_spans, -1, dtype=np.int32)
            assigned_peak_map[qualifying] = qualifying
            cluster_map = assigned_peak_map[cluster_map]
            peak_indices = qualifying
        else:
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
                                           min_dens=min_dom_dens,
                                           point_specific=point_specific)

            assigned_peak_map = np.full(n_spans, -1, dtype=np.int32)
            for peak_id, member_ids in peak_to_cluster.items():
                assigned_peak_map[member_ids] = peak_id
            cluster_map = assigned_peak_map[cluster_map]

            if len(peak_to_cluster) > 1:
                unique_domain_indices = np.array(list(peak_to_cluster.keys()), dtype=np.int32)
                peak_mask = np.isin(rows, unique_domain_indices) & np.isin(cols, unique_domain_indices)
                if np.any(peak_mask):
                    # Compute alignment hit support per unique span, excluding negative noise (-1)
                    valid_mask = cluster_map >= 0
                    support_counts = np.bincount(cluster_map[valid_mask], minlength=n_spans)
                    # Lexsort: -support (desc), length (asc), index (asc)
                    sort_order = np.lexsort((np.arange(n_spans), span_lengths, -support_counts))
                    domain_ranks = np.empty(n_spans, dtype=np.int32)
                    domain_ranks[sort_order] = np.arange(n_spans, dtype=np.int32)

                    cluster_map = filter_spurious_domains(cluster_map=cluster_map,
                                                          suspect_rows=rows[peak_mask],
                                                          suspect_cols=cols[peak_mask],
                                                          suspect_intersections=intersections[peak_mask],
                                                          suspect_lengths=span_lengths,
                                                          min_containment=min_containment,
                                                          reassign_targets=reassign_targets,
                                                          domain_ranks=domain_ranks)

            active_peaks = np.unique(cluster_map)
            active_peaks = active_peaks[active_peaks >= 0]
            peak_indices = active_peaks

    if len(peak_indices) == 0:
        return [], np.empty((0, 4), dtype=np.int32)

    # Re-index local domains contiguously: 0, 1, ..., K-1
    domain_id_map = {peak_id: local_idx for local_idx, peak_id in enumerate(peak_indices)}
    local_domain_ptrs = np.array([domain_id_map.get(c, -1) for c in cluster_map], dtype=np.int32)
    valid_mask = local_domain_ptrs >= 0

    t_seqs = target_spans[:, 0][valid_mask]
    t_starts = target_spans[:, 1][valid_mask]
    t_ends = target_spans[:, 2][valid_mask]
    t_strands = target_spans[:, 3][valid_mask]
    t_doms = local_domain_ptrs[valid_mask]

    local_domains: List[Tuple[int, int, int, str]] = []
    for local_id, peak_id in enumerate(peak_indices):
        peak_span = unique_query_spans[peak_id]
        assigned_strands = t_strands[t_doms == local_id]
        dom_strand = determine_strand(assigned_strands)
        local_domains.append((query_id, int(peak_span[0]), int(peak_span[1]), dom_strand))

    # Supplement targets with self-hits: (query_id, peak_start, peak_end, local_domain_id)
    n_self = len(peak_indices)
    self_targets = np.empty((n_self, 4), dtype=np.int32)
    for local_id, peak_id in enumerate(peak_indices):
        peak_span = unique_query_spans[peak_id]
        self_targets[local_id] = [query_id, peak_span[0], peak_span[1], local_id]

    if valid_mask.any():
        partner_targets = np.column_stack((t_seqs, t_starts, t_ends, t_doms))
        all_targets = np.vstack((partner_targets, self_targets))
    else:
        all_targets = self_targets

    return local_domains, all_targets


def _query_worker_dispatch(shared_kwargs: Dict[str, Any],
                           batch: List[Tuple[int, np.ndarray, np.ndarray]]) \
        -> List[Tuple[List[Tuple[int, int, int, str]], np.ndarray]]:
    """
    Worker dispatch function receiving shared query_kwargs and query batch.
    """
    return batch_query_processor(batch=batch, **shared_kwargs)


def batch_query_processor(batch: List[Tuple[int, np.ndarray, np.ndarray]],
                          min_dom_sim: float,
                          min_containment: float,
                          reassign_targets: bool,
                          halo_stringency: float,
                          min_dom_dens: float,
                          halo_method: Optional[str],
                          point_specific: bool = False) \
        -> List[Tuple[List[Tuple[int, int, int, str]], np.ndarray]]:
    """
    Process a batch of queries and identify domain instances and associated targets.

    :param batch: List of tuples containing (query_id, sorted_query_spans, target_spans).
    :param min_dom_sim: Minimum similarity for domain identification.
    :param min_containment: Minimum containment for domain filtering.
    :param reassign_targets: Whether to reassign targets of spurious domains.
    :param halo_stringency: Stringency for halo filtering.
    :param min_dom_dens: Minimum density for domain identification.
    :param halo_method: Halo filtering algorithm.
    :param point_specific: Whether to evaluate halo border filtering individually per point.
    :return: List of tuples (local_domains, all_targets) for each query in the batch.
    """
    batch_results = []
    for query_id, sorted_query_spans, target_spans in batch:
        res = domain_instances_on_query(query_id=query_id,
                                        sorted_query_spans=sorted_query_spans,
                                        target_spans=target_spans,
                                        min_dom_sim=min_dom_sim,
                                        min_containment=min_containment,
                                        reassign_targets=reassign_targets,
                                        halo_stringency=halo_stringency,
                                        min_dom_dens=min_dom_dens,
                                        halo_method=halo_method,
                                        point_specific=point_specific)
        batch_results.append(res)
    return batch_results


# =============================================================================
# Stage 4: Target Overlap Calculation
# =============================================================================

def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
    """
    Merge overlapping target spans into non-overlapping intervals.

    :param targets_on_sequence: 2D NumPy array of target spans (start, end).
    :return: 2D NumPy array of merged non-overlapping target spans.
    """
    if len(targets_on_sequence) == 0:
        return targets_on_sequence.copy()
    s = np.ascontiguousarray(targets_on_sequence[:, 0])
    e = np.ascontiguousarray(targets_on_sequence[:, 1])
    order = np.lexsort((e, s))
    s = s[order]
    e = e[order]
    e_cum = np.maximum.accumulate(e)
    brk = np.flatnonzero(s[1:] > e_cum[:-1])
    out = np.empty((len(brk) + 1, 2), dtype=targets_on_sequence.dtype)
    out[0, 0] = s[0]
    out[1:, 0] = s[brk + 1]
    out[:-1, 1] = e_cum[brk]
    out[-1, 1] = e_cum[-1]
    return out


def overlaps_on_target(sorted_target_spans: np.ndarray,
                       aligned_domain_pointers: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate overlaps between target spans and aligned domain pointers on a sequence.

    :param sorted_target_spans: Start-sorted 2D NumPy array of target spans.
    :param aligned_domain_pointers: 1D NumPy array of domain pointers for each span.
    :return: Tuple of (global_rows, global_cols, intersections).
    """
    n_targets = sorted_target_spans.shape[0]
    if n_targets > 1:
        local_rows, local_cols, intersections = compare_sorted_spans(sorted_target_spans)
        if local_rows.size > 0:
            mask = aligned_domain_pointers[local_rows] != aligned_domain_pointers[local_cols]
            if np.any(mask):
                global_rows = aligned_domain_pointers[local_rows][mask]
                global_cols = aligned_domain_pointers[local_cols][mask]
                return global_rows, global_cols, intersections[mask]
    return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)


def target_batch_processor(batch_targets: List[Tuple[int, np.ndarray, np.ndarray]],
                           chunk_id: Optional[int] = None,
                           shard_dir_path: Optional[str] = None) \
        -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]],
                 Tuple[int, int, Dict[int, int]]]:
    """
    Process a batch of target sequences to merge same-domain intervals, compute
    per-domain target lengths, and calculate pairwise target overlaps.
    Maintains 32-bit integers for all physical genomic coordinate operations.

    If shard_dir_path is provided, writes coordinate arrays directly to binary `.npy` shards
    and returns (chunk_id, n_edges, domain_lengths_dict) to avoid duplicating memory in the parent process.

    :param batch_targets: List of (target_id, target_spans, domain_pointers).
    :param chunk_id: Optional chunk identifier for disk sharding.
    :param shard_dir_path: Optional directory string where .npy shards are written.
    :return: Either (rows, cols, intersections, domain_lengths_dict) or (chunk_id, n_edges, domain_lengths_dict).
    """
    batch_rows, batch_cols, batch_data = [], [], []
    batch_dom_lengths: Dict[int, int] = defaultdict(int)

    for target_id, target_spans, domain_pointers in batch_targets:
        if domain_pointers.size == 0:
            continue

        ## mem_opt ## Group domain intervals contiguously using boundary indexing instead of repeated unvectorized boolean masks
        d_order = np.argsort(domain_pointers)
        s_doms = domain_pointers[d_order]
        s_spans = target_spans[d_order]

        dom_boundaries = np.flatnonzero(np.diff(s_doms)) + 1
        dom_splits = np.split(s_doms, dom_boundaries)
        span_splits = np.split(s_spans, dom_boundaries)

        merged_spans_list = []
        aligned_domains_list = []

        for d_group, sp_group in zip(dom_splits, span_splits):
            d_id = int(d_group[0])
            m_spans = merge_overlapping_spans(sp_group)
            for sp in m_spans:
                merged_spans_list.append(sp)
                aligned_domains_list.append(d_id)
                batch_dom_lengths[d_id] += int(sp[1] - sp[0])

        if len(merged_spans_list) <= 1:
            continue

        merged_spans = np.asarray(merged_spans_list, dtype=np.int32)
        aligned_domains = np.asarray(aligned_domains_list, dtype=np.int32)

        sort_order = np.argsort(merged_spans[:, 0])
        sorted_spans = merged_spans[sort_order]
        aligned_sorted_domains = aligned_domains[sort_order]

        rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
        if rows.size > 0:
            batch_rows.append(rows)
            batch_cols.append(cols)
            batch_data.append(intersections.astype(np.int32, copy=False))

    r_arr = np.concatenate(batch_rows) if batch_rows else np.empty(0, dtype=np.int32)
    c_arr = np.concatenate(batch_cols) if batch_cols else np.empty(0, dtype=np.int32)
    i_arr = np.concatenate(batch_data) if batch_data else np.empty(0, dtype=np.int32)

    if shard_dir_path is not None and chunk_id is not None:
        shard_dir = Path(shard_dir_path)
        n_edges = r_arr.size
        if n_edges > 0:
            np.save(shard_dir / f"{chunk_id}_r.npy", r_arr)
            np.save(shard_dir / f"{chunk_id}_c.npy", c_arr)
            np.save(shard_dir / f"{chunk_id}_i.npy", i_arr)
        return chunk_id, n_edges, dict(batch_dom_lengths)

    return r_arr, c_arr, i_arr, dict(batch_dom_lengths)


def _target_sharded_worker_dispatch(shared_shard_dir: str,
                                    task_pkg: Tuple[int, List[Tuple[int, np.ndarray, np.ndarray]]]) \
        -> Tuple[int, int, Dict[int, int]]:
    """
    Worker dispatch function for disk-sharded Stage 4 target batch processing.
    """
    chunk_id, batch_targets = task_pkg
    return target_batch_processor(batch_targets=batch_targets,
                                  chunk_id=chunk_id,
                                  shard_dir_path=shared_shard_dir)


# =============================================================================
# Stage 5: Family Finding
# =============================================================================

def family_finder(rows: np.ndarray,
                  cols: np.ndarray,
                  intersections: np.ndarray,
                  domain_lengths: np.ndarray,
                  n_domains: int,
                  min_fam_sim: float,
                  min_fam_dens: float,
                  halo_stringency: float,
                  halo_method: Optional[str] = "similarity_weighted",
                  point_specific: bool = False) \
        -> Tuple[Dict[int, np.ndarray], int]:
    """
    Cluster domain instances into domain families using DPC based on Jaccard similarity
    of accumulated target overlaps and total domain target lengths.

    :param rows: 1D NumPy array of domain index pairs (row).
    :param cols: 1D NumPy array of domain index pairs (col).
    :param intersections: 1D NumPy array of accumulated target overlap lengths.
    :param domain_lengths: 1D NumPy array of total target lengths per domain.
    :param n_domains: Total number of domain instances.
    :param min_fam_sim: Minimum family similarity for DPC clustering.
    :param min_fam_dens: Minimum family density for DPC clustering.
    :param halo_stringency: Halo stringency threshold.
    :param halo_method: Halo filtering algorithm.
    :return: Tuple of (peak_to_cluster dictionary, n_noise count).
    """
    if n_domains == 0:
        return {}, 0

    # Ensure canonical undirected edges (rows < cols)
    if rows.size > 0:
        u_r = np.minimum(rows, cols).astype(np.int64)
        u_c = np.maximum(rows, cols).astype(np.int64)
        diff = u_r != u_c
        u_r, u_c, inters = u_r[diff], u_c[diff], intersections[diff].astype(np.int64)
    else:
        u_r = np.empty(0, dtype=np.int64)
        u_c = np.empty(0, dtype=np.int64)
        inters = np.empty(0, dtype=np.int64)

    if u_r.size == 0:
        # Trivial self-similarity diagonal correction (self-density = 1.0)
        # Qualified solitary domains must meet min_fam_dens
        qualifying = np.flatnonzero(np.ones(n_domains, dtype=np.float64) >= min_fam_dens)
        if qualifying.size == 0:
            return {}, int(n_domains)
        return {int(i): np.array([i], dtype=np.int64) for i in qualifying}, int(n_domains - qualifying.size)

    keys = (u_r << 32) | u_c
    order = np.argsort(keys)
    keys_sorted = keys[order]
    inters_sorted = inters[order]

    starts = np.flatnonzero(np.concatenate(([True], keys_sorted[1:] != keys_sorted[:-1])))
    summed_inters = np.add.reduceat(inters_sorted, starts)
    uniq_keys = keys_sorted[starts]

    final_r = (uniq_keys >> 32).astype(np.int32)
    final_c = (uniq_keys & 0xFFFFFFFF).astype(np.int32)

    unions = domain_lengths[final_r] + domain_lengths[final_c] - summed_inters
    valid = (unions > 0) & (summed_inters > 0)
    final_r = final_r[valid]
    final_c = final_c[valid]
    similarities = summed_inters[valid] / unions[valid]

    logger.info(
        f"Clustering {n_domains} domains into families using DPC with min_fam_sim={min_fam_sim}, "
        f"halo_stringency={halo_stringency}, min_fam_dens={min_fam_dens}, halo_method={halo_method}"
    )

    peak_to_cluster, n_halo = dpc(rows=final_r,
                                  cols=final_c,
                                  sims=similarities,
                                  n=n_domains,
                                  min_sim=min_fam_sim,
                                  backend="scipy",
                                  filter_transparency=halo_stringency,
                                  halo_method=halo_method,
                                  min_dens=min_fam_dens,
                                  point_specific=point_specific)
    return peak_to_cluster, n_halo


# =============================================================================
# Stage 6: Evaluation & GFF3 Export
# =============================================================================

def evaluate_clusters(peak_to_cluster: Dict[int, np.ndarray],
                      domain_registry: List[Tuple[int, int, int, str]]) \
        -> Generator[Tuple[str, List[Tuple[int, int, int, str]]], None, None]:
    """
    Evaluate DPC clusters, group member domains by query sequence, and merge
    overlapping same-cluster domains on the same query sequence.

    :param peak_to_cluster: Dictionary mapping peak ID to member domain index array.
    :param domain_registry: Global list of domain instances: (query_id, start, end, strand).
    :return: Generator yielding (cluster_id_str, list_of_final_domains).
    """
    n_clusters = len(peak_to_cluster)
    with tqdm(total=n_clusters, desc="Evaluating clusters") as pbar:
        for peak_id, member_ids in peak_to_cluster.items():
            cluster_domains_per_seq: Dict[int, List[Tuple[int, int, int, str]]] = defaultdict(list)
            for d_id in member_ids:
                d = domain_registry[int(d_id)]
                cluster_domains_per_seq[d[0]].append(d)

            component_domains: List[Tuple[int, int, int, str]] = []
            for seq_id, domains in cluster_domains_per_seq.items():
                if len(domains) == 1:
                    component_domains.append(domains[0])
                    continue

                sorted_domains = sorted(domains, key=lambda d: d[1])
                ## mem_opt ## Standardize interval spans array to np.int32
                spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int32)
                merged_spans = merge_overlapping_spans(spans_arr)

                for m_start, m_end in merged_spans:
                    overlapping_strands = [
                        d[3] for d in sorted_domains
                        if max(d[1], m_start) < min(d[2], m_end)
                    ]
                    if not overlapping_strands:
                        consensus_strand = "+"
                    else:
                        neg = overlapping_strands.count("-")
                        pos = overlapping_strands.count("+")
                        consensus_strand = "-" if neg > pos else "+"
                    component_domains.append((seq_id, int(m_start), int(m_end), consensus_strand))

            pbar.update(1)
            yield str(peak_id), component_domains


def find_domains(fasta_path: Path,
                 attributes: Dict[str, Any],
                 output_gff: Path,
                 tmp_dir: Optional[Path] = None,
                 result_dir: Optional[Path] = None,
                 cpu: int = default_threads) -> Path:
    """
    Main orchestration function for domain instance extraction and family clustering.
    Operates on a pre-computed alignment TSV file specified in attributes['tsv'].

    :param fasta_path: Path to input FASTA file containing query sequences.
    :param attributes: Dictionary with clustering parameters and pre-computed TSV path.
    :param output_gff: Output GFF3 file path.
    :param tmp_dir: Optional temporary directory path.
    :param result_dir: Optional result directory path.
    :param cpu: Number of parallel CPU worker threads.
    :return: Path to the generated GFF3 file.
    """
    fasta_path = Path(fasta_path)
    output_gff = Path(output_gff)
    output_gff.parent.mkdir(parents=True, exist_ok=True)

    tsv_path_val = attributes.get("tsv")
    if not tsv_path_val:
        raise ValueError("🛑 'tsv' path must be specified in attributes. Aligners must be run externally.")

    tsv_file = Path(tsv_path_val)
    if not tsv_file.exists():
        raise FileNotFoundError(f"🛑 Pre-computed TSV file not found: {tsv_file}")

    aligner_name = str(attributes.get("aligner", "mmseqs2")).lower()
    min_dom_sim = float(attributes.get("min_dom_sim", 0.35))
    dom_halo = float(attributes.get("dom_halo", 0.5))
    min_dom_dens = float(attributes.get("min_dom_dens", 1.0))
    min_fam_sim = float(attributes.get("min_fam_sim", 0.25))
    fam_halo = float(attributes.get("fam_halo", 0.5))
    min_fam_dens = float(attributes.get("min_fam_dens", 1.0))
    halo_method = attributes.get("halo_method", "similarity_weighted")
    if halo_method is not None and str(halo_method).lower() == "none":
        halo_method = None
    reassign_targets = bool(attributes.get("reassign_targets", True))
    min_containment = float(attributes.get("min_containment", 0.6))

    sequence_registry = populate_registry_from_fasta(fasta_path)

    # 1. Obtain generator of alignment query batches from pre-computed TSV
    batch_gen = batch_parse_mmseqs_tsv(tsv_file=tsv_file,
                                       n=1000,
                                       registry=sequence_registry)

    point_specific_val = bool(attributes.get("point_specific", False))

    # 2. Stage 2 & 3: Parallel query processing and domain identification
    logger.info("Executing Stage 2: Finding domain instances across queries")
    query_kwargs = {
        "min_dom_sim": min_dom_sim,
        "min_containment": min_containment,
        "reassign_targets": reassign_targets,
        "halo_stringency": dom_halo,
        "min_dom_dens": min_dom_dens,
        "halo_method": halo_method,
        "point_specific": point_specific_val,
    }

    target_dtype = np.dtype([
        ('t_seq', np.int32),
        ('start', np.int32),
        ('end', np.int32),
        ('dom_id', np.int32)
    ])

    global_domains: List[Tuple[int, int, int, str]] = []
    global_offset = 0

    ## mem_opt ## Empirical line sampling to accurately estimate initial capacity and avoid reallocation
    sample_lines_len = 0
    sample_lines_count = 0
    with tsv_file.open("rb") as fh:
        for _ in range(10_000):
            line = fh.readline()
            if not line:
                break
            sample_lines_len += len(line)
            sample_lines_count += 1
    mean_line_len = (sample_lines_len / sample_lines_count) if sample_lines_count > 0 else 75.0
    est_capacity = max(100_000, int(tsv_file.stat().st_size / max(1.0, mean_line_len)))
    target_buffer = np.empty(est_capacity, dtype=target_dtype)
    write_offset = 0

    ## mem_opt ## Persistent WorkerPool streaming queries with bounded active queue to prevent repeated process spawning
    def _batch_tuple_generator() -> Generator[Tuple[List[Tuple[int, np.ndarray, np.ndarray]]], None, None]:
        for b in batch_gen:
            yield (b,)

    with WorkerPool(n_jobs=cpu,
                    start_method=_safe_ctx_name,
                    shared_objects=query_kwargs) as pool:
        with tqdm(desc="Finding domain instances", unit="batch") as pbar:
            for batch_res in pool.imap(_query_worker_dispatch,
                                       _batch_tuple_generator(),
                                       worker_lifespan=200,
                                       chunk_size=1,
                                       max_tasks_active=max(1, cpu * 2)):
                for local_domains, all_targets in batch_res:
                    n_doms = len(local_domains)
                    if n_doms == 0:
                        continue
                    global_domains.extend(local_domains)
                    n_hits = all_targets.shape[0] if all_targets.size > 0 else 0
                    if n_hits > 0:
                        if write_offset + n_hits > target_buffer.shape[0]:
                            ## mem_opt ## Safe 1.25x buffer expansion copying only active prefix without cyclic data duplication
                            new_size = int((write_offset + n_hits) * 1.25) + 100_000
                            new_buffer = np.empty(new_size, dtype=target_dtype)
                            new_buffer[:write_offset] = target_buffer[:write_offset]
                            target_buffer = new_buffer

                        dest = target_buffer[write_offset:write_offset + n_hits]
                        dest['t_seq'] = all_targets[:, 0]
                        dest['start'] = all_targets[:, 1]
                        dest['end'] = all_targets[:, 2]
                        dest['dom_id'] = all_targets[:, 3] + global_offset
                        write_offset += n_hits
                    global_offset += n_doms
                pbar.update(1)

    n_total_domains = len(global_domains)

    if write_offset > 0:
        all_targets_arr = target_buffer[:write_offset]
        del target_buffer
        all_targets_arr.sort(order=['t_seq', 'start'])
        ## mem_opt ## Index target sequence groups directly via boundary offsets instead of creating intermediate Python list slices with np.split
        seq_breaks = np.flatnonzero(np.diff(all_targets_arr['t_seq'])) + 1
        seq_starts = np.concatenate(([0], seq_breaks))
        seq_ends = np.concatenate((seq_breaks, [all_targets_arr.shape[0]]))
        n_target_seqs = len(seq_starts)
    else:
        del target_buffer
        seq_starts = np.empty(0, dtype=np.intp)
        seq_ends = np.empty(0, dtype=np.intp)
        n_target_seqs = 0

    logger.info(f"Found {n_total_domains} domain instances across {n_target_seqs} target sequences")

    # 3. Stage 4: Parallel target sequence overlap calculation
    ## mem_opt ## Adaptive workstation batch sizing ensuring uniform multi-core utilization without thread starvation
    if n_target_seqs <= cpu * 4:
        target_batch_size = max(1, int(np.ceil(n_target_seqs / max(1, cpu * 2))))
    else:
        target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))

    def target_batch_generator() -> Generator[Tuple[int, List[Tuple[int, np.ndarray, np.ndarray]]], None, None]:
        batch_pkg: List[Tuple[int, np.ndarray, np.ndarray]] = []
        chunk_idx = 0
        for st, en in zip(seq_starts, seq_ends):
            grp = all_targets_arr[st:en]
            t_seq = int(grp['t_seq'][0])
            spans = np.column_stack((grp['start'], grp['end']))
            dom_ptrs = grp['dom_id']
            batch_pkg.append((t_seq, spans, dom_ptrs))
            if len(batch_pkg) >= target_batch_size:
                yield (chunk_idx, batch_pkg)
                chunk_idx += 1
                batch_pkg = []
        if batch_pkg:
            yield (chunk_idx, batch_pkg)

    # Establish dedicated shard directory under tmp_dir or system temp
    if tmp_dir is not None:
        stage4_tmp_dir = Path(tmp_dir) / "stage4_shards"
        stage4_tmp_dir.mkdir(parents=True, exist_ok=True)
        shard_cleanup_needed = False
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="domain_finder_stage4_")
        stage4_tmp_dir = Path(temp_dir_obj.name)
        shard_cleanup_needed = True

    try:
        logger.info("Executing Stage 4: Calculating target overlaps across sequences")
        meta_results: List[Tuple[int, int]] = []
        global_dom_lengths = np.zeros(n_total_domains, dtype=np.int64)
        total_edges = 0

        def _target_task_feeder() -> Generator[Tuple[Tuple[int, List[Tuple[int, np.ndarray, np.ndarray]]]], None, None]:
            for pkg in target_batch_generator():
                yield (pkg,)

        with WorkerPool(n_jobs=cpu,
                        start_method=_safe_ctx_name,
                        shared_objects=str(stage4_tmp_dir)) as pool:
            with tqdm(desc="Calculating target overlaps", unit="batch") as pbar:
                for chunk_id, n_edges, dom_lens_dict in pool.imap(_target_sharded_worker_dispatch,
                                                                  _target_task_feeder(),
                                                                  worker_lifespan=200,
                                                                  chunk_size=1,
                                                                  max_tasks_active=max(1, cpu * 2)):
                    meta_results.append((chunk_id, n_edges))
                    total_edges += n_edges
                    for d_id, d_len in dom_lens_dict.items():
                        if d_id < n_total_domains:
                            global_dom_lengths[d_id] += d_len
                    pbar.update(1)

        ## mem_opt ## Pre-allocate destination coordinate arrays once and memory-map copy shards
        meta_results.sort(key=lambda x: x[0])
        all_rows = np.empty(total_edges, dtype=np.int32)
        all_cols = np.empty(total_edges, dtype=np.int32)
        all_inters = np.empty(total_edges, dtype=np.int32)

        write_pos = 0
        for cid, n in meta_results:
            if n > 0:
                for dest, tag in ((all_rows, "r"), (all_cols, "c"), (all_inters, "i")):
                    shard_file = stage4_tmp_dir / f"{cid}_{tag}.npy"
                    dest[write_pos:write_pos + n] = np.load(shard_file, mmap_mode="r")
                    shard_file.unlink(missing_ok=True)
                write_pos += n
    finally:
        if shard_cleanup_needed:
            temp_dir_obj.cleanup()
        elif stage4_tmp_dir.exists():
            for f in stage4_tmp_dir.glob("*.npy"):
                f.unlink(missing_ok=True)
            try:
                stage4_tmp_dir.rmdir()
            except OSError:
                pass

    # 4. Stage 5: Family finding with DPC
    logger.info("Executing Stage 5: Finding domain families")
    peak_to_cluster, n_halo = family_finder(rows=all_rows,
                                            cols=all_cols,
                                            intersections=all_inters,
                                            domain_lengths=global_dom_lengths,
                                            n_domains=n_total_domains,
                                            min_fam_sim=min_fam_sim,
                                            min_fam_dens=min_fam_dens,
                                            halo_stringency=fam_halo,
                                            halo_method=halo_method,
                                            point_specific=point_specific_val)

    # 5. Stage 6: Evaluate clusters and export to GFF3
    logger.info(f"Writing predicted domain families to {output_gff}")
    with output_gff.open("w", encoding="utf-8") as gff_writer:
        gff_writer.write("##gff-version 3\n")
        for fam_id, fam_domains in evaluate_clusters(peak_to_cluster=peak_to_cluster,
                                                     domain_registry=global_domains):
            for seq_int, start, end, strand in fam_domains:
                seq_name = sequence_registry.get_str(seq_int)
                gff_start = start + 1
                gff_end = end
                gff_line = f"{seq_name}\t{aligner_name}\tDomain\t{gff_start}\t{gff_end}\t.\t{strand}\t.\tID={fam_id}\n"
                gff_writer.write(gff_line)

    logger.info(f"Successfully finished domain finding. Output written to {output_gff}")
    return output_gff
# fmt: on
