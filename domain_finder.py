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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# standard libraries
import numpy as np
from scipy.sparse import coo_matrix, csgraph
from tqdm import tqdm

# internal repository/package imports
from aligners import (SequenceRegistry, count_fasta,
                      populate_registry_from_fasta,
                      batch_parse_mmseqs_tsv)
from clustering import (compare_sorted_spans,
                        jaccard_index, max_containment,
                        dpc)
from tweaks import Parallel, default_threads, logger


def count_fasta_queries(fasta_path: Path) -> int:
    """
    Count sequence headers (starting with '>') in a FASTA sequence file.

    :param fasta_path: Path to the FASTA sequence file.
    :return: Integer count of query sequence headers.
    """
    return count_fasta(fasta_path=fasta_path)


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
                            reassign_targets: bool) -> np.ndarray:
    """
    Filter out low-scoring overlapping domains.

    :param cluster_map: Cluster map to filter.
    :param suspect_rows: Indices of the suspect overlaps (in the unique_spans array).
    :param suspect_cols: Indices of the suspect overlaps (in the unique_spans array).
    :param suspect_intersections: Intersection lengths of suspect domain instances.
    :param suspect_lengths: Lengths of unique domain instances.
    :param min_containment: Minimum containment threshold for filtering.
    :param reassign_targets: Whether targets should be reassigned to strongest neighbors.
    :return: Refined cluster map with spurious domains removed or reassigned.
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
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)

    removed_nodes = []
    for node in sorted(neighbors, key=lambda n: (node_weights[n], n)):
        live_neighbors = neighbors.pop(node)
        if live_neighbors:
            removed_nodes.append(node)
            if reassign_targets:
                best_neighbor = max(live_neighbors, key=lambda n: node_weights[n])
                cluster_map[cluster_map == node] = best_neighbor
            else:
                cluster_map[cluster_map == node] = -1
            for neighbor in live_neighbors:
                neighbors[neighbor].remove(node)

    return cluster_map


def domain_instances_on_query(query_id: int,
                              sorted_query_spans: np.ndarray,
                              target_spans: np.ndarray,
                              min_dom_sim: float,
                              min_containment: float,
                              reassign_targets: bool,
                              halo_stringency: float,
                              min_dom_dens: float,
                              halo_method: Optional[str]) \
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
        cluster_map[:] = 0
        peak_indices = np.array([0], dtype=np.int32)
    else:
        rows, cols, intersections = compare_sorted_spans(unique_query_spans)
        if rows.size == 0:
            peak_indices = np.arange(n_spans, dtype=np.int32)
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
                                           min_dens=min_dom_dens)

            assigned_peak_map = np.full(n_spans, -1, dtype=np.int32)
            for peak_id, member_ids in peak_to_cluster.items():
                assigned_peak_map[member_ids] = peak_id
            cluster_map = assigned_peak_map[cluster_map]

            if len(peak_to_cluster) > 1:
                unique_domain_indices = np.array(list(peak_to_cluster.keys()), dtype=np.int32)
                peak_mask = np.isin(rows, unique_domain_indices) & np.isin(cols, unique_domain_indices)
                if np.any(peak_mask):
                    cluster_map = filter_spurious_domains(cluster_map=cluster_map,
                                                          suspect_rows=rows[peak_mask],
                                                          suspect_cols=cols[peak_mask],
                                                          suspect_intersections=intersections[peak_mask],
                                                          suspect_lengths=span_lengths,
                                                          min_containment=min_containment,
                                                          reassign_targets=reassign_targets)

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


def batch_query_processor(batch: List[Tuple[int, np.ndarray, np.ndarray]],
                          min_dom_sim: float,
                          min_containment: float,
                          reassign_targets: bool,
                          halo_stringency: float,
                          min_dom_dens: float,
                          halo_method: Optional[str]) \
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
                                        halo_method=halo_method)
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


def target_batch_processor(batch_targets: List[Tuple[int, np.ndarray, np.ndarray]]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Process a batch of target sequences to merge same-domain intervals, compute
    per-domain target lengths, and calculate pairwise target overlaps.

    :param batch_targets: List of (target_id, target_spans, domain_pointers).
    :return: Tuple of (rows, cols, intersections, domain_lengths_dict).
    """
    batch_rows, batch_cols, batch_data = [], [], []
    batch_dom_lengths: Dict[int, int] = defaultdict(int)

    for target_id, target_spans, domain_pointers in batch_targets:
        uniq_domains = np.unique(domain_pointers)
        merged_spans_list = []
        aligned_domains_list = []

        for d_id in uniq_domains:
            dom_mask = domain_pointers == d_id
            d_spans = target_spans[dom_mask]
            m_spans = merge_overlapping_spans(d_spans)
            for sp in m_spans:
                merged_spans_list.append(sp)
                aligned_domains_list.append(d_id)
                batch_dom_lengths[int(d_id)] += int(sp[1] - sp[0])

        if len(merged_spans_list) <= 1:
            continue

        merged_spans = np.array(merged_spans_list, dtype=np.int32)
        aligned_domains = np.array(aligned_domains_list, dtype=np.int32)

        sort_order = np.argsort(merged_spans[:, 0])
        sorted_spans = merged_spans[sort_order]
        aligned_sorted_domains = aligned_domains[sort_order]

        rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
        if rows.size > 0:
            batch_rows.append(rows)
            batch_cols.append(cols)
            batch_data.append(intersections)

    if batch_rows:
        return (np.concatenate(batch_rows),
                np.concatenate(batch_cols),
                np.concatenate(batch_data),
                dict(batch_dom_lengths))

    return (np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            dict(batch_dom_lengths))


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
                  halo_method: Optional[str]) \
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
    if rows.size == 0 or n_domains <= 1:
        return {int(i): np.array([i], dtype=np.int64) for i in range(n_domains)}, 0

    # Ensure canonical undirected edges (rows < cols)
    u_r = np.minimum(rows, cols).astype(np.int64)
    u_c = np.maximum(rows, cols).astype(np.int64)
    diff = u_r != u_c
    u_r, u_c, inters = u_r[diff], u_c[diff], intersections[diff].astype(np.int64)

    if u_r.size == 0:
        return {int(i): np.array([i], dtype=np.int64) for i in range(n_domains)}, 0

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
                                  min_dens=min_fam_dens)
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
                    component_domains.extend(domains)
                    continue

                sorted_domains = sorted(domains, key=lambda d: d[1])
                spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int32)
                rows, cols, intersections = compare_sorted_spans(spans_arr)
                if rows.size == 0:
                    component_domains.extend(sorted_domains)
                else:
                    similarity_matrix = coo_matrix(
                        (intersections, (rows, cols)),
                        shape=(len(sorted_domains), len(sorted_domains))
                    ).tocsr()
                    components, labels = csgraph.connected_components(
                        similarity_matrix, directed=False, return_labels=True
                    )
                    for component_id in range(components):
                        comp = [sorted_domains[i] for i in range(len(sorted_domains))
                                if labels[i] == component_id]
                        if len(comp) == 1:
                            component_domains.append(comp[0])
                        else:
                            merged_start = min(s[1] for s in comp)
                            merged_end = max(s[2] for s in comp)
                            strands = [s[3] for s in comp]
                            merged_strand = "-" if strands.count("-") > strands.count("+") else "+"
                            component_domains.append((seq_id, merged_start, merged_end, merged_strand))

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
    halo_method = attributes.get("halo_method", "knn")
    if halo_method is not None and str(halo_method).lower() == "none":
        halo_method = None
    reassign_targets = bool(attributes.get("reassign_targets", True))
    min_containment = float(attributes.get("min_containment", 0.6))

    sequence_registry = populate_registry_from_fasta(fasta_path)

    # 1. Obtain generator of alignment query batches from pre-computed TSV
    batch_gen = batch_parse_mmseqs_tsv(tsv_file=tsv_file,
                                       n=1000,
                                       registry=sequence_registry)

    # 2. Stage 2 & 3: Parallel query processing and domain identification
    logger.info("Executing Stage 2: Finding domain instances across queries")
    query_kwargs = {
        "min_dom_sim": min_dom_sim,
        "min_containment": min_containment,
        "reassign_targets": reassign_targets,
        "halo_stringency": dom_halo,
        "min_dom_dens": min_dom_dens,
        "halo_method": halo_method,
    }

    query_workers = Parallel(function=batch_query_processor,
                             input_collection=batch_gen,
                             kwargs=query_kwargs,
                             n_jobs=cpu,
                             bar=True,
                             description="Finding domain instances",
                             chunk_size=1)

    global_domains: List[Tuple[int, int, int, str]] = []
    targets_by_seq: Dict[int, List[np.ndarray]] = defaultdict(list)
    global_offset = 0

    for batch_res in query_workers.results:
        for local_domains, all_targets in batch_res:
            n_doms = len(local_domains)
            if n_doms == 0:
                continue
            global_domains.extend(local_domains)
            if all_targets.size > 0:
                offset_targets = all_targets.copy()
                offset_targets[:, 3] += global_offset
                order = np.argsort(offset_targets[:, 0])
                sorted_t = offset_targets[order]
                seq_breaks = np.flatnonzero(np.diff(sorted_t[:, 0])) + 1
                seq_groups = np.split(sorted_t, seq_breaks)
                for grp in seq_groups:
                    t_seq = int(grp[0, 0])
                    targets_by_seq[t_seq].append(grp[:, 1:4])
            global_offset += n_doms

    n_total_domains = len(global_domains)
    logger.info(f"Found {n_total_domains} domain instances across {len(targets_by_seq)} target sequences")

    # 3. Stage 4: Parallel target sequence overlap calculation
    def target_batch_generator() -> Generator[List[Tuple[int, np.ndarray, np.ndarray]], None, None]:
        batch_pkg: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for t_seq, chunks in targets_by_seq.items():
            combined = np.vstack(chunks)
            spans = combined[:, 0:2]
            dom_ptrs = combined[:, 2]
            batch_pkg.append((t_seq, spans, dom_ptrs))
            if len(batch_pkg) >= 1000:
                yield batch_pkg
                batch_pkg = []
        if batch_pkg:
            yield batch_pkg

    logger.info("Executing Stage 4: Calculating target overlaps across sequences")
    target_workers = Parallel(function=target_batch_processor,
                              input_collection=target_batch_generator(),
                              n_jobs=cpu,
                              bar=True,
                              description="Calculating target overlaps",
                              chunk_size=1)

    accumulated_rows: List[np.ndarray] = []
    accumulated_cols: List[np.ndarray] = []
    accumulated_inters: List[np.ndarray] = []
    global_dom_lengths = np.zeros(n_total_domains, dtype=np.int64)

    for rows, cols, inters, dom_lens_dict in target_workers.results:
        if rows.size > 0:
            accumulated_rows.append(rows)
            accumulated_cols.append(cols)
            accumulated_inters.append(inters)
        for d_id, d_len in dom_lens_dict.items():
            if d_id < n_total_domains:
                global_dom_lengths[d_id] += d_len

    all_rows = np.concatenate(accumulated_rows) if accumulated_rows else np.empty(0, dtype=np.int32)
    all_cols = np.concatenate(accumulated_cols) if accumulated_cols else np.empty(0, dtype=np.int32)
    all_inters = np.concatenate(accumulated_inters) if accumulated_inters else np.empty(0, dtype=np.int32)

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
                                            halo_method=halo_method)

    # 5. Stage 6: Evaluate clusters and export to GFF3
    logger.info(f"Writing predicted domain families to {output_gff}")
    with output_gff.open("w", encoding="utf-8") as gff_writer:
        gff_writer.write("##gff-version 3\n")
        for fam_id, fam_domains in evaluate_clusters(peak_to_cluster=peak_to_cluster,
                                                     domain_registry=global_domains):
            for seq_int, start, end, strand in fam_domains:
                seq_name = sequence_registry.get_str(seq_int)
                gff_line = f"{seq_name}\t{aligner_name}\tDomain\t{start}\t{end}\t.\t{strand}\t.\tID={fam_id}\n"
                gff_writer.write(gff_line)

    logger.info(f"Successfully finished domain finding. Output written to {output_gff}")
    return output_gff
# fmt: on
