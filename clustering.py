#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Peak Clustering (DPC) methods
"""
from typing import Dict, List, Literal, Tuple, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from aligners import Span


def compare_sorted_spans(spans: Union[List[Span], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract positional intersections from local pairwise alignments on a single sequence.
    Expects input to be start-position sorted. Maintains O(N) range-bound early exit.

    :param spans: START-SORTED list of Spans or 2D NumPy array shape (N, 2) [[start, end], ...]
    :return: row_indices, col_indices, intersection_lengths
    """
    if isinstance(spans, np.ndarray):
        n = spans.shape[0]
        if n < 2:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        starts = spans[:, 0]
        ends = spans[:, 1]
    else:
        n = len(spans)
        if n < 2:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        starts = [s.start for s in spans]
        ends = [s.end for s in spans]

    if not isinstance(starts, np.ndarray):
        starts = np.array(starts, dtype=np.int64)
        ends = np.array(ends, dtype=np.int64)

    lows, highs, inters = [], [], []

    # Vectorized searchsorted binary range slice for O(N log N + K) complexity
    for i in range(n):
        end_i = ends[i]
        r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)
        if r > i + 1:
            st_js = starts[i + 1:r]
            en_js = ends[i + 1:r]
            overlap = np.minimum(end_i, en_js) - st_js
            valid = overlap > 0
            if np.any(valid):
                js = np.arange(i + 1, r, dtype=np.int32)[valid]
                lows.append(np.full(js.size, i, dtype=np.int32))
                highs.append(js)
                inters.append(overlap[valid].astype(np.int32))

    if not lows:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return np.concatenate(lows), np.concatenate(highs), np.concatenate(inters)


def jaccard_index(intersections: np.ndarray,
                  row_lengths: np.ndarray,
                  column_lengths: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard similarity from union and intersection arrays.
    Returns only sparse non-zero values aligned with original intersection coordinates
    :param intersections: sparse matrix of intersection lengths
    :param row_lengths: array of lengths for the first set of elements (to calculate union)
    :param column_lengths: array of lengths for the second set of elements (to calculate union)
    :return: sparse matrix of Jaccard similarities
    """
    unions = row_lengths + column_lengths - intersections.data
    ji = intersections.data / unions
    # check for indices outside the valid range  >1 and <0, which can occur due to numerical issues or incorrect input data
    if np.any((ji < 0) | (ji > 1)):
        raise ValueError(f'Jaccard similarity values out of bounds\n'
                         f'values:{ji}\n')
    return ji


def max_containment(intersections: np.ndarray,
                    row_lengths: np.ndarray,
                    column_lengths: np.ndarray):
    """
    Compute maximal containment from intersection array and min(A, B)
    Returns only sparse non-zero values aligned with original intersection coordinates
    :param intersections: sparse matrix of intersection lengths
    :param row_lengths: array of lengths for the first set of elements (to calculate union)
    :param column_lengths: array of lengths for the second set of elements (to calculate union)
    :return: sparse matrix of Jaccard similarities
    """
    smaller = np.minimum(row_lengths, column_lengths)
    containment = intersections.data / smaller
    # check for indices outside the valid range  >1 and <0, which can occur due to numerical issues or incorrect input data
    if np.any((containment < 0) | (containment > 1)):
        raise ValueError(f'Jaccard similarity values out of bounds\n'
                         f'values:{containment}\n')
    return containment


"""
Density Peak Clustering (DPC) modified for bounded direct-similarity metrics s(i, j) in [0, 1] range.
Basically a linear kernel for density estimation that works directly on fractional similarity matrices
(Jaccard similarity graphs etc.) where the distance can be defined as d(i, j) = 1 - s(i, j). Hence:

1. Local Density (rho):
   rho_i = sum_j s(i, j)
   This sum includes self-similarity s(i, i) == 1.0 and is weighted if weights are provided.
   Self-similarity is explicitly set to 1.0 during both sparse and dense matrix reconstruction.

2. Similarity to Nearest Higher-Density Neighbour (reverse-delta):
   In similarity metric space: s_max_higher(i) = max_j s(i, j) for all j such that rho_j > rho_i
   If a point has no higher-density neighbour, s_max_higher(i) = 0.0 by definition.

3. Decision Graph & Peak Selection Score (gamma):
   gamma_i = rho_i * (1 - s_max_higher(i))
   Cluster centers (peaks) are local density maxima isolated from other high-density points,
   characterised by high rho and low s_max_higher.

4. Cascading Assignment (Label Propagation):
   Non-peak points inherit the cluster label of their most similar higher-density neighbour.

5. Halo / Noise Filtering:
   Identifies low-density border points between clusters.
   Supports traditional DPC halo filtering and similarity-weighted halo filtering based on linear similarity scoring
   between a point and its foreign-cluster neighbours, normalised by total similarity to all neighbours.
"""

Matrix = Union[np.ndarray, csr_matrix]

HALO_METHODS = ('traditional', 'similarity_weighted', 'knn', 'ndcg')

# Default number of nearest neighbours used in neighbour-based halo criteria.
DEFAULT_HALO_K = 20

# Chunk size ceiling (number of elements) for row-block memory optimization in dense operations.
_DENSE_CHUNK_CELLS = 1_000_000

# Element ceiling for sparse CSR row gather blocks to prevent memory overhead during indexing.
_SPARSE_GATHER_CELLS = 250_000

# Cache for NDCG discount vectors and cumulative IDCG tables indexed by halo_k.
_NDCG_CACHE = {}


def _row_chunks(n: int) -> int:
    """
    Calculate the number of rows per chunk to fit dense matrix operations within memory limits.

    :param n: Total number of points in the dataset.
    :return: Number of matrix rows to process in each chunk.
    """
    return max(1, min(n, _DENSE_CHUNK_CELLS // max(n, 1)))


# --------------------------------------------------------------------------------------
# Similarity graph construction
# --------------------------------------------------------------------------------------

def build_similarity_matrix(rows: np.ndarray,
                            cols: np.ndarray,
                            similarities: np.ndarray,
                            n: int,
                            backend: str) -> Matrix:
    """
    Construct a full symmetric similarity matrix with unit self-similarity along the diagonal.

    Duplicate coordinate entries are merged by keeping the maximum similarity value.
    The output is returned either as a SciPy sparse CSR matrix or a NumPy dense array.

    :param rows: 1D array of row indices for non-zero similarities.
    :param cols: 1D array of column indices for non-zero similarities.
    :param similarities: 1D array of pairwise similarity values bounded in [0, 1].
    :param n: Total number of data points.
    :param backend: Matrix format string, either 'scipy' (sparse CSR) or 'numpy' (dense array).
    :return: n x n symmetric similarity matrix with diagonal values s(i, i) = 1.0.
    """
    mask = rows != cols
    r = rows[mask]
    c = cols[mask]
    s = similarities[mask]

    # Fast path for very small dense NumPy matrices: avoid CSR allocation and conversion
    if backend == 'numpy' and n <= 128:
        mat = np.zeros((n, n), dtype=np.float64)
        if r.size > 0:
            np.maximum.at(mat, (r, c), s)
            np.maximum.at(mat, (c, r), s)
        np.fill_diagonal(mat, 1.0)
        return mat

    if r.size > 0:
        r_sym = np.concatenate((r, c))
        c_sym = np.concatenate((c, r))
        s_sym = np.concatenate((s, s))

        if n <= 2**32:
            key = (r_sym.astype(np.int64) << 32) | c_sym.astype(np.int64)
            order = np.argsort(key)
            key_s = key[order]
            s_s = s_sym[order]

            starts = np.flatnonzero(np.concatenate(([True], key_s[1:] != key_s[:-1])))
            max_s = np.maximum.reduceat(s_s, starts)
            uniq_key = key_s[starts]

            u_r = (uniq_key >> 32).astype(np.int64)
            u_c = (uniq_key & 0xFFFFFFFF).astype(np.int64)
        else:
            struct_arr = np.empty(r_sym.size, dtype=[('r', np.int64), ('c', np.int64)])
            struct_arr['r'] = r_sym
            struct_arr['c'] = c_sym
            order = np.argsort(struct_arr, order=('r', 'c'))
            r_s, c_s, s_s = r_sym[order], c_sym[order], s_sym[order]
            mask_diff = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
            starts = np.flatnonzero(np.concatenate(([True], mask_diff)))
            max_s = np.maximum.reduceat(s_s, starts)
            u_r, u_c = r_s[starts], c_s[starts]

        diag = np.arange(n, dtype=np.int64)
        all_r = np.concatenate((u_r, diag))
        all_c = np.concatenate((u_c, diag))
        all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
    else:
        all_r = np.arange(n, dtype=np.int64)
        all_c = all_r
        all_s = np.ones(n, dtype=np.float64)

    mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))
    mat.sort_indices()

    if backend == 'numpy':
        return mat.toarray()
    return mat


# --------------------------------------------------------------------------------------
# Density and nearest higher-density neighbour
# --------------------------------------------------------------------------------------

def compute_density_and_higher_neighbours(
    rows: np.ndarray,
    cols: np.ndarray,
    similarities: np.ndarray,
    n: int,
    weights: Optional[np.ndarray],
    backend: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Matrix]:
    """
    Calculate local density (rho) and identify the nearest higher-density neighbour for each point.

    Density is computed as the sum of similarities to all neighbours:
        rho_i = Σ_j (w_j * s(i, j))
    For each point i, the algorithm finds neighbour j having strictly higher density (rho_j > rho_i)
    that maximizes similarity s(i, j).

    :param rows: 1D array of row indices.
    :param cols: 1D array of column indices.
    :param similarities: 1D array of similarity values in [0, 1].
    :param n: Total number of points.
    :param weights: Optional 1D array of sample weights for density calculation.
    :param backend: Computation backend ('scipy' for sparse, 'numpy' for dense).
    :return: Tuple containing:
             - 1D array of local densities (rho).
             - 1D array of maximum similarities to higher-density neighbours (s_max_higher).
             - 1D array of indices for the nearest higher-density neighbours (-1 if none exists).
             - n x n similarity matrix.
    """
    mat = build_similarity_matrix(rows, cols, similarities, n, backend)

    if weights is None:
        densities = np.asarray(mat.sum(axis=1)).reshape(-1)
    else:
        densities = np.asarray(mat @ weights).reshape(-1)

    # Global ranking ordered by descending density with ascending index as tie-breaker.
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), -densities))] = np.arange(n, dtype=np.int64)

    if backend == 'scipy':
        nearest_higher, s_max_higher = _nearest_higher_sparse(mat, rank, n)
    else:
        nearest_higher, s_max_higher = _nearest_higher_dense(mat, rank, n)

    return densities, s_max_higher, nearest_higher, mat


def _nearest_higher_sparse(mat: csr_matrix, rank: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the most similar higher-density neighbour for each row in a sparse CSR matrix.

    Scans non-zero matrix entries to locate the candidate neighbour with higher density rank
    that maximizes the pairwise similarity value.

    :param mat: n x n sparse similarity matrix in CSR format.
    :param rank: 1D array of density ranks (0 = highest density).
    :param n: Total number of points.
    :return: Tuple of (nearest higher-density indices, maximum higher similarity values).
    """
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    lengths = indptr[1:] - indptr[:-1]
    row_of = np.repeat(np.arange(n, dtype=indices.dtype), lengths)

    # Identify neighbours that have strictly higher density rank and non-zero similarity.
    candidate = np.where((rank[indices] < rank[row_of]) & (data > 0.0), data, -1.0)

    nonempty = lengths > 0
    best_val = np.full(n, -1.0, dtype=np.float64)
    if np.any(nonempty):
        best_val[nonempty] = np.maximum.reduceat(candidate, indptr[:-1][nonempty])

    nearest_higher = np.full(n, -1, dtype=np.int64)
    s_max_higher = np.zeros(n, dtype=np.float64)

    hit = (candidate > 0.0) & (candidate == best_val[row_of])
    if np.any(hit):
        hit_pos = np.flatnonzero(hit)
        hit_row = row_of[hit_pos]
        first = np.empty(hit_row.size, dtype=bool)
        first[0] = True
        np.not_equal(hit_row[1:], hit_row[:-1], out=first[1:])
        chosen = hit_pos[first]
        target = hit_row[first]
        nearest_higher[target] = indices[chosen]
        s_max_higher[target] = data[chosen]

    return nearest_higher, s_max_higher


def _nearest_higher_dense(mat: np.ndarray, rank: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the most similar higher-density neighbour for each row in a dense matrix.

    Processes the matrix in row chunks to optimize memory usage and cache locality.

    :param mat: n x n dense similarity matrix.
    :param rank: 1D array of density ranks (0 = highest density).
    :param n: Total number of points.
    :return: Tuple of (nearest higher-density indices, maximum higher similarity values).
    """
    nearest_higher = np.full(n, -1, dtype=np.int64)
    s_max_higher = np.zeros(n, dtype=np.float64)

    if n <= 2000:
        mask = rank[np.newaxis, :] < rank[:, np.newaxis]
        block = mat * mask
        best = block.argmax(axis=1)
        best_val = block[np.arange(n), best]
        found = np.flatnonzero(best_val > 0.0)
        nearest_higher[found] = best[found]
        s_max_higher[found] = best_val[found]
        return nearest_higher, s_max_higher

    step = _row_chunks(n)
    scratch = np.empty((step, n), dtype=np.float64)
    for start in range(0, n, step):
        stop = min(n, start + step)
        height = stop - start
        block = scratch[:height]
        np.multiply(mat[start:stop], rank[np.newaxis, :] < rank[start:stop, np.newaxis], out=block)
        best = block.argmax(axis=1)
        best_val = block[np.arange(height), best]
        found = np.flatnonzero(best_val > 0.0)
        nearest_higher[found + start] = best[found]
        s_max_higher[found + start] = best_val[found]

    return nearest_higher, s_max_higher


# --------------------------------------------------------------------------------------
# Top-K neighbourhood extraction
# --------------------------------------------------------------------------------------

def _rank_top_k(values: np.ndarray, k: int) -> np.ndarray:
    """
    Extract column indices of the K highest values in each row of a 2D matrix.

    Uses partial sorting (argpartition) for linear-time performance per row.

    :param values: 2D array of numerical scores.
    :param k: Number of top elements to select.
    :return: 2D array of shape (rows, k) containing column indices ordered by descending score.
    """
    width = values.shape[1]
    if width <= k:
        return np.argsort(-values, axis=1, kind='stable')
    part = np.argpartition(-values, k, axis=1)[:, :k]
    part_values = np.take_along_axis(values, part, axis=1)
    return np.take_along_axis(part, np.argsort(-part_values, axis=1, kind='stable'), axis=1)


def top_k_neighbours(
    mat: Matrix, n: int, k: int, backend: str, rows: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the K most similar neighbours for specified or all points in the dataset.

    Self-similarity edges and zero-similarity pairs are excluded. If a point has fewer than
    K valid neighbours, remaining entries are padded with index -1 and similarity 0.0.

    :param mat: n x n similarity matrix (sparse CSR or dense array).
    :param n: Total number of data points.
    :param k: Number of nearest neighbours to retrieve.
    :param backend: Computation backend ('scipy' or 'numpy').
    :param rows: Optional 1D array of target row indices for which to compute top-K.
    :return: Tuple of two 2D arrays of shape (m, k) where m = len(rows) if rows is provided, else n:
             - Neighbour indices.
             - Neighbour similarity values.
    """
    target_rows = np.arange(n, dtype=np.int64) if rows is None else rows
    m = target_rows.size
    nbr_idx = np.full((m, k), -1, dtype=np.int32)
    nbr_sim = np.zeros((m, k), dtype=np.float64)
    if m == 0 or n == 0:
        return nbr_idx, nbr_sim

    if backend == 'numpy':
        if rows is None:
            # Whole-matrix fast path for writable arrays
            if mat.flags.writeable:
                diag_idx = np.arange(n)
                diag_orig = mat[diag_idx, diag_idx].copy()
                mat[diag_idx, diag_idx] = -1.0
                try:
                    if n <= 2000:
                        idx = _rank_top_k(mat, k)[:, :k]
                        vals = np.take_along_axis(mat, idx, axis=1)
                        keep = vals > 0.0
                        nbr_idx[:, :idx.shape[1]] = np.where(keep, idx, -1)
                        nbr_sim[:, :idx.shape[1]] = np.where(keep, vals, 0.0)
                    else:
                        step = _row_chunks(n)
                        for start in range(0, n, step):
                            stop = min(n, start + step)
                            idx = _rank_top_k(mat[start:stop], k)[:, :k]
                            vals = np.take_along_axis(mat[start:stop], idx, axis=1)
                            keep = vals > 0.0
                            nbr_idx[start:stop, :idx.shape[1]] = np.where(keep, idx, -1)
                            nbr_sim[start:stop, :idx.shape[1]] = np.where(keep, vals, 0.0)
                finally:
                    mat[diag_idx, diag_idx] = diag_orig
                return nbr_idx, nbr_sim

            # Read-only dense array handling in row blocks to avoid memory duplication
            step = _row_chunks(n) if n > 2000 else n
            for start in range(0, n, step):
                stop = min(n, start + step)
                height = stop - start
                block = np.array(mat[start:stop], copy=True)
                local = np.arange(height)
                block[local, start + local] = -1.0
                idx = _rank_top_k(block, k)[:, :k]
                vals = np.take_along_axis(block, idx, axis=1)
                keep = vals > 0.0
                nbr_idx[start:stop, :idx.shape[1]] = np.where(keep, idx, -1)
                nbr_sim[start:stop, :idx.shape[1]] = np.where(keep, vals, 0.0)
            return nbr_idx, nbr_sim

        # Dense subset path: process target rows in chunks
        step = _row_chunks(n) if n > 2000 else m
        for start in range(0, m, step):
            pos = np.arange(start, min(m, start + step))
            target = target_rows[pos]
            block = np.array(mat[target], copy=True)
            block[np.arange(pos.size), target] = -1.0
            idx = _rank_top_k(block, k)[:, :k]
            vals = np.take_along_axis(block, idx, axis=1)
            keep = vals > 0.0
            nbr_idx[pos, :idx.shape[1]] = np.where(keep, idx, -1)
            nbr_sim[pos, :idx.shape[1]] = np.where(keep, vals, 0.0)
        return nbr_idx, nbr_sim

    # SciPy sparse CSR backend: global vectorized top-K avoids heavy mat[rows] CSR slicing overhead
    all_idx, all_sim = _top_k_scipy_global(mat, n, k)
    if rows is None:
        return all_idx, all_sim
    return all_idx[target_rows], all_sim[target_rows]


def _top_k_scipy_global(mat: csr_matrix, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient global CSR top-K traversal across all N rows.

    :param mat: n x n sparse CSR matrix.
    :param n: Total number of rows.
    :param k: Top-K neighbours to extract.
    :return: Tuple of (n, k) neighbour indices and (n, k) similarity values.
    """
    nbr_idx = np.full((n, k), -1, dtype=np.int32)
    nbr_sim = np.zeros((n, k), dtype=np.float64)

    indptr, indices, data = mat.indptr, mat.indices, mat.data
    row_lengths = indptr[1:] - indptr[:-1]
    row_of = np.repeat(np.arange(n, dtype=indices.dtype), row_lengths)
    valid = (indices != row_of) & (data > 0.0)

    v_data = data[valid]
    v_indices = indices[valid]
    v_row = row_of[valid]

    if v_data.size > 0:
        v_counts = np.bincount(v_row, minlength=n)
        v_indptr = np.empty(n + 1, dtype=np.intp)
        v_indptr[0] = 0
        np.cumsum(v_counts, out=v_indptr[1:])

        active_rows = np.flatnonzero(v_counts > 0)
        for r in active_rows:
            st, sp = v_indptr[r], v_indptr[r + 1]
            c_vals = v_data[st:sp]
            c_cols = v_indices[st:sp]
            m = sp - st
            if m <= k:
                sort_order = np.argsort(-c_vals, kind='stable')
                nbr_idx[r, :m] = c_cols[sort_order]
                nbr_sim[r, :m] = c_vals[sort_order]
            else:
                top_part = np.argpartition(-c_vals, k)[:k]
                top_order = top_part[np.argsort(-c_vals[top_part], kind='stable')]
                nbr_idx[r, :] = c_cols[top_order]
                nbr_sim[r, :] = c_vals[top_order]

    return nbr_idx, nbr_sim


def _foreign_mask(labels: np.ndarray, nbr_idx: np.ndarray, query_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boolean masks identifying valid neighbour slots and foreign cluster members.

    :param labels: 1D array of cluster labels assigned to data points.
    :param nbr_idx: 2D array of shape (m, k) containing neighbour indices.
    :param query_labels: Optional 1D array of labels for query rows (if m != n).
    :return: Tuple of boolean arrays (valid_mask, foreign_mask), both of shape (m, k).
    """
    valid = nbr_idx >= 0
    safe_idx = np.maximum(nbr_idx, 0)
    reference = labels if query_labels is None else query_labels
    foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])
    return valid, foreign


def _ndcg_discounts(k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rank discount factors 1 / log2(rank + 1) and cumulative Ideal DCG lookup table.
    Caches tables by K to avoid repeated transcendental math on repeated DPC evaluations.

    :param k: Maximum neighbourhood rank size.
    :return: Tuple of (1D discount factors array of length k, 1D cumulative IDCG lookup array).
    """
    cached = _NDCG_CACHE.get(k)
    if cached is not None:
        return cached
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    idcg = np.empty(k + 1, dtype=np.float64)
    idcg[0] = 1.0
    idcg[1:] = np.cumsum(discounts)
    cached = (discounts, idcg)
    _NDCG_CACHE[k] = cached
    return cached


# --------------------------------------------------------------------------------------
# Border-evidence scores for halo detection
# --------------------------------------------------------------------------------------

def traditional_scores(labels: np.ndarray, mat: Matrix, backend: str, filter_transparency: float = 0.0,
                       rows: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute border contamination scores using the maximum similarity to foreign cluster points.

    Following Rodriguez & Laio (2014), the border score is defined as:
        score_i = max { s(i, j) : label_j != label_i and s(i, j) >= cutoff_sim }

    :param labels: 1D array of current cluster assignments for all points.
    :param mat: n x n similarity matrix (sparse CSR or dense array).
    :param backend: Backend string ('scipy' or 'numpy').
    :param filter_transparency: Transparency / leniency parameter in [0, 1].
    :param rows: Optional target rows subset to process.
    :return: 1D array of border scores in [0, 1].
    """
    cutoff_sim = max(0.0, min(1.0, float(filter_transparency)))
    target_rows = np.arange(labels.size, dtype=np.int64) if rows is None else rows
    labels_for_rows = labels[target_rows]
    n = labels.size

    if rows is not None:
        if target_rows.size == 0:
            return np.zeros(0, dtype=np.float64)

        if backend == 'scipy':
            sub = mat[target_rows]
            indptr, indices, data = sub.indptr, sub.indices, sub.data
            lengths = indptr[1:] - indptr[:-1]
            row_labels = np.repeat(labels_for_rows, lengths)
            valid = (labels[indices] != row_labels) & (data >= cutoff_sim) & (data > 0.0)
            vals = np.where(valid, data, 0.0)
            scores = np.zeros(target_rows.size, dtype=np.float64)
            nonempty = lengths > 0
            if np.any(nonempty):
                scores[nonempty] = np.maximum.reduceat(vals, indptr[:-1][nonempty])
            return scores

        scores = np.zeros(target_rows.size, dtype=np.float64)
        step = max(1, _row_chunks(n) if n > 2000 else target_rows.size)
        for start in range(0, target_rows.size, step):
            stop = min(target_rows.size, start + step)
            target = target_rows[start:stop]
            is_foreign = labels[np.newaxis, :] != labels[target, np.newaxis]
            is_above_cutoff = mat[target] >= cutoff_sim
            scores[start:stop] = (mat[target] * (is_foreign & is_above_cutoff) * (mat[target] > 0.0)).max(axis=1)
        return scores

    if backend == 'scipy':
        indptr, indices, data = mat.indptr, mat.indices, mat.data
        scores = np.zeros(n, dtype=np.float64)
        if indices.size == 0:
            return scores
        lengths = indptr[1:] - indptr[:-1]
        row_labels = np.repeat(labels, lengths)
        is_valid_foreign = (labels[indices] != row_labels) & (data >= cutoff_sim) & (data > 0.0)
        foreign_vals = np.where(is_valid_foreign, data, 0.0)
        nonempty = lengths > 0
        if np.any(nonempty):
            scores[nonempty] = np.maximum.reduceat(foreign_vals, indptr[:-1][nonempty])
        return np.maximum(scores, 0.0)

    scores = np.empty(n, dtype=np.float64)
    if n <= 2000:
        is_foreign = labels[np.newaxis, :] != labels[:, np.newaxis]
        is_above_cutoff = mat >= cutoff_sim
        scores = (mat * (is_foreign & is_above_cutoff) * (mat > 0.0)).max(axis=1)
        return scores

    step = _row_chunks(n)
    scratch = np.empty((step, n), dtype=np.float64)
    for start in range(0, n, step):
        stop = min(n, start + step)
        height = stop - start
        is_foreign = labels[np.newaxis, :] != labels[start:stop, np.newaxis]
        is_above_cutoff = mat[start:stop] >= cutoff_sim
        np.multiply(mat[start:stop], is_foreign & is_above_cutoff & (mat[start:stop] > 0.0), out=scratch[:height])
        scores[start:stop] = scratch[:height].max(axis=1)
    return scores


def similarity_weighted_scores(labels: np.ndarray, nbr_idx: np.ndarray, nbr_sim: np.ndarray,
                               query_labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute similarity-weighted border contamination scores.

    :param labels: 1D array of cluster labels assigned to data points.
    :param nbr_idx: 2D array of shape (m, k) containing neighbour indices.
    :param nbr_sim: 2D array of shape (m, k) containing neighbour similarity values.
    :param query_labels: Optional 1D array of labels for query rows.
    :return: 1D array of border contamination scores in [0, 1].
    """
    _, foreign = _foreign_mask(labels, nbr_idx, query_labels)
    total = nbr_sim.sum(axis=1)
    foreign_mass = np.where(foreign, nbr_sim, 0.0).sum(axis=1)
    return np.divide(foreign_mass, total, out=np.zeros_like(total), where=total > 0.0)


def knn_scores(labels: np.ndarray, nbr_idx: np.ndarray,
              query_labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute border contamination scores based on foreign neighbour counts in top-K.

    :param labels: 1D array of cluster labels assigned to data points.
    :param nbr_idx: 2D array of shape (m, k) containing neighbour indices.
    :param query_labels: Optional 1D array of labels for query rows.
    :return: 1D array of border contamination scores in [0, 1].
    """
    valid, foreign = _foreign_mask(labels, nbr_idx, query_labels)
    counts = valid.sum(axis=1).astype(np.float64)
    return np.divide(foreign.sum(axis=1), counts, out=np.zeros_like(counts), where=counts > 0.0)


def ndcg_scores(labels: np.ndarray, nbr_idx: np.ndarray,
                query_labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute rank-aware border contamination scores using Normalized Discounted Cumulative Gain (NDCG).

    :param labels: 1D array of cluster labels assigned to data points.
    :param nbr_idx: 2D array of shape (m, k) containing neighbour indices.
    :param query_labels: Optional 1D array of labels for query rows.
    :return: 1D array of rank-discounted border scores in [0, 1].
    """
    _, foreign = _foreign_mask(labels, nbr_idx, query_labels)
    k = nbr_idx.shape[1]
    discounts, idcg = _ndcg_discounts(k)
    relevant = foreign.sum(axis=1)
    dcg = foreign.astype(np.float64) @ discounts
    scores = np.divide(dcg, idcg[relevant], out=np.zeros_like(dcg), where=relevant > 0)
    return np.clip(scores, 0.0, 1.0)


def border_evidence(labels: np.ndarray,
                    mat: Matrix,
                    method: str,
                    backend: str,
                    filter_transparency: float,
                    halo_k: int = DEFAULT_HALO_K,
                    rows: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate border contamination evidence scores across points using a chosen halo method.

    :param labels: 1D array of cluster labels assigned to data points.
    :param mat: n x n similarity matrix (sparse CSR or dense array).
    :param method: Halo detection method ('traditional', 'similarity_weighted', 'knn', or 'ndcg').
    :param backend: Computation backend ('scipy' or 'numpy').
    :param filter_transparency: Transparency parameter in [0, 1].
    :param halo_k: Number of nearest neighbours used for neighbour-based criteria.
    :param rows: Optional target row indices.
    :return: 1D array of border evidence scores in [0, 1].
    """
    if method == 'traditional':
        return traditional_scores(labels, mat, backend, filter_transparency=filter_transparency, rows=rows)

    if method not in HALO_METHODS:
        raise ValueError(f"Unknown halo method {method!r} (supported: {', '.join(HALO_METHODS)})")

    n = labels.size
    nbr_idx, nbr_sim = top_k_neighbours(mat, n, max(1, min(halo_k, max(n - 1, 1))), backend, rows=rows)
    query_labels = None if rows is None else labels[rows]
    if method == 'similarity_weighted':
        return similarity_weighted_scores(labels, nbr_idx, nbr_sim, query_labels)
    if method == 'knn':
        return knn_scores(labels, nbr_idx, query_labels)
    return ndcg_scores(labels, nbr_idx, query_labels)


# --------------------------------------------------------------------------------------
# Halo / core partition
# --------------------------------------------------------------------------------------

def halo_filter(labels: np.ndarray,
                similarity_mat: Matrix,
                filter_transparency: float,
                halo_min_cluster_size: int,
                backend: str,
                method: str,
                rho: np.ndarray,
                halo_k: int = DEFAULT_HALO_K) -> np.ndarray:
    """
    Partition clusters into core regions and sparse halo regions, reassigning halo points to -1.

    Points with border scores exceeding filter_transparency are marked as border points.
    For each cluster, the maximum local density among its border points (rho_b) is computed.
    Cluster members with density strictly less than rho_b (rho_i < rho_b) are designated as halo (noise, -1).

    Small clusters (<3) are exempt from halo procedure,
    as they lack sufficient statistical sample size to define a meaningful boundary region density.

    :param labels: 1D array of initial cluster labels (0 to C-1, with -1 for noise).
    :param similarity_mat: n x n similarity matrix (sparse CSR or dense array).
    :param filter_transparency: Sensitivity threshold in [0, 1]. Lower values filter more halo points.
    :param backend: Matrix computation backend ('scipy' or 'numpy').
    :param method: Halo filtering method ('traditional', 'similarity_weighted', 'knn', or 'ndcg').
    :param rho: 1D array of precomputed local density values for all points.
    :param halo_k: Number of nearest neighbours used for neighbour-based methods.
    :return: 1D array of updated cluster labels where halo/noise points are set to -1.
    """
    new_labels = labels.copy()
    if filter_transparency >= 1.0:
        return new_labels

    n_clusters = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    if n_clusters <= 1:
        return new_labels

    sizes = np.bincount(labels[labels >= 0], minlength=n_clusters)
    # Early exit: return immediately if all clusters are below minimum size threshold
    eligible_clusters = sizes >= halo_min_cluster_size
    if not np.any(eligible_clusters):
        return new_labels

    eligible_points = int(sizes[eligible_clusters].sum())

    # Optimize target rows computation: evaluate evidence only for points in non-exempt clusters
    evidence_rows = None
    if eligible_points * 4 < labels.size * 3:
        evidence_rows = np.flatnonzero((labels >= 0) & eligible_clusters[labels.clip(min=0)])

    scores = border_evidence(labels, similarity_mat, method, backend, filter_transparency=filter_transparency,
                             halo_k=halo_k, rows=evidence_rows)

    is_border = (scores >= filter_transparency) & (scores > 0.0)
    rho_b = np.full(n_clusters, -1.0, dtype=np.float64)

    border = np.flatnonzero(is_border) if evidence_rows is None else evidence_rows[is_border]
    if evidence_rows is None:
        border = border[labels[border] >= 0]

    if border.size > 0:
        # In-place maximum reduction O(N_b) without sorting overhead
        np.maximum.at(rho_b, labels[border], rho[border])

    # Exempt small clusters (< halo_min_cluster_size) from halo reassignment
    rho_b[sizes < halo_min_cluster_size] = -1.0

    assigned = labels >= 0
    cut = assigned.copy()
    cut[assigned] = rho[assigned] < rho_b[labels[assigned]]
    new_labels[cut] = -1
    return new_labels


# --------------------------------------------------------------------------------------
# Main Density Peak Clustering entry point
# --------------------------------------------------------------------------------------

def dpc(rows: np.ndarray,
        cols: np.ndarray,
        sims: np.ndarray,
        n: int,
        min_sim: float,
        min_dens: float,
        filter_transparency: float,
        halo_method: Optional[str],
        backend: str,
        weights: Optional[np.ndarray] = None,
        halo_min_cluster_size: int = 3,
        halo_k: int = DEFAULT_HALO_K) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Execute Density Peak Clustering directly on a pairwise similarity graph.

    Processes similarity data to identify cluster centers (peaks), assign points to clusters via
    higher-density parent paths, and filter low-density border points into halo noise (-1).

    :param rows: 1D integer array of row indices for non-zero pairwise similarities.
    :param cols: 1D integer array of column indices for non-zero pairwise similarities.
    :param sims: 1D float array of similarity values in [0, 1].
    :param n: Total number of points in the dataset.
    :param min_sim: Minimum higher-density similarity threshold in (0, 1) for candidate peak selection.
    :param min_dens: Minimum local density threshold for candidate peaks.
    :param filter_transparency: Leniency parameter in [0, 1] for halo filtering. Lower values generate more halo.
    :param halo_method: Halo criterion ('traditional', 'similarity_weighted', 'knn', 'ndcg', or None).
    :param backend: Matrix processing backend ('scipy' for sparse CSR, 'numpy' for dense array).
    :param weights: Optional 1D array of per-point weights for density calculation.
    :param halo_min_cluster_size: Minimum cluster size required to trigger halo filtering.
    :param halo_k: Neighbourhood width K for neighbour-based halo criteria.
    :return: Tuple containing:
             - Dictionary mapping cluster peak indices (int) to 1D arrays of member point indices.
             - Integer count of unassigned or halo/noise points.
    """
    if rows.size == 0:
        return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0

    if not (0.0 < min_sim < 1.0):
        raise ValueError('min_sim must be in (0, 1)')
    if rows.shape != cols.shape or rows.shape != sims.shape:
        raise ValueError('rows, cols and sims must have matching shapes')
    if backend not in ('numpy', 'scipy'):
        raise ValueError("backend must be 'numpy' or 'scipy'")

    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    sims = np.ascontiguousarray(sims, dtype=np.float64)

    if sims.size and (sims.min() < 0.0 or sims.max() > 1.0):
        raise ValueError('sims must lie in [0, 1]')

    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError('weights must have one entry per point')

    densities, s_max_higher, nearest_higher, mat = compute_density_and_higher_neighbours(
        rows, cols, sims, n, weights, backend
    )

    # --- Peak selection on the decision graph ---
    gamma = densities * (1.0 - s_max_higher)
    sorted_by_gamma = np.lexsort((np.arange(n), -gamma))
    is_candidate_peak = (densities > min_dens) & ((s_max_higher < min_sim) | (nearest_higher == -1))
    peak_indices = sorted_by_gamma[is_candidate_peak[sorted_by_gamma]]
    if peak_indices.size == 0:
        peak_indices = sorted_by_gamma[:1]

    # --- Cascading assignment via vectorised pointer doubling ---
    parent = np.where((nearest_higher >= 0) & (s_max_higher > 0.0), nearest_higher, np.arange(n))
    parent[peak_indices] = peak_indices
    while True:
        grandparent = parent[parent]
        if np.array_equal(grandparent, parent):
            break
        parent = grandparent

    labels = np.full(n, -1, dtype=np.int64)
    peak_rank = np.full(n, -1, dtype=np.int64)
    peak_rank[peak_indices] = np.arange(peak_indices.size)
    labels = peak_rank[parent]

    # --- Halo refinement ---
    if halo_method is not None and str(halo_method).lower() != 'none' and filter_transparency < 1.0:
        labels = halo_filter(labels,
                             similarity_mat=mat,
                             filter_transparency=filter_transparency,
                             halo_min_cluster_size=halo_min_cluster_size,
                             backend=backend,
                             method=str(halo_method).lower(),
                             rho=densities,
                             halo_k=halo_k)

    # --- Output formatting ---
    peak_to_cluster: Dict[int, np.ndarray] = {int(p): np.empty(0, dtype=np.int64) for p in peak_indices}
    assigned = np.flatnonzero(labels >= 0)
    if assigned.size:
        order = assigned[np.argsort(labels[assigned], kind='stable')]
        sorted_labels = labels[order]
        starts = np.flatnonzero(np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1])))
        for group, start in zip(np.split(order, starts[1:]), starts):
            peak_to_cluster[int(peak_indices[sorted_labels[start]])] = group

    return peak_to_cluster, int(np.count_nonzero(labels < 0))
