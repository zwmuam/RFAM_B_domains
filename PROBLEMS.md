# PROBLEMS: Comprehensive Audit of the Domain Finding Codebase and Dependencies

The pipeline extracts conserved structural RNA domain instances from pairwise alignments (MMseqs2 or RiboSeek) and clusters them into global domain families via Density Peak Clustering (DPC). Operating on massive multi-gigabyte or >100GB alignment datasets imposes strict operational requirements:
1. Peak system memory must stay bounded below 200 GB.
2. Parallel worker memory overhead must not exceed 200 MB per CPU core.
3. Multi-core execution must scale efficiently up to 30+ threads without worker starvation or IPC communication bottlenecks.
4. Noise instances ($-1$) must never be assigned to valid clusters or corrupt downstream metrics.

---

## 2. Re-Assessment and Audit of PREVIOUS_UPDATE.md Claims

Two claims from `PREVIOUS_UPDATE.md` were incompletely implemented or introduced subsequent complications:

### Disjoint Graph Density Thresholding in family_finder Was Partially Bypassed
- **Audit Finding:** In `domain_finder.py:family_finder` (lines 430-442), when domain instances share zero target overlaps across other sequences (`u_r.size == 0`), solitary domains are filtered against `min_fam_dens` under the assumption that their local density is $1.0$ (`np.ones(n_domains, dtype=np.float64)`). However, in biological reality, solitary domain instances possess alignment target support proportional to their number of target occurrences $N_{\text{targets}}$. Forcing their local density to $1.0$ discards high-confidence solitary domains if `min_fam_dens` > 1.0, while admitting zero-overlap spurious singletons if `min_fam_dens` <= 1.0.
- **Subsequent Problem:** In `clustering.py:dpc` (lines 871-877), the empty edge handler checks `densities >= min_dens`. However, if `rows.size > 0` and points are disconnected components with zero edge weights, disconnected points bypass candidate peak selection unless their self-density meets `min_dens`. When `family_finder` delegates to `dpc`, it passes `weights=None`, causing `dpc` to compute densities purely from edge sums plus diagonal 1.0, losing track of real alignment support counts.

### Coordinate Typing Harmonization and Docstring Drift
- **Audit Finding:** `PREVIOUS_UPDATE.md` claimed coordinates and sparse edge arrays were strictly standardized to 32-bit integers (`np.int32`). In `domain_finder.py:evaluate_clusters` (line 500), interval spans were previously cast to `np.int64`, and `target_batch_processor` docstrings stated 64-bit coordinate operations were maintained. While line 500 now uses `np.int32`, `clustering.py:build_similarity_matrix` (lines 190-210) still upcasts 32-bit indices to `np.int64` and allocates 64-bit diagonals unconditionally, doubling memory usage during global similarity graph construction.

---

## 3. Critical Algorithmic, Mathematical, and Biological Problems

### Problem A: Halo Filtering Border Contamination Treats Noise Points as Foreign Cluster Neighbors
- **Module / Component:** `clustering.py` (`_foreign_mask`, `halo_filter`)
- **Severity / Category:** CRITICAL | Mathematical & Algorithmic Correctness
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:_foreign_mask` (lines 535-546), the mask defining foreign cluster neighbors is computed as:
  ```python
  valid = nbr_idx >= 0
  safe_idx = np.maximum(nbr_idx, 0)
  reference = labels if query_labels is None else query_labels
  foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])
  ```
  If a neighbor point $j$ in `nbr_idx` was classified as noise ($-1$) during DPC cascading assignment or density thresholding, `labels[safe_idx]` equals $-1$. For an authentic cluster member $i$ with `reference[i] == c` ($c \ge 0$), the condition `c != -1` evaluates to `True`.
  Consequently, **unassigned noise points and low-confidence alignment artifacts are treated as foreign cluster points**. When a legitimate domain point sits adjacent to discarded alignment noise, its foreign contamination score (`foreign_mass / total`) artificially spikes. In `point_specific` or traditional halo filtering, authentic core RNA domain instances are erroneously classified as border contamination and pruned to noise simply because they border discarded noise.
  Biologically, border points must measure contamination between two competing *bona fide* clusters, not proximity to discarded background noise.
- **Evidence in Code:**
  - `clustering.py:544`: `foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])`
- **Recommended Solution:**
  Update `_foreign_mask` so that a neighbor is recognized as foreign if and only if both the reference point and the neighbor point belong to valid, distinct clusters ($\ge 0$):

```python
# FAULTY (clustering.py:544 - Noise (-1) neighbors count as foreign cluster contamination):
foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])

# REVISED (clustering.py - Only genuine, distinct clusters qualify as foreign neighbors):
nbr_labels = labels[safe_idx]
foreign = valid & (reference[:, np.newaxis] >= 0) & (nbr_labels >= 0) & (reference[:, np.newaxis] != nbr_labels)
```

---

### Problem B: Unbounded Accumulation of Intermediate Edge Triplets in Stage 4 Master Process
- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 4)
- **Severity / Category:** HIGH | Memory Scaling & RAM Ceiling Risk on >100GB TSVs
- **Detailed Bioinformatic & Technical Analysis:**
  In Stage 4 of `domain_finder.py` (lines 695-714), parallel workers compute pairwise domain intersections for each target sequence batch. The master process collects these results:
  ```python
  accumulated_rows: List[np.ndarray] = []
  accumulated_cols: List[np.ndarray] = []
  accumulated_inters: List[np.ndarray] = []
  for rows, cols, inters, dom_lens_dict in target_workers.results:
      if rows.size > 0:
          accumulated_rows.append(rows)
          accumulated_cols.append(cols)
          accumulated_inters.append(inters)
  all_rows = np.concatenate(accumulated_rows)
  ```
  While Stage 2 streams query batches in chunk windows (`chunk_window=50`), Stage 4 holds the full `target_workers.results` list in memory, appends hundreds of thousands of NumPy arrays into three Python lists, and then calls `np.concatenate`.
  On an alignment dataset exceeding 100 GB with hundreds of millions of target hits, the number of non-zero pairwise domain overlaps can exceed $5 \times 10^8$ edges:
  1. Storing $5 \times 10^8$ entries across `accumulated_rows`, `accumulated_cols`, and `accumulated_inters` requires $3 \times (5 \times 10^8 \times 4\text{ bytes}) = 6.0\text{ GB}$.
  2. Calling `np.concatenate` allocates another 6.0 GB concurrently before freeing the list elements.
  3. Storing this inside the MPIRE master process while keeping all worker results causes peak memory spikes that can breach system RAM limits.
- **Evidence in Code:**
  - `domain_finder.py:695-712`: Appending worker outputs into growing lists before global concatenation.
- **Recommended Solution:**
  Stream Stage 4 target batches using chunk windows or dynamically aggregate edges into a compact sparse coordinate buffer, or perform early edge summation across batches to prevent memory duplication.

---

### Problem C: Discarding Biological Domain Support Counts in family_finder
- **Module / Component:** `domain_finder.py` (`family_finder`), `clustering.py` (`dpc`)
- **Severity / Category:** HIGH | Scientific Rigor & Biological Validity
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:family_finder` (lines 465-487), `dpc` is invoked with `weights=None`:
  ```python
  peak_to_cluster, n_halo = dpc(rows=final_r,
                                cols=final_c,
                                sims=similarities,
                                n=n_domains,
                                min_sim=min_fam_sim,
                                backend="scipy",
                                min_dens=min_fam_dens)
  ```
  When `weights` is `None`, `compute_density_and_higher_neighbours` computes local density as $\rho_i = \sum_{j} s(i, j)$, where $s(i, i) = 1.0$.
  Each domain instance $i$ represents an ncRNA domain identified on a query sequence supported by $N_i$ target alignments across the sequence database. In DPC family clustering, a domain instance supported by 5,000 alignment hits has identical weight ($w_i = 1.0$) to a domain instance supported by a single alignment hit.
  Furthermore, if two domain instances have high mutual Jaccard overlap similarity but few overall hits, their density can equal or exceed that of a domain with thousands of distributed homologous hits if the latter has lower pairwise Jaccard scores.
  Biologically, alignment hit support ($N_{\text{targets}}$) represents evolutionary conservation depth. Weighting domain instances by $\log(1 + N_i)$ or hit counts $N_i$ grounds DPC density in biological conservation depth rather than treating all domain instances uniformly.
- **Evidence in Code:**
  - `domain_finder.py:476-486`: `dpc` invoked without `weights` parameter.
- **Recommended Solution:**
  Compute per-domain hit counts during Stage 3 or Stage 4, pass normalized weights into `family_finder`, and forward them to `dpc(..., weights=dom_weights)`.

---

### Problem D: BLAST-TAB TSV Parsing Inefficiency with Python String Splitting
- **Module / Component:** `aligners.py` (`parse_mmseqs_tsv`)
- **Severity / Category:** MEDIUM | Performance Bottleneck on >100GB Alignments
- **Detailed Bioinformatic & Technical Analysis:**
  `aligners.py:parse_mmseqs_tsv` streams alignment lines:
  ```python
  for line in handle:
      line = line.rstrip("\n")
      if not line or line.startswith("#"):
          continue
      fields = line.split("\t")
      raw_qs, raw_qe = int(fields[6]), int(fields[7])
      raw_ts, raw_te = int(fields[8]), int(fields[9])
  ```
  A 100 GB BLAST-TAB TSV file contains approximately $1.3 \times 10^9$ lines. In pure Python:
  1. Calling `line.rstrip("\n")`, `line.split("\t")`, and creating a list of 10-12 string objects per line instantiates over 15 billion transient Python string objects.
  2. Python's memory allocator (`pymalloc`) experiences heavy allocation overhead, limiting parser throughput to $\approx 200,000\text{ lines/sec}$. At that rate, streaming a 100 GB TSV takes nearly 2 hours purely in Python string parsing before any clustering occurs.
- **Evidence in Code:**
  - `aligners.py:150-165`: Line-by-line `line.split("\t")` and integer casting in Python.
- **Recommended Solution:**
  Use chunked byte buffer parsing (e.g., using `csv.reader` with large buffer size or memory-mapped PyArrow / NumPy column readers) to parse tab-separated integer coordinates in native C blocks.

---

### Problem E: Suboptimal Single-Worker Dispatch in target_batch_generator on Low Sequence Counts
- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 4)
- **Severity / Category:** MEDIUM | Parallel Concurrency Bottleneck
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:657`:
  ```python
  target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))
  ```
  If an analysis involves a compact genome or filtered sequence set where `n_target_seqs = 35` and `cpu = 32`, `n_target_seqs // (32 * 4) = 0`, so `target_batch_size = max(10, 0) = 10`.
  The 35 sequences are packaged into $\lceil 35 / 10 \rceil = 4$ batches. With 4 batches dispatched to 32 worker threads, 28 CPU cores sit idle.
  Setting the minimum floor to 10 prevents fine-grained worker distribution on small to medium sequence sets.
- **Evidence in Code:**
  - `domain_finder.py:657`: Hard floor of `10` in `target_batch_size`.
- **Recommended Solution:**
  Scale `target_batch_size` dynamically down to 1 when `n_target_seqs < cpu * 4`:
  $$\text{target\_batch\_size} = \max\left(1, \min\left(250, \left\lceil \frac{n\_target\_seqs}{\text{cpu} \times 4} \right\rceil\right)\right)$$

```python
# FAULTY (domain_finder.py:657 - Hard floor of 10 starves workers on <= 40 sequences):
target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))

# REVISED (Scale batch size down to 1 to ensure full core saturation):
target_batch_size = max(1, min(250, int(np.ceil(n_target_seqs / max(1, cpu * 4)))))
```

---

### Problem F: Redundant Memory Allocation in Sparse build_similarity_matrix
- **Module / Component:** `clustering.py` (`build_similarity_matrix`)
- **Severity / Category:** MEDIUM | Memory Optimization on Graph Construction
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:build_similarity_matrix` (lines 190-215):
  ```python
  key = (r_sym.astype(np.int64) << 32) | c_sym.astype(np.int64)
  order = np.argsort(key)
  ...
  diag = np.arange(n, dtype=np.int64)
  all_r = np.concatenate((u_r, diag))
  all_c = np.concatenate((u_c, diag))
  all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
  mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))
  ```
  When constructing the symmetric CSR matrix:
  1. `u_r` and `u_c` are unpackaged from 64-bit keys into `np.int64`.
  2. Diagonal vectors (`diag`, `np.ones(n)`) are concatenated to coordinate arrays.
  3. SciPy's `csr_matrix` constructor internally casts indices to `np.int32` if $N < 2^{31} - 1$ and non-zero entries fit in 32-bit limits.
  Allocating `all_r` and `all_c` as `np.int64` and appending $N$ diagonal entries creates unnecessary temporary arrays. SciPy's `csr_matrix.setdiag(1.0)` or directly constructing the CSR with 32-bit indices avoids copying large coordinate arrays twice.
- **Evidence in Code:**
  - `clustering.py:192`: `(r_sym.astype(np.int64) << 32)`
  - `clustering.py:204-208`: Allocating 64-bit diagonal arrays and concatenating.
- **Recommended Solution:**
  Use `np.int32` for `all_r` and `all_c` when $N < 2^{31}$, construct CSR directly, and set diagonal values without allocating auxiliary vectors.

---

### Problem G: Inconsistent Point-Specific Parameter Exposure in domain_finder.py
- **Module / Component:** `domain_finder.py` (`domain_instances_on_query`, `find_domains`)
- **Severity / Category:** LOW | API & Pipeline Consistency
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:find_domains`, `point_specific_val` is retrieved from `attributes.get("point_specific", False)` and passed to Stage 5 `family_finder`.
  However, in Stage 2 `domain_instances_on_query` (lines 160-220), `dpc` is invoked for query domain instance extraction **without** `point_specific`:
  ```python
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
  ```
  `dpc` defaults to `point_specific=False`. Consequently, halo filtering in Stage 2 uses traditional cluster-wide $\rho_b$ thresholding (which aggressively prunes members of ragged RNA domain clusters), while Stage 5 allows `point_specific=True`.
  To ensure consistent halo mechanics across all stages, `point_specific` should be configurable in `domain_instances_on_query` and passed via `query_kwargs`.
- **Evidence in Code:**
  - `domain_finder.py:208-218`: `dpc` call in `domain_instances_on_query` lacks `point_specific`.
- **Recommended Solution:**
  Add `point_specific: bool = False` to `domain_instances_on_query` and forward it from `find_domains`.

---

## 4. Phased Implementation Plan for Future Revisions

### Phase 1: Mathematical Integrity and Halo Noise Mask Correction
1. In `clustering.py:_foreign_mask`, update the boolean mask so that neighbor points with label $-1$ (noise) are excluded from foreign contamination calculations (`(reference >= 0) & (nbr_labels >= 0) & (reference != nbr_labels)`).
2. In `domain_finder.py:family_finder`, update solitary domain density calculation when `u_r.size == 0` to use actual domain target occurrence counts rather than uniform $1.0$.
3. Forward `point_specific` parameter consistently to `domain_instances_on_query` in Stage 2.

### Phase 2: Memory Ceiling Optimization for >100GB Alignments
1. In `domain_finder.py:find_domains` Stage 4, replace monolithic accumulation in `accumulated_rows`, `accumulated_cols`, and `accumulated_inters` with streaming batch processing or pre-allocated chunk buffers to eliminate 2x memory spikes during `np.concatenate`.
2. In `clustering.py:build_similarity_matrix`, maintain 32-bit integer coordinate representations when $N < 2^{31}$ to avoid doubling memory during sparse matrix assembly.

### Phase 3: High-Throughput Streaming Parser Optimization
1. In `aligners.py:parse_mmseqs_tsv`, optimize BLAST-TAB line parsing to reduce intermediate Python string allocations during multi-gigabyte file streaming.
2. Ensure `target_batch_size` dynamically scales down to 1 when target sequence counts are small to guarantee 100% CPU thread saturation.

### Phase 4: Test Verification & Benchmarking
1. Validate cluster results and noise assignments across synthetic benchmarks with known ground-truth borders.
2. Verify that running `python -m pytest -W error::DeprecationWarning` passes cleanly without warnings or regressions.
3. Confirm peak RAM remains below 200 GB on simulated high-depth datasets.
