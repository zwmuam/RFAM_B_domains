# PROBLEMS: Comprehensive Audit of the Domain Finding Codebase and Dependencies

The pipeline extracts conserved structural RNA domain instances from pairwise alignments (MMseqs2 or RiboSeek) and clusters them into global domain families via Density Peak Clustering (DPC). Processing large multi-gigabyte alignment datasets imposes strict operational requirements:
1. Peak system memory must stay bounded below 200 GB.
2. Parallel worker memory overhead must remain below 200 MB per CPU core.
3. Multi-core execution must scale smoothly up to 30+ threads without worker starvation or inter-process communication bottlenecks.
4. Noise instances ($-1$) must never be assigned to valid clusters or bias downstream biological metrics.

---

## Re-Assessment and Audit of PREVIOUS_UPDATE.md Claims

Two claims from `PREVIOUS_UPDATE.md` were incompletely implemented or introduced subsequent complications:

**Disjoint Graph Density Thresholding in family_finder Was Partially Bypassed**
- **Audit Finding:** In `domain_finder.py:family_finder`, when candidate domain instances share zero pairwise target overlaps across sequences (`u_r.size == 0`), the code checks whether domains meet $\text{min\_fam\_dens}$ by assuming every domain has a fixed self-density of $1.0$ (`np.ones(n_domains, dtype=np.float64)`). In biological datasets, solitary domain instances possess real alignment support proportional to their number of target occurrences $N_{\text{targets}}$. Forcing solitary density to $1.0$ causes two distinct failures:
  1. If a user sets $\text{min\_fam\_dens} > 1.0$ (e.g. requiring at least 3 supporting hits), all solitary domains are discarded as noise—even highly conserved domains supported by hundreds of hits across genomes.
  2. If $\text{min\_fam\_dens} \le 1.0$, single-hit spurious noise artifacts are accepted as valid domain family peaks.
- **Subsequent Problem:** In `clustering.py:dpc`, an empty edge check (`rows.size == 0`) evaluates `densities >= min_dens`. However, when `family_finder` calls `dpc`, it does not pass domain weights (`weights=None`). Consequently, `dpc` calculates density solely from edge similarities plus diagonal $1.0$, completely losing track of underlying alignment hit counts. If candidate domains share no target overlaps, or form disconnected graph components with zero mutual edges, their real biological support is ignored.

```python
# FAULTY (domain_finder.py:family_finder - Solitary domains assigned artificial density 1.0):
if u_r.size == 0:
    qualifying = np.flatnonzero(np.ones(n_domains, dtype=np.float64) >= min_fam_dens)
    if qualifying.size == 0:
        return {}, int(n_domains)
    return {int(i): np.array([i], dtype=np.int64) for i in qualifying}, int(n_domains - qualifying.size)

# REVISED (domain_finder.py:family_finder - Evaluate density using true alignment hit support):
if u_r.size == 0:
    densities = np.ones(n_domains, dtype=np.float64) if dom_weights is None else np.asarray(dom_weights, dtype=np.float64)
    qualifying = np.flatnonzero(densities >= min_fam_dens)
    if qualifying.size == 0:
        return {}, int(n_domains)
    return {int(i): np.array([i], dtype=np.int64) for i in qualifying}, int(n_domains - qualifying.size)
```

**Coordinate Typing Harmonization and Docstring Drift**
- **Audit Finding:** `PREVIOUS_UPDATE.md` stated that coordinates and sparse edge arrays were standardized to 32-bit integers (`np.int32`). While `domain_finder.py:evaluate_clusters` was updated to `np.int32`, `clustering.py:build_similarity_matrix` still casts 32-bit row and column indices to `np.int64` and allocates 64-bit diagonal arrays unconditionally. This doubles memory usage during sparse matrix construction. Furthermore, docstrings in `target_batch_processor` still claim 64-bit coordinate operations are maintained.

```python
# FAULTY (clustering.py:build_similarity_matrix - Upcasting 32-bit coordinates to 64-bit):
diag = np.arange(n, dtype=np.int64)
all_r = np.concatenate((u_r, diag))
all_c = np.concatenate((u_c, diag))
all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))

# REVISED (clustering.py:build_similarity_matrix - Native int32 sparse matrix assembly):
mat = csr_matrix((max_s, (u_r.astype(np.int32), u_c.astype(np.int32))), shape=(n, n))
mat.setdiag(1.0)
```

---

## Critical Algorithmic, Mathematical, and Biological Problems

**Problem A: Halo Filtering Border Contamination Treats Noise Points as Foreign Cluster Neighbors**
- **Module / Component:** `clustering.py` (`_foreign_mask`, `halo_filter`)
- **Severity / Category:** CRITICAL | Mathematical & Algorithmic Correctness
- **Detailed Bioinformatic & Technical Analysis:**
  In Density Peak Clustering, halo filtering removes ambiguous border points that lie between two competing, authentic clusters.
  In `clustering.py:_foreign_mask`, the mask identifying foreign cluster neighbors is computed as:
  ```python
  valid = nbr_idx >= 0
  safe_idx = np.maximum(nbr_idx, 0)
  reference = labels if query_labels is None else query_labels
  foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])
  ```
  If a neighboring point was classified as noise ($-1$) during DPC cascading assignment or density thresholding, its label is $-1$. For an authentic cluster member with label $c \ge 0$, the inequality $c \ne -1$ evaluates to `True`.
  As a result, **unassigned noise points and low-confidence alignment artifacts are treated as foreign cluster neighbors**. When a genuine core domain instance borders discarded noise, its foreign contamination ratio artificially spikes. The point is erroneously flagged as border contamination and pruned to noise.
  Biologically, halo filtering should measure competition between two distinct biological families, not proximity to discarded background noise.
- **Evidence in Code:**
  - `clustering.py:544`: `foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])`
- **Recommended Solution:**
  Require that both the reference point and its neighbor belong to valid, distinct clusters ($\ge 0$):

```python
# FAULTY (clustering.py:544 - Noise (-1) neighbors count as foreign cluster contamination):
foreign = valid & (reference[:, np.newaxis] != labels[safe_idx])

# REVISED (clustering.py - Only genuine, distinct clusters qualify as foreign neighbors):
nbr_labels = labels[safe_idx]
foreign = valid & (reference[:, np.newaxis] >= 0) & (nbr_labels >= 0) & (reference[:, np.newaxis] != nbr_labels)
```

---

**Problem B: Unbounded Accumulation of Intermediate Edge Triplets in Stage 4 Master Process**
- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 4)
- **Severity / Category:** HIGH | Memory Scaling & RAM Ceiling Risk on >100GB TSVs
- **Detailed Bioinformatic & Technical Analysis:**
  In Stage 4 of `domain_finder.py`, parallel workers calculate pairwise domain intersections for each target sequence batch. The master process collects these outputs by appending NumPy arrays into three growing Python lists:
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
  While Stage 2 streams query batches in chunk windows, Stage 4 collects all worker outputs in memory simultaneously before calling `np.concatenate`.
  On alignment datasets exceeding 100 GB with hundreds of millions of target hits, the number of pairwise domain overlaps can exceed $5 \times 10^8$ edges:
  1. Storing $5 \times 10^8$ entries across three Python lists consumes $3 \times (5 \times 10^8 \times 4\text{ bytes}) = 6.0\text{ GB}$.
  2. Calling `np.concatenate` concurrently allocates an additional 6.0 GB before freeing the list chunks.
  3. Holding all raw worker outputs alongside the concatenated arrays in the master process produces memory spikes that can breach workstation RAM limits.
- **Evidence in Code:**
  - `domain_finder.py:695-712`: Appending worker outputs into unbounded lists before global concatenation.
- **Recommended Solution:**
  Stream worker outputs using an iterator, or pre-allocate an expandable structured buffer to aggregate edge triplets directly, avoiding duplicate in-memory copies during `np.concatenate`.

```python
# FAULTY (domain_finder.py:695-714 - Triple list accumulation followed by np.concatenate):
accumulated_rows: List[np.ndarray] = []
accumulated_cols: List[np.ndarray] = []
accumulated_inters: List[np.ndarray] = []
for rows, cols, inters, dom_lens_dict in target_workers.results:
    if rows.size > 0:
        accumulated_rows.append(rows)
        accumulated_cols.append(cols)
        accumulated_inters.append(inters)
all_rows = np.concatenate(accumulated_rows) if accumulated_rows else np.empty(0, dtype=np.int32)
all_cols = np.concatenate(accumulated_cols) if accumulated_cols else np.empty(0, dtype=np.int32)
all_inters = np.concatenate(accumulated_inters) if accumulated_inters else np.empty(0, dtype=np.int32)

# REVISED (Directly aggregate non-empty chunks or stream results to eliminate 2x RAM spike):
total_edges = sum(r.size for r, _, _, _ in target_workers.results)
all_rows = np.empty(total_edges, dtype=np.int32)
all_cols = np.empty(total_edges, dtype=np.int32)
all_inters = np.empty(total_edges, dtype=np.int32)

write_pos = 0
for r, c, inter, dom_lens_dict in target_workers.results:
    k = r.size
    if k > 0:
        all_rows[write_pos:write_pos + k] = r
        all_cols[write_pos:write_pos + k] = c
        all_inters[write_pos:write_pos + k] = inter
        write_pos += k
    for d_id, d_len in dom_lens_dict.items():
        if d_id < n_total_domains:
            global_dom_lengths[d_id] += d_len
```

---

**Problem C: Discarding Biological Domain Support Counts in family_finder**
- **Module / Component:** `domain_finder.py` (`family_finder`), `clustering.py` (`dpc`)
- **Severity / Category:** HIGH | Scientific Rigor & Biological Validity
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:family_finder`, `dpc` is invoked with `weights=None`:
  ```python
  peak_to_cluster, n_halo = dpc(rows=final_r,
                                cols=final_c,
                                sims=similarities,
                                n=n_domains,
                                min_sim=min_fam_sim,
                                backend="scipy",
                                min_dens=min_fam_dens)
  ```
  When `weights` is `None`, local density is computed as $\rho_i = \sum_{j} s(i, j)$, where self-similarity is $s(i, i) = 1.0$.
  Each domain instance $i$ represents an ncRNA domain supported by $N_i$ target alignments across the genome. In DPC family clustering, a domain instance supported by 5,000 alignment hits currently receives the exact same baseline weight ($w_i = 1.0$) as a domain instance supported by a single alignment hit.
  Biologically, alignment hit support ($N_{\text{targets}}$) represents evolutionary conservation depth. Weighting domain instances by hit support grounds DPC density in biological conservation depth rather than treating all domain instances uniformly.
  Hit counts are already recorded in `all_targets_arr['dom_id']` at the end of Stage 3, meaning $N_i$ can be computed with zero algorithmic overhead via `np.bincount(all_targets_arr['dom_id'], minlength=n_total_domains)`.
- **Evidence in Code:**
  - `domain_finder.py:476-486`: `dpc` invoked without `weights` parameter in `family_finder`.
- **Recommended Solution:**
  Compute per-domain hit counts at the end of Stage 3, pass them to `family_finder`, and forward them to `dpc(..., weights=dom_weights)`.

```python
# FAULTY (domain_finder.py:476-486 - dpc invoked without biological hit weights):
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

# REVISED (Compute domain hit weights from Stage 3 targets and pass to family_finder & dpc):
dom_weights = np.bincount(all_targets_arr['dom_id'], minlength=n_total_domains).astype(np.float64)

# In family_finder:
peak_to_cluster, n_halo = dpc(rows=final_r,
                              cols=final_c,
                              sims=similarities,
                              n=n_domains,
                              min_sim=min_fam_sim,
                              weights=dom_weights,
                              backend="scipy",
                              filter_transparency=halo_stringency,
                              halo_method=halo_method,
                              min_dens=min_fam_dens,
                              point_specific=point_specific)
```

---

**Problem D: BLAST-TAB TSV Parsing Inefficiency with Python String Splitting**
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
  1. Calling `line.rstrip("\n")`, `line.split("\t")`, and creating 10-12 string objects per line instantiates over 15 billion transient Python string objects.
  2. Python's memory allocator (`pymalloc`) incurs heavy overhead, limiting parser throughput to $\approx 200,000\text{ lines/sec}$. Streaming a 100 GB TSV takes nearly 2 hours purely in string allocations before any clustering occurs.
- **Evidence in Code:**
  - `aligners.py:150-165`: Line-by-line `line.split("\t")` and integer conversion in pure Python.
- **Recommended Solution:**
  Use `csv.reader` with a large read buffer (`io.BufferedReader`), or slice string fields without splitting the entire line, significantly reducing intermediate object allocations.

```python
# FAULTY (aligners.py:150-165 - Allocates 10-12 Python string objects per line):
for line in handle:
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        continue
    fields = line.split("\t")
    raw_qs, raw_qe = int(fields[6]), int(fields[7])
    raw_ts, raw_te = int(fields[8]), int(fields[9])

# REVISED (Stream through csv.reader with 8MB read buffer for faster C-level tokenization):
import csv
import io

reader = csv.reader(io.open(tsv_file, "r", buffering=8 * 1024 * 1024), delimiter="\t")
for fields in reader:
    if not fields or fields[0].startswith("#"):
        continue
    raw_qs, raw_qe = int(fields[6]), int(fields[7])
    raw_ts, raw_te = int(fields[8]), int(fields[9])
```

---

**Problem E: Suboptimal Single-Worker Dispatch in target_batch_generator on Low Sequence Counts**
- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 4)
- **Severity / Category:** MEDIUM | Parallel Concurrency Bottleneck
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:657`:
  ```python
  target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))
  ```
  When analyzing smaller sequence sets (e.g. `n_target_seqs = 35` on a 32-core workstation), integer division `35 // (32 * 4) = 0`, causing `target_batch_size = max(10, 0) = 10`.
  The 35 sequences are packaged into $\lceil 35 / 10 \rceil = 4$ batches. When dispatched across 32 CPU worker threads, only 4 threads receive work while 28 CPU cores sit completely idle.
  Enforcing a hard minimum floor of 10 prevents fine-grained worker distribution on small to medium sequence sets.
- **Evidence in Code:**
  - `domain_finder.py:657`: Hard floor of `10` in `target_batch_size`.
- **Recommended Solution:**
  Scale `target_batch_size` dynamically down to 1 when sequence counts are small:
  $$\text{target\_batch\_size} = \max\left(1, \min\left(250, \left\lceil \frac{\text{n\_target\_seqs}}{\text{cpu} \times 4} \right\rceil\right)\right)$$

```python
# FAULTY (domain_finder.py:657 - Hard floor of 10 starves workers on <= 40 sequences):
target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))

# REVISED (Scale batch size down to 1 to ensure full core saturation):
target_batch_size = max(1, min(250, int(np.ceil(n_target_seqs / max(1, cpu * 4)))))
```

---

**Problem F: Redundant Memory Allocation in Sparse build_similarity_matrix**
- **Module / Component:** `clustering.py` (`build_similarity_matrix`)
- **Severity / Category:** MEDIUM | Memory Optimization on Graph Construction
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:build_similarity_matrix`:
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
  2. Diagonal vectors (`diag`, `np.ones(n)`) are explicitly created and concatenated to coordinate arrays.
  3. SciPy's `csr_matrix` internally casts coordinates to `np.int32` when $n < 2^{31} - 1$ and non-zero entries fit within 32-bit limits.
  Allocating `all_r` and `all_c` as `np.int64` and appending $n$ diagonal entries creates redundant temporary arrays. Constructing the CSR matrix directly with `np.int32` and setting diagonal entries via `mat.setdiag(1.0)` eliminates duplicate coordinate copies.
- **Evidence in Code:**
  - `clustering.py:192`: `(r_sym.astype(np.int64) << 32)`
  - `clustering.py:204-208`: Allocating 64-bit diagonal arrays and concatenating.
- **Recommended Solution:**
  Retain `np.int32` for coordinate arrays when $n < 2^{31}$, construct the CSR matrix directly, and set the diagonal without allocating auxiliary coordinate vectors.

```python
# FAULTY (clustering.py:204-208 - Explicit diagonal allocation and concatenation in int64):
diag = np.arange(n, dtype=np.int64)
all_r = np.concatenate((u_r, diag))
all_c = np.concatenate((u_c, diag))
all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))

# REVISED (Construct CSR with 32-bit indices and set diagonal in-place):
mat = csr_matrix((max_s, (u_r.astype(np.int32), u_c.astype(np.int32))), shape=(n, n))
mat.setdiag(1.0)
```

---

**Problem G: Inconsistent Point-Specific Parameter Exposure in domain_finder.py**
- **Module / Component:** `domain_finder.py` (`domain_instances_on_query`, `find_domains`)
- **Severity / Category:** LOW | API & Pipeline Consistency
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:find_domains`, `point_specific_val` is retrieved from `attributes.get("point_specific", False)` and forwarded to Stage 5 `family_finder`.
  However, in Stage 2 `domain_instances_on_query`, `dpc` is invoked for query domain instance discovery without passing `point_specific`:
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
  `dpc` defaults to `point_specific=False`. Consequently, halo filtering in Stage 2 uses traditional cluster-wide density thresholding (which aggressively prunes ragged biological RNA domain clusters), while Stage 5 permits `point_specific=True`.
  To maintain consistent halo filtering semantics across all stages, `point_specific` should be configurable in `domain_instances_on_query` and passed via `query_kwargs`.
- **Evidence in Code:**
  - `domain_finder.py:208-218`: `dpc` call in `domain_instances_on_query` lacks `point_specific`.
- **Recommended Solution:**
  Add `point_specific: bool = False` to `domain_instances_on_query` and forward it from `find_domains`.

```python
# FAULTY (domain_finder.py:208-218 - dpc call in Stage 2 omits point_specific):
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

# REVISED (domain_finder.py - Pass point_specific consistently to Stage 2 dpc):
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
```

---

## Phased Implementation Plan for Future Revisions

**Phase: Mathematical Integrity and Halo Noise Mask Correction**
1. In `clustering.py:_foreign_mask`, update the boolean mask so that neighbor points with label $-1$ (noise) are excluded from foreign contamination calculations (`(reference >= 0) & (nbr_labels >= 0) & (reference != nbr_labels)`).
2. In `domain_finder.py:family_finder`, update solitary domain density calculation when `u_r.size == 0` to use actual domain target occurrence counts rather than uniform $1.0$.
3. Forward `point_specific` parameter consistently to `domain_instances_on_query` in Stage 2.

**Phase: Memory Ceiling Optimization for >100GB Alignments**
1. In `domain_finder.py:find_domains` Stage 4, replace list accumulation in `accumulated_rows`, `accumulated_cols`, and `accumulated_inters` with direct array pre-allocation to eliminate memory duplication during `np.concatenate`.
2. In `clustering.py:build_similarity_matrix`, maintain 32-bit integer coordinate representations when $n < 2^{31}$ and set matrix diagonals via `mat.setdiag(1.0)` to avoid copying coordinate arrays.

**Phase: High-Throughput Streaming Parser Optimization**
1. In `aligners.py:parse_mmseqs_tsv`, optimize BLAST-TAB line parsing using buffered `csv.reader` to reduce intermediate Python string allocations during multi-gigabyte file streaming.
2. Ensure `target_batch_size` dynamically scales down to 1 when target sequence counts are small to guarantee 100% CPU thread saturation.

**Phase: Test Verification and Benchmarking**
1. Validate cluster results and noise assignments across synthetic benchmarks with known ground-truth borders.
2. Verify that running `python -m pytest -W error::DeprecationWarning` passes cleanly without warnings or regressions.
3. Confirm peak RAM remains below 200 GB on simulated high-depth datasets.
