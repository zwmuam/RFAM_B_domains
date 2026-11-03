# Abstract

The pipeline extracts conserved structural RNA domain instances from pairwise alignments (MMseqs2 or RiboSeek) and clusters them into global domain families via Density Peak Clustering (DPC). Processing large multi-gigabyte alignment datasets imposes strict operational requirements:
1. Peak system memory must stay bounded strictly below 200 GB.
2. Parallel worker memory overhead must remain strictly below 200 MB per worker process.
3. Multi-core execution must scale smoothly up to 30+ threads without worker starvation or inter-process communication bottlenecks.
4. Non-overlapping or 0-similarity pairs must never link domain instances:
   - In domain finding: spans with zero overlap are mutually disjoint.
   - In halo filtering: 0-similarity never serves as border evidence to foreign clusters.
   - In cluster merging: non-overlapping domain instances remain distinct unless bridged by intermediate support.
   - In family finding: domain instances with 0-similarity are strictly unlinked.

---

## Memory Accumulation in Stage 4 Target Overlap Collection

- **Evidence**:
  In `domain_finder.py:695-714`, `find_domains` collects parallel worker outputs from `target_workers.results` by appending triplets into Python lists:
  ```python
  accumulated_rows: List[np.ndarray] = []
  accumulated_cols: List[np.ndarray] = []
  accumulated_inters: List[np.ndarray] = []
  for rows, cols, inters, dom_lens_dict in target_workers.results:
      if rows.size > 0:
          accumulated_rows.append(rows)
          accumulated_cols.append(cols)
          accumulated_inters.append(inters)
  all_rows = np.concatenate(accumulated_rows) if accumulated_rows else np.empty(0, dtype=np.int32)
  ```
- **Impact**:
  For an alignment dataset with 500 million pairwise target overlaps:
  $$\text{Memory}_{\text{workers}} = 5 \times 10^8 \times 3 \times 4\text{ bytes} \approx 6.0\text{ GB}$$
  Because `target_workers.results` holds all completed worker arrays simultaneously in memory, executing `np.concatenate` (or naive two-pass pre-allocation) allocates an additional $6.0\text{ GB}$ before the original arrays are dereferenced, doubling peak memory consumption to $12.0\text{ GB}$. On massive multi-gigabyte TSVs, this memory spike threatens the 200 GB RAM budget.
- **Proposed Fix**:
  Eliminating peak duplication requires genuine incremental consumption, streaming generator yields, or bounded chunked aggregation where worker outputs are dereferenced and garbage-collected as they are processed. Consume worker outputs incrementally in bounded chunks or pop worker result references as they are processed so that intermediate memory is immediately reclaimed. When total size is known or streaming, write triplets directly into pre-allocated memory or disk-backed structures and eagerly delete worker result references.

```python
# FAULTY (domain_finder.py:695-714 - List accumulation retaining worker results before concatenation):
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

# REVISED (domain_finder.py - Eager popping/consumption of worker results with bounded memory):
results_list = target_workers.results
total_edges = 0
for r, _, _, _ in results_list:
    total_edges += r.size

all_rows = np.empty(total_edges, dtype=np.int32)
all_cols = np.empty(total_edges, dtype=np.int32)
all_inters = np.empty(total_edges, dtype=np.int32)
global_dom_lengths = np.zeros(n_total_domains, dtype=np.int64)

write_pos = 0
while results_list:
    r, c, inter, dom_lens_dict = results_list.pop(0)
    k = r.size
    if k > 0:
        all_rows[write_pos:write_pos + k] = r
        all_cols[write_pos:write_pos + k] = c
        all_inters[write_pos:write_pos + k] = inter
        write_pos += k
    del r, c, inter
    for d_id, d_len in dom_lens_dict.items():
        if d_id < n_total_domains:
            global_dom_lengths[d_id] += d_len
del target_workers.results
```

- **Validation**:
  1. Profile master and worker memory using `psutil` or `tracemalloc` to demonstrate memory does not scale with completed batch count.
  2. Test synthetic edge collections scaling up to $5 \times 10^8$ entries and verify peak memory remains bounded below $6.5\text{ GB}$.
  3. Ensure final coordinate arrays (`all_rows`, `all_cols`, `all_inters`) and `global_dom_lengths` are numerically identical to the unoptimized output.

---

## Sparse Matrix Construction Typing and Redundant Diagonal Allocation

- **Evidence**:
  In `clustering.py:build_similarity_matrix`:
  ```python
  diag = np.arange(n, dtype=np.int64)
  all_r = np.concatenate((u_r, diag))
  all_c = np.concatenate((u_c, diag))
  all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
  mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))
  ```
  `u_r` and `u_c` are unpacked into 64-bit coordinates, and $n$ self-similarity diagonal entries are explicitly constructed and concatenated into coordinate arrays.
- **Impact**:
  1. Allocating separate 64-bit coordinate arrays `all_r` and `all_c` and float array `all_s` doubles the memory footprint during sparse matrix construction.
  2. For $n = 10^7$ nodes, explicit diagonal vectors allocate $3 \times 10^7$ unnecessary elements.
  3. Conversely, casting coordinates unconditionally to `np.int32` without validating non-zero count ($nnz$) risks integer overflow in SciPy's CSR `indptr` when $nnz \ge 2^{31} - 1$.
- **Proposed Fix**:
  Sparse index typing must check both $\max(n, nnz) < 2^{31} - 1$. When satisfied, coordinate arrays should use `np.int32` directly, eliminating intermediate `np.int64` upcasting and redundant explicit diagonal concatenation arrays. Select index dtype dynamically based on both matrix dimensions and stored entry count:
  $$\text{dtype} = \begin{cases} \text{np.int32} & \text{if } n < 2^{31}-1 \text{ and } (u_r.\text{size} + n) < 2^{31}-1 \\ \text{np.int64} & \text{otherwise} \end{cases}$$
  Construct the CSR matrix directly from non-zero edge entries without auxiliary diagonal coordinate arrays, and set self-similarities via `mat.setdiag(1.0)` with canonical index sorting.

```python
# FAULTY (clustering.py:204-208 - Explicit diagonal allocation and 64-bit coordinates):
diag = np.arange(n, dtype=np.int64)
all_r = np.concatenate((u_r, diag))
all_c = np.concatenate((u_c, diag))
all_s = np.concatenate((max_s, np.ones(n, dtype=np.float64)))
mat = csr_matrix((all_s, (all_r, all_c)), shape=(n, n))
mat.sort_indices()

# REVISED (clustering.py - Bounds-checked index typing and in-place diagonal setting):
idx_dtype = np.int32 if (n < (1 << 31) - 1 and (u_r.size + n) < (1 << 31) - 1) else np.int64
mat = csr_matrix((max_s, (u_r.astype(idx_dtype), u_c.astype(idx_dtype))), shape=(n, n))
mat.setdiag(1.0)
mat.sort_indices()
```

- **Validation**:
  1. Verify index dtypes: confirm `mat.indices.dtype` and `mat.indptr.dtype` match `idx_dtype`.
  2. Compare matrix symmetry ($A = A^T$) and numerical equivalence against the reference explicit-diagonal matrix: $\max |A_{\text{new}} - A_{\text{old}}| = 0.0$.
  3. Validate on boundary cases: empty edge sets ($u_r.\text{size} == 0$), matrices with existing diagonal entries, and matrices exceeding $2^{31}-1$ entries.


---

## Dynamic Batch Dispatch Scaling Balancing Core Saturation and IPC Overhead

- **Evidence**:
  In `domain_finder.py:657`:
  ```python
  target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))
  ```
- **Impact**:
  For datasets with few target sequences (e.g. $n_{\text{target\_seqs}} = 35$ on a 32-core machine), integer division yields $35 // (32 \times 4) = 0$, giving `target_batch_size = max(10, 0) = 10`. This partitions the work into only $\lceil 35 / 10 \rceil = 4$ batches, leaving 28 worker threads completely idle.
  Conversely, setting a flat batch size of 1 on large sequence sets ($n_{\text{target\_seqs}} = 100,000$) creates 100,000 separate IPC tasks, causing severe queue contention and context switching in MPIRE.
- **Proposed Fix**:
  Batch sizing must scale dynamically: small sequence sets ($n < \text{cpu} \times 4$) should distribute 1-sequence batches to ensure all cores receive tasks, while large sequence sets should scale batch sizes to amortize IPC overhead. Scale `target_batch_size` adaptively: for small target counts ($n_{\text{target\_seqs}} \le \text{cpu} \times 4$), reduce batch size down to 1 to guarantee full multi-core saturation. For larger target counts, scale dynamically between 10 and 1000 queries  per batch to minimize IPC overhead.

```python
# FAULTY (domain_finder.py:657 - Hard floor of 10 starves workers when sequence count <= 40):
target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))

# REVISED (domain_finder.py - Dynamic batch scaling balancing saturation and IPC overhead):
if n_target_seqs <= cpu * 4:
    target_batch_size = max(1, int(np.ceil(n_target_seqs / max(1, cpu))))
else:
    target_batch_size = max(10, min(250, n_target_seqs // (cpu * 4)))
```

- **Validation**:
  1. On $n_{\text{target\_seqs}} = 35$ with 32 worker threads, verify $\ge 32$ batches are generated and all worker cores receive tasks.
  2. On $n_{\text{target\_seqs}} = 50,000$ sequences, verify batch size scales between 10 and 250, avoiding task flooding.
  3. Measure elapsed runtime across 4, 16, and 32 threads, confirming monotonic scaling without performance regressions.

---

## Point-Specific Parameter Exposure Across DPC Stages

- **Evidence**:
  In `domain_finder.py:find_domains`, `point_specific_val` is parsed from attributes and passed to Stage 5 `family_finder`:
  ```python
  point_specific_val = bool(attributes.get("point_specific", False))
  peak_to_cluster, n_halo = family_finder(..., point_specific=point_specific_val)
  ```
  However, in Stage 2 `domain_instances_on_query`, `dpc` is invoked without passing `point_specific`:
  ```python
  peak_to_cluster, n_noise = dpc(..., min_dens=min_dom_dens)
  ```
- **Impact**:
  In Stage 2, DPC defaults to `point_specific=False` (traditional cluster-wide halo filtering). Under traditional halo filtering, if a single ragged border span exhibits elevated local density, the entire cluster threshold $\rho_b$ is elevated, inappropriately purging valid, lower-density biological RNA domain variants into noise (-1). Stage 5 permits `point_specific=True` to preserve sparse domain families, leading to inconsistent halo filtering behavior across stages.
- **Proposed Fix**:
  Rather than hardcoding behavior, `point_specific` should be consistently exposed across both Stage 2 (`domain_instances_on_query`) and Stage 5 (`family_finder`), enabling uniform biological parameter propagation from pipeline configuration and CLI arguments while preserving regression invariants. Expose `point_specific: bool = False` in `domain_instances_on_query` and pass `point_specific_val` through `query_kwargs` from `find_domains`. This permits users and pipelines to configure consistent border filtering behavior across both query instance identification and family clustering while retaining default backwards compatibility.

```python
# FAULTY (domain_finder.py:208-218 - Stage 2 omits point_specific parameter):
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

# REVISED (domain_finder.py - Pass point_specific through query_kwargs to Stage 2 dpc):
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

- **Validation**:
  1. Regression testing: verify that when `point_specific=False` (default), Stage 2 clustering output is identical to prior releases.
  2. Test with `point_specific=True`: confirm that valid lower-density border spans in ragged ncRNA alignments are retained rather than pruned.
  3. Validate that noise points (-1) are never assigned to valid clusters or included in cluster property calculations.

---

# Phased Implementation Plan for Future Revisions

This implementation plan provides direct, step-by-step instructions for a future engineering run. Each phase specifies exact file modifications, coding conventions, anti-patterns to avoid, and risk controls.

## Phase: Interface and Pipeline Parameter Harmonization
1. **Target File:** `domain_finder.py`
   - **Target Functions:** `domain_instances_on_query` and `find_domains`
   - **Specific Instructions:**
     - Update signature of `domain_instances_on_query` to include `point_specific: bool = False`.
     - Forward `point_specific=point_specific` in the `dpc(...)` call within `domain_instances_on_query`.
     - In `find_domains`, add `'point_specific': point_specific_val` to `query_kwargs` passed to `batch_query_processor`.
   - **Practices to Adhere to:**
     - Ensure the default remains `False` to maintain full backward compatibility with existing unit tests.
     - Document parameter semantics with clear docstrings explaining point-specific border filtering vs cluster-wide $\rho_b$ thresholding.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not modify noise point mapping (`labels < 0` $\to$ `-1`); unassigned halo points must remain strictly excluded from domain instances.

## Phase: Master Memory Bounding in Stage 4 Target Overlaps
1. **Target File:** `domain_finder.py`
   - **Target Function:** `find_domains` (Stage 4 result collection)
   - **Specific Instructions:**
     - Replace the three unbounded accumulation lists (`accumulated_rows`, `accumulated_cols`, `accumulated_inters`) with a single-pass extraction loop.
     - First, sum total edges from `target_workers.results` to pre-allocate `all_rows`, `all_cols`, and `all_inters` with dtype `np.int32`.
     - Then, consume results using `results_list.pop(0)` or an incremental iterator, copying data slices directly into destination arrays and immediately deleting the popped worker array references to allow immediate garbage collection.
     - Accumulate `dom_lens_dict` entries directly into `global_dom_lengths` during the consumption loop.
     - Immediately delete `target_workers.results` upon loop completion.
   - **Practices to Adhere to:**
     - Use `## mem_opt ##` annotations above the allocation and popping loop.
     - Maintain strict `np.int32` typing for rows, cols, and inters arrays.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not retain worker array references in memory while holding pre-allocated arrays; references must be popped and dereferenced eagerly.
     - Do not use `np.concatenate` on worker lists, as it forces concurrent duplicate memory allocation.

## Phase: Sparse Similarity Matrix Index Typing and Assembly
1. **Target File:** `clustering.py`
   - **Target Function:** `build_similarity_matrix`
   - **Specific Instructions:**
     - Check index safety against both dimension and stored-entry counts: `idx_dtype = np.int32 if (n < (1 << 31) - 1 and (u_r.size + n) < (1 << 31) - 1) else np.int64`.
     - Cast `u_r` and `u_c` to `idx_dtype` and construct `csr_matrix((max_s, (u_r, u_c)), shape=(n, n))`.
     - Set diagonal self-similarities using `mat.setdiag(1.0)` and ensure canonical format via `mat.sort_indices()`.
     - Eliminate the allocation of `diag = np.arange(n, dtype=np.int64)`, `all_r`, `all_c`, and `all_s`.
   - **Practices to Adhere to:**
     - Verify both `mat.indices.dtype` and `mat.indptr.dtype`.
     - Ensure sparse matrix operations preserve diagonal entries without mutating original off-diagonal duplicate sums.
   - **Anti-Patterns & Risks to Avoid:**
     - Never assume $n < 2^{31} - 1$ alone guarantees 32-bit CSR safety; non-zero count $nnz$ can exceed 32-bit limits even when $n$ is small.
     - Avoid converting sparse matrices to dense arrays unless `backend == 'numpy'` is explicitly requested.

## Phase: High-Throughput Vectorized Alignment Streaming
1. **Target File:** `aligners.py`
   - **Target Function:** `parse_mmseqs_tsv`
   - **Specific Instructions:**
     - Implement chunked parsing using `pandas.read_csv(chunksize=100_000)` with `engine='c'`, reading columns `[0, 1, 6, 7, 8, 9]`.
     - Perform vectorized coordinate transformations:
       - Relative strand: `t_strand = np.where((raw_qs > raw_qe) ^ (raw_ts > raw_te), -1, 1).astype(np.int8)`
       - 0-based half-open query intervals: `q_start = np.minimum(raw_qs, raw_qe) - 1`, `q_end = np.maximum(raw_qs, raw_qe)`
       - 0-based half-open target intervals: `t_start = np.minimum(raw_ts, raw_te) - 1`, `t_end = np.maximum(raw_ts, raw_te)`
     - Filter self-hits (`q_seq_str == t_seq_str`) in vectorized boolean masks.
     - Map string identifiers against the frozen `SequenceRegistry`, raising an explicit error (🛑) if an unknown sequence is encountered.
   - **Practices to Adhere to:**
     - Ensure query batch ordering matches the deterministic sort order required downstream.
     - Maintain closed-set verification without memory leaks.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not use `csv.reader`, as it does not improve parsing throughput over standard line splitting in pure Python.
     - Avoid loading entire multi-gigabyte TSVs into memory at once; always stream via chunk windows.

## Phase: Adaptive Worker Batch Scaling
1. **Target File:** `domain_finder.py`
   - **Target Function:** `find_domains` (Stage 4 target batch sizing)
   - **Specific Instructions:**
     - Update `target_batch_size` computation:
       ```python
       if n_target_seqs <= cpu * 4:
           target_batch_size = max(1, int(np.ceil(n_target_seqs / max(1, cpu))))
       else:
           target_batch_size = max(10, min(250, n_target_seqs // (cpu * 4)))
       ```
   - **Practices to Adhere to:**
     - Retain `chunk_size = 1` in `Parallel` mapping to ensure fine-grained worker dispatch.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not enforce a hard floor of 10 on small sequence datasets, which leads to core starvation.
     - Do not unconditionally set batch size to 1 on massive sequence datasets, which causes IPC bottlenecks.

## Phase: Quality Assurance, Regression, and Performance Verification
1. **Target Files:** `tests/test_aligners.py`, `tests/test_domain_finder.py`, `tests/test_dpc_and_spurious.py`
   - **Specific Instructions:**
     - Run the full test suite with deprecation warnings treated as errors: `PYTHONPATH=. python3 -m pytest -W error::DeprecationWarning`.
     - Add regression tests comparing clustering outputs before and after optimizations on synthetic benchmarks with known ground-truth borders.
     - Validate that peak RAM remains strictly below 200 GB and per-worker memory below 200 MB on simulated high-depth datasets.
     - Confirm all pre-commit checks pass cleanly.
