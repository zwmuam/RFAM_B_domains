# Abstract
The pipeline extracts conserved structural RNA domain instances from pairwise alignments (MMseqs2 or RiboSeek) and clusters them into global domain families via Density Peak Clustering (DPC). Processing large hundreds-gigabyte alignment datasets imposes strict operational requirements:

1. Peak system memory must stay bounded strictly below 200 GB.
2. Parallel worker memory overhead must remain strictly below 500 MB per worker process.
3. Multi-core execution must scale smoothly up to 30+ threads without worker starvation or inter-process communication bottlenecks.
4. Non-overlapping or 0-similarity pairs must never link domain instances.

---

# Evaluation of Previous Update Claims

## Worker Result Memory Retained Prior to Pre-allocation in Stage 4

- **Evidence**:
  `PREVIOUS_UPDATE.md` claimed to eliminate duplicate peak memory in Stage 4 by summing edge counts, pre-allocating contiguous coordinate arrays (`all_rows`, `all_cols`, `all_inters`), and popping items via `results_list.pop(0)`.
  In `domain_finder.py:627-646`:
  ```python
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
      ...
  ```
- **Impact**:
  In `executor.py:301`, `target_workers` collects all completed worker returns in `pool.map(...)` into `self.result` (and `self.results`). When `find_domains` accesses `target_workers.results`, all worker arrays are already resident in memory simultaneously.
  Pre-allocating `all_rows`, `all_cols`, and `all_inters` before popping duplicates the memory footprint ($2 \times 6.0\text{ GB} = 12.0\text{ GB}$ for $5 \times 10^8$ overlaps) at the moment of allocation. True peak memory mitigation requires either streaming worker returns or writing directly into shared or disk-backed buffers during worker execution.
- **Proposed Fix**:
  Stream worker outputs lazily or yield results chunk-by-chunk using `pool.imap_unordered` rather than eager `pool.map`. If using list popping, write coordinates incrementally into memory-mapped or dynamically grown buffers without retaining the full completed worker result list.

```python
# FAULTY (domain_finder.py:627-635 - Pre-allocates destination buffers while holding full worker results list):
results_list = target_workers.results
total_edges = 0
for r, _, _, _ in results_list:
    total_edges += r.size

all_rows = np.empty(total_edges, dtype=np.int32)
all_cols = np.empty(total_edges, dtype=np.int32)
all_inters = np.empty(total_edges, dtype=np.int32)

# REVISED (domain_finder.py - Stream worker outputs to avoid holding both collections simultaneously):
all_rows_list = []
all_cols_list = []
all_inters_list = []
global_dom_lengths = np.zeros(n_total_domains, dtype=np.int64)

for r, c, inter, dom_lens_dict in target_workers.imap_unordered():
    if r.size > 0:
        all_rows_list.append(r)
        all_cols_list.append(c)
        all_inters_list.append(inter)
    for d_id, d_len in dom_lens_dict.items():
        if d_id < n_total_domains:
            global_dom_lengths[d_id] += d_len
```

- **Validation**:
  1. Profile memory during Stage 4 aggregation under synthetic 100M edge loads.
  2. Confirm peak resident set size (RSS) does not exhibit a $2\times$ memory step during pre-allocation.

---

# Scientific, Mathematical, and Algorithmic Vulnerabilities

## Arbitrary Nearest-Higher Tie-Breaking in Sparse Matrix Traversal

- **Evidence**:
  In `clustering.py:270-279` (`_nearest_higher_sparse`):
  ```python
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
  ```
- **Impact**:
  When multiple higher-density neighbors have identical maximal similarity values with node $i$, `chosen` selects the first matching entry in CSR column order (`indices`), which corresponds to the smallest node ID.
  Under standard DPC (Rodriguez & Laio 2014), ties should be broken by higher density rank (highest $\rho$), or deterministically by density rank rather than arbitrary CSR storage order. In dense evaluation (`_nearest_higher_dense`), `argmax(axis=1)` similarly selects the lowest column index. This creates subtle topological divergence between sparse and dense backends if matrix columns are reordered.
- **Proposed Fix**:
  Encode density rank as a tie-breaker when selecting the best higher-density neighbor.

```python
# FAULTY (clustering.py:270 - Arbitrary first match in CSR column order):
hit = (candidate > 0.0) & (candidate == best_val[row_of])

# REVISED (clustering.py - Tie-break by highest density / lowest rank index):
# Composite priority score: similarity as primary, (n - rank) as fractional tie-breaker
priority = np.where(candidate > 0.0, candidate + (n - rank[indices]) / (n * 1e9), -1.0)
best_prio = np.full(n, -1.0, dtype=np.float64)
if np.any(nonempty):
    best_prio[nonempty] = np.maximum.reduceat(priority, indptr[:-1][nonempty])
hit = (priority > 0.0) & (priority == best_prio[row_of])
```

- **Validation**:
  1. Construct a triangular similarity graph where node 0 has identical similarity (0.8) to node 1 ($\rho=5$) and node 2 ($\rho=10$).
  2. Verify node 0 deterministically selects node 2 ($\rho=10$) under both sparse and dense backends.

---

## Legacy Memoryview Buffer Extraction in Array Arithmetic

- **Evidence**:
  In `clustering.py:48-50` and `70-71`:
  ```python
  unions = row_lengths + column_lengths - intersections.data
  ji = intersections.data / unions
  ...
  containment = intersections.data / smaller
  ```
- **Impact**:
  In `domain_finder.py:164` and `filter_spurious_domains`, `intersections` is a 1D contiguous NumPy array (`np.ndarray`), not a SciPy sparse matrix. Accessing `.data` on a NumPy array returns a Python `memoryview` object.
  While NumPy overloads memoryview operations in simple 1D expressions, passing non-contiguous array slices, custom subclasses, or memoryviews into downstream routines creates subtle typing bugs, unpicklable references, and bypasses NumPy array dispatching.
- **Proposed Fix**:
  Accept array-like inputs and extract `.data` only if `isinstance(intersections, sp.spmatrix)`, or operate directly on the NumPy array.

```python
# FAULTY (clustering.py:48 - Accesses .data memoryview on 1D NumPy arrays):
unions = row_lengths + column_lengths - intersections.data
ji = intersections.data / unions

# REVISED (clustering.py - Robust array-like handling):
inter_data = intersections.data if hasattr(intersections, "data") and not isinstance(intersections, np.ndarray) else np.asarray(intersections)
unions = row_lengths + column_lengths - inter_data
ji = inter_data / unions
```

- **Validation**:
  1. Test `jaccard_index` and `max_containment` with standard NumPy arrays, array slices, and SciPy CSR matrices.
  2. Verify identical numerical output and lack of memoryview objects.

---

# Performance Bottlenecks and Avoidable Bloat

## Process Pool Re-spawning in Stage 2 Query Streaming

- **Evidence**:
  In `domain_finder.py:545-572`:
  ```python
  with tqdm(desc="Finding domain instances", unit="batch") as pbar:
      for batch_window in stream_query_batches(batch_gen, chunk_window=50):
          query_workers = Parallel(function=batch_query_processor,
                                   input_collection=batch_window,
                                   kwargs=query_kwargs,
                                   n_jobs=cpu,
                                   bar=False,
                                   chunk_size=1)

          for batch_res in query_workers.results:
              ...
          pbar.update(len(batch_window))
          del query_workers
  ```
- **Impact**:
  For an alignment dataset with 5,000 batches (5,000,000 queries), `stream_query_batches` loops 100 times (`chunk_window = 50`). In each iteration, `Parallel` instantiates a brand new `mpire.WorkerPool`, spawning and terminating `cpu` (e.g. 32) worker processes 100 consecutive times (3,200 process spawns and IPC teardowns).
  Process initialization in Python (especially under `forkserver` or `spawn`) incurs substantial CPU and wall-clock overhead, causing worker launch latency and memory churn.
- **Proposed Fix**:
  Instantiate a single persistent `WorkerPool` outside the loop, streaming batches through the long-lived pool.

```python
# FAULTY (domain_finder.py:546-552 - Spawns and destroys WorkerPool every 50 batches):
for batch_window in stream_query_batches(batch_gen, chunk_window=50):
    query_workers = Parallel(...)

# REVISED (domain_finder.py - Persistent worker pool streaming all batches):
with WorkerPool(n_jobs=cpu, start_method='forkserver') as pool:
    for batch_res in pool.imap_unordered(batch_query_processor, batch_gen):
        ...
```

- **Validation**:
  1. Benchmark total runtime on a 500k-hit TSV dataset with 32 worker threads.
  2. Measure execution time reduction from eliminating repeated worker pool re-initializations.

---

## Memory Expansion During Geometric Target Buffer Reallocation

- **Evidence**:
  In `domain_finder.py:558-566`:
  ```python
  if write_offset + n_hits > target_buffer.shape[0]:
      new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
      new_buffer = np.empty(new_size, dtype=target_dtype)
      new_buffer[:write_offset] = target_buffer[:write_offset]
      target_buffer = new_buffer
  ```
- **Impact**:
  When `target_buffer` reaches 20 GB ($1.25 \times 10^9$ hits), expanding it geometrically requires allocating a new 40 GB buffer while the 20 GB buffer is still retained, causing a 60 GB instantaneous memory peak.
  Furthermore, `est_capacity = max(100_000, int(tsv_file.stat().st_size // 75))` uses a hardcoded 75 bytes/line estimate. When TSV formatting includes long sequence IDs, lines exceed 120 bytes, causing `est_capacity` to overestimate or underestimate significantly.
- **Proposed Fix**:
  Accurately sample the first 10,000 lines to determine the true mean line length, and allocate memory-mapped storage or chunked pre-allocated blocks to eliminate reallocation copy spikes.

```python
# FAULTY (domain_finder.py:560-565 - Monolithic doubling creates 3x transient memory footprint):
new_buffer = np.empty(new_size, dtype=target_dtype)
new_buffer[:write_offset] = target_buffer[:write_offset]
target_buffer = new_buffer

# REVISED (domain_finder.py - Block-list allocation avoiding reallocation spikes):
target_blocks = [np.empty(block_size, dtype=target_dtype)]
# Store slices across fixed-size blocks, avoiding duplicate memory copies
```

- **Validation**:
  1. Track peak memory under synthetic multi-gigabyte inputs.
  2. Verify total RAM usage never exceeds $1.1\times$ final buffer size.

---

# Data Integrity, Concurrency, and Programming Practices

## Deprecated Fork Start Method Deadlock Vulnerability

- **Evidence**:
  In `executor.py:35-46`:
  ```python
  _avail_start_methods = mp.get_all_start_methods()
  _safe_ctx_name = 'forkserver' if 'forkserver' in _avail_start_methods else 'spawn'
  _safe_ctx = mp.get_context(_safe_ctx_name)
  ```
  `Parallel` attempts to default to `'forkserver'` or `'spawn'`.
  However, in multi-threaded environments or nested worker loops where workers call external compiled libraries (e.g. OpenBLAS, MKL, or SciPy), starting child processes via unmanaged contexts can hang or trigger deadlocks.
- **Impact**:
  When running under interactive scripts or without proper module-level pickling guards (`if __name__ == '__main__':`), `forkserver` or `spawn` crashes with unpicklable target functions or hangs indefinitely waiting on IPC sockets.
- **Proposed Fix**:
  Enforce explicit multiprocessing start-method validation, ensuring worker target callables are module-level functions with clear fallback error messages.

```python
# REVISED (executor.py - Robust start method initialization with explicit validation):
if mp.get_start_method(allow_none=True) != _safe_ctx_name:
    try:
        mp.set_start_method(_safe_ctx_name, force=True)
    except RuntimeError:
        pass
```

- **Validation**:
  1. Run unit test suite under Python 3.12+ with `-W error::DeprecationWarning`.
  2. Confirm zero deprecation warnings and zero deadlocks.

---

## Asymmetric Point-Specific Parameter Exposure

- **Evidence**:
  In `domain_finder.py:find_domains`:
  ```python
  point_specific_val = bool(attributes.get("point_specific", False))
  ```
  `point_specific_val` is passed uniformly to both Stage 2 `batch_query_processor` and Stage 5 `family_finder`.
  In `domain_finder_precomputed.py`:
  ```python
  def find_domains(..., point_specific: bool = False, ...):
      attributes = { ..., "point_specific": point_specific }
  ```
- **Impact**:
  Stage 2 operates on query alignment spans (identifying domain instances on single RNA sequences), whereas Stage 5 operates on target overlaps (grouping domain instances across genomes into families).
  Biologically, query spans often have ragged ends requiring point-specific halo filtering, whereas family clustering on Jaccard graphs may benefit from traditional halo filtering. Users cannot configure Stage 2 and Stage 5 independently.
- **Proposed Fix**:
  Support independent controls `point_specific_spans` and `point_specific_families`, falling back to `point_specific` when individual flags are omitted.

```python
# FAULTY (domain_finder.py:512 - Single parameter forces identical mode in both stages):
point_specific_val = bool(attributes.get("point_specific", False))

# REVISED (domain_finder.py - Decoupled stage-specific parameters with backward fallback):
ps_spans = bool(attributes.get("point_specific_spans", attributes.get("point_specific", False)))
ps_families = bool(attributes.get("point_specific_families", attributes.get("point_specific", False)))
```

- **Validation**:
  1. Test running with `point_specific_spans=True` and `point_specific_families=False`.
  2. Verify Stage 2 uses point-specific filtering while Stage 5 uses cluster-wide halo filtering.

---

# Phased Implementation Plan for Future Revisions

This implementation plan provides direct, step-by-step instructions for future engineering runs. Each phase specifies exact file modifications, coding conventions, anti-patterns to avoid, and risk controls.

## Phase: Correcting Inequality Inconsistencies and Ties in Clustering
1. **Target File:** `clustering.py`
   - **Target Functions:** `dpc` and `_nearest_higher_sparse`
   - **Specific Instructions:**
     - In `_nearest_higher_sparse`, introduce density rank as a secondary tie-breaker when higher-density neighbors have identical similarity scores.
     - In `jaccard_index` and `max_containment`, eliminate the `.data` access on NumPy arrays.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not break backward compatibility with unit tests where `min_sim` is strictly within $(0, 1)$.
     - Ensure sparse matrix indexing remains `np.int32` unless total entries exceed $2^{31} - 1$.

## Phase: Streaming Target Overlap Results in Stage 4
1. **Target File:** `domain_finder.py`
   - **Target Function:** `find_domains`
   - **Specific Instructions:**
     - Refactor `target_workers` aggregation to consume worker results incrementally or stream them directly into destination structures.
     - Eliminate the simultaneous retention of `results_list` and pre-allocated destination arrays.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not use `np.concatenate` on large worker result lists, as it forces concurrent duplicate memory allocation.
     - Ensure `dom_lens_dict` updates to `global_dom_lengths` remain numerically identical.

## Phase: Aligners Ingestion Acceleration and Temp File Sanitization
1. **Target File:** `aligners.py`
   - **Target Functions:** `parse_mmseqs_tsv`, `riboseek_easy_search`, `mmseq_easy_search`
   - **Specific Instructions:**
     - Accelerate TSV parsing using chunked tabular reading (`pd.read_csv` with C engine) and vectorized NumPy coordinate validation.
     - Fix directory naming in `mmseq_easy_search` where temporary directories are mistakenly labeled `.riboseek.tmp`.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not alter the 0-based half-open $[s-1, e)$ coordinate output contract expected by `domain_finder.py`.
     - Enforce error raising on malformed TSV inputs without silent skipping.

## Phase: Multi-Core Saturation and Process Lifecycle Optimization
1. **Target Files:** `domain_finder.py` and `executor.py`
   - **Target Functions:** `find_domains` and `Parallel`
   - **Specific Instructions:**
     - Adjust `target_batch_size` in Stage 4 to set `target_batch_size = 1` when $n_{\text{target\_seqs}} \le \text{cpu} \times 2$, ensuring all worker threads receive tasks.
     - Move `WorkerPool` creation in Stage 2 outside the batch streaming loop to avoid spawning and killing worker processes every 50 batches.
     - Decouple `point_specific` configuration into `point_specific_spans` and `point_specific_families`.
   - **Anti-Patterns & Risks to Avoid:**
     - Avoid using `os.fork` directly on Linux under Python 3.12+ to prevent deadlocks with compiled threading runtimes.
