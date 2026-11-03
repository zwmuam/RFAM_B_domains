# 1. Problem Analysis

**Problem 1: Bypassing Minimum Density Filtering in Disjoint DPC (`clustering.py`, `domain_finder.py`)**

- **Module / Component:** `clustering.py` (`dpc`), `domain_finder.py` (`domain_instances_on_query`, `family_finder`)
- **Severity / Category:** CRITICAL | Mathematical & Biological Edge-Case Inconsistency
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:dpc`, when candidate points have no pairwise overlaps or connections (`rows.size == 0`), the function executes an immediate early-exit check:
  `if rows.size == 0: return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0`
  This unconditionally turns every candidate point into an independent cluster peak and reports zero noise points (`n_noise = 0`), completely bypassing the minimum density cutoff (`min_dens`).
  While `domain_finder.py:domain_instances_on_query` introduced a local density filter for disjoint query spans, the core clustering engine `clustering.dpc` was left uncorrected.
  Consequently, when `domain_finder.py:family_finder` clusters domain instances where candidates share zero target overlaps (`rows.size == 0`), every solitary domain instance is unconditionally approved as an authentic domain family peak, even when its alignment support is strictly below $\text{min\_fam\_dens}$.
  Biologically, this undermines the purpose of minimum density filtering: low-confidence alignment noise hits and spurious fragments are preserved rather than filtered out.
- **Evidence in Code:**
  - `clustering.py:871-872`: `if rows.size == 0: return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0`
  - `domain_finder.py:422-423`: `if rows.size == 0 or n_domains <= 1: return {int(i): np.array([i], dtype=np.int64) for i in range(n_domains)}, 0`
  - `domain_finder.py:431-432`: `if u_r.size == 0: return {int(i): np.array([i], dtype=np.int64) for i in range(n_domains)}, 0`
- **Recommended Solution:**
  Enforce uniform density thresholding in `clustering.py:dpc` when `rows.size == 0`. When no non-zero similarity edges exist, each point's local density is determined solely by its diagonal self-similarity (1.0) or its assigned weight: $\rho\_i = 1.0$ (or $w\_i$ if `weights` is provided). Only instances satisfying $\rho\_i \ge \text{min\_dens}$ qualify as peaks. Points failing the cutoff must be classified as noise ($-1$). Similarly, `family_finder` must ensure solitary domains meet $\text{min\_fam\_dens}$.

```python
# FAULTY (clustering.py:871-872 - density threshold bypassed when rows.size == 0):
if rows.size == 0:
    return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0

# REVISED (clustering.py - Uniformly enforce rho >= min_dens across empty edge graphs):
if rows.size == 0:
    densities = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    qualifying = np.flatnonzero(densities >= min_dens)
    if qualifying.size == 0:
        return {}, int(n)
    peak_to_cluster = {int(p): np.array([p], dtype=np.int64) for p in qualifying}
    return peak_to_cluster, int(n - qualifying.size)
```

---

**Problem 3: Unsafe Array Resizing with `np.resize` in Target Hit Buffer (`domain_finder.py`)**

- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 3 Target Hit Aggregation)
- **Severity / Category:** HIGH | Numerical & Algorithmic Bug / Memory Corruption Risk
- **Detailed Bioinformatic & Technical Analysis:**
  During Stage 3 target hits aggregation in `domain_finder.py:find_domains`, the contiguous target buffer is expanded dynamically when `write_offset + n_hits > target_buffer.shape[0]`:
  ```python
  new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
  target_buffer = np.resize(target_buffer, new_size)
  ```
  `np.resize(a, new_shape)` does **not** behave like C `realloc` or Python `list.extend`. When expanding an array, `np.resize` fills the new entries by **cyclically repeating copies of the original array**.
  While the pipeline immediately writes incoming records starting at `write_offset`, if an unexpected exception occurs, if downstream logic accesses unwritten positions beyond `write_offset`, or if slice boundaries are miscalculated, the buffer contains phantom duplicated alignment hits rather than uninitialized or empty entries.
  Furthermore, `np.resize` unconditionally allocates a full copy of the entire buffer and duplicates all elements, triggering unnecessary memory thrashing.
- **Evidence in Code:**
  - `domain_finder.py:613-614`: `target_buffer = np.resize(target_buffer, new_size)`.
- **Recommended Solution:**
  Replace `np.resize` with explicit, safe geometric pre-allocation: allocate `new_buffer = np.empty(new_size, dtype=target_dtype)`, copy strictly the valid prefix `new_buffer[:write_offset] = target_buffer[:write_offset]`, and delete the previous array reference (`del target_buffer`). This prevents cyclic data duplication while preserving $\mathcal{O}(1)$ amortized memory allocation.

```python
# FAULTY (domain_finder.py:613-614 - np.resize cyclically duplicates previous elements):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
    target_buffer = np.resize(target_buffer, new_size)

# REVISED (Explicit, safe geometric reallocation copying only the active prefix):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
    new_buffer = np.empty(new_size, dtype=target_dtype)
    new_buffer[:write_offset] = target_buffer[:write_offset]
    target_buffer = new_buffer
```

---

**Problem 4: Sub-Array View Overhead in `compare_sorted_spans` Range Queries (`clustering.py`)**

- **Module / Component:** `clustering.py` (`compare_sorted_spans`)
- **Severity / Category:** MEDIUM | Performance Bottleneck / Excessive Object Allocations
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:compare_sorted_spans`, pairwise interval overlaps are detected using a binary search range query:
  ```python
  for i in range(n):
      end_i = ends[i]
      r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)
  ```
  `starts[i + 1:]` creates an ephemeral NumPy slice object and array view on every iteration ($N$ allocations per function call). For dense genomic regions or high-depth alignment files with $N = 50,000$ query spans, this triggers tens of thousands of object allocations, increasing Python garbage collection pressure.
  Because the global `starts` array is guaranteed to be sorted in ascending order, `np.searchsorted(starts, end_i, side="left")` can be performed directly on the entire `starts` array in $\mathcal{O}(\log N)$ time, clamping the lower bound to `i + 1`. This eliminates $N$ sub-array allocations while producing identical intersection coordinates.
  In addition, lines 27-29 contain redundant type checks (`if not isinstance(starts, np.ndarray)`), which are dead branches because `starts = spans[:, 0]` is guaranteed to be an `ndarray`.
- **Evidence in Code:**
  - `clustering.py:27-29`: Dead type assertion on `starts`.
  - `clustering.py:36`: `r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)`.
- **Recommended Solution:**
  Run `np.searchsorted` directly on the full `starts` array, bounding the resulting index to $\max(i + 1, r)$. Remove the dead type assertions on lines 27-29.

```python
# FAULTY (clustering.py:36 - generates intermediate slice object per iteration):
for i in range(n):
    end_i = ends[i]
    r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)

# REVISED (clustering.py - Direct binary search on full sorted starts array):
for i in range(n):
    end_i = ends[i]
    r = max(i + 1, int(np.searchsorted(starts, end_i, side="left")))
    if r > i + 1:
        st_js = starts[i + 1:r]
        en_js = ends[i + 1:r]
        overlap = np.minimum(end_i, en_js) - st_js
        valid = overlap > 0
        if np.any(valid):
            js = np.arange(i + 1, r, dtype=np.int32)[valid]
            lows.append(np.full(js.size, i, dtype=np.int32))
            highs.append(js)
            inters.append(overlap[valid].astype(np.int32, copy=False))
```

---

**Problem 5: Inefficient Nested Sequence Grouping and IPC Starvation in Stage 4 (`domain_finder.py`)**

- **Module / Component:** `domain_finder.py` (`find_domains`, `target_batch_processor`), `executor.py` (`Parallel`)
- **Severity / Category:** MEDIUM | Parallel Worker Starvation & IPC Bottlenecks
- **Detailed Bioinformatic & Technical Analysis:**
  In Stage 4 of `domain_finder.py:find_domains`, target sequences are grouped and processed across parallel workers:
  1. `target_batch_generator` packs target sequences into packages of 1,000 sequences and yields them to `Parallel`.
  2. In `executor.py:Parallel` (lines 284-285), generator inputs default to `chunk_size = 32`. Consequently, MPIRE groups $32 \times 1,000 = 32,000$ target sequences into each task chunk sent through inter-process communication (IPC).
  This introduces significant performance bottlenecks:
  - **Worker Starvation:** When a benchmark or dataset contains fewer than 32,000 target sequences (e.g., 5,000 target sequences distributed across 16 CPU cores), all sequences are yielded in fewer than 32 generator items. MPIRE packages all items into a single worker task chunk, leaving 15 CPU cores completely idle while a single core processes the entire workload.
  - **Repeated Boolean Masking:** Inside each worker process, `target_batch_processor` performs an unvectorized loop over `uniq_domains = np.unique(domain_pointers)` for every sequence, applying boolean masks `domain_pointers == d_id` one domain at a time rather than sorting records by `(domain_pointers, start)` and splitting on contiguous boundaries.
- **Evidence in Code:**
  - `domain_finder.py:648-649`: Hard-coded batch threshold `len(batch_pkg) >= 1000`.
  - `executor.py:284-285`: Generator chunk size defaulting to `chunk_size = 32`.
  - `domain_finder.py:355-363`: Unvectorized loop with dynamic boolean masks `domain_pointers == d_id`.
- **Recommended Solution:**
  1. In `domain_finder.py:find_domains`, compute target batch size adaptively based on available CPU worker threads: $\text{batch\_size} = \max(10, \min(250, \text{n\_target\_seqs} // \max(1, \text{cpu} \times 4)))$.
  2. Set `chunk_size = 1` explicitly when invoking `Parallel` on batch generator collections, ensuring smooth worker distribution without idle cores.
  3. In `target_batch_processor`, sort target spans by `domain_pointers` and use boundary indexing (`np.flatnonzero(np.diff(s_doms)) + 1`) to group domain intervals contiguously.

```python
# FAULTY (domain_finder.py:648-656 - Coarse batch size combined with generator chunking):
# domain_finder.py:
if len(batch_pkg) >= 1000:
    yield batch_pkg
# executor.py:
if chunk_size == 'auto':
    chunk_size = 32

# REVISED (Adaptive workstation batch sizing ensuring uniform multi-core utilization):
target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))

def target_batch_generator() -> Generator[List[Tuple[int, np.ndarray, np.ndarray]], None, None]:
    batch_pkg: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for grp in target_seq_groups:
        t_seq = int(grp['t_seq'][0])
        spans = np.column_stack((grp['start'], grp['end']))
        dom_ptrs = grp['dom_id']
        batch_pkg.append((t_seq, spans, dom_ptrs))
        if len(batch_pkg) >= target_batch_size:
            yield batch_pkg
            batch_pkg = []
    if batch_pkg:
        yield batch_pkg

target_workers = Parallel(function=target_batch_processor,
                          input_collection=target_batch_generator(),
                          n_jobs=cpu,
                          bar=True,
                          description="Calculating target overlaps",
                          chunk_size=1)
```

---

**Problem 6: Contradictory Data Typing Claims and Outdated Docstrings (`domain_finder.py`, `clustering.py`)**

- **Module / Component:** `domain_finder.py` (`target_batch_processor`, `evaluate_clusters`), `clustering.py` (`dpc`)
- **Severity / Category:** MEDIUM | Data Typing Inconsistency & Documentation Drift
- **Detailed Bioinformatic & Technical Analysis:**
  Previous update summaries claimed that coordinates and interval arrays were strictly standardized to 32-bit integers (`int32`) across `domain_finder.py` and `clustering.py`.
  Auditing the current codebase reveals multiple unresolved inconsistencies:
  1. `domain_finder.py:target_batch_processor` docstring (line 342) states:
     `Maintains 64-bit integers for all physical genomic coordinate operations.`
     This directly contradicts the internal implementation, which standardizes coordinates to `np.int32`.
  2. `domain_finder.py:evaluate_clusters` (line 500) still explicitly casts interval spans to `np.int64`:
     `spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int64)`
  3. `clustering.py:dpc` (lines 881-882) unconditionally converts input `rows` and `cols` from `np.int32` to `np.int64`:
     `rows = np.ascontiguousarray(rows, dtype=np.int64)`
     `cols = np.ascontiguousarray(cols, dtype=np.int64)`
     This doubles the memory footprint of graph coordinates from 4 bytes to 8 bytes per coordinate across millions of edges.
  4. Dead import: `count_fasta` is imported from `aligners` in `domain_finder.py:31` but is never referenced anywhere in `domain_finder.py`.
- **Evidence in Code:**
  - `domain_finder.py:31`: Unused import `count_fasta`.
  - `domain_finder.py:342`: Outdated docstring claiming 64-bit coordinate operations.
  - `domain_finder.py:500`: Redundant `dtype=np.int64` in `spans_arr`.
  - `clustering.py:881-882`: Unnecessary upcasting of `rows` and `cols` to `np.int64`.
- **Recommended Solution:**
  Standardize coordinate and sparse graph edge arrays to `np.int32` throughout `evaluate_clusters` and `clustering.py:dpc`. Reserve `np.int64` strictly for 64-bit packed edge keys (`(u_r << 32) | u_c`) and global accumulated nucleotide lengths (`global_dom_lengths`). Remove the unused `count_fasta` import and update the docstrings to reflect 32-bit coordinate representations.

```python
# FAULTY (domain_finder.py:500 & clustering.py:881-882 - Redundant int64 upcasting):
# domain_finder.py:
spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int64)
# clustering.py:
rows = np.ascontiguousarray(rows, dtype=np.int64)
cols = np.ascontiguousarray(cols, dtype=np.int64)

# REVISED (Consistent int32 typing for coordinates and graph indices):
# domain_finder.py:
spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int32)
# clustering.py:
rows = np.ascontiguousarray(rows, dtype=np.int32)
cols = np.ascontiguousarray(cols, dtype=np.int32)
```

---

## 2. Architectural Risks and Warning Guidelines

The following risks should be considered prior to code refactoring:

1. **AVOID Euclidean Halo Filtering - it does not work with Structured ncRNA :**
   Traditional Rodriguez-Laio halo filtering assumes Gaussian clusters with dense spherical centers. Structured ncRNA domains display non-uniform coverage and ragged alignment boundaries. Applying traditional halo filtering can aggressively discard legitimate biological variants and multimeric duplications. The `similarity_weighted` halo method with `point_specific=True` must remain the default.

2. **Multiprocessing Context Under Linux (`forkserver` vs `spawn`):**
   The monkey-patch in `executor.py` redirects MPIRE's `SyncManager` to `forkserver`. However, if `forkserver` is initialized after background threads have been spawned, worker processes can still inherit corrupted lock states. Any synchronization manager initialization must strictly precede worker pool dispatch.

3. **Peak Memory Ceiling on Multi-Gigabyte Alignments:**
   For multi-GB alignment files ($> 10^7$ hits), sorting `all_targets_arr` in-place requires up to 32 bytes per hit. For 100 million hits, this requires 3.2 GB of RAM. The pipeline must avoid converting structured arrays into intermediate Python dictionaries or object lists. Additionally accumulation of target hits still requires immense chunk of RAM. If any memory optimisations are available try to include them at this stage.

---

## 3. Implementation Plan

**Phase 1: Algorithmic & Biological Correctness**
   - Correct disjoint density thresholding in `clustering.py:dpc` so that candidate instances failing $\text{min\_dens}$ are assigned to noise ($-1$).
   - Replace `np.resize` in `domain_finder.py:find_domains` with safe geometric buffer reallocation.

**Phase 2: Performance & Concurrency Optimization**
   - Optimize `compare_sorted_spans` in `clustering.py` by performing binary search directly on the full `starts` array without slice allocations.
   - Configure adaptive target batch sizing in `domain_finder.py:find_domains` and set `chunk_size = 1` for generator inputs in `Parallel`.
   - Refactor `target_batch_processor` to group domain intervals by sorting `domain_pointers` and splitting on contiguous boundaries.

**Phase 3: Coordinate Type Harmonization & Cleanup**
   - Standardize all coordinates, spans, and local index arrays to `np.int32` in `domain_finder.py:evaluate_clusters` and `clustering.py:dpc`.
   - Remove unused import `count_fasta` from `domain_finder.py`.
   - Correct outdated docstrings in `target_batch_processor`.

**Phase 4: Test Verification**
   - Run `python3 -m pytest -v -W error::DeprecationWarning` to ensure zero regressions across the complete test suite.
