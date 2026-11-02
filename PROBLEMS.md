# Domain Finding Pipeline Codebase Audit & Problem Analysis

---

## 1. Problem Analysis

**Problem 1: Unsafe Fork Trigger in MPIRE Progress Bar SyncManager on Python 3.12+ (`executor.py`)**

- **Module / Component:** `executor.py` (`Parallel.__init__`), `mpire` dependency interaction
- **Severity / Category:** CRITICAL | Concurrency / Subprocess Deadlock / Deprecation Warning
- **Detailed Bioinformatic & Technical Analysis:**
  The repository aims to eliminate unsafe `os.fork()` on Linux under Python 3.12+ by passing `start_method='forkserver'` (or `'spawn'`) to `mpire.WorkerPool`. However, when `bar=True` is enabled (the default across `domain_finder.py`), MPIRE invokes `TqdmManager.start_manager()`.
  Under the hood, `TqdmManager` creates a standard Python `multiprocessing.managers.SyncManager`. In Python's standard library, `SyncManager` defaults to the global default multiprocessing context, which remains `'fork'` on Linux unless explicitly configured.
  Consequently, executing `Parallel(..., bar=True)` triggers:
  `DeprecationWarning: This process (pid=...) is multi-threaded, use of fork() may lead to deadlocks in the child. self.pid = os.fork()`
  When a multi-threaded parent process (such as a runner executing background I/O, Loguru logging, or Pytest runners) invokes `os.fork()`, mutex locks in C-extensions and file descriptors can be cloned in an acquired state, leading to child process deadlocks.
- **Evidence in Code:**
  Executing `pytest` triggers 2 `DeprecationWarning`s originating from `multiprocessing/popen_fork.py:66` when `SyncManager` is started during Stage 2 and Stage 4 progress tracking.
- **Recommended Solution:**
  Explicitly initialize multiprocessing contexts using `mp.get_context('forkserver')` and ensure `mpire.tqdm_utils.SyncManager` or the multiprocessing manager uses the safe non-fork context, or inject a forkserver-bound manager before invoking MPIRE mapping.

```python
# FAULTY (Current executor.py:272 - WorkerPool receives forkserver, but internal SyncManager still forks):
worker_kwargs['start_method'] = 'forkserver' if 'forkserver' in avail else 'spawn'

# REVISED (Explicitly configure manager context to forkserver/spawn to prevent SyncManager forks):
import multiprocessing as mp
ctx_name = 'forkserver' if 'forkserver' in mp.get_all_start_methods() else 'spawn'
worker_kwargs['start_method'] = ctx_name
if hasattr(mp, 'set_forkserver_preload'):
    try:
        mp.set_forkserver_preload(['numpy', 'scipy', 'mpire'])
    except Exception:
        pass
```

---

**Problem 2: Bypassing Minimum Density and Noise Classification for Single or Disjoint Spans (`domain_finder.py`)**

- **Module / Component:** `domain_finder.py` (`domain_instances_on_query`)
- **Severity / Category:** HIGH | Mathematical Inconsistency / Biological Edge Case
- **Detailed Bioinformatic & Technical Analysis:**
  In `domain_finder.py:domain_instances_on_query`, sequence alignment spans on a query sequence are grouped into unique spans with hit count weights:
  ```python
  unique_query_spans, cluster_map, weights = np.unique(sorted_query_spans, axis=0, return_inverse=True, return_counts=True)
  ```
  If `n_spans == 1`, the code sets:
  ```python
  cluster_map[:] = 0
  peak_indices = np.array([0], dtype=np.int32)
  ```
  Similarly, if `n_spans > 1` but `rows.size == 0` (no two spans share any overlapping nucleotides):
  ```python
  peak_indices = np.arange(n_spans, dtype=np.int32)
  ```
  In both scenarios, Density Peak Clustering (`dpc`) is entirely bypassed. This creates a severe biological discrepancy:
  If a user configures `min_dom_dens = 5.0` (requiring at least 5 supporting alignment hits to validate a conserved biological domain candidate), a solitary span supported by only 1 hit (`weights[0] == 1`) bypasses `dpc` and is automatically accepted as a validated domain instance! Solitary, low-evidence noise hits thus escape biological density thresholds.
- **Evidence in Code:**
  `domain_finder.py:171-177`: Unconditional assignment of `cluster_map[:] = 0` and `peak_indices = np.arange(n_spans)` without checking whether `weights >= min_dom_dens`.
- **Recommended Solution:**
  Verify that unique spans satisfy `weights >= min_dom_dens` before accepting them as domain instances. Spans failing the density threshold must be classified as noise (`-1`).

```python
# FAULTY (domain_finder.py:171-177 - bypasses density filtering when no intersections exist):
if n_spans == 1:
    cluster_map[:] = 0
    peak_indices = np.array([0], dtype=np.int32)
else:
    rows, cols, intersections = compare_sorted_spans(unique_query_spans)
    if rows.size == 0:
        peak_indices = np.arange(n_spans, dtype=np.int32)

# REVISED (Enforce density thresholds uniformly across all hit topologies):
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
```

---

**Problem 4: Coordinate Data Type Inconsistency Between Modules (`domain_finder.py` vs `aligners.py`)**

- **Module / Component:** `domain_finder.py` (`target_batch_processor`, `find_domains`), `aligners.py`
- **Severity / Category:** MEDIUM | Numerical Precision / Memory Bloat
- **Detailed Bioinformatic & Technical Analysis:**
  `PREVIOUS_UPDATE.md` claimed that all physical genomic coordinates were standardized to `np.int32` to halve memory and fit four records into a 64-byte CPU cache line.
  However, in `domain_finder.py:target_batch_processor`:
  ```python
  merged_spans = np.asarray(merged_spans_list, dtype=np.int64)
  ...
  batch_data.append(intersections.astype(np.int64, copy=False))
  ```
  The function explicitly casts `merged_spans` and `intersections` to `np.int64`. Then, in `find_domains`:
  ```python
  all_inters = np.concatenate(accumulated_inters) if accumulated_inters else np.empty(0, dtype=np.int32)
  ```
  If `accumulated_inters` is non-empty, `all_inters` becomes `int64`. But if it is empty, it initializes as `np.int32`.
  While 64-bit integers prevent potential overflow when accumulating large multi-megabase overlaps, physical RNA sequence coordinates in transcriptomic datasets never exceed millions of bases. Casting back and forth creates type confusion and increases cache line misses.
  On the other hand, global domain index can reach the billions of items so `np.int64` may be required to safely handle the registry.

- **Recommended Solution:**
  Standardize `intersections` and `spans` to `np.int32` throughout all local processing stages, reserving `np.int64` strictly for 64-bit packed composite edge keys (`(u_r << 32) | u_c`) and global genome-wide summation accumulators (`global_dom_lengths`) AND GLOBAL DOMAIN INDEXING (needs update).


  1. **Claim of Zero Warnings on Python 3.12:**
     `PREVIOUS_UPDATE.md:534, 558` claimed that child worker deadlocks and `DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks` were completely eliminated.
     *Reality:* Running `python3 -m pytest` on Python 3.12.13 emits 2 `DeprecationWarning`s because `mpire`'s `TqdmManager` internally starts a `SyncManager` using `os.fork()`.
  2. **Claim of Uniform int32 Coordinates:**
     `PREVIOUS_UPDATE.md:282, 381` claimed physical coordinates were converted to `int32` throughout.
     *Reality:* `domain_finder.py:378` and `domain_finder.py:388` still explicitly cast `merged_spans` and `intersections` to `np.int64`. Conversely, global doamain index risks overflowing during massive multi-GB analyses.
- **Recommended Solution:**
  Sort out final range requirements and ensure no DeprecationWarning occurs.

---

**Problem 8: Redundant FASTA Header Counting and I/O Pass (`domain_finder.py`, `aligners.py`)**

- **Module / Component:** `domain_finder.py` (`count_fasta_queries`), `aligners.py` (`count_fasta`)
- **Severity / Category:** LOW | Redundant Computation / Code Cleanliness
- **Detailed Bioinformatic & Technical Analysis:**
  `domain_finder.py` defines `count_fasta_queries(fasta_path: Path) -> int`, which is an alias for `aligners.count_fasta(fasta_path)`.
  However, in `domain_finder.find_domains`:
  `sequence_registry = populate_registry_from_fasta(fasta_path)` is called, which already scans the entire FASTA file and counts headers to populate the registry. Calling `count_fasta_queries` separately is completely redundant.
  Furthermore, `count_fasta_queries` is no longer used in `find_domains` because buffer pre-sizing now relies on `tsv_file.stat().st_size // 75`. The function is unused dead code in `domain_finder.py`.
- **Evidence in Code:**
  `domain_finder.py:40-47`: Unused wrapper function `count_fasta_queries`.
- **Recommended Solution:**
  Remove `count_fasta_queries` from `domain_finder.py` and rely directly on `len(sequence_registry)` or `aligners.count_fasta` where needed.


---

## 3. Phased Implementation Architecture (For Future Agent Run)

This section provides a clear, sequential implementation guide for a future code revision run.

---

**Phase 1: Concurrency Safety and Subprocess Hardening**

**Objective:** Ensure that subprocess can be spawned seamlessly and their lifecycle ensures optimal CPU thread utilisation on linux Ubuntu machine with up to 36 threads.

1. **Context-Bound SyncManager Initialization:**
   In `executor.py`, before launching `WorkerPool(..., progress_bar=True)`, configure `mpire.tqdm_utils.SyncManager` to use `mp.get_context('forkserver')` instead of the default `'fork'`.
2. **Explicit Process Tree Validation:**
   Ensure worker processes spawned by `WorkerPool` terminate cleanly upon task completion, with zero orphan processes and zero warnings under `pytest -W error::DeprecationWarning`.

---

**Phase 2: Biological Density Enforcement and Algorithmic Precision**

**Objective:** Correct edge-case handling in DPC and domain candidate pruning to maintain strict biological fidelity.

1. **Uniform Minimum Density Filtering:**
   In `domain_finder.py:domain_instances_on_query`:
   - If `n_spans == 1`, check `weights[0] >= min_dom_dens`. If not, assign `cluster_map[:] = -1` and return empty local domains.
   - If `n_spans > 1` and `rows.size == 0`, filter unique spans by `weights >= min_dom_dens`, assigning failing spans to `-1`.

---

**Phase 3: Memory Streamlining and Coordinate Type Harmonization**

**Objective:** Eliminate redundant memory conversions and optimize Stage 3 hit buffer allocation.

1. **Strict 32-bit Coordinate Typing:**
   In `domain_finder.py:target_batch_processor`:
   - Cast `merged_spans` to `np.int32`.
   - Cast `intersections` to `np.int32`.
   - Ensure `accumulated_inters` in `find_domains` initializes as `np.int32`.
   - the global domain index is exempt and should be protected from overflowing by use of NymPy 64 bit integer.
2. **Robust Hit Buffer Capacity Sizing:**
   In `domain_finder.py:find_domains`, use `est_capacity = max(100_000, int(tsv_file.stat().st_size // 50))` to prevent memory reallocations even when sequence identifiers are long.
3. **Dead Code Cleanup:**
   Remove `count_fasta_queries` from `domain_finder.py` and replace any external references with `len(sequence_registry)` or `aligners.count_fasta`.

---

**Phase 4: Test Suite Hardening and Warning Elimination**


2. **Pytest Verification:**
   Run `python3 -m pytest -W error` across the entire repository to ensure:
   - 0 test failures
   - 0 deprecation warnings
   - 0 resource warnings
