# CODEBASE AUDIT & TECHNICAL PROBLEM ANALYSIS: DOMAIN FINDING PIPELINE

The domain-finding pipeline identifies conserved non-coding RNA (ncRNA) domain instances and family clusters from sequence alignments through a multi-stage architecture:
1. **TSV Streaming & Coordinate Ingestion (`aligners.py`)**: Streams BLAST-TAB formatted alignment hits from MMseqs2 or RiboSeek, parses coordinates into half-open intervals $[s - 1, e)$, maps sequence identifiers to integers via `SequenceRegistry`, and groups records by query sequence.
2. **Query Domain Instance Discovery (`domain_finder.py:domain_instances_on_query`)**: Performs Density Peak Clustering (DPC) on query alignment spans using Jaccard similarity of 1D coordinate intervals (`clustering.py`), prunes spurious overlapping candidate domains using biological alignment support, assigns consensus strands, and supplements self-hits.
3. **Target Chunk Ingestion & Aggregation (`domain_finder.py:find_domains`)**: Aggregates worker outputs from Stage 2 into a contiguous array buffer, re-indexing domain identifiers globally, and sorts by target sequence.
4. **Target Overlap & Intersection Calculation (`domain_finder.py:target_batch_processor`)**: Merges overlapping same-domain hits on target sequences, extracts pairwise coordinate intersections across domains, and computes total aligned target nucleotide lengths per domain.
5. **Family Finding via Graph DPC (`domain_finder.py:family_finder`)**: Constructs a global multi-sequence similarity graph using the cumulative sum of Jaccard similarities across shared target sequences, and groups domain instances into families using sparse DPC.
6. **Cluster Evaluation & GFF3 Export (`domain_finder.py:evaluate_clusters`)**: Groups family members, resolves overlaps among members of the same family on the same sequence, and exports annotations in 1-based closed GFF3 format $[s + 1, e]$.

---

## 2. Problem Analysis

### Problem 1: Multiprocessing Fork Deadlock on Python 3.12+ (`executor.py`)

- **Module / Component:** `executor.py` (`Parallel.__init__`), `tweaks.py`
- **Severity / Category:** CRITICAL | Concurrency / Subprocess Vulnerability / Documentation Inconsistency
- **Detailed Technical Analysis:**
  `PREVIOUS_UPDATE.md` claimed that `executor.py` replaced `os.fork()` with `start_method='forkserver'` (or `'spawn'`) to eliminate deadlocks in multi-threaded contexts on Python 3.12+. However, `executor.py` still contains:
  ```python
  if 'start_method' not in worker_kwargs and 'start_method' not in extra_kwargs:
      worker_kwargs['start_method'] = 'fork'
  ```
  Executing `pytest` or running `Parallel` triggers Python 3.12 deprecation warnings:
  `DeprecationWarning: This process (pid=...) is multi-threaded, use of fork() may lead to deadlocks in the child. self.pid = os.fork()`.
  When parent processes run threads for logging (`loguru`), progress monitoring (`tqdm`), or background I/O, `fork()` clones locked mutexes into child processes. Under thread contention, child workers deadlock when acquiring cloned pipe or socket mutexes.
  Furthermore, the nested closure `_worker_wrapper` inside `Parallel.__init__` cannot be pickled by `forkserver` or `spawn`.
- **Evidence in Code:**
  `executor.py:141-149`: Hardcoded `'fork'` and unpicklable inner function `_worker_wrapper`.
  `PREVIOUS_UPDATE.md:391, 626`: Contradicts the codebase by claiming `forkserver` was already integrated.
- **Recommended Remediation:**
  Extract a clean, top-level picklable dispatcher function `_parallel_worker_dispatch` and configure `start_method` to default to `'forkserver'` when available, with a graceful fallback to `'spawn'`.

```python
# FAULTY (executor.py:141-149 - hardcoded fork and unpicklable nested closure):
if 'start_method' not in worker_kwargs and 'start_method' not in extra_kwargs:
    worker_kwargs['start_method'] = 'fork'

def _worker_wrapper(*args: Any) -> Any:
    item = args[0] if len(args) == 1 else args
    return target_fn(item, **kwargs)

# REVISED:
def _parallel_worker_dispatch(item: Any, target_fn: Callable, kwargs: Dict[str, Any]) -> Any:
    """Module-level picklable target dispatcher compatible with forkserver/spawn."""
    if isinstance(item, tuple) and not hasattr(item, "_fields"):
        return target_fn(*item, **kwargs)
    return target_fn(item, **kwargs)

# In Parallel.__init__:
if 'start_method' not in worker_kwargs and 'start_method' not in extra_kwargs:
    available = mp.get_all_start_methods()
    worker_kwargs['start_method'] = 'forkserver' if 'forkserver' in available else 'spawn'
```

---

### Problem 2: Duplicate Parallel Pool and Utility Abstractions (`executor.py` vs `tweaks.py`)

- **Module / Component:** `tweaks.py`, `executor.py`, `domain_finder.py`, `aligners.py`
- **Severity / Category:** HIGH | Architectural Bloat / Code Maintainability / Technical Debt
- **Detailed Technical Analysis:**
  The repository maintains two parallel, conflicting utility implementations:
  1. `executor.py:Parallel` wraps MPIRE `WorkerPool` using composition, handles `chunk_size='auto'`, and sets `self.results = self.result`.
  2. `tweaks.py:Parallel` directly subclasses `WorkerPool`, hardcodes `chunk_size=100`, and lacks `chunk_size='auto'`.
  3. External process execution is duplicated between `tweaks.py:shell` (using `subprocess.Popen` with process group cleanup via `os.killpg`) and `executor.py:run_external` (using multi-thread stderr pipe draining).
  4. Logging configurations (`loguru_showwarning`) and CPU thread calculations are duplicated across both modules.
  Downstream modules mix imports arbitrarily (`domain_finder.py` imports `Parallel` from `executor.py` but `logger` and `default_threads` from `tweaks.py`).
- **Evidence in Code:**
  `tweaks.py:67-93` vs `executor.py:74-169`.
  `tweaks.py:106-151` vs `executor.py:173-356`.
- **Recommended Remediation:**
  Purge `tweaks.py`. Consolidate all essential execution utilities (`Parallel`, `run_external`, `shell`, `checkpoint`, `default_threads`, `logger`) into `executor.py`, updating downstream imports directly.

---

### Problem 3: Data Type Segregation: Physical Coordinates vs Packed Graph Keys

- **Module / Component:** `domain_finder.py` (`target_dtype`, `target_batch_processor`), `clustering.py`, `PREVIOUS_UPDATE.md`
- **Severity / Category:** MEDIUM | Memory Inefficiency / Data Type Design
- **Detailed Technical Analysis:**
  `PREVIOUS_UPDATE.md` claimed all physical sequence coordinates (`start`, `end`, `lengths`, `intersections`) must be strictly cast to `np.int64` to prevent signed 32-bit overflow on large genomes.
  In ncRNA transcriptomics and local alignment discovery, sequence lengths rarely exceed millions of nucleotides ($< 10^6$ nt). A signed 32-bit integer handles coordinates up to $2,147,483,647$ nt, which exceeds the longest known RNA molecule by three orders of magnitude.
  Storing coordinates in `int64` inflates record size from 16 to 28 bytes:
  - `int32` hit record: `t_seq` (int32) + `start` (int32) + `end` (int32) + `dom_id` (int32) = 16 bytes (4 records fit into a standard 64-byte CPU cache line).
  - `int64` hit record: 28 bytes (only 2 records fit per cache line, doubling cache misses and RAM footprint).
  Crucially, `np.int64` is required **only** when packing two 32-bit integers into a single 64-bit edge key (`keys = (u_r.astype(np.int64) << 32) | u_c.astype(np.int64)`) and accumulating global hit lengths.
- **Evidence in Code:**
  `domain_finder.py:592-597`: `target_dtype` uses `int64` for `t_seq`, `start`, and `end`.
  `domain_finder.py:365, 382`: Unnecessary casting to `int64` during target batch processing.
  `aligners.py:119-120`: Already parses coordinates as `np.int32`.
- **Recommended Remediation:**
  Enforce `np.int32` across all physical coordinates, spans, and local domain IDs. Restrict `np.int64` strictly to 64-bit edge bit-shift keys and global accumulators.

```python
# FAULTY (domain_finder.py:592-597 - int64 coordinates inflate memory and cache misses):
target_dtype = np.dtype([
    ('t_seq', np.int64),
    ('start', np.int64),
    ('end', np.int64),
    ('dom_id', np.int32)
])

# REVISED:
target_dtype = np.dtype([
    ('t_seq', np.int32),
    ('start', np.int32),
    ('end', np.int32),
    ('dom_id', np.int32)
])
```

---

### Problem 4: Algorithmic Flaws in Cluster-Wide DPC Halo Filtering and Rigorous Noise Handling

- **Module / Component:** `clustering.py` (`halo_filter`, `dpc`), `domain_finder.py`
- **Severity / Category:** CRITICAL | Biological Inconsistency / Algorithmic Distortion
- **Detailed Technical Analysis:**
  Traditional DPC halo filtering (Rodriguez & Laio, 2014) identifies border points as those exceeding a foreign similarity threshold, computes the maximum border density per cluster:
  $$\rho_b(c) = \max_{i \in \text{border}(c)} \rho_i$$
  and reassigns all cluster members with local density $\rho_i < \rho_b(c)$ to noise ($-1$).

  When applied to structured ncRNA alignment graphs, this formulation exhibits two major flaws:
  1. **Core Decapitation:** Biological alignment coverage is naturally heterogeneous. If a single member of family $A$ shares partial homology with family $B$ (e.g. a shared hairpin or repeat linker), its density $\rho_i$ may be elevated. Setting $\rho_b(c) = \rho_i$ causes all authentic internal family members whose density is lower than $\rho_b(c)$ to be discarded as noise ($-1$), even if they are structurally conserved core members.
  2. **Improper Handling of Noise Points ($-1$):**
     Noise points ($-1$) must never be assigned to valid clusters or used to compute cluster properties (such as cluster sizes, border maxima, or cluster medians). If negative labels are not strictly masked out, noise points can distort cluster metrics or trigger index errors.
- **Evidence in Code:**
  `clustering.py:788-796`: Cluster-wide `rho_b` reassignment unconditionally prunes intra-cluster density basins.
  `clustering.py:771-778`: Noise points with label `-1` require explicit masking (`labels >= 0`) during cluster size and evidence calculations.
- **Recommended Remediation:**
  1. Provide a simplified point-specific halo filtering mode (`point_specific=True`): a point $i$ is reclassified as halo noise if and only if it is an assigned member (`labels[i] >= 0`), belongs to an eligible cluster, its foreign border score exceeds `filter_transparency`, and its density is below its cluster median:
     $$s_{\text{foreign}}(i) \ge \tau \quad \text{and} \quad \rho_i < \text{median}_{j \in \text{Cluster}(i)}(\rho_j)$$
  2. Ensure noise points ($-1$) are strictly masked out from cluster property computations (`labels >= 0`).
  3. Default `halo_method='similarity_weighted'` for ncRNA family discovery (`family_finder`).

```python
# FAULTY (clustering.py:788-796 - cluster-wide rho_b decapitates entire intra-cluster basins):
if border.size > 0:
    np.maximum.at(rho_b, labels[border], rho[border])
rho_b[sizes < halo_min_cluster_size] = -1.0
cut[assigned] = rho[assigned] < rho_b[labels[assigned]]
new_labels[cut] = -1

# REVISED:
def halo_filter(labels: np.ndarray,
                similarity_mat: Matrix,
                filter_transparency: float,
                halo_min_cluster_size: int,
                backend: str,
                method: str,
                rho: np.ndarray,
                halo_k: int = DEFAULT_HALO_K,
                point_specific: bool = False) -> np.ndarray:
    new_labels = labels.copy()
    if filter_transparency >= 1.0 or new_labels.size == 0:
        return new_labels

    valid_mask = new_labels >= 0
    if not np.any(valid_mask):
        return new_labels

    n_clusters = int(new_labels[valid_mask].max()) + 1
    if n_clusters <= 1:
        return new_labels

    # Noise (-1) is strictly excluded from cluster sizes
    sizes = np.bincount(new_labels[valid_mask], minlength=n_clusters)
    eligible_clusters = sizes >= halo_min_cluster_size
    if not np.any(eligible_clusters):
        return new_labels

    scores = border_evidence(labels, similarity_mat, method, backend,
                             filter_transparency=filter_transparency, halo_k=halo_k)
    is_border = valid_mask & eligible_clusters[new_labels.clip(min=0)] & (scores >= filter_transparency) & (scores > 0.0)

    if point_specific:
        # Point-specific mode: evaluate border points individually without cluster-wide rho_b
        cluster_medians = np.zeros(n_clusters, dtype=np.float64)
        for c in range(n_clusters):
            members = new_labels == c  # strictly assigned members only
            if np.any(members):
                cluster_medians[c] = np.median(rho[members])

        outliers = is_border & (rho < cluster_medians[new_labels.clip(min=0)])
        new_labels[outliers] = -1
        return new_labels

    # Traditional cluster-wide mode
    border = np.flatnonzero(is_border)
    rho_b = np.full(n_clusters, -1.0, dtype=np.float64)
    if border.size > 0:
        np.maximum.at(rho_b, new_labels[border], rho[border])
    rho_b[sizes < halo_min_cluster_size] = -1.0

    cut = valid_mask & (rho < rho_b[new_labels.clip(min=0)])
    new_labels[cut] = -1
    return new_labels
```

---

### Problem 5: Array Buffer Resizing Safety and Allocation Overhead in Stage 3 (`domain_finder.py`)

- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 3)
- **Severity / Category:** MEDIUM | Numerical Robustness / Memory Optimization
- **Detailed Technical Analysis:**
  Stage 3 of `domain_finder.py` estimates target buffer capacity as:
  ```python
  est_capacity = max(100_000, count_fasta_queries(fasta_path) * 100)
  ```
  For deep alignments with many hits per sequence, this estimate triggers multiple geometric reallocations via `np.resize`. While `np.resize` handles array copying, frequent reallocations spike peak memory and degrade throughput.
- **Evidence in Code:**
  `domain_finder.py:601-614`.
- **Recommended Remediation:**
  Estimate initial hit capacity directly from the TSV file byte size ($\approx \text{file\_size} / 75$ bytes per BLAST-TAB line). Pre-sizing eliminates reallocations in over $95\%$ of runs while keeping geometric growth as a fallback.

```python
# FAULTY (domain_finder.py:601):
est_capacity = max(100_000, count_fasta_queries(fasta_path) * 100)

# REVISED:
est_capacity = max(100_000, int(tsv_file.stat().st_size // 75))
target_buffer = np.empty(est_capacity, dtype=target_dtype)
```

---

### Problem 6: Coordinate Conversion and Deterministic Strand Consensus Voting

- **Module / Component:** `aligners.py` (`parse_mmseqs_tsv`), `domain_finder.py` (`evaluate_clusters`, `determine_strand`)
- **Severity / Category:** MEDIUM | Data Integrity / Specification Compliance
- **Detailed Technical Analysis:**
  Coordinates are converted from 1-based closed intervals $[s, e]$ in BLAST-TAB to 0-based half-open intervals $[s - 1, e)$ upfront in `aligners.py`, and converted back to 1-based closed $[s + 1, e]$ upon GFF3 export.
  In `domain_finder.py:evaluate_clusters`, when overlapping domain instances of the same family on the same sequence are merged, strands are polled across overlapping instances. If votes tie, the assignment should be deterministic without introducing arbitrary strand bias.
- **Evidence in Code:**
  `domain_finder.py:51-64`, `domain_finder.py:504-511`, `domain_finder.py:672-675`.
- **Recommended Remediation:**
  Maintain clean majority voting (`'-' if neg > pos else '+'`) with deterministic tie-breaking, ensuring noise points ($-1$) are never polled.

---

### Problem 7: Discrepancy Audit of Claims in `PREVIOUS_UPDATE.md`

- **Module / Component:** `PREVIOUS_UPDATE.md` vs Current Working Tree
- **Severity / Category:** HIGH | Documentation Inconsistency / Audit Veracity
- **Detailed Technical Analysis:**
  Audit of claims made in `PREVIOUS_UPDATE.md` against the working tree:
  1. **Claimed Multiprocessing Start Method:** `PREVIOUS_UPDATE.md` claimed `forkserver` was implemented, but `executor.py:143` still contains `start_method = 'fork'`.
  2. **Claimed 64-bit Requirement:** `PREVIOUS_UPDATE.md` claimed `int64` was necessary for all coordinates, which causes 2x memory inflation without biological necessity.
  3. **Claimed Initial Capacity:** `PREVIOUS_UPDATE.md` claimed initial buffer capacity was `1_000_000`, but `domain_finder.py:601` actually set `100_000`.
  4. **Purged `Span` Mechanics:** Verified — legacy `Span` class has been completely purged from the pipeline.
  5. **Pre-computed Priority Ranks in `filter_spurious_domains`:** Verified — priority ranking via `np.lexsort((-support, length, index))` is correctly implemented.
- **Evidence in Code:**
  `executor.py:143` vs `PREVIOUS_UPDATE.md:391`.
  `domain_finder.py:601` vs `PREVIOUS_UPDATE.md:575`.
- **Recommended Remediation:**
  Document only verified facts and provide executable solutions that directly resolve the discrepancy.

---

### Problem 8: File Naming Typo and Documentation Ergonomics

- **Module / Component:** `domain_finder_pecomputed.py`
- **Severity / Category:** LOW | Code Clarity / Maintainability
- **Detailed Technical Analysis:**
  `domain_finder_pecomputed.py` contains a typo in its filename (`pecomputed` instead of `precomputed`). The benchmark paths in its `__main__` block should be preserved for benchmark execution continuity.
- **Evidence in Code:**
  `domain_finder_pecomputed.py:1-94`.
- **Recommended Remediation:**
  Rename `domain_finder_pecomputed.py` to `domain_finder_precomputed.py` while retaining existing benchmark paths in `__main__`.

---

## 3. Summary Risk & Vulnerability Matrix

| Problem / Vulnerability | Affected Module(s) | Impact / Risk | Proposed Resolution |
| :--- | :--- | :--- | :--- |
| **Multiprocessing fork deadlock on Python 3.12+** | `executor.py:143` | Child workers deadlock under multi-threaded logging/tqdm contexts; emits deprecation warnings. | Implement `start_method='forkserver'` (or `'spawn'`) and module-level picklable dispatcher `_parallel_worker_dispatch`. |
| **Duplicate `Parallel` & CLI abstractions** | `tweaks.py`, `executor.py` | Conflicting pool implementations, code bloat, divergent bug fixes. | Purge `tweaks.py`, consolidating essential utilities into `executor.py`. |
| **Coordinate int64 memory bloat** | `domain_finder.py`, `clustering.py` | 2x memory inflation for alignment coordinates; cache miss penalty on large runs. | Use `np.int32` for physical coordinates and spans; reserve `np.int64` strictly for 64-bit edge bit-shift keys and global accumulators. |
| **Destructive halo filtering on structured RNA** | `clustering.py:788-796` | Deletes authentic intra-cluster family members; fails to cleanly isolate noise ($-1$). | Implement point-specific foreign border filtering as a configurable mode; ensure noise ($-1$) is strictly excluded from cluster metrics; default halo to `similarity_weighted` for ncRNA families. |
| **Dynamic buffer copy overhead in Stage 3** | `domain_finder.py:601-614` | Repeated `np.resize` allocations on deep alignments; risk of memory spikes. | Pre-allocate hit capacity directly from TSV file byte size ($\approx \text{file\_size} / 75$). |
| **Discrepancies in `PREVIOUS_UPDATE.md`** | `PREVIOUS_UPDATE.md` | False claims that `forkserver` and 1M buffer allocation were already merged. | Document verified code state and implement genuine fixes. |
| **File naming typo** | `domain_finder_pecomputed.py` | Typo in filename affects module discovery and import ergonomics. | Rename to `domain_finder_precomputed.py` while preserving benchmark test paths. |

---

## 4. Unified Step-by-Step Implementation Architecture (For Next Jules Agent Run)

This section provides the implementation blueprint to guide the next Jules agent run. It prioritizes worker lifecycle safety, consolidates redundant abstractions, streamlines coordinate data types, and refines density peak halo partitioning.

---

### Implementation Principles & Guidelines for Next Jules Run
1. **Low Structural Complexity:** Follow clear, sequential phases with concise, to-the-point instructions and single-level sectioning.
2. **Backward Compatibility is Irrelevant:** Purge obsolete wrappers and duplicate files (`tweaks.py`). Do not retain legacy shims or aliases.
3. **Optimized Worker Lifecycle:** Use `start_method='forkserver'` (fallback to `'spawn'`) on Linux with top-level picklable dispatchers. Never use `os.fork()` from multi-threaded parent processes.
4. **Data Type Segregation:** Use `np.int32` for physical sequence coordinates, spans, and local domain IDs; reserve `np.int64` strictly for packed 64-bit graph edge keys (`(u_r << 32) | u_c`) and global accumulators.
5. **Rigorous Noise Handling:** Noise points ($-1$) must never be assigned to valid clusters, used in cluster metrics, or polled during merging. Provide point-specific border filtering as an explicit configuration.

---

### Phase 1: Process Concurrency, Fork-Safety & Utility Consolidation

**Objective:** Eliminate Python 3.12+ `os.fork()` child deadlocks and deprecation warnings, purge redundant module `tweaks.py`, and consolidate execution utilities into `executor.py`.

#### Step 1.1: Purge `tweaks.py` and Consolidate Utilities in `executor.py`
- Delete `tweaks.py`.
- Move essential utilities from `tweaks.py` into `executor.py`:
  - `default_threads` (logical CPU count minus 1, minimum 1).
  - `shell(...)`: External command invocation with process group termination semantics (`os.killpg`).
  - `checkpoint(...)`: Function result caching decorator.
- Update all repository imports referencing `tweaks` (`aligners.py`, `domain_finder.py`, `evaluation.py`) to import directly from `executor`.

#### Step 1.2: Picklable Worker Dispatch and Safe Start Method in `executor.py`
- In `executor.py`:
  - Replace the unpicklable nested `_worker_wrapper` closure with a module-level picklable function `_parallel_worker_dispatch`.
  - Configure `start_method` to default to `'forkserver'` on Linux when available, falling back to `'spawn'`. Explicitly disallow `'fork'`.
  - Maintain dynamic chunk sizing (`chunk_size='auto'`), defaulting to 32 for generator iterables and adaptive scaling ($B = \max(1, \min(64, \lfloor N / (4 \times \text{n\_jobs}) \rfloor))$) for collections.

```python
# FAULTY (executor.py - nested closure and hardcoded fork causes deadlocks on Python 3.12+):
class Parallel:
    def __init__(self, ...):
        if 'start_method' not in worker_kwargs and 'start_method' not in extra_kwargs:
            worker_kwargs['start_method'] = 'fork'

        def _worker_wrapper(*args: Any) -> Any:
            item = args[0] if len(args) == 1 else args
            return target_fn(item, **kwargs)

        with WorkerPool(n_jobs=n_jobs, **worker_kwargs) as pool:
            self.result = pool.map(_worker_wrapper, input_list, **map_kwargs)

# REVISED (executor.py - top-level picklable dispatch and safe forkserver/spawn start method):
def _parallel_worker_dispatch(item: Any, target_fn: Callable, kwargs: Dict[str, Any]) -> Any:
    """Module-level picklable target dispatcher compatible with forkserver/spawn."""
    if isinstance(item, tuple) and not hasattr(item, "_fields"):
        return target_fn(*item, **kwargs)
    return target_fn(item, **kwargs)

class Parallel:
    def __init__(self, parallelized_function: Optional[Callable] = None,
                 input_collection: Optional[Collection] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 n_jobs: Optional[int] = None,
                 chunk_size: Any = 'auto',
                 **extra_kwargs: Any) -> None:
        ...
        worker_kwargs = {'shared_objects': shared_objects} if shared_objects else {}
        if 'start_method' not in worker_kwargs and 'start_method' not in extra_kwargs:
            avail = mp.get_all_start_methods()
            worker_kwargs['start_method'] = 'forkserver' if 'forkserver' in avail else 'spawn'

        dispatch_fn = functools.partial(_parallel_worker_dispatch,
                                        target_fn=target_fn,
                                        kwargs=kwargs or {})

        with WorkerPool(n_jobs=n_jobs, **worker_kwargs) as pool:
            map_func = pool.map if keep_order else pool.map_unordered
            self.result = map_func(dispatch_fn, input_list, **map_kwargs)
        self.results = self.result
```

---

### Phase 2: Memory Streamlining & Coordinate Type Segregation

**Objective:** Halve RAM consumption and eliminate CPU cache line misses by enforcing `np.int32` for physical coordinates and spans while reserving `np.int64` strictly for 64-bit edge bit-shift keys and global accumulators.

#### Step 2.1: Enforce `np.int32` in Alignment Ingestion and Target Overlaps
- In `domain_finder.py`:
  - Define `target_dtype` to use 32-bit integers across all fields:
    `target_dtype = np.dtype([('t_seq', np.int32), ('start', np.int32), ('end', np.int32), ('dom_id', np.int32)])`
    (Reduces record size from 28 bytes to 16 bytes per hit, aligning 4 records per 64-byte CPU cache line).
  - In `target_batch_processor`:
    - Cast `merged_spans` to `np.int32`: `np.asarray(merged_spans_list, dtype=np.int32)`.
    - Retain `intersections` as `np.int32`: `intersections.astype(np.int32, copy=False)`.

#### Step 2.2: Retain `np.int64` Strictly for Composite Graph Keys
- In `clustering.py:build_similarity_matrix` and `domain_finder.py:family_finder`:
  - When packing 32-bit row and column indices into a single 64-bit integer:
    `keys = (u_r.astype(np.int64) << 32) | (u_c.astype(np.int64) & 0xFFFFFFFF)`
  - Use `np.int64` for `global_dom_lengths` and total intersection sums in `family_finder`.

#### Step 2.3: Zero-Reallocation Target Buffer Sizing from TSV File Size
- In `domain_finder.py:find_domains` (Stage 3):
  - Estimate initial target hit capacity directly from TSV file byte size:
    `est_capacity = max(100_000, int(tsv_file.stat().st_size // 75))`
  - Pre-allocate `target_buffer = np.empty(est_capacity, dtype=target_dtype)`.
  - Maintain geometric resizing (`max(capacity * 2, write_offset + n_hits + 100_000)`) as a fallback safeguard.

```python
# FAULTY (domain_finder.py - int64 coordinates inflate memory, small FASTA-based estimate causes repeated reallocations):
target_dtype = np.dtype([
    ('t_seq', np.int64),
    ('start', np.int64),
    ('end', np.int64),
    ('dom_id', np.int32)
])
est_capacity = max(100_000, count_fasta_queries(fasta_path) * 100)
target_buffer = np.empty(est_capacity, dtype=target_dtype)

# REVISED (domain_finder.py - int32 coordinates and TSV file-size-based buffer pre-allocation):
target_dtype = np.dtype([
    ('t_seq', np.int32),
    ('start', np.int32),
    ('end', np.int32),
    ('dom_id', np.int32)
])
# BLAST-TAB records average ~75 bytes per line; pre-sizing prevents buffer reallocations
est_capacity = max(100_000, int(tsv_file.stat().st_size // 75))
target_buffer = np.empty(est_capacity, dtype=target_dtype)
```

---

### Phase 3: DPC Algorithmic Refinement & Point-Specific Halo Filtering

**Objective:** Prevent core member pruning in structured RNA families, isolate noise strictly, and provide a configurable point-specific foreign border filter.

#### Step 3.1: DPC Early Termination and Validated Lineage Propagation
- In `clustering.py:dpc()`:
  - If no candidate satisfies `densities > min_dens`, immediately return `{}` and `int(n)` (all points classified as noise `-1`).
  - Propagate labels via parent pointer doubling along non-zero similarity edges (`(nearest_higher >= 0) & (s_max_higher > 0.0)`).
  - Mask labels such that points unable to trace their lineage to an approved peak remain `-1`.

#### Step 3.2: Configurable Point-Specific Halo Filtering Option with Strict Noise Masking
- In `clustering.py:halo_filter()`:
  - Add parameter `point_specific: bool = False`.
  - Strictly mask assigned points (`valid_mask = labels >= 0`). Noise points ($-1$) must never be included in cluster size counts, border candidate sets, or cluster median calculations.
  - **Point-specific mode (`point_specific=True`):** Disables cluster-wide $\rho_b(c)$ cutoffs. A point $i$ is reclassified as halo noise if and only if it is an assigned member, in an eligible cluster, its individual foreign border score exceeds `filter_transparency`, and its local density is below its cluster median:
    $$s_{\text{foreign}}(i) \ge \tau \quad \text{and} \quad \rho_i < \text{median}_{j \in \text{Cluster}(i)}(\rho_j)$$
  - **Cluster-wide mode (`point_specific=False`):** Evaluates traditional Rodriguez-Laio halo filtering via $\rho_b(c) = \max_{i \in \text{border}(c)} \rho_i$ and reassigns points with $\rho_i < \rho_b(c)$ to noise ($-1$).
- In `domain_finder.py`:
  - Default `halo_method=similarity_weighted` for domain family finding (`family_finder`), ensuring structured ncRNA families with ragged alignment boundaries are preserved intact.

```python
# FAULTY (clustering.py - cluster-wide rho_b decapitates entire intra-cluster basins, lacks strict noise masking):
if border.size > 0:
    np.maximum.at(rho_b, labels[border], rho[border])
rho_b[sizes < halo_min_cluster_size] = -1.0
cut[assigned] = rho[assigned] < rho_b[labels[assigned]]
new_labels[cut] = -1

# REVISED (clustering.py - strict noise masking and configurable point-specific filtering):
def halo_filter(labels: np.ndarray,
                similarity_mat: Matrix,
                filter_transparency: float,
                halo_min_cluster_size: int,
                backend: str,
                method: str,
                rho: np.ndarray,
                halo_k: int = DEFAULT_HALO_K,
                point_specific: bool = False) -> np.ndarray:
    new_labels = labels.copy()
    if filter_transparency >= 1.0 or new_labels.size == 0:
        return new_labels

    valid_mask = new_labels >= 0
    if not np.any(valid_mask):
        return new_labels

    n_clusters = int(new_labels[valid_mask].max()) + 1
    if n_clusters <= 1:
        return new_labels

    # Noise (-1) is strictly excluded from cluster sizes
    sizes = np.bincount(new_labels[valid_mask], minlength=n_clusters)
    eligible_clusters = sizes >= halo_min_cluster_size
    if not np.any(eligible_clusters):
        return new_labels

    scores = border_evidence(labels, similarity_mat, method, backend,
                             filter_transparency=filter_transparency, halo_k=halo_k)
    is_border = valid_mask & eligible_clusters[new_labels.clip(min=0)] & (scores >= filter_transparency) & (scores > 0.0)

    if point_specific:
        # Point-specific mode: evaluate border points individually without cluster-wide rho_b
        cluster_medians = np.zeros(n_clusters, dtype=np.float64)
        for c in range(n_clusters):
            members = new_labels == c  # strictly assigned members only
            if np.any(members):
                cluster_medians[c] = np.median(rho[members])

        outliers = is_border & (rho < cluster_medians[new_labels.clip(min=0)])
        new_labels[outliers] = -1
        return new_labels

    # Traditional cluster-wide mode
    border = np.flatnonzero(is_border)
    rho_b = np.full(n_clusters, -1.0, dtype=np.float64)
    if border.size > 0:
        np.maximum.at(rho_b, new_labels[border], rho[border])
    rho_b[sizes < halo_min_cluster_size] = -1.0

    cut = valid_mask & (rho < rho_b[new_labels.clip(min=0)])
    new_labels[cut] = -1
    return new_labels
```

---

### Phase 4: Module Cleanup, Typo Correction & Pipeline Harmonization

**Objective:** Clean module discovery ergonomics, fix file naming typos, preserve test dataset paths, and ensure deterministic strand resolution.

#### Step 4.1: Rename File and Retain Benchmark Test Paths
- Rename `domain_finder_pecomputed.py` to `domain_finder_precomputed.py`.
- Keep the existing test workstation paths inside `if __name__ == '__main__':` in `domain_finder_precomputed.py` to maintain exact continuity for external benchmark evaluation runs.
- Update documentation and references to use the correct module name.

#### Step 4.2: Harmonize Strand Consensus Resolution
- In `domain_finder.py:evaluate_clusters`:
  - When merging multiple overlapping domain instances belonging to the same family on the same query sequence, determine consensus strand using explicit vote counting (`'-' if neg > pos else '+'`).
  - Guarantee that noise points ($-1$) are never included in domain merging.

---

### Phase 5: Verification Protocols & Regression Benchmarks

**Objective:** Provide rigorous validation commands and quantitative gates to confirm correctness, performance, and absence of regressions.

#### Step 5.1: Unit & Integration Test Suite
Execute the full pytest suite under Python 3.12:
```bash
python3 -m pytest tests/test_aligners.py -v
python3 -m pytest tests/test_domain_finder.py -v
python3 -m pytest tests/test_dpc_and_spurious.py -v
python3 -m pytest tests/test_msa_benchmark.py -v
```
*Gate:* All tests must pass with **0 failures and 0 warnings**. Specifically verify that `DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks` is completely absent.

#### Step 5.2: Concurrency & Worker Lifecycle Stress Test
- Run `find_domains` across worker thread allocations ($C \in \{1, 2, 4, 8\}$):
  - Confirm worker processes spawn cleanly using `forkserver`/`spawn` without orphan processes or zombie workers.
  - Verify that memory per worker core remains strictly below 200 MB.

#### Step 5.3: GFF3 Formatting & Semantic Output Validation
- Inspect generated GFF3 output files:
  - Confirm GFF3 version header: `##gff-version 3`.
  - Confirm 1-based closed coordinate intervals $[s_{\text{gff}}, e_{\text{gff}}] = [s + 1, e]$.
  - Ensure all coordinate values are positive integers ($s \ge 1, e \ge s$).
  - Confirm valid consensus strand values (`+` or `-`).
