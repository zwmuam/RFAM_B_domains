# PROBLEMS TO SOLVE

## Concurrency Hazards and Shared State in Temporary Shard Directories

- **Evidence**:
  In `domain_finder.py:750-760`:
  ```python
  if tmp_dir is not None:
      stage4_tmp_dir = Path(tmp_dir) / "stage4_shards"
      stage4_tmp_dir.mkdir(parents=True, exist_ok=True)
      shard_cleanup_needed = False
  else:
      temp_dir_obj = tempfile.TemporaryDirectory(prefix="domain_finder_stage4_")
      stage4_tmp_dir = Path(temp_dir_obj.name)
      shard_cleanup_needed = True
  ```
  And in the `finally` cleanup block (`domain_finder.py:800-810`):
  ```python
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
  ```
- **Impact**:
  1. **Race Conditions in Concurrent Executions**: When multiple pipeline runs share a common `tmp_dir` (such as `/tmp` or a shared cluster scratch mount), all processes write to the exact same directory path `tmp_dir / "stage4_shards"` using identical chunk filenames (`0_r.npy`, `1_r.npy`, etc.). One run will overwrite or prematurely unlink another run's binary shards while reading, resulting in missing file errors, corrupted edge arrays, or non-deterministic cluster outputs.
  2. **Unintended Deletion of Foreign Files**: The cleanup routine executes `for f in stage4_tmp_dir.glob("*.npy"): f.unlink()`. If a separate pipeline execution or worker process is writing shards to that directory, its data files will be wiped out.
  3. **Resource Leaks on Failure**: If an unhandled exception or crash occurs before shard reading completes, shard files left in the shared folder can interfere with subsequent executions.
- **Proposed Fix**:
  Always create a dedicated, uniquely named subdirectory using `tempfile.TemporaryDirectory` with a process-specific prefix, regardless of whether `tmp_dir` is supplied by the caller or defaulted. Restrict all shard writes, reads, and cleanup operations strictly to that isolated per-run directory. Use context management to guarantee directory removal upon normal completion or handled exceptions.

```python
# FAULTY (domain_finder.py - Hardcoded shared directory path causes collisions in concurrent runs):
if tmp_dir is not None:
    stage4_tmp_dir = Path(tmp_dir) / "stage4_shards"
    stage4_tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_cleanup_needed = False
else:
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="domain_finder_stage4_")
    stage4_tmp_dir = Path(temp_dir_obj.name)
    shard_cleanup_needed = True

# REVISED (domain_finder.py - Guaranteed unique temporary directory per run):
base_tmp = Path(tmp_dir) if tmp_dir is not None else None
temp_dir_obj = tempfile.TemporaryDirectory(prefix="domain_finder_stage4_", dir=base_tmp)
stage4_tmp_dir = Path(temp_dir_obj.name)
try:
    # Execute Stage 4 worker dispatch and shard reading within isolated stage4_tmp_dir
    ...
finally:
    temp_dir_obj.cleanup()
```

- **Validation**:
  1. Execute two concurrent pipeline instances pointing to the same `--tmp_dir` simultaneously.
  2. Confirm both runs complete without shard overwrite errors, missing file exceptions, or cross-talk corruption.
  3. Verify the temporary directory is cleaned up upon process termination or exception.

---

## Memory Spikes and Algorithmic Bottlenecks in Overlap Streaming and Graph Aggregation

- **Evidence**:
  In `clustering.py:27-44`:
  ```python
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
  In Stage 4 worker accumulation (`domain_finder.py:420-435`), each worker accumulates all edge lists for its assigned sequence batch, concatenates them into single arrays (`r_arr`, `c_arr`, `i_arr`), and only then writes them to disk.
  In Stage 5 graph aggregation (`domain_finder.py:785-800` and `domain_finder.py:475-515`), the parent process loads all edge shards into contiguous arrays `all_rows`, `all_cols`, and `all_inters`, sorts 64-bit composite keys, and computes Jaccard similarities for the entire pairwise graph at once.
- **Impact**:
  1. **Unbounded Worker Memory on Dense Targets**: Sharding coordinates to disk after an entire batch completes does not protect individual workers while overlaps are being generated. If a single target sequence contains dense overlapping intervals (e.g., thousands of domain hits on a conserved locus), `compare_sorted_spans` builds millions of pairwise combinations in memory before returning. A batch-level threshold applied only after `compare_sorted_spans` completes is too late to prevent worker Out-Of-Memory crashes.
  2. **Python Allocation and View Overhead**: In `target_batch_processor` (`domain_finder.py:388-403`), `merged_spans_list` and `aligned_domains_list` append individual NumPy views and Python integers per interval. For large interval counts, storing Python objects incurs higher memory overhead than flat numeric buffers.
  3. **Whole-Graph Materialization Overhead**: While Stage 4 disk shards protect parent IPC during worker execution, Stage 5 still reconstructs all three full edge arrays (`all_rows`, `all_cols`, `all_inters`) simultaneously in memory. For large datasets with hundreds of millions of pairwise overlaps, building and sorting composite 64-bit edge keys `(u_r << 32) | u_c` creates a dominant memory peak in the parent process.
- **Proposed Fix**:
  1. **In-Loop Bounded Pair Streaming**: Refactor `compare_sorted_spans` and `target_batch_processor` to emit and flush pairwise overlap chunks directly to disk shards when an edge buffer budget is reached (e.g., every 500,000 edges), rather than accumulating all edges in memory across an entire batch or dense sequence.
  2. **Workload-Aware Batch Sizing**: Scale Stage 4 task batching by interval count rather than target sequence count alone, preventing single batches from packing disproportionately large numbers of domain spans.
  3. **Stage Aggregation Profiling**: Profile the ratio of emitted interval-pair contributions ($Q$) to unique canonical domain pairs ($E$). When $\frac{Q}{E}$ is high, investigate partial pre-aggregation or chunked sparse accumulation before constructing the final similarity matrix.
  4. **Vectorized Search Bounds as an Empirical Speed Trial**: Vectorizing binary search bounds via `np.searchsorted` across all intervals avoids Python loop overhead for intervals with no overlaps, but introduces $O(M)$ index arrays. This should be treated as an experimental performance trial and verified against memory profiles before adoption.

```python
# FAULTY (domain_finder.py - Workers accumulate all edges for an entire batch before writing to disk):
batch_rows, batch_cols, batch_data = [], [], []
for target_id, target_spans, domain_pointers in batch_targets:
    ...
    rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
    if rows.size > 0:
        batch_rows.append(rows)
        batch_cols.append(cols)
        batch_data.append(intersections)

# Memory spike: All edge arrays concatenated in memory before sharding
r_arr = np.concatenate(batch_rows) if batch_rows else np.empty(0, dtype=np.int32)
c_arr = np.concatenate(batch_cols) if batch_cols else np.empty(0, dtype=np.int32)
i_arr = np.concatenate(batch_data) if batch_data else np.empty(0, dtype=np.int32)
if shard_dir_path is not None:
    np.save(shard_dir / f"{chunk_id}_r.npy", r_arr)
    ...

# REVISED (domain_finder.py - Bounded in-stream chunk flushing to limit worker RSS):
class ShardedEdgeWriter:
    def __init__(self, shard_dir: Path, chunk_id: int, max_buffer_edges: int = 500_000):
        self.shard_dir = shard_dir
        self.chunk_id = chunk_id
        self.max_buffer_edges = max_buffer_edges
        self.part_idx = 0
        self.buf_r, self.buf_c, self.buf_i = [], [], []
        self.buf_size = 0
        self.total_edges = 0

    def add(self, r: np.ndarray, c: np.ndarray, inter: np.ndarray):
        if r.size == 0:
            return
        self.buf_r.append(r)
        self.buf_c.append(c)
        self.buf_i.append(inter)
        self.buf_size += r.size
        self.total_edges += r.size
        if self.buf_size >= self.max_buffer_edges:
            self.flush()

    def flush(self):
        if self.buf_size == 0:
            return
        r_cat = np.concatenate(self.buf_r)
        c_cat = np.concatenate(self.buf_c)
        i_cat = np.concatenate(self.buf_i)
        part_name = f"{self.chunk_id}_part{self.part_idx}"
        np.save(self.shard_dir / f"{part_name}_r.npy", r_cat)
        np.save(self.shard_dir / f"{part_name}_c.npy", c_cat)
        np.save(self.shard_dir / f"{part_name}_i.npy", i_cat)
        self.part_idx += 1
        self.buf_r.clear()
        self.buf_c.clear()
        self.buf_i.clear()
        self.buf_size = 0
```

- **Validation**:
  1. Stress-test worker memory on synthetic loci containing $> 20,000$ mutually overlapping domain intervals.
  2. Verify that worker process peak RSS remains strictly bounded by the buffer chunk size.
  3. Confirm that edge indices, intersection sums, and final Jaccard similarities match bit-for-bit against the baseline implementation.

---

## Target Buffer Allocation, Lifetime, and Unused Capacity Retention

- **Evidence**:
  In `domain_finder.py:692-715`:
  ```python
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
  ```
  And in `domain_finder.py:710-720`:
  ```python
  if write_offset > 0:
      all_targets_arr = target_buffer[:write_offset]
      del target_buffer
      all_targets_arr.sort(order=['t_seq', 'start'])
  ```
- **Impact**:
  1. **Slice View Retains Entire Buffer Allocation**: `all_targets_arr = target_buffer[:write_offset]` creates a NumPy slice view over the active records. Deleting `target_buffer` via `del target_buffer` removes only the variable name; the underlying base array—including all unwritten trailing capacity—remains pinned in memory because `all_targets_arr.base` references the original buffer. If initial line-sampling capacity estimates significantly overshoot actual retained hits, gigabytes of unused capacity remain allocated throughout Stage 4 until Stage 5.
  2. **Transient Reallocation Peaks**: Safe allocate-copy-rebind growth temporarily holds both the existing buffer and the newly expanded buffer during copying. For large datasets with $> 100$ million target hits (16 bytes per hit), buffers occupy $1.6$ to $2.4\text{ GB}$, meaning a reallocation can transiently double RSS during the copy.
  3. **Batch List Lifetime**: In Stage 2, `WorkerPool.imap` yields `batch_res` containing lists of query results. The parent process processes items sequentially, but all result arrays in the batch remain alive until the entire batch loop iteration completes, adding transient memory pressure.
  4. **Unsafe Anti-Pattern Warning**: Switching to in-place `target_buffer.resize(new_size, refcheck=False)` would be hazardous if any slice views exist in active Python scopes. NumPy documents that in-place resize does not guarantee zero-copy operating system page remapping (`mremap`) on heap memory and will reallocate, which produces dangling pointers and silent memory corruption if views persist.
- **Proposed Fix**:
  1. **Explicit Shrink and Memory Detachment**: At the end of Stage 2, if `write_offset < target_buffer.shape[0]`, detach the active slice from the oversized buffer by creating an exact-fit copy: `all_targets_arr = target_buffer[:write_offset].copy()`, then delete `target_buffer` to immediately release all unused capacity before Stage 4.
  2. **Prompt Batch Result Reclamation**: Clear or pop processed query elements from `batch_res` as they are written into `target_buffer` to allow the Python garbage collector to reclaim intermediate worker arrays immediately.
  3. **Empirical Buffer Profiling**: Log `write_offset` versus allocated capacity and benchmark chunked block accumulation (e.g., fixed blocks of 10 million hits) against single-buffer geometric growth to evaluate heap fragmentation.

```python
# FAULTY (domain_finder.py - Slicing target_buffer leaves unused capacity alive in memory):
if write_offset > 0:
    all_targets_arr = target_buffer[:write_offset]
    del target_buffer  # Base buffer remains pinned in memory via all_targets_arr.base!
    all_targets_arr.sort(order=['t_seq', 'start'])

# REVISED (domain_finder.py - Explicitly detach active prefix to release unused buffer capacity):
if write_offset > 0:
    # If substantial unused capacity remains, create a compact copy to free the oversized buffer
    if write_offset < target_buffer.shape[0]:
        all_targets_arr = target_buffer[:write_offset].copy()
    else:
        all_targets_arr = target_buffer[:write_offset]
    del target_buffer
    all_targets_arr.sort(order=['t_seq', 'start'])
```

- **Validation**:
  1. Measure `write_offset`, allocated buffer size, and process RSS before and after Stage 2 completion.
  2. Verify that `all_targets_arr.base` is detached or exact-sized, ensuring no excess memory is retained.
  3. Validate that target coordinates, domain indices, and sorting order remain identical.

---

## Target-Span De-Replication Trade-offs and Quadratic Pair Scaling

- **Evidence**:
  In `domain_finder.py:375-410` (`target_batch_processor`):
  ```python
  dom_boundaries = np.flatnonzero(np.diff(s_doms)) + 1
  dom_splits = np.split(s_doms, dom_boundaries)
  span_splits = np.split(s_spans, dom_boundaries)

  for d_group, sp_group in zip(dom_splits, span_splits):
      d_id = int(d_group[0])
      m_spans = merge_overlapping_spans(sp_group)
      for sp in m_spans:
          merged_spans_list.append(sp)
          aligned_domains_list.append(d_id)
          batch_dom_lengths[d_id] += int(sp[1] - sp[0])
  ```
  And in `overlaps_on_target` (`domain_finder.py:340-350`):
  ```python
  local_rows, local_cols, intersections = compare_sorted_spans(sorted_target_spans)
  if local_rows.size > 0:
      mask = aligned_domain_pointers[local_rows] != aligned_domain_pointers[local_cols]
      if np.any(mask):
          global_rows = aligned_domain_pointers[local_rows][mask]
          global_cols = aligned_domain_pointers[local_cols][mask]
          return global_rows, global_cols, intersections[mask]
  ```
- **Impact Analysis**:
  1. **Post-Merge Interval Invariant**: Stage 4 partitions alignment hits per target sequence and domain ID, running `merge_overlapping_spans` independently for each domain. This merges all overlapping and abutting hits into a non-overlapping interval union. Consequently, merged intervals belonging to the same domain on a given target sequence are strictly disjoint and cannot have positive overlap. The domain inequality filter `aligned_domain_pointers[local_rows] != aligned_domain_pointers[local_cols]` in `overlaps_on_target` serves as a defensive check; it does not discard substantial same-domain overlap calculations during normal operation.
  2. **Mathematical Validity of Exact-Coordinate Grouping**: Multiple distinct domains $D\_{1}, D\_{2}, \dots, D\_{K}$ often hit the exact same target interval $[start, end)$ on a conserved target sequence. If intervals are grouped into unique coordinate spans with associated domain ID sets, the geometric intersection between two overlapping coordinate groups can be evaluated once and shared across domain pairs:
     - For identical coordinate groups sharing $K$ domains, the internal intersection length equals $(end - start)$ for all $\binom{K}{2}$ domain pairs.
     - For two overlapping coordinate groups with $K\_1$ and $K\_2$ domains, the cross-intersection length equals $\min(end\_1, end\_2) - \max(start\_1, start\_2)$ for all $K\_1 \times K\_2$ domain pairs.
  3. **The Pair Emission Bottleneck**: Grouping coordinates saves coordinate storage and reduces interval arithmetic, but it **does not eliminate pair emissions**. DPC similarity graph construction in Stage 5 requires explicit pairwise Jaccard similarities:
     $$J(D\_{i}, D\_{j}) = \frac{\sum \text{overlap}}{\text{Length}(D\_{i}) + \text{Length}(D\_{j}) - \sum \text{overlap}}$$
     Every pair among the $K$ domains sharing an interval must still emit an intersection contribution. For $K = 10,000$ domains on a conserved target sequence, this requires:
     $$Q = \binom{K}{2} = \frac{K(K - 1)}{2} = 49,995,000 \text{ pairwise contributions}$$
     Three `int32` arrays (`row`, `col`, `intersection`) require ~600 MB for those contributions alone. Thus, coordinate grouping reduces geometric evaluation from $O(M \log M + Q)$ to $O(M \log M + P + Q)$ (where $P$ is the number of overlapping coordinate groups and $Q$ is the emitted pair contributions), but leaves the quadratic $Q$ term intact.
  4. **Strand Voting Integrity**: Alignment hits cannot be deduplicated prior to Stage 2 domain instance discovery and `determine_strand()`, as hit frequency across strands determines strand consensus. Deduplication can only be applied after domain discovery and per-domain interval merging.
- **Proposed Fix**:
  1. **Formalize Redundancy Metrics**: Replace arbitrary percentage thresholds with precise tracked metrics per target sequence:
     - $M$: Number of merged intervals on the target sequence.
     - $G$: Number of unique $(start, end)$ coordinate groups.
     - $\text{duplicate\_fraction} = 1 - \frac{G}{M}$ (defined as 0 when $M = 0$).
     - $Q$: Total emitted interval-pair contributions.
     - $E$: Total unique canonical domain pairs in Stage 5.
     - $\frac{Q}{E}$: Average number of contributions per final canonical domain pair.
  2. **Singleton Merge Fast Path**: In `merge_overlapping_spans`, add an immediate early exit for solitary intervals (`if targets_on_sequence.shape[0] <= 1: return targets_on_sequence.copy()`). This eliminates `lexsort`, cumulative operations, and slicing for non-overlapping single-span domains.
  3. **Controlled Coordinate Grouping Experiment**: Treat exact-coordinate grouping as an optional, benchmark-driven optimization within Stage 4. Grouping must only occur *after* per-domain union merging, must retain all domain IDs in flat integer index arrays, and must accumulate intersection lengths into 64-bit integers.

```python
# FAULTY (domain_finder.py - Executes full lexsort and accumulate even for solitary intervals):
def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
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

# REVISED (domain_finder.py - Early exit for solitary intervals preserving exact coordinates):
def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
    n = targets_on_sequence.shape[0]
    if n <= 1:
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
```

- **Validation**:
  1. Test `merge_overlapping_spans` on empty, single-span, and multi-span inputs to verify coordinate equivalence.
  2. In experimental coordinate grouping, assert that no domain ID appears twice in any coordinate group and that accumulated pairwise intersections match ungrouped outputs exactly.
  3. Measure runtime and peak memory on datasets with varying $\text{duplicate\_fraction}$ to assess whether coordinate compression yields net benefits over pair output overhead.

---

## Domain Integrity, Sequence Filtering, and DPC Graph Semantics

- **Evidence**:
  In `aligners.py:145-152`:
  ```python
  q_seq_str = fields[0]
  t_seq_str = fields[1]

  if q_seq_str == t_seq_str:
      continue
  ```
  In `domain_finder.py:480-495` (`family_finder`), disconnected graphs ($u\_r.\text{size} == 0$) evaluate density against `min_fam_dens`:
  ```python
  if u_r.size == 0:
      qualifying = np.flatnonzero(np.ones(n_domains, dtype=np.float64) >= min_fam_dens)
      if qualifying.size == 0:
          return {}, int(n_domains)
      return {int(i): np.array([i], dtype=np.int64) for i in qualifying}, int(n_domains - qualifying.size)
  ```
  In `clustering.py:440-455` (`dpc`), nearest-higher neighbor links enforce positive similarity:
  ```python
  # Ensure s_max_higher > 0 for valid parent connections
  valid_parents = (parent_nodes != np.arange(n)) & (s_max_higher > 0.0)
  ```
- **Impact Analysis**:
  1. **Same-Sequence Alignment Filtering**: The MMseqs2 parser currently skips all alignment hits where $q\_\text{seq} == t\_\text{seq}$. In the biological context of ncRNA domain discovery, filtering same-sequence hits is an intentional scientific choice: self-sequence duplications do not constitute independent comparative evidence of a conserved domain family across sequences. While recovering tandem duplications is biologically relevant in some contexts, doing so alters domain lengths, background density, and family clustering. It should not be modified as an incidental performance change.
  2. **DPC Singleton Handling**: When a similarity graph has no edges, points rely solely on self-similarity ($\rho\_i = 1.0$). If `min_fam_dens` exceeds 1.0, all points are classified as noise (-1). This logic is correctly implemented in `family_finder` and `dpc`, but requires ongoing regression testing to ensure disconnected subgraphs remain noise-safe.
  3. **Strict Lineage Propagation**: Points must only inherit cluster labels along edges with non-zero similarity ($s\_\text{max\_higher} > 0.0$). Unconnected components or points that fail to reach a valid density peak must resolve to noise (-1) rather than forming arbitrary clusters.
- **Proposed Fix**:
  1. Retain the same-sequence filter in `aligners.py` as a deliberate biological policy. Any future relaxation (e.g., admitting non-identical tandem duplicates) must be isolated as an explicit biological feature flag with dedicated regression tests.
  2. Maintain comprehensive regression suites covering singletons, empty edge lists, zero-similarity boundaries, and multi-component graphs in `clustering.py` and `domain_finder.py`.

- **Validation**:
  1. Execute unit tests verifying that edge-free datasets evaluate solitary points against `min_fam_dens` and return correct noise counts.
  2. Confirm that points with $s\_\text{max\_higher} = 0.0$ are classified as noise (-1) when their local density fails peak thresholds.
  3. Ensure same-sequence alignments remain excluded from comparative domain discovery unless explicitly requested.

---

# Phased Implementation Plan for Future Revisions

This phased plan provides concise, mathematically sound, step-by-step guidance for future engineering agents implementing code updates.

## Phase: Concurrency Isolation and Directory Sandboxing
1. **Target File:** `domain_finder.py`
   - **Target Function:** `find_domains`
   - **Specific Instructions:**
     - Replace the hardcoded `stage4_tmp_dir = Path(tmp_dir) / "stage4_shards"` path with a unique `tempfile.TemporaryDirectory(prefix="domain_finder_stage4_", dir=tmp_dir)` instance.
     - Ensure the cleanup handler strictly purges only the isolated run-specific directory.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not hardcode shared shard folder names across concurrent runs.
     - Do not execute wildcard unlinks (`*.npy`) across a shared caller directory.

## Phase: Stage Bounded Pair Streaming and Workload Batching
1. **Target Files:** `domain_finder.py`, `clustering.py`
   - **Target Functions:** `compare_sorted_spans`, `target_batch_processor`
   - **Specific Instructions:**
     - Implement in-loop edge streaming in `compare_sorted_spans` and `target_batch_processor` that writes partial chunks to temporary disk shards once an edge buffer threshold (e.g., 500,000 edges) is reached.
     - Update Stage 4 batch sizing to account for interval count per batch in addition to sequence count.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not defer edge chunking until after `compare_sorted_spans` returns, as dense loci will exhaust worker memory during array creation.
     - Do not alter half-open interval boundaries `[start, end)` during span comparisons.

## Phase: Stage Target Buffer Capacity and Worker Result Lifecycle
1. **Target File:** `domain_finder.py`
   - **Target Function:** `find_domains`
   - **Specific Instructions:**
     - At the end of Stage 2, detach the active prefix via `all_targets_arr = target_buffer[:write_offset].copy()` if `write_offset < target_buffer.shape[0]`, then `del target_buffer` to release unused allocation before Stage 4.
     - Promptly clear processed query items in `batch_res` during Stage 2 result consumption.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not use `target_buffer.resize(..., refcheck=False)` while slice views exist in active Python scope, as this creates dangling pointers and silent memory corruption.
     - Do not assume chunked array concatenation automatically saves memory; measure peak RSS during array reconstruction.

## Phase: Stage Pair Aggregation and Memory Graph Construction
1. **Target Files:** `domain_finder.py`, `clustering.py`
   - **Target Functions:** `family_finder`, `dpc`
   - **Specific Instructions:**
     - Profile parent peak memory during shard reassembly and composite key sorting (`keys = (u_r << 32) | u_c`).
     - Investigate chunked or sparse edge reduction where $\frac{Q}{E}$ is high to eliminate redundant pairwise contributions before CSR construction.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not sum Jaccard values directly across batches; raw integer intersection lengths must be summed before computing final Jaccard similarities.
     - Do not use 32-bit integers for global pairwise intersection sums; enforce 64-bit integer accumulators.

## Phase: Interval Geometry Micro Optimizations and Coordinate Grouping Experiments
1. **Target Files:** `domain_finder.py`, `clustering.py`
   - **Target Functions:** `merge_overlapping_spans`, `compare_sorted_spans`
   - **Specific Instructions:**
     - In `merge_overlapping_spans`, add an immediate fast path for single-span targets (`if targets_on_sequence.shape[0] <= 1: return targets_on_sequence.copy()`).
     - Benchmark exact-coordinate grouping strictly after per-domain interval merging, logging metrics $M$, $G$, $Q$, and $E$.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not deduplicate coordinates across different target sequences or before consensus strand voting.
     - Do not drop domain identities when grouping identical coordinates; all domain pairs must be emitted.

## Phase: Regression Testing for Cluster Invariants and Sequence Filtering
1. **Target Files:** `aligners.py`, `domain_finder.py`, `clustering.py`
   - **Target Functions:** `parse_mmseqs_tsv`, `family_finder`, `dpc`
   - **Specific Instructions:**
     - Retain the same-sequence filter in `parse_mmseqs_tsv` as a deliberate biological policy.
     - Add regression tests verifying that edge-free graphs and points with $s\_\text{max\_higher} = 0.0$ are classified as noise (-1) when failing peak density criteria.
   - **Anti-Patterns & Risks to Avoid:**
     - Never assign noise points (-1) to valid clusters or include them in median density calculations.
     - Never alter the half-open `[start, end)` coordinate convention between aligners and clustering.
