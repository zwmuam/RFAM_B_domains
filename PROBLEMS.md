# PROBLEMS TO SOLVE

## Memory Safety and Stale Views in Target Buffer Growth

- **Evidence**:
  In `domain_finder.py:700-710`, the Stage 2 worker results collection loop processes streamed domain findings:
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
  Prior proposals recommended replacing buffer allocation with `target_buffer.resize(new_size, refcheck=False)`. However, `dest` is defined in the outer loop scope and retains a slice view into `target_buffer`.
- **Impact**:
  1. **Memory Safety Risk**: When `target_buffer.resize(..., refcheck=False)` is executed while `dest` remains alive in scope, the underlying buffer memory may be reallocated to a new base address by the system allocator. Because `refcheck=False` bypasses NumPy's reference check, `dest` is not updated and becomes a dangling pointer to deallocated or stale memory. Any subsequent access or deferred evaluation produces undefined behavior or silent memory corruption.
  2. **Memory Overhead**: While allocating `new_buffer` and copying the active prefix does temporarily hold both the old buffer and new buffer in memory, NumPy's `resize()` also does not guarantee zero-copy operating system page remapping (`mremap`) on heap memory. Uncontrolled geometric reallocation can spike resident memory when buffers hold over 100 million hits (~1.6 to 2.4 GB).
- **Proposed Fix**:
  1. Ensure slice references are explicitly deleted or properly scoped so no views exist across loop iterations.
  2. Rather than relying on fragile `refcheck=False` in-place resizing, eliminate the unconstrained single giant contiguous array allocation entirely. Use a chunked buffer strategy: accumulate target hit chunks in a list of structured arrays or pre-allocated fixed-size blocks (e.g., blocks of 10 million hits), and concatenate or stream them directly into sorted runs.
  3. If retaining single-buffer expansion, explicitly scope and delete `dest` before checking capacity and reallocating, ensuring deterministic memory handling without dangling pointer vulnerabilities.

```python
# FAULTY (domain_finder.py:700-712 - dest view leaks across iterations; naive resize creates dangling pointer):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = int((write_offset + n_hits) * 1.25) + 100_000
    target_buffer.resize(new_size, refcheck=False) # UNSAFE: dest from prior iteration holds dangling pointer

dest = target_buffer[write_offset:write_offset + n_hits]
dest['t_seq'] = all_targets[:, 0]
dest['start'] = all_targets[:, 1]
dest['end'] = all_targets[:, 2]
dest['dom_id'] = all_targets[:, 3] + global_offset
write_offset += n_hits

# REVISED (domain_finder.py - Scoped view deletion and safe chunked accumulation or guarded reallocation):
# Explicitly delete any slice reference to prevent dangling views
if 'dest' in locals():
    del dest

if write_offset + n_hits > target_buffer.shape[0]:
    new_size = max(int((write_offset + n_hits) * 1.25) + 100_000, target_buffer.shape[0] * 2)
    new_buffer = np.empty(new_size, dtype=target_dtype)
    new_buffer[:write_offset] = target_buffer[:write_offset]
    target_buffer = new_buffer

dest = target_buffer[write_offset:write_offset + n_hits]
dest['t_seq'] = all_targets[:, 0]
dest['start'] = all_targets[:, 1]
dest['end'] = all_targets[:, 2]
dest['dom_id'] = all_targets[:, 3] + global_offset
write_offset += n_hits
del dest  # Ensure slice view does not persist into the next iteration
```

- **Validation**:
  1. Execute a synthetic test generating $> 50$ buffer reallocations with verified data integrity across all indices.
  2. Confirm that no memory corruption, segmentation faults, or stale pointer overwrites occur.
  3. Measure peak Resident Set Size (RSS) during Stage 2 to verify bounded memory footprint.

---

## Target Span De-Replication Trade-offs and Downstream Scaling Risks

- **Evidence**:
  In `domain_finder.py:220-226`, discovered query domains emit partner target coordinates and synthetic self-targets:
  ```python
  if valid_mask.any():
      partner_targets = np.column_stack((t_seqs, t_starts, t_ends, t_doms))
      all_targets = np.vstack((partner_targets, self_targets))
  else:
      all_targets = self_targets
  ```
  In Stage 4 (`target_batch_processor`, `domain_finder.py:380-410`), target intervals are partitioned per target sequence and grouped by domain ID (`s_doms`):
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
- **Impact Analysis**:
  A superficial inspection suggests compressing identical target coordinates across alignments or domains. However, rigorous algorithmic and biological analysis reveals strict trade-offs:
  1. **Intra-Domain vs. Inter-Domain Redundancy**:
     - *Intra-domain duplicates*: Identical alignment coordinates mapped to the same domain instance occur when multiple query hits cover the same region. However, `merge_overlapping_spans` in Stage 4 already calculates the union of all intervals for each domain instance before computing domain lengths and cross-domain overlaps. Running `np.unique` per query domain in Stage 2 incurs sorting and memory allocation overhead ($O(N \log N)$ per query) that often exceeds the minor memory saved in `target_buffer`, particularly when duplicate rates are low.
     - *Inter-domain identical coordinates*: Multiple distinct domains $D\_{1}, D\_{2}, \dots, D\_{K}$ frequently hit the exact same target coordinates $[start, end)$ on a conserved target sequence. Compressing these into an inverted index (storing the coordinate once with a list of $K$ domain IDs) avoids storing duplicate coordinates. However, in Stage 5, DPC family clustering requires pairwise Jaccard similarities:
       $$J(D\_{i}, D\_{j}) = \frac{\sum \text{overlap}}{\text{Length}(D\_{i}) + \text{Length}(D\_{j}) - \sum \text{overlap}}$$
       To compute this, every pair among the $K$ domains sharing the interval $[start, end)$ must emit an overlap intersection of length $(end - start)$.
  2. **Combinatorial Explosion Risk**:
     If $K$ domains share an identical interval, the number of distinct pairwise edges is:
     $$\binom{K}{2} = \frac{K(K - 1)}{2}$$
     For a conserved domain locus shared across $10,000$ sequences or instances, this single interval generates $49,995,000$ pairwise edges. Storing coordinates compactly does nothing to alleviate this downstream quadratic edge explosion. If a compression scheme requires decompressing or expanding these pairs into intermediate Python structures, memory usage and serialization overhead spike dramatically.
  3. **Biological Model Integrity**:
     Deduplicating hits before Stage 2 completes would alter alignment coverage density and strand consensus voting (`determine_strand`), distorting the biological model. Merging must only happen after query domain peak discovery and strand resolution.
- **Proposed Fix**:
  Adopt a balanced, low-overhead optimization strategy:
  1. **Keep 1D Contiguous Structured Records**: Retain the flat structured array representation `[('t_seq', int32), ('start', int32), ('end', int32), ('dom_id', int32)]`. Avoid ragged lists or nested dictionaries that degrade multiprocessing serialization speed and cause IPC bottlenecks.
  2. **Fast-Path Interval Merging**: In `merge_overlapping_spans`, add an immediate $O(1)$ fast-path when `len(spans) <= 1`. This bypasses `lexsort`, `maximum.accumulate`, and array slicing for solitary spans, saving substantial CPU time across millions of single-hit targets without changing output coordinates.
  3. **Profile Redundancy Ratio Before Inverted Indexing**: Do not impose an invasive inverted-index coordinate deduplication engine unless an empirical target redundancy check ($U/M$ ratio of unique coordinates over total occurrences) proves significant coordinate replication ($> 60\%$). If implemented, perform coordinate grouping only *after* per-domain union merging, and emit pairwise edges in bounded streaming batches directly to sparse accumulator shards to prevent worker memory exhaustion.

```python
# FAULTY (domain_finder.py:308-329 - Executes full lexsort and accumulate even for single-span targets):
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

# REVISED (domain_finder.py - O(1) early exit for solitary intervals preserving exact coordinates):
def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
    n = len(targets_on_sequence)
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
  1. Verify identical output coordinates and domain lengths from `merge_overlapping_spans` across edge cases (empty, single span, disjoint spans, nested spans, identical duplicates, adjacent abutting spans).
  2. Benchmark Stage 4 runtime and memory on datasets with varying redundancy levels ($0\%$, $25\%$, $50\%$, $80\%$).
  3. Validate that final family partitions and Jaccard similarities remain identical.

---

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
  And in the `finally` cleanup block (`domain_finder.py:808-816`):
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
  1. **Race Conditions in Concurrent Executions**: When multiple pipeline runs or worker tasks share a common `tmp_dir` (e.g., `/tmp` or a shared cluster scratch mount), all processes write to the exact same directory `tmp_dir / "stage4_shards"` using identical chunk filenames (`0_r.npy`, `1_r.npy`, etc.). One run will overwrite or prematurely unlink another run's binary shards while reading, resulting in file not found errors, corrupt edge arrays, or wrong cluster outputs.
  2. **Unintended Deletion of Foreign Files**: The cleanup routine executes `for f in stage4_tmp_dir.glob("*.npy"): f.unlink()`. If a separate pipeline execution is writing shards to that directory, its data files will be wiped out.
  3. **Resource Leaks on Failure**: If a worker crashes or an unhandled exception occurs before shard creation completes, partially written shards in shared temporary directories may be orphaned.
- **Proposed Fix**:
  Always create a dedicated, uniquely named subdirectory using `tempfile.TemporaryDirectory` with a process-specific and UUID/timestamp suffix, regardless of whether `tmp_dir` is supplied by the user or defaulted. Restrict all shard writes, reads, and cleanup operations strictly to that isolated per-run directory. Use context managers to guarantee cleanup even upon worker exceptions.

```python
# FAULTY (domain_finder.py:750-760 - Hardcoded shared directory path causes collisions in concurrent runs):
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

## Memory Spikes and Algorithmic Bottlenecks in Pairwise Overlap Search

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
  And in Stage 4 worker accumulation (`domain_finder.py:420-435`), each worker accumulates all edge lists for its assigned sequence batch, concatenates them into single arrays (`r_arr`, `c_arr`, `i_arr`), and writes them to disk.
- **Impact**:
  1. **Python Loop Allocation Overhead**: For loci with thousands of spans, `compare_sorted_spans` executes repeated slice operations and allocates multiple ephemeral arrays per interval (`overlap`, `valid`, `js`, `lows`). This places high pressure on Python's memory manager.
  2. **Unbounded Worker Memory on Dense Targets**: Sharding coordinates to disk after processing a batch bounds memory in the parent process, but does **not** protect individual workers. If a batch contains a highly repetitive or dense target sequence where thousands of domains overlap, a single worker can generate millions of pairwise intersections in memory before writing to disk, triggering an uncatchable Out-Of-Memory (OOM) crash in worker processes.
  3. **Stage 5 Graph Retention**: In `family_finder`, all edges from all shards are loaded into `all_rows`, `all_cols`, and `all_inters` simultaneously before canonicalizing keys, sorting, and building CSR matrices.
- **Proposed Fix**:
  1. **Batch Sizing by Workload, Not Just Sequence Count**: Scale Stage 4 task chunking not only by sequence count, but by the number of domain intervals per batch, preventing heavily populated sequences from overloading single workers.
  2. **Bounded Worker Edge Buffers**: Inside `target_batch_processor`, flush accumulated edge arrays to intermediate shard chunks if the accumulated edge count exceeds a threshold (e.g., 2 million edges), keeping worker heap consumption strictly bounded.
  3. **Direct Sparse Edge Accumulation**: Canonicalize edges `(min(u, v), max(u, v))` and sum intersections using 64-bit composite keys or a sparse coordinate accumulator to avoid duplicate symmetric sort passes.

```python
# FAULTY (clustering.py:27-44 - Repeated Python allocations per active interval):
for i in range(n):
    end_i = ends[i]
    r = max(i + 1, int(np.searchsorted(starts, end_i, side="left")))
    if r > i + 1:
        st_js = starts[i + 1:r]
        en_js = ends[i + 1:r]
        overlap = np.minimum(end_i, en_js) - st_js
        ...

# REVISED (clustering.py - Vectorized active search and pre-filtered boundaries):
r_indices = np.maximum(np.arange(1, n + 1, dtype=np.int32),
                       np.searchsorted(starts, ends, side="left").astype(np.int32))
active_i = np.flatnonzero(r_indices > (np.arange(n, dtype=np.int32) + 1))

lows, highs, inters = [], [], []
for i in active_i:
    r = r_indices[i]
    st_js = starts[i + 1:r]
    en_js = ends[i + 1:r]
    overlap = np.minimum(ends[i], en_js) - st_js
    valid = overlap > 0
    if np.any(valid):
        js = np.arange(i + 1, r, dtype=np.int32)[valid]
        lows.append(np.full(js.size, i, dtype=np.int32))
        highs.append(js)
        inters.append(overlap[valid])
```

- **Validation**:
  1. Benchmark `compare_sorted_spans` on simulated dense loci ($> 20,000$ overlapping spans).
  2. Verify edge pairs, intersections, and coordinates match existing implementation bit-for-bit.
  3. Confirm worker process RSS remains strictly bounded when processing high-depth target sequences.

---

# Phased Implementation Plan for Future Revisions

This phased plan provides concise, mathematically sound, step-by-step guidance for future engineering agents implementing code updates.

## Phase: Buffer Lifetime and Concurrency Isolation
1. **Target File:** `domain_finder.py`
   - **Target Functions:** `find_domains`, `target_batch_processor`
   - **Specific Instructions:**
     - In `find_domains` Stage 2, eliminate cross-iteration view retention by explicitly deleting `dest` slices before checking capacity and reallocating `target_buffer`.
     - In `find_domains` Stage 4, replace the hardcoded `stage4_tmp_dir = Path(tmp_dir) / "stage4_shards"` path with a unique `tempfile.TemporaryDirectory(prefix="domain_finder_stage4_", dir=tmp_dir)` instance.
     - Ensure the cleanup handler strictly purges only files within the run-specific temporary directory and gracefully releases directory handles upon completion or failure.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not use `target_buffer.resize(..., refcheck=False)` while slice views exist in active Python scope, as this leaves dangling pointers and causes silent memory corruption.
     - Do not hardcode shared shard folder names across concurrent runs.

## Phase: Target Span Merging and Overlap Vectorization
1. **Target Files:** `domain_finder.py`, `clustering.py`
   - **Target Functions:** `merge_overlapping_spans`, `compare_sorted_spans`, `target_batch_processor`
   - **Specific Instructions:**
     - In `merge_overlapping_spans`, insert an early exit `if len(targets_on_sequence) <= 1: return targets_on_sequence.copy()`.
     - In `compare_sorted_spans`, precompute binary search upper bounds using vectorized `np.searchsorted` and filter to active indices (`active_i`) before the loop.
     - In `target_batch_processor`, track accumulated edge counts per batch and write partial chunks to disk if edge count exceeds 2,000,000 edges to bound worker RSS.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not apply `np.unique` deduplication across different domain IDs before pairwise overlap calculations; multi-domain identical loci must emit complete pairwise overlaps for valid Stage 5 DPC Jaccard similarities.
     - Do not alter half-open interval semantics `[start, end)` during span comparisons.

## Phase: Algorithmic Integrity and Scaling
1. **Target Files:** `aligners.py`, `domain_finder.py`, `clustering.py`
   - **Target Functions:** `parse_mmseqs_tsv`, `family_finder`, `dpc`
   - **Specific Instructions:**
     - In `parse_mmseqs_tsv`, preserve legitimate tandem duplications on the same sequence by filtering only exact coordinate identity (`raw_qs == raw_ts and raw_qe == raw_te`).
     - In `family_finder`, ensure solitary domains with zero graph edges are assigned to valid clusters only if their self-similarity density meets `min_fam_dens`; otherwise map them to noise (`-1`).
     - In `dpc`, preserve non-zero similarity requirements (`s_max_higher > 0.0`) when propagating cluster assignments to density peaks, ensuring disconnected components do not coalesce into arbitrary clusters.
   - **Anti-Patterns & Risks to Avoid:**
     - Never assign noise points (`-1`) to valid clusters or include them in median density calculations.
     - Avoid floating-point arithmetic perturbations for tie-breaking; use deterministic integer lexicographic sorting.
