# Core Domain-Finding Pipeline Audit and Architectural Risk Assessment

---

## 1. Problem Analysis

**Problem 1: Bypassing Minimum Density Filtering in Disjoint DPC (`clustering.py`, `domain_finder.py`)**

- **Module / Component:** `clustering.py` (`dpc`), `domain_finder.py` (`domain_instances_on_query`, `family_finder`)
- **Severity / Category:** CRITICAL | Mathematical & Biological Edge-Case Inconsistency
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:dpc`, when no pairwise similarities exist across candidate elements (`rows.size == 0`), the function evaluates an immediate early-exit check:
  `if rows.size == 0: return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0`
  This unconditionally classifies every solitary or mutually disjoint point as a valid cluster peak and returns zero noise points (`n_noise = 0`), completely bypassing the minimum local density threshold (`min_dens`).
  `PREVIOUS_UPDATE.md` claimed in Problem 2 that minimum density filtering was enforced across all topologies. While a specialized branch was added inside `domain_finder.py:domain_instances_on_query` for query instances, the core algorithm `clustering.dpc` was left untouched.
  Consequently, when `clustering.dpc` is invoked by `domain_finder.py:family_finder` on domain families where candidate domains share zero target overlaps (`rows.size == 0`), every single domain instance is unconditionally formed into a validated domain family peak, even if its alignment support is strictly below $\text{min\_fam\_dens}$.
  Biologically, this defeats the primary objective of minimum density filtering: preventing isolated, low-confidence alignment noise hits from creating spurious domain families.
- **Evidence in Code:**
  - `clustering.py:586-587`: `if rows.size == 0: return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0`
  - `domain_finder.py:389-390`: `if rows.size == 0 or n_domains <= 1: return {int(i): np.array([i], dtype=np.int64) for i in range(n_domains)}, 0`
- **Recommended Solution:**
  Enforce uniform density thresholding in `clustering.py:dpc` regardless of whether `rows` contains elements. Compute the effective density `rho = np.ones(n)` (or `weights` if supplied) when `rows.size == 0`. Only points satisfying $\rho\_i \ge \text{min\_dens}$ qualify as peaks. Points failing the density cutoff must be assigned to noise ($-1$). Similarly, `family_finder` must validate candidate domains against $\text{min\_fam\_dens}$.

```python
# FAULTY (clustering.py:586-587 - density threshold bypassed when rows.size == 0):
if rows.size == 0:
    return {int(p): np.array([p], dtype=np.int64) for p in range(n)}, 0

# REVISED (clustering.py: Uniformly enforce rho >= min_dens across empty edge graphs):
if rows.size == 0:
    densities = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    qualifying = np.flatnonzero(densities >= min_dens)
    if qualifying.size == 0:
        return {}, int(n)
    peak_to_cluster = {int(p): np.array([p], dtype=np.int64) for p in qualifying}
    return peak_to_cluster, int(n - qualifying.size)
```

---

**Problem 2: Strand-Blind Merging in Target Hits and GFF3 Cluster Evaluation (`domain_finder.py`)**

- **Module / Component:** `domain_finder.py` (`domain_instances_on_query`, `target_batch_processor`, `evaluate_clusters`)
- **Severity / Category:** HIGH | Biological Validity & Annotation Corruption
- **Detailed Bioinformatic & Technical Analysis:**
  Non-coding RNA transcripts and regulatory elements function strictly in a directionally constrained manner. Transcription from opposite genomic strands (`+` vs `-`) produces distinct sequence molecules with inverted secondary structure folding thermodynamics and distinct biological functions.
  In the current pipeline:
  1. `domain_finder.py:domain_instances_on_query` determines a consensus strand for the query domain instance, but subsequently strips strand information when emitting `all_targets`.
  2. `target_batch_processor` aggregates all target hits on a target sequence and merges same-domain intervals using `merge_overlapping_spans(d_spans)` regardless of hit orientation. If a domain matches both the forward and reverse strand of a locus (e.g., in inverted repeats, convergent promoters, or bidirectional ncRNA loci), hits on opposite strands are merged into a single chimeric interval.
  3. `domain_finder.py:evaluate_clusters` groups all member domain instances on the same query sequence and unconditionally merges overlapping spans using `merge_overlapping_spans(spans_arr)`. The consensus strand is decided by simple majority voting of the merged instances (`neg > pos`).
  This causes biologically distinct antisense domains located on opposite strands of the same locus to be merged into a single feature, erasing the antisense feature and misrepresenting genomic coordinates.
- **Evidence in Code:**
  - `domain_finder.py:214-219`: `partner_targets = np.column_stack((t_seqs, t_starts, t_ends, t_doms))` completely drops target alignment strands (`t_strands`).
  - `domain_finder.py:447-458`: `cluster_domains_per_seq[d[0]].append(d)` groups strictly by `seq_id` (`d[0]`) rather than `(seq_id, strand)`.
  - `domain_finder.py:459-467`: Overlapping domains on opposite strands are merged and assigned a single consensus strand.
- **Recommended Solution:**
  Partition domain instances by `(seq_id, strand)` prior to interval merging in `evaluate_clusters`. Same-cluster domain instances on opposite strands must be preserved as separate GFF3 records. Similarly, retain strand qualifiers in target hit processing where bidirectional transcriptomics requires strand discrimination.

```python
# FAULTY (domain_finder.py:447-458 - grouping strictly by sequence ID, ignoring strand):
cluster_domains_per_seq: Dict[int, List[Tuple[int, int, int, str]]] = defaultdict(list)
for d_id in member_ids:
    d = domain_registry[int(d_id)]
    cluster_domains_per_seq[d[0]].append(d)

# REVISED (domain_finder.py: Group by (seq_id, strand) to preserve antisense features):
cluster_domains_per_seq: Dict[Tuple[int, str], List[Tuple[int, int, int, str]]] = defaultdict(list)
for d_id in member_ids:
    d = domain_registry[int(d_id)]
    cluster_domains_per_seq[(d[0], d[3])].append(d)

component_domains: List[Tuple[int, int, int, str]] = []
for (seq_id, dom_strand), domains in cluster_domains_per_seq.items():
    if len(domains) == 1:
        component_domains.append(domains[0])
        continue
    sorted_domains = sorted(domains, key=lambda x: x[1])
    spans_arr = np.array([[x[1], x[2]] for x in sorted_domains], dtype=np.int32)
    merged_spans = merge_overlapping_spans(spans_arr)
    for m_start, m_end in merged_spans:
        component_domains.append((seq_id, int(m_start), int(m_end), dom_strand))
```

---

**Problem 3: Unsafe Array Resizing with `np.resize` in Target Hit Buffer (`domain_finder.py`)**

- **Module / Component:** `domain_finder.py` (`find_domains`, Stage 3 Target Hit Aggregation)
- **Severity / Category:** HIGH | Numerical & Algorithmic Bug / Memory Corruption Risk
- **Detailed Bioinformatic & Technical Analysis:**
  During Stage 3 target hits aggregation in `domain_finder.py:find_domains`, the target buffer is dynamically resized when `write_offset + n_hits > target_buffer.shape[0]`:
  ```python
  new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
  target_buffer = np.resize(target_buffer, new_size)
  ```
  `np.resize(a, new_shape)` does **not** behave like C `realloc` or Python `list.extend`. If the new size is larger than the original array, `np.resize` fills the new space by **repeating copies of the original array cyclically**.
  While current code immediately overwrites newly added elements starting at `write_offset`, if any error or bounds mismatch occurs, or if future logic accesses `target_buffer[write_offset:]`, the buffer contains phantom duplicated alignment hits rather than uninitialized or zero-filled memory.
  Furthermore, `np.resize` unconditionally allocates a full copy of the array and copies all bytes, triggering memory thrashing and temporary 2x RAM spikes.
- **Evidence in Code:**
  - `domain_finder.py:596-615`: Use of `np.resize(target_buffer, new_size)`.
- **Recommended Solution:**
  Replace `np.resize` with explicit pre-allocation via `np.empty(new_size, dtype=target_dtype)` and copy strictly the active prefix `dest[:write_offset] = target_buffer[:write_offset]`, followed by `del target_buffer`. Alternatively, rely on an expanding list of 1D structured arrays chunked at 500,000 entries and concatenate once upon completion.

```python
# FAULTY (domain_finder.py:614 - np.resize tiles previous elements into the buffer):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
    target_buffer = np.resize(target_buffer, new_size)

# REVISED (Explicit, safe allocation without cyclic repeating):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = max(target_buffer.shape[0] * 2, write_offset + n_hits + 100_000)
    new_buffer = np.empty(new_size, dtype=target_dtype)
    new_buffer[:write_offset] = target_buffer[:write_offset]
    target_buffer = new_buffer
```

---

**Problem 4: O(N) Sub-Array Slicing in `compare_sorted_spans` Range Queries (`clustering.py`)**

- **Module / Component:** `clustering.py` (`compare_sorted_spans`)
- **Severity / Category:** MEDIUM | Performance Bottleneck / Excessive Micro-Allocations
- **Detailed Bioinformatic & Technical Analysis:**
  In `clustering.py:compare_sorted_spans`, pairwise interval overlaps are computed using a binary search range scan:
  ```python
  for i in range(n):
      end_i = ends[i]
      r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)
  ```
  `starts[i + 1:]` instantiates a new NumPy slice object and array view on every iteration of the loop ($N$ slice allocations). For large query sequences or dense target loci with $N = 50,000$ spans, this allocates tens of thousands of intermediate NumPy objects, triggering substantial Python interpreter overhead and garbage collection cycles.
  Because the global `starts` array is already sorted in non-descending order, `np.searchsorted(starts, end_i, side="left")` can be executed directly on the whole array without slicing, obtaining the boundary index $r$ directly in $\mathcal{O}(\log N)$ without sub-array allocations.
  Additionally, redundant type-checking `if not isinstance(starts, np.ndarray)` is executed on `starts = spans[:, 0]`, which is guaranteed to be an `ndarray`.
- **Evidence in Code:**
  - `clustering.py:25-28`: Dead type check on `starts`.
  - `clustering.py:33-35`: `r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)`.
- **Recommended Solution:**
  Perform `np.searchsorted` on the global `starts` array directly, clamping the lower bound to `i + 1`. Remove redundant type assertions.

```python
# FAULTY (clustering.py:34 - slice allocation per iteration):
for i in range(n):
    end_i = ends[i]
    r = np.searchsorted(starts[i + 1:], end_i, side="left") + (i + 1)

# REVISED (Direct searchsorted on full sorted array without sub-array slicing):
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

- **Module / Component:** `domain_finder.py` (`find_domains`, `target_batch_processor`)
- **Severity / Category:** MEDIUM | Memory Bottleneck & Parallel Worker Starvation
- **Detailed Bioinformatic & Technical Analysis:**
  Following Stage 3 hit collection, `find_domains` partitions `all_targets_arr` into per-sequence arrays:
  ```python
  seq_breaks = np.flatnonzero(np.diff(all_targets_arr['t_seq'])) + 1
  target_seq_groups = np.split(all_targets_arr, seq_breaks)
  ```
  For large genomic benchmarks with 100,000 target sequences, `np.split` creates 100,000 individual array slices stored in a Python list.
  Then, `target_batch_generator` groups these into batches of 1,000 sequences and submits them to `Parallel`.
  In `executor.py:Parallel`, generator inputs default to `chunk_size = 32`. MPIRE therefore groups $32 \times 1,000 = 32,000$ target sequences into a single task chunk.
  This introduces two critical performance flaws:
  1. **Worker Starvation on Small-to-Medium Datasets:** When a dataset has fewer than 32,000 target sequences (e.g., 5,000 targets across 16 CPU cores), all target sequences are yielded in fewer than 32 batches, causing MPIRE to dispatch all work to a single worker while 15 cores remain idle.
  2. **Suboptimal Grouping in `target_batch_processor`:** Within each worker, `target_batch_processor` runs an unvectorized loop over `uniq_domains = np.unique(domain_pointers)` for every sequence, applying boolean masks `domain_pointers == d_id` one domain at a time rather than sorting by `(t_seq, dom_id)` and performing contiguous grouping.
- **Evidence in Code:**
  - `domain_finder.py:630-631`: `np.split` creating lists of tens of thousands of array objects.
  - `domain_finder.py:648-650`: Batch size of 1,000 coupled with `chunk_size = 32` in `executor.py:279`.
  - `domain_finder.py:351-360`: Unvectorized domain loop with dynamic boolean masking inside workers.
- **Recommended Solution:**
  In `target_batch_generator`, adapt batch sizing dynamically based on `cpu` thread count: $\text{batch\_size} = \max(10, \min(500, \text{n\_target\_seqs} // (\text{cpu} \times 4)))$, and set `chunk_size = 1` for batch generators. Within `target_batch_processor`, sort records by `(domain_pointers, start)` and use `np.split` on domain boundaries rather than repeated boolean masking.

```python
# FAULTY (executor.py & domain_finder.py: coarse generator chunking causing starvation):
# domain_finder.py:
if len(batch_pkg) >= 1000:
    yield batch_pkg
# executor.py:
if chunk_size == 'auto':
    chunk_size = 32

# REVISED (Adaptive batch sizing to guarantee thread saturation across all worker cores):
target_batch_size = max(10, min(250, n_target_seqs // max(1, cpu * 4)))
def target_batch_generator():
    batch_pkg = []
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

**Problem 6: Contradictory Data Typing Claims in `PREVIOUS_UPDATE.md` and Docstrings (`domain_finder.py`, `clustering.py`)**

- **Module / Component:** `domain_finder.py` (`target_batch_processor`, `evaluate_clusters`), `clustering.py` (`dpc`)
- **Severity / Category:** MEDIUM | Data Typing Inconsistency & Documentation Drift
- **Detailed Bioinformatic & Technical Analysis:**
  `PREVIOUS_UPDATE.md` Claim 3 stated that coordinates and interval arrays were strictly standardized to `int32` across `target_batch_processor` and `find_domains`.
  However, an audit of the current codebase reveals multiple unresolved inconsistencies:
  1. `target_batch_processor` docstring (`domain_finder.py:342`) explicitly states:
     `Maintains 64-bit integers for all physical genomic coordinate operations.`
     This directly contradicts the internal code which was changed to `np.int32`.
  2. `domain_finder.py:evaluate_clusters` (`line 454`) still explicitly casts interval spans to `np.int64`:
     `spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int64)`
  3. `clustering.py:dpc` (`lines 594-595`) unconditionally converts input `rows` and `cols` from `np.int32` to `np.int64`:
     `rows = np.ascontiguousarray(rows, dtype=np.int64)`
     `cols = np.ascontiguousarray(cols, dtype=np.int64)`
     Doubling array allocations from 4 bytes to 8 bytes per coordinate across millions of edges.
  4. Dead import: `count_fasta` is imported from `aligners` into `domain_finder.py:24` but is never referenced anywhere in `domain_finder.py`.
- **Evidence in Code:**
  - `domain_finder.py:342`: Outdated docstring claiming 64-bit coordinate operations.
  - `domain_finder.py:454`: `dtype=np.int64` in `spans_arr`.
  - `domain_finder.py:24`: Unused import `count_fasta`.
  - `clustering.py:594-595`: Unnecessary upcasting to `int64` for sparse edge indices.
- **Recommended Solution:**
  Harmonize all coordinate and span arrays to `np.int32` throughout `evaluate_clusters` and `clustering.py`. Update all docstrings to reflect the true 32-bit coordinate boundary, and purge the unused `count_fasta` import from `domain_finder.py`.

```python
# FAULTY (domain_finder.py:454 & clustering.py:594 - redundant int64 allocations):
# domain_finder.py:
spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int64)
# clustering.py:
rows = np.ascontiguousarray(rows, dtype=np.int64)
cols = np.ascontiguousarray(cols, dtype=np.int64)

# REVISED (Harmonized int32 representations throughout):
# domain_finder.py:
spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int32)
# clustering.py:
rows = np.ascontiguousarray(rows, dtype=np.int32)
cols = np.ascontiguousarray(cols, dtype=np.int32)
```

---

**Problem 7: Style Violations of Repository PEP 8 Visual Indent and Line Length (`domain_finder.py`, `clustering.py`)**

- **Module / Component:** `domain_finder.py`, `clustering.py`, `executor.py`
- **Severity / Category:** LOW | Code Quality & PEP 8 Visual Indent Compliance
- **Detailed Bioinformatic & Technical Analysis:**
  The repository specification requires strict delimiter-anchored visual alignment (Visual Indent Style) and an absolute maximum line length of $\le 99$ characters.
  An automated line length audit revealed:
  - `domain_finder.py`: 11 lines exceeding 99 characters (up to 105 characters, e.g., lines 101, 192, 205).
  - `clustering.py`: 31 lines exceeding 99 characters (up to 123 characters, e.g., lines 22, 49, 67, 107).
  - `executor.py`: 5 lines exceeding 99 characters (up to 111 characters, e.g., lines 22, 190, 258).
  These lines violate the project's formatting standard (`templates/formatting_template.py`) and must be wrapped to maintain visual consistency.
- **Evidence in Code:**
  - `clustering.py:67`: `# check for indices outside the valid range >1 and <0, which can occur due to numerical issues...` (123 chars)
  - `domain_finder.py:205`: `suspect_intersections=intersections[peak_mask], suspect_lengths=span_lengths, ...` (105 chars)
- **Recommended Solution:**
  Refactor all lines exceeding 99 characters to comply strictly with the visual alignment style defined in `templates/formatting_template.py`.

---

## 2. Architectural Risks and Warning Guidelines

The following risks should be considered prior to code refactoring:

1. **Euclidean Halo Filtering on Structured RNA:**
   Traditional Rodriguez-Laio halo filtering assumes Gaussian clusters with dense spherical centers. ncRNA domains have non-uniform coverage and ragged alignment boundaries. Applying traditional halo filtering can aggressively discard legitimate biological variants and multimeric duplications. The `similarity_weighted` halo method with `point_specific=True` must remain the default.

2. **Multiprocessing Context Under Linux (`forkserver` vs `spawn`):**
   The monkey-patch in `executor.py` correctly redirects MPIRE's `SyncManager` to `forkserver`. However, if `forkserver` is initialized after background threads have been spawned, worker processes can still inherit corrupted lock states. Any synchronization manager initialization must strictly precede worker pool dispatch.

3. **Peak Memory Ceiling on Multi-Gigabyte Alignments:**
   For multi-GB alignment files ($> 10^7$ hits), sorting `all_targets_arr` in-place requires up to 32 bytes per hit. For 100 million hits, this requires 3.2 GB of RAM. The pipeline must avoid converting structured arrays into intermediate Python dictionaries or object lists.

---

## 3. Phased Implementation Plan

This phased plan outlines the sequence of updates recommended for future execution:

1. **Phase 1: Algorithmic & Biological Correctness**
   - Correct disjoint density thresholding in `clustering.py:dpc` so that points failing $\text{min\_dens}$ are assigned to noise ($-1$).
   - Refactor `domain_finder.py:evaluate_clusters` to group domain instances by `(seq_id, strand)` prior to interval merging, preventing antisense domain obliteration.
   - Replace `np.resize` in `domain_finder.py:find_domains` with safe buffer copying.

2. **Phase 2: Performance & Concurrency Optimization**
   - Optimize `compare_sorted_spans` in `clustering.py` by eliminating sub-array slicing in `np.searchsorted`.
   - Update Stage 4 batch generation in `domain_finder.py` with dynamic batch sizing and `chunk_size = 1` to eliminate worker starvation on multi-core systems.
   - Refactor `target_batch_processor` to use sorted domain slicing rather than repeated boolean masking.

3. **Phase 3: Coordinate Type Harmonization & Cleanup**
   - Standardize all coordinates, spans, and local index arrays to `np.int32` in `evaluate_clusters` and `clustering.py`.
   - Purge dead import `count_fasta` from `domain_finder.py`.
   - Correct outdated docstrings in `target_batch_processor`.

4. **Phase 4: Style Compliance & Test Verification**
   - Reformat all lines exceeding 99 characters across `domain_finder.py`, `clustering.py`, and `executor.py` to adhere to `templates/formatting_template.py`.
   - Run `python3 -m pytest -v -W error::DeprecationWarning` to ensure zero regressions across the complete test suite.
