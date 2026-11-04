# PROBLEMS TO SOLVE

## Incomplete Implementation of In-Place Target Buffer Growth in Stage 2

- **Evidence**:
  In `PREVIOUS_UPDATE.md`, under "Memory Expansion During Geometric Target Buffer Reallocation", it is claimed that the pipeline replaces duplicate buffer allocations with in-place buffer growth via `target_buffer.resize(new_size, refcheck=False)`.
  However, in `domain_finder.py:526-532`:
  ```python
  if write_offset + n_hits > target_buffer.shape[0]:
      ## mem_opt ## Safe 1.25x buffer expansion copying only active prefix without cyclic data duplication
      new_size = int((write_offset + n_hits) * 1.25) + 100_000
      new_buffer = np.empty(new_size, dtype=target_dtype)
      new_buffer[:write_offset] = target_buffer[:write_offset]
      target_buffer = new_buffer
  ```
- **Impact**:
  The claim in `PREVIOUS_UPDATE.md` was not fulfilled in the actual codebase. Whenever `target_buffer` expands, allocating `new_buffer` while retaining `target_buffer` duplicates memory allocation in the parent heap. For a buffer holding $1.5 \times 10^8$ target hits ($2.4\text{ GB}$ of structured records), allocating a replacement $3.0\text{ GB}$ buffer incurs a transient peak of over $5.4\text{ GB}$ and triggers redundant array copying.
- **Proposed Fix**:
  Because no external views or slices of `target_buffer` are retained across loop iterations, `target_buffer.resize(new_size, refcheck=False)` can be called directly to reallocate memory in-place via operating system page remapping (`mremap` on Linux) without creating duplicate buffer allocations.

```python
# FAULTY (domain_finder.py:526-532 - Still allocates new_buffer and copies active prefix):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = int((write_offset + n_hits) * 1.25) + 100_000
    new_buffer = np.empty(new_size, dtype=target_dtype)
    new_buffer[:write_offset] = target_buffer[:write_offset]
    target_buffer = new_buffer

# REVISED (domain_finder.py - In-place buffer expansion without duplicate allocation):
if write_offset + n_hits > target_buffer.shape[0]:
    new_size = int((write_offset + n_hits) * 1.25) + 100_000
    target_buffer.resize(new_size, refcheck=False)
```

- **Validation**:
  1. Profile memory allocation during Stage 2 under high target hit counts ($> 5 \times 10^7$ hits).
  2. Confirm that peak Resident Set Size (RSS) during buffer growth never exceeds the target buffer size by more than single-page allocations, eliminating duplicate buffer memory peaks.

---

## Redundancy in Target Spans and Feasibility of Compression

- **Evidence**:
  In `domain_finder.py:535-545`:
  ```python
  dest = target_buffer[write_offset:write_offset + n_hits]
  dest['t_seq'] = all_targets[:, 0]
  dest['start'] = all_targets[:, 1]
  dest['end'] = all_targets[:, 2]
  dest['dom_id'] = all_targets[:, 3] + global_offset
  write_offset += n_hits
  ```
  In Stage 4 (`target_batch_processor`):
  ```python
  for d_group, sp_group in zip(dom_splits, span_splits):
      d_id = int(d_group[0])
      m_spans = merge_overlapping_spans(sp_group)
      for sp in m_spans:
          merged_spans_list.append(sp)
          aligned_domains_list.append(d_id)
          batch_dom_lengths[d_id] += int(sp[1] - sp[0])
  ```
- **Impact Analysis**:
  In large homology datasets, many domain instances hit identical target sequence regions with identical coordinates $[start, end)$. Two distinct replication scenarios occur:
  1. **Intra-domain duplicate hits**: Multiple query alignments mapped to the same domain instance produce identical target hits on sequence $T$. Retaining duplicates in `target_buffer` wastes heap memory and incurs sorting overhead.
  2. **Inter-domain identical spans**: Multiple distinct domain instances $D_1, D_2, \dots, D_K$ align to the exact same physical coordinates $[start, end)$ on target sequence $T$.

  *Downstream Dependency Consideration*:
  If $K$ domains hit the exact same target interval $[s, e)$ on sequence $T$, the pipeline requires pairwise overlaps between all pairs $(D_i, D_j)$ to compute Jaccard similarities in Stage 5:
  $$J(D_i, D_j) = \frac{\sum \text{overlap}}{\text{Length}(D_i) + \text{Length}(D_j) - \sum \text{overlap}}$$
  If identical spans are compressed into an inverted index (e.g., coordinates mapping to a list of domain IDs), generating the $K(K-1)/2$ pairwise overlap entries cannot be avoided without restructuring Stage 5 DPC. Storing ragged domain lists in `target_buffer` breaks the contiguous structured 1D NumPy array representation, increasing Python object allocation overhead and complicating disk sharding.
- **Proposed Fix**:
  Apply targeted, low-overhead deduplication where it yields immediate memory savings without corrupting downstream parallel operations:
  1. **Stage 2 Target Deduplication**: Deduplicate target hits within each query domain instance before writing to `target_buffer`.
  2. **Fast Path in `merge_overlapping_spans`**: Add an $O(1)$ fast path for single-span groups (`len == 1`) to eliminate unnecessary sorting, accumulation, and slice allocations.
  3. **Preserve Contiguous 1D Array Structure**: Retain the structured NumPy array `target_dtype = [('t_seq', int32), ('start', int32), ('end', int32), ('dom_id', int32)]` for Stage 4 worker distribution, avoiding complex inverted index structures that degrade IPC serialization speed.

```python
# FAULTY (domain_finder.py:220-225 - Emits raw un-deduplicated target spans per query domain):
if valid_mask.any():
    partner_targets = np.column_stack((t_seqs, t_starts, t_ends, t_doms))
    all_targets = np.vstack((partner_targets, self_targets))
else:
    all_targets = self_targets

# REVISED (domain_finder.py - Deduplicates identical target hits per domain instance in Stage 2):
if valid_mask.any():
    partner_targets = np.column_stack((t_seqs, t_starts, t_ends, t_doms))
    # Deduplicate identical (t_seq, start, end, dom_id) rows before buffer insertion
    partner_targets = np.unique(partner_targets, axis=0)
    all_targets = np.vstack((partner_targets, self_targets))
else:
    all_targets = self_targets
```

```python
# FAULTY (domain_finder.py:240-255 - Redundant lexsort and accumulate on single-interval spans):
def merge_overlapping_spans(targets_on_sequence: np.ndarray) -> np.ndarray:
    if len(targets_on_sequence) == 0:
        return targets_on_sequence.copy()
    s = np.ascontiguousarray(targets_on_sequence[:, 0])
    e = np.ascontiguousarray(targets_on_sequence[:, 1])
    order = np.lexsort((e, s))
    ...

# REVISED (domain_finder.py - O(1) early exit for solitary intervals):
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
  1. Benchmark Stage 2 and Stage 4 on simulated repetitive hits with $50\%$ redundant target coordinates.
  2. Confirm memory consumption in `target_buffer` drops proportionally without altering downstream cluster assignments.

---

## Redundant Range-Slicing Overhead in Pairwise Span Comparison

- **Evidence**:
  In `clustering.py:27-37`:
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
- **Impact**:
  For sequence loci containing thousands of alignment spans, iterating row-by-row in Python executes thousands of individual `np.searchsorted`, `np.minimum`, `np.arange`, and `np.full` calls. Each iteration allocates ephemeral sub-arrays that stress Python's small object allocator.
- **Proposed Fix**:
  Vectorize the overlap discovery using two-pointer cumulative indices or sweep-line event points when $N > 1,000$, or pre-allocate flat coordinate index buffers to eliminate repeated list appending and concatenation.

```python
# FAULTY (clustering.py:27-37 - Per-span Python loop creating ephemeral slice allocations):
for i in range(n):
    end_i = ends[i]
    r = max(i + 1, int(np.searchsorted(starts, end_i, side="left")))
    ...
    lows.append(np.full(js.size, i, dtype=np.int32))

# REVISED (clustering.py - Vectorized span comparison reducing per-iteration allocation overhead):
# Pre-filter search boundaries to active spans with r > i + 1
r_indices = np.maximum(np.arange(1, n + 1), np.searchsorted(starts, ends, side="left"))
active_i = np.flatnonzero(r_indices > np.arange(1, n + 1))

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
  1. Benchmark runtime on sequences containing $> 10,000$ overlapping spans.
  2. Confirm 2- to 3-fold speedup in `compare_sorted_spans` without altering coordinate output.

---

# Phased Implementation Plan for Future Revisions

This phased plan outlines precise instructions for future engineering agents to implement and validate the audited solutions.

## Phase: Memory and Deduplication Optimizations
1. **Target File:** `domain_finder.py`
   - **Target Functions:** `domain_instances_on_query`, `merge_overlapping_spans`, and `find_domains`
   - **Specific Instructions:**
     - In `domain_instances_on_query`, deduplicate `partner_targets` rows via `np.unique` before stacking with `self_targets`.
     - In `merge_overlapping_spans`, add an $O(1)$ fast path returning immediately when input span length is $\le 1$.
     - In `find_domains`, fulfill the in-place reallocation optimization by using `target_buffer.resize(new_size, refcheck=False)` during Stage 2 expansion.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not create intermediate array views or slices of `target_buffer` prior to calling `resize`.
     - Do not convert structured NumPy target arrays into nested Python lists or dictionaries.

## Phase: Algorithmic Integrity and Tandem Alignment Preservation
1. **Target Files:** `aligners.py` and `domain_finder.py`
   - **Target Functions:** `parse_mmseqs_tsv`, `family_finder`, and `populate_registry_from_fasta`
   - **Specific Instructions:**
     - In `parse_mmseqs_tsv`, filter only identical coordinates on the same sequence (`q_seq_str == t_seq_str and raw_qs == raw_ts and raw_qe == raw_te`), allowing valid tandem domain duplications to be parsed.
     - In `family_finder`, calculate domain instance alignment support (`domain_weights`) and pass them to `dpc` as sample weights.
     - In `domain_finder.py`, default `point_specific` to `True` to prevent cluster-wide halo decapitation.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not freeze `SequenceRegistry` prematurely when reading pre-computed alignment files against external target databases.
     - Ensure sample weights in `dpc` remain non-negative floats with minimum weight $\ge 1.0$.

## Phase: Secondary Module Hygiene and Thread Management
1. **Target Files:** `evaluation.py` and `domain_finder_precomputed.py`
   - **Target Functions:** `evaluation.read_fasta` and `domain_finder_precomputed.find_domains`
   - **Specific Instructions:**
     - In `evaluation.py`, replace the non-existent `annotations.NucleotideSeq` import with `Bio.SeqIO` or repository-native FASTA parsing.
     - Replace the 300-byte-per-nucleotide character matrix with 32-bit integer arrays or coordinate interval queries.
     - In `domain_finder_precomputed.py`, expose the `cpu` argument and pass it to `df_find_domains`.
   - **Anti-Patterns & Risks to Avoid:**
     - Never allocate full-genome nucleotide string arrays in evaluation routines.
     - Avoid omitting core thread controls in public orchestration wrappers.
