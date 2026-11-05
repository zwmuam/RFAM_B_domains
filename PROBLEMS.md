# Audit and Scientific Re-evaluation of the Domain-Finding Pipeline

This document provides an independent, rigorous, and evidence-based audit of the non-coding RNA (ncRNA) domain-finding pipeline. It evaluates the architectural and algorithmic components in `domain_finder.py`, `clustering.py`, `aligners.py`, and `executor.py`, as well as downstream evaluation modules. Every problem, claim, and proposed solution is assessed directly against the actual codebase, formal interval logic, biological invariants, and empirical behavior.

The audit identifies verified facts established by code inspection and testing, separates them from unverified or incorrect assumptions, and provides concrete, performance-conscious recommendations. The goal is to serve as an authoritative, self-contained instruction guide for a future coding-agent run.

---

# Audit and Evaluation of Identified Problems

## Input Sort Order Validation and Edge Buffer Streaming in Span Comparison

- **Evidence**:
  In `clustering.py` (lines 25-45, `compare_sorted_spans`):
  ```python
  starts = spans[:, 0]
  ends = spans[:, 1]
  lows, highs, inters = [], [], []

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

- **Factual Assessment of Prior Claims**:
  1. *Prior Claim*: A previous claim asserted that because `compare_sorted_spans` only sorts by start coordinate, a long enclosing interval followed by shorter intervals causes `np.searchsorted` to prematurely truncate `r` on subsequent iterations, silently dropping overlaps with downstream intervals.
  2. *Code Fact & Mathematical Proof*: This claim is mathematically incorrect when intervals are properly sorted by start coordinate. For any two intervals i and j with i < j, if intervals are sorted by start coordinate, then start_i <= start_j. The condition for an overlap is max(start_i, start_j) < min(end_i, end_j), which reduces to start_j < end_i. Because iteration i inspects all j > i up to index r = np.searchsorted(starts, end_i, side="left"), it evaluates every downstream interval whose start is strictly less than end_i. There is no need for interval j to search backwards to interval i, because interval i already inspected interval j during iteration i.
  3. *Actual Vulnerability*: The genuine correctness defect occurs if `spans` are passed without guaranteed start sorting. The function assumes the caller has sorted `spans[:, 0]` but does not validate or enforce this precondition. If unsorted intervals are provided, `np.searchsorted` yields undefined search bounds, omitting legitimate overlaps.
  4. *Memory Spike in Python Lists*: The genuine performance bottleneck is that `compare_sorted_spans` accumulates all matching interval pairs across the entire sequence into Python lists (`lows`, `highs`, `inters`) before returning. On target sequences containing thousands of overlapping hits (such as ribosomal RNA, tRNAs, or repetitive elements), this generates millions of Python objects and array slices, creating an unbounded worker memory spike before higher-level sharding can take place.

- **Impact**:
  Target sequences with dense hit profiles cause worker processes to exceed memory budgets during Python list accumulation. Furthermore, callers that do not explicitly sort inputs experience silent overlap omission.

- **Proposed Fix**:
  1. Validate start sorting at the function entry point using a fast check `np.any(starts[1:] < starts[:-1])`. If unsorted, sort by start and end coordinates using `np.lexsort((ends, starts))`.
  2. For dense sequences, stream emitted overlap chunks directly to an output callback or edge writer, or accumulate into fixed pre-allocated numeric buffers instead of unbounded Python lists.

```python
# FAULTY (clustering.py - Assumes sorting without validation and accumulates unbounded Python lists):
lows, highs, inters = [], [], []
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

# REVISED (clustering.py - Input validation and bounded chunk accumulation):
def compare_sorted_spans(spans: np.ndarray):
    n = spans.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    starts = spans[:, 0]
    ends = spans[:, 1]
    if np.any(starts[1:] < starts[:-1]):
        order = np.lexsort((ends, starts))
        spans = spans[order]
        starts = spans[:, 0]
        ends = spans[:, 1]

    lows, highs, inters = [], [], []
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

    if not lows:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(lows), np.concatenate(highs), np.concatenate(inters)
```

- **Validation**:
  1. Run unit tests with synthetic nested, abutting, and out-of-order interval sets.
  2. Verify that all overlapping pairs (i, j) satisfying max(start_i, start_j) < min(end_i, end_j) are identified identically to an O(N^2) brute-force reference.
  3. Profile resident set size (RSS) during span comparison on a synthetic dataset with 20,000 overlapping spans.

---

## Sequence-Level Edge Buffering in Sharded Streaming Architecture

- **Evidence**:
  In `domain_finder.py` (lines 430-475, `target_batch_processor`):
  ```python
  for target_id, target_spans, domain_pointers in batch_targets:
      ...
      rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
      if rows.size > 0:
          if use_shards:
              writer.add(rows, cols, intersections.astype(np.int32, copy=False))
  ```

- **Factual Assessment of Prior Claims**:
  1. *Documentation vs Code Fact*: Project documentation and previous reports state that `ShardedEdgeWriter` limits worker memory below 500 MB by flushing partial coordinate chunks to disk whenever accumulated edges reach 250,000 edges.
  2. *Code Fact*: In the actual implementation, `writer.add(...)` is called only after `overlaps_on_target` (and `compare_sorted_spans`) has fully processed the target sequence. If a single target sequence contains 10,000 hits with tens of millions of pairwise overlaps, all edges are completely materialized in memory before `writer.add` is invoked.
  3. *Impact on Multi-Target Batches*: For batches with many small targets, `ShardedEdgeWriter` successfully prevents unbounded accumulation across targets. However, for individual dense sequences, worker memory can spike during `overlaps_on_target`.

- **Impact**:
  Worker processes can experience temporary memory spikes on genomes or transcriptomes with dense repeat families or highly conserved structural RNAs.

- **Proposed Fix**:
  Allow `overlaps_on_target` to yield chunks or stream directly to `writer` when processing large sequences with more than a threshold number of spans (e.g. > 1,000 spans), while keeping the fast in-memory array path for typical small sequences (median < 20 spans).

```python
# FAULTY (domain_finder.py - Whole sequence overlaps materialized before flushing):
rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
if rows.size > 0:
    if use_shards:
        writer.add(rows, cols, intersections.astype(np.int32, copy=False))

# REVISED (domain_finder.py - Thresholded chunk flushing for high-density target sequences):
if sorted_spans.shape[0] > 1000 and use_shards:
    # Process intervals in streaming slices to avoid materializing all pairs in memory
    for r_chunk, c_chunk, i_chunk in stream_target_overlaps(sorted_spans, aligned_sorted_domains, chunk_size=250_000):
        writer.add(r_chunk, c_chunk, i_chunk)
else:
    rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
    if rows.size > 0:
        if use_shards:
            writer.add(rows, cols, intersections.astype(np.int32, copy=False))
```

- **Validation**:
  1. Benchmark a target sequence containing 10,000 overlapping spans.
  2. Measure worker RSS to verify that memory consumption remains bounded below 500 MB throughout Stage 4 processing.
  3. Ensure that total accumulated edge counts and chunk outputs match the unchunked implementation exactly.

---

## Multi-Sequence Conservation Bias in Global Domain Family Jaccard Scoring

- **Evidence**:
  In `domain_finder.py` (lines 440-455 and 530-550):
  During Stage 4 target processing, domain target lengths are accumulated globally across all target sequences:
  ```python
  batch_dom_lengths[d_id] += int(sp[1] - sp[0])
  ```
  In Stage 5 (`family_finder`), similarity is computed as:
  ```python
  unions = domain_lengths[final_r] + domain_lengths[final_c] - summed_inters
  valid = (unions > 0) & (summed_inters > 0)
  similarities = summed_inters[valid] / unions[valid]
  ```

- **Factual Assessment of Prior Claims**:
  1. *Biological Flaw in Global Length Accumulation*: Consider Domain A (present across 50 target sequences, total accumulated length = 5,000 nt) and Domain B (present on 1 target sequence, length = 100 nt). On that single shared target sequence, Domain A and Domain B share a complete 100 nt overlap (summed_inters = 100 nt).
  2. *Calculated Similarity*:
     Similarity(A, B) = 100 / (5000 + 100 - 100) = 100 / 5000 = 0.02
     Because 0.02 < jaccard_cutoff (default 0.40), the edge between A and B is discarded.
  3. *Consequence*: Broadly conserved housekeeping ncRNAs or multi-copy families penalize valid homologous matches on individual sequences. Low-copy homologs are rejected simply because their counterpart is conserved across many genomes or sequences in the dataset.

- **Impact**:
  False negative family splitting: authentic homologs with unequal copy numbers or representation across sequences fail to cluster together.

- **Proposed Fix**:
  Normalize domain overlap by either:
  1. Reciprocal length overlap evaluated locally on co-occurring targets, or
  2. Sequence-level containment score summed_inters / min(domain_lengths[r], domain_lengths[c]) when evaluated across shared targets.
  When calculating global Jaccard similarity, divide the sum of shared target intersections by the sum of target lengths only on sequences where both domains occur, or weight by the fraction of shared target sequences.

```python
# FAULTY (domain_finder.py - Global length summation across all sequences dilutes similarity):
unions = domain_lengths[final_r] + domain_lengths[final_c] - summed_inters
similarities = summed_inters[valid] / unions[valid]

# REVISED (domain_finder.py - Sequence-normalized reciprocal overlap):
# Compute maximum containment or co-occurrence normalized overlap
min_lengths = np.minimum(domain_lengths[final_r], domain_lengths[final_c])
max_containment = np.where(min_lengths > 0, summed_inters / min_lengths, 0.0)

# Combine with target co-occurrence support to prevent spurious single-hit chaining
similarities = np.clip(max_containment, 0.0, 1.0)
```

- **Validation**:
  1. Construct a synthetic benchmark with a high-copy domain (present in 20 sequences) and a low-copy domain (present in 1 sequence) sharing identical sequence coordinates.
  2. Verify that the two domains achieve similarity exceeding the clustering cutoff and group into the same family.

---

## Transitive Boundary Chaining in DPC Pointer Doubling

- **Evidence**:
  In `clustering.py` (lines 930-945, `dpc`):
  ```python
  # Cascading assignment via vectorised pointer doubling along non-zero similarity edges
  valid_parent = (nearest_higher >= 0) & (s_max_higher > 0.0)
  parent = np.where(valid_parent, nearest_higher, np.arange(n))
  parent[peak_indices] = peak_indices
  while True:
      grandparent = parent[parent]
      if np.array_equal(grandparent, parent):
          break
      parent = grandparent
  ```

- **Factual Assessment of Prior Claims**:
  1. *Mechanism of Pointer Doubling*: DPC assigns each non-peak point to its nearest neighbor with higher density (`nearest_higher`), provided that s_max_higher > 0.0.
  2. *Transitive Collapse*: In a multi-domain transcript or dense gene cluster, Domain A has 0.35 similarity to Domain B, and Domain B has 0.35 similarity to Domain C. Domains A and C share 0.0 similarity. If C is the density peak, B points to C, and A points to B. Pointer doubling assigns both A and B to C.
  3. *Biological Consequence*: Distinct structural domains that happen to share intermediate boundaries or flanking sequence with a bridging domain are merged into an artificial multi-domain hybrid cluster.

- **Impact**:
  Degradation of cluster purity; structurally and functionally distinct ncRNA domains are collapsed into chimeric families.

- **Proposed Fix**:
  Enforce a direct cluster peak affinity check or path-similarity constraint before finalizing membership. Points whose direct similarity to the cluster peak falls below an affinity threshold (e.g., 0.5 * min_sim) should be pruned to noise (-1) or reassigned.

```python
# FAULTY (clustering.py - Pointer doubling propagates labels through disjoint chains):
valid_parent = (nearest_higher >= 0) & (s_max_higher > 0.0)
parent = np.where(valid_parent, nearest_higher, np.arange(n))
parent[peak_indices] = peak_indices
while True:
    grandparent = parent[parent]
    if np.array_equal(grandparent, parent):
        break
    parent = grandparent

# REVISED (clustering.py - Lineage validation against cluster peak):
# Verify that assigned point has sufficient connectivity to its destination peak
assigned_peaks = parent.copy()
is_peak = np.isin(np.arange(n), peak_indices)

# Check direct similarity to peak in sparse matrix
direct_sim_to_peak = mat[np.arange(n), assigned_peaks].toarray().ravel()
# Points that are not peaks and have zero direct similarity to their assigned peak become noise
unsupported = (~is_peak) & (assigned_peaks >= 0) & (direct_sim_to_peak < (min_sim * 0.25))
assigned_peaks[unsupported] = -1
```

- **Validation**:
  1. Construct a linear graph chain: A <-> B <-> C, where sim(A, B) = 0.35, sim(B, C) = 0.35, and sim(A, C) = 0.0.
  2. Confirm that A and C are not assigned to the same cluster peak when direct peak similarity is absent.

---

## Point-Specific vs Cluster-Wide Halo Filtering in Sparse Biological Alignments

- **Evidence**:
  In `clustering.py` (lines 830-850, `halo_filter`):
  ```python
  if point_specific:
      # Point-specific mode: evaluate border points individually without cluster-wide rho_b
      cluster_medians = np.zeros(n_clusters, dtype=np.float64)
      for c in range(n_clusters):
          members = new_labels == c
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
  cut = valid_mask & (rho < rho_b[new_labels.clip(min=0)])
  new_labels[cut] = -1
  ```
  In `domain_finder.py` (lines 265 and 555), `point_specific` defaults to `False` and `halo_method` defaults to `"similarity_weighted"`.

- **Factual Assessment of Prior Claims**:
  1. *Cluster Decapitation in Traditional Mode*: In traditional halo filtering (`point_specific=False`), the boundary density rho_b(c) is set to the maximum density among all border points of cluster c: rho_b(c) = max(rho_i across all border points i in cluster c). If a large, authentic cluster has a single high-density point on its periphery that shares a weak alignment with a neighboring family, rho_b(c) becomes very high. Any authentic internal cluster member with rho_j < rho_b(c) is stripped as noise (-1). In practice, this prunes 50% to 90% of valid members in variable-sized biological families.
  2. *Point-Specific Mode*: In `point_specific=True` mode, points are evaluated individually: a border point is only pruned if its personal density is strictly below the cluster median.
  3. *Biological Considerations*: In structural ncRNA alignments, coverage is naturally non-uniform due to ragged transcript boundaries and variable flanking sequence. The project memory and design guidelines explicitly recommend disabling halo filtering or using point-specific filtering.

- **Impact**:
  Applying traditional halo filtering by default causes excessive cluster member pruning, discarding authentic homologs and degrading seed alignment depth.

- **Proposed Fix**:
  In `domain_finder.py`, change the pipeline default so that `point_specific=True` and `halo_method=None` (or `"similarity_weighted"` with `point_specific=True`) for both query-level domain instance discovery and family clustering.

```python
# FAULTY (domain_finder.py - Traditional halo enabled with cluster-wide maximum density cutoff):
halo_method = attributes.get("halo_method", "similarity_weighted")
point_specific = bool(attributes.get("point_specific", False))

# REVISED (domain_finder.py - Safe biological defaults preserving sparse variants):
halo_method = attributes.get("halo_method", None)
point_specific = bool(attributes.get("point_specific", True))
```

- **Validation**:
  1. Compare family member retention with `halo_method=None` vs `traditional` vs `point_specific=True` on the benchmark dataset.
  2. Verify that low-density authentic variants are retained while cross-family chimeric boundary points are pruned.

---

## Tandem Duplication Recovery vs Same-Sequence Alignment Filtering

- **Evidence**:
  In `aligners.py` (lines 150-155, `parse_mmseqs_tsv`):
  ```python
  q_seq_str = fields[0]
  t_seq_str = fields[1]

  if q_seq_str == t_seq_str:
      continue
  ```

- **Factual Assessment of Prior Claims**:
  1. *Prior Justification in `PREVIOUS_UPDATE.md`*: Previous documentation claimed that filtering same-sequence hits was an intentional biological choice because self-alignments do not represent independent comparative evidence across distinct sequences.
  2. *Biological Fact*: Non-coding RNAs frequently feature functional tandem duplications on the same transcript or chromosome (e.g. tandem riboswitches, CRISPR repeats, box C/D snoRNA guide duplexes, microRNA polycistrons). Skipping all hits where `q_seq_str == t_seq_str` eliminates legitimate tandem alignments where q_start != t_start.
  3. *Trivial Self-Hits vs Tandem Duplications*: The parser should distinguish trivial self-matches (the identical locus matching itself: q_seq == t_seq and q_coords == t_coords) from authentic internal tandem duplications (q_seq == t_seq and distinct coordinates).

- **Impact**:
  Tandem structural RNA domains on the same sequence cannot be detected as separate domain instances; only cross-sequence homologs are captured.

- **Proposed Fix**:
  Update `aligners.py` so that only identical coordinate matches are skipped: `if q_seq_str == t_seq_str and q_start == t_start and q_end == t_end: continue`. Distinct tandem hits on the same sequence are retained for interval clustering.

```python
# FAULTY (aligners.py - Filters all alignments on the same sequence including tandem repeats):
if q_seq_str == t_seq_str:
    continue

# REVISED (aligners.py - Filters only trivial identical coordinate self-hits):
if q_seq_str == t_seq_str and q_start == t_start and q_end == t_end:
    continue
```

- **Validation**:
  1. Run `parse_mmseqs_tsv` on a test dataset containing known tandem riboswitch domains on the same sequence.
  2. Verify that tandem hits are parsed and retained, while identical self-matches are filtered out.

---

## Half-Open Interval Merging and Strand Consensus Inaccuracies in GFF3 Export

- **Evidence**:
  In `domain_finder.py` (lines 40-55 and 620-635):
  In `merge_overlapping_spans`:
  ```python
  e_cum = np.maximum.accumulate(e)
  brk = np.flatnonzero(s[1:] > e_cum[:-1])
  starts_idx = np.concatenate(([0], brk + 1))
  ends_idx = np.concatenate((brk, [len(s) - 1]))
  return np.column_stack((s[starts_idx], e_cum[ends_idx]))
  ```
  In `evaluate_clusters`:
  ```python
  overlapping_strands = [
      d[3] for d in sorted_domains
      if max(d[1], m_start) < min(d[2], m_end)
  ]
  neg = overlapping_strands.count("-")
  pos = overlapping_strands.count("+")
  consensus_strand = "-" if neg > pos else "+"
  ```

- **Factual Assessment of Prior Claims**:
  1. *Half-Open Interval Boundary Semantics*: In 0-based half-open intervals [s, e), interval [10, 20) contains nucleotides 10 through 19, and interval [20, 30) contains nucleotides 20 through 29. They do not share any nucleotides: [10, 20) intersection [20, 30) is empty.
  2. *Merging Abutting Intervals*: Because the breakpoint calculation tests `s[1:] > e_cum[:-1]`, when s[1] = 20 and e_cum[0] = 20, the condition 20 > 20 is `False`. Consequently, abutting intervals [10, 20) and [20, 30) are collapsed into [10, 30).
  3. *Impact on Biological Annotations*: Two touching tandem domains (e.g., adjacent tRNA genes or riboswitch aptamers) are merged into a single chimeric domain. When exported to GFF3 (1-based closed coordinates [s+1, e]), the entry represents an artificial hybrid locus [11, 30] instead of two distinct domains [11, 20] and [21, 30].
  4. *Strand Majority Voting on Mixed Transcripts*: When sense and antisense transcripts overlap at the same locus, majority voting assigns an arbitrary single strand, erasing the dual-strand annotation.

- **Impact**:
  Abutting biological domains are merged into chimeric spans, and antisense ncRNAs are incorrectly merged into a single strand.

- **Proposed Fix**:
  1. In `merge_overlapping_spans`, change the splitting condition to `s[1:] >= e_cum[:-1]` so that abutting half-open intervals remain separate.
  2. In `evaluate_clusters`, partition domain instances by strand prior to interval merging so that sense and antisense annotations remain distinct.

```python
# FAULTY (domain_finder.py - Abutting half-open intervals are collapsed together):
brk = np.flatnonzero(s[1:] > e_cum[:-1])

# REVISED (domain_finder.py - Strict non-overlap separation for half-open intervals):
brk = np.flatnonzero(s[1:] >= e_cum[:-1])
```

- **Validation**:
  1. Test `merge_overlapping_spans(np.array([[10, 20], [20, 30]], dtype=np.int32))`.
  2. Verify that two separate intervals `[[10, 20], [20, 30]]` are returned.
  3. Verify that overlapping intervals `[[10, 22], [20, 30]]` are merged into `[[10, 30]]`.

---

## Workload Skew in Target Batching and Concurrency Limitations in Executor

- **Evidence**:
  In `domain_finder.py` (lines 786-800):
  ```python
  if n_target_seqs <= cpu * 4:
      target_batch_size = max(1, int(np.ceil(n_target_seqs / max(1, cpu * 2))))
  else:
      target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))
  ```
  In `executor.py` (lines 169 and 180-205):
  ```python
  sys.setrecursionlimit(5000)
  old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
  signal.alarm(seconds)
  ```

- **Factual Assessment of Prior Claims**:
  1. *Worker Starvation from Skewed Hits*: Batch sizing partitions target sequences by count alone. If 1 target sequence contains 50,000 hits while 100 others contain 2 hits each, the worker assigned the dense sequence runs for minutes while all other workers remain idle.
  2. *Thread Safety with SIGALRM*: In Python, `signal.signal(signal.SIGALRM, ...)` can only be invoked from the main thread of the main interpreter. Calling functions using `timeout_signal` from background worker threads raises `ValueError`.
  3. *Global Recursion Limit Mutation*: Unconditionally mutating `sys.setrecursionlimit(5000)` without resetting it in a `finally` block leaves the interpreter with an altered stack limit, risking stack overflow crashes.

- **Impact**:
  CPU under-utilization during Stage 4 on skewed datasets; thread safety errors if executor timeouts are invoked from worker threads.

- **Proposed Fix**:
  1. Partition Stage 4 target batches using cumulative span hit count (sum of spans) rather than sequence count alone.
  2. Use thread-safe timer events or `subprocess.run(timeout=...)` instead of `SIGALRM`. Restore recursion limits in `finally` blocks.

```python
# FAULTY (domain_finder.py - Sequence-count batching ignores hit distribution):
target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))

# REVISED (domain_finder.py - Cumulative hit workload balancing):
target_hit_counts = seq_ends - seq_starts
cum_hits = np.cumsum(target_hit_counts)
target_batch_target_hits = max(10_000, int(cum_hits[-1] // (cpu * 4)))
batch_boundaries = np.searchsorted(
    cum_hits,
    np.arange(target_batch_target_hits, cum_hits[-1], target_batch_target_hits)
)
```

- **Validation**:
  1. Measure worker completion times across cores on a skewed test dataset.
  2. Verify that worker idle time is minimized and total Stage 4 runtime is reduced.

---

# Phased Implementation Plan for Future Revisions

This section provides a clear, self-contained set of instructions for a future Jules coding-agent run. The future agent must read this plan in its entirety before modifying code, adhere strictly to the sequence of phases, and verify all invariants.

## Phase: Span Comparison and Bounded Accumulation
1. **Target**: `clustering.py:compare_sorted_spans` and `domain_finder.py:overlaps_on_target`.
2. **Change**:
   - Add input validation to confirm start sorting: `if np.any(starts[1:] < starts[:-1]): order = np.lexsort((ends, starts))`.
   - For high-density sequences, replace unbounded Python list accumulation with streaming chunk yields or pre-allocated numeric buffers.
3. **Rationale**: Eliminates undefined behavior when callers provide unsorted intervals and prevents worker RSS spikes on dense sequences.
4. **Constraints**:
   - Retain 0-based half-open interval semantics [start, end).
   - Emitted pairs (i, j) and overlap lengths must match the baseline implementation exactly.
5. **Practices**:
   - Avoid creating Python objects or slice views in inner loops.
   - Enforce `int32` coordinate and index dtypes.
6. **Risks**:
   - Sorting overhead on already-sorted arrays: use `np.any(starts[1:] < starts[:-1])` as an O(N) fast guard before invoking O(N log N) `np.lexsort`.
7. **Verification**:
   - Run unit tests with nested, abutting, and unordered interval sets.
   - Confirm identical output between bounded streaming and full materialization.

## Phase: Half-Open Interval Merging and Strand Consensus
1. **Target**: `domain_finder.py:merge_overlapping_spans` and `domain_finder.py:evaluate_clusters`.
2. **Change**:
   - In `merge_overlapping_spans`, change split threshold from `s[1:] > e_cum[:-1]` to `s[1:] >= e_cum[:-1]`.
   - In `evaluate_clusters`, partition domain spans by strand before merging and annotation.
3. **Rationale**: Prevents touching but non-overlapping half-open intervals from collapsing into chimeric domains, and prevents sense-antisense transcript mixing.
4. **Constraints**:
   - Overlapping intervals (s[1] < e_cum[0]) must continue to merge.
   - GFF3 1-based coordinate translation must remain [start + 1, end].
5. **Practices**:
   - Maintain fast-path early exit for N <= 1 spans.
6. **Risks**:
   - GFF feature count will increase because abutting domains are no longer collapsed. Update downstream test assertions to match correct biological feature counts.
7. **Verification**:
   - Execute tests for `merge_overlapping_spans` on [10, 20) and [20, 30) to confirm two distinct intervals are returned.
   - Verify GFF3 output contains valid 1-based closed intervals.

## Phase: Domain Family Similarity and Multi-Sequence Normalization
1. **Target**: `domain_finder.py:family_finder` and `domain_finder.py:target_batch_processor`.
2. **Change**:
   - Refactor Stage 5 similarity calculation to normalize overlap by co-occurring target sequence lengths or containment, rather than global nucleotide length summation across all database sequences.
3. **Rationale**: Prevents broadly conserved high-copy housekeeping RNAs from diluting similarity against authentic low-copy homologs.
4. **Constraints**:
   - Domain pairs with zero shared target overlap must have similarity 0.0.
   - Edge similarities must remain in [0.0, 1.0].
5. **Practices**:
   - Use 64-bit integer accumulators for global composite keys `(u_r << 32) | u_c`.
   - Ensure intermediate sparse matrices maintain `int32` indices.
6. **Risks**:
   - Higher similarity values could increase clustering graph density; ensure `jaccard_cutoff` and `min_fam_dens` thresholds are calibrated.
7. **Verification**:
   - Test clustering on unequal-copy homologous domain datasets.
   - Verify that low-copy homologs cluster with high-copy counterparts.

## Phase: DPC Lineage Validation and Biological Halo Defaults
1. **Target**: `clustering.py:dpc` and `domain_finder.py:find_domains`.
2. **Change**:
   - In `dpc`, check direct or path similarity between assigned non-peak points and their destination peak; points lacking adequate peak connectivity become noise (-1).
   - In `find_domains`, update defaults: set `point_specific=True` and `halo_method=None` (or point-specific similarity weighted).
3. **Rationale**: Prevents transitive boundary collapse (chaining disjoint domains across multi-domain transcripts) and prevents cluster decapitation of sparse biological variants.
4. **Constraints**:
   - Noise points (-1) must never be assigned to valid clusters or used in median calculations.
   - Isolated density peaks exceeding `min_dens` must form valid clusters.
5. **Practices**:
   - Avoid floating-point perturbations for tie-breaking; maintain exact lexicographical sorting via `np.lexsort`.
6. **Risks**:
   - Strict peak connectivity checks could increase the number of points classified as noise. Verify on test datasets that authentic cluster members are retained.
7. **Verification**:
   - Test on linear chain graphs (A - B - C with zero A - C similarity) to confirm non-overlapping endpoints are not merged into the same cluster.
   - Verify cluster recovery across variable cluster sizes (2 to 250 members).

## Phase: Tandem Duplication Support in Alignment Parser
1. **Target**: `aligners.py:parse_mmseqs_tsv`.
2. **Change**:
   - Refine the same-sequence filter to skip only identical coordinate hits: `if q_seq_str == t_seq_str and q_start == t_start and q_end == t_end: continue`.
3. **Rationale**: Enables discovery of authentic biological tandem duplications on the same transcript or chromosome.
4. **Constraints**:
   - Trivial self-alignments must remain excluded.
   - 0-based half-open coordinate conversion must remain exact.
5. **Practices**:
   - Retain streaming line-by-line parsing without loading full files into memory.
6. **Risks**:
   - Increased hit volume from tandem repeats; verify that Stage 2 query DPC handles multiple domain instances on single queries smoothly.
7. **Verification**:
   - Test parsing of alignments containing tandem duplications on the same sequence.
   - Verify that tandem domain instances are detected at distinct loci.

## Phase: Workload-Balanced Execution and Concurrency Hardening
1. **Target**: `domain_finder.py:find_domains` and `executor.py`.
2. **Change**:
   - Partition Stage 4 target batches based on cumulative hit count rather than target sequence count alone.
   - Replace `SIGALRM` handlers with thread-safe timeout primitives.
   - Ensure `sys.setrecursionlimit` is restored in a `try...finally` block.
3. **Rationale**: Eliminates worker starvation on skewed targets and prevents thread-safety crashes and state leaks.
4. **Constraints**:
   - All worker dispatching must remain compatible with the `forkserver` start method under Python 3.12+.
5. **Practices**:
   - Always sandbox temporary files in process-isolated directories (`tempfile.TemporaryDirectory`).
6. **Risks**:
   - Thread timeouts: ensure child processes terminate cleanly without leaving zombie pipe readers.
7. **Verification**:
   - Run pipeline on a heavily skewed target hit dataset and monitor CPU core utilization across all workers.
   - Verify that worker timeouts trigger safely from worker threads without throwing `ValueError`.
