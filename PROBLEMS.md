# Comprehensive Audit and Scientific Evaluation of the Domain-Finding Pipeline

This audit presents an independent, rigorous evaluation of the ncRNA domain-finding pipeline, focusing on its core architectural modules: `domain_finder.py`, `clustering.py`, `aligners.py`, and `executor.py`. It establishes whether the current implementation is scientifically valid, mathematically and numerically sound, statistically justified, computationally and memory-efficient, and robust against data corruption. Discrepancies between claims in `PREVIOUS_UPDATE.md` and the actual codebase have been systematically investigated and documented below.

## Truncated Interval Search in Span Intersection Engine

- **Evidence**:
  In `clustering.py:27-44` (`compare_sorted_spans`):
  ```python
  starts = spans[:, 0]
  ends = spans[:, 1]
  for i in range(n):
      end_i = ends[i]
      r = max(i + 1, int(np.searchsorted(starts, end_i, side="left")))
      if r > i + 1:
          st_js = starts[i + 1:r]
          en_js = ends[i + 1:r]
          overlap = np.minimum(end_i, en_js) - st_js
          valid = overlap > 0
          ...
  ```
- **Impact**:
  1. **Algorithmic Correctness Defect**: The input intervals are sorted by start coordinate: $\text{starts}[i] \le \text{starts}[j]$ for $i < j$. The binary search computes $r = \text{np.searchsorted}(\text{starts}, \text{end}_i, \text{side}="left")$. This correctly identifies indices $j$ where $\text{starts}[j] < \text{end}_i$, meaning span $i$ and span $j$ can overlap. However, on subsequent iterations for interval $k > i$, interval $k$'s start coordinate $\text{starts}[k]$ may be less than $\text{end}_i$, but if interval $k$'s end coordinate $\text{end}_k$ extends beyond interval $m$ (where $\text{starts}[m] \ge \text{end}_k$), any long enclosing interval earlier in the array is omitted from subsequent pairings if not captured in the forward pass.
  2. **Severe Incompleteness Under Nested Spans**: When a large domain span (e.g., $[10, 100)$) contains multiple smaller domain spans (e.g., $[20, 30)$ and $[90, 105)$), `compare_sorted_spans` checks overlaps forward from the primary span. While span $0$ ($[10, 100)$) finds span $1$ and span $2$, iteration $i=1$ ($[20, 30)$) searches only up to $\text{starts}[j] < 30$, correctly skipping span $2$ ($[90, 105)$). However, if intervals are not strictly sorted by both start and end, or if an outer interval occurs after an inner interval, pairwise intersections are silently dropped.
  3. **Discrepancy with `PREVIOUS_UPDATE.md`**: `PREVIOUS_UPDATE.md` claimed that Stage 4 implements in-loop bounded edge streaming directly inside `compare_sorted_spans`. In reality, `compare_sorted_spans` still accumulates all internal lists `lows`, `highs`, and `inters` in memory and concatenates them at the end of the sequence, deferring sharding until the higher-level caller `overlaps_on_target`. A target sequence with tens of thousands of dense overlapping intervals will still cause an unconstrained memory spike inside `compare_sorted_spans`.
- **Proposed Fix**:
  Enforce strict primary and secondary lexicographic sorting (`order = np.lexsort((spans[:, 1], spans[:, 0]))`) at function entry, and stream detected overlaps directly into an output buffer or consumer callback when accumulated interval pairs exceed a predefined chunk threshold.

```python
# FAULTY (clustering.py - compare_sorted_spans accumulates all pairs in Python lists in memory):
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

# REVISED (clustering.py - Validated sort order and chunked edge emissions):
def compare_sorted_spans_bounded(spans: np.ndarray, max_edges: int = 250_000):
    n = spans.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    starts = spans[:, 0]
    ends = spans[:, 1]
    # Verify or enforce strict start ordering
    if np.any(starts[1:] < starts[:-1]):
        order = np.lexsort((ends, starts))
        starts = starts[order]
        ends = ends[order]
    ...
```

- **Validation**:
  1. Construct synthetic test cases with nested, staggered, and abutting intervals.
  2. Verify that all intersecting pairs $(i, j)$ where $\max(\text{start}_i, \text{start}_j) < \min(\text{end}_i, \text{end}_j)$ are identified without omissions.
  3. Profile memory consumption on a benchmark locus containing $50,000$ intersecting spans to guarantee memory is strictly bounded.

---

## Discrepancies in Sharded Edge Streaming Architecture

- **Evidence**:
  In `PREVIOUS_UPDATE.md` (lines 142–155 and 338–348), the report asserts:
  > "Implemented `ShardedEdgeWriter` inside `target_batch_processor` to flush partial coordinate chunks (`r`, `c`, `i`) to disk whenever accumulated edges reach 250,000 edges. Individual worker processes no longer accumulate unbounded lists of edges across dense loci, bounding worker RSS strictly under 500 MB."
  In `domain_finder.py:355-385` and `domain_finder.py:440-475`:
  `ShardedEdgeWriter` is defined and called in `target_batch_processor`, but its buffering is only evaluated at the target sequence loop boundary (`for target_id, target_spans, domain_pointers in batch_targets:`), *after* `overlaps_on_target` and `compare_sorted_spans` have already run to completion for that entire sequence.
- **Impact**:
  1. **Worker Memory Spikes on Dense Conserved Loci**: If a single target sequence contains a dense cluster of hits (e.g., $10,000$ domain hits yielding $50,000,000$ pairwise intersections), all $50,000,000$ edges are fully materialized in `compare_sorted_spans` and `overlaps_on_target` before `writer.add(...)` is ever invoked. The claim that worker RSS is bounded under 500 MB is invalid for dense target sequences.
  2. **Intermittent Memory Crashes in Stage 4**: In large-scale genomic or metagenomic runs where structural RNAs (like 5S, 16S, or tRNA) produce extreme target hit depth at specific loci, worker processes exceed operating system memory limits, causing task termination.
- **Proposed Fix**:
  Push the `ShardedEdgeWriter` (or an edge-yielding generator) down into `compare_sorted_spans` so that when the number of accumulated edge pairs exceeds the buffer limit, the partial chunk is flushed immediately before processing additional interval comparisons on that same sequence.

```python
# FAULTY (domain_finder.py - Target sequence processes all spans before flushing to writer):
rows, cols, intersections = overlaps_on_target(sorted_spans, aligned_sorted_domains)
if rows.size > 0:
    if use_shards:
        writer.add(rows, cols, intersections.astype(np.int32, copy=False))

# REVISED (domain_finder.py - Stream edges directly from overlap engine):
def stream_overlaps_on_target(sorted_spans, aligned_sorted_domains, writer):
    for r_chunk, c_chunk, i_chunk in compare_sorted_spans_generator(sorted_spans, chunk_size=250_000):
        mask = aligned_sorted_domains[r_chunk] != aligned_sorted_domains[c_chunk]
        if np.any(mask):
            writer.add(aligned_sorted_domains[r_chunk][mask],
                       aligned_sorted_domains[c_chunk][mask],
                       i_chunk[mask].astype(np.int32, copy=False))
```

- **Validation**:
  1. Generate a single target sequence with $25,000$ mutually overlapping domain hits.
  2. Track worker process resident set size (RSS) continuously during Stage 4.
  3. Ensure worker RSS stays flat below 200 MB regardless of total edge volume generated.

---

## Asymmetric Multi-Sequence Dilution in Domain Family Jaccard Scoring

- **Evidence**:
  In `domain_finder.py:440-455` and `domain_finder.py:530-550`:
  During Stage 4, domain target lengths are accumulated globally across all target sequences:
  ```python
  batch_dom_lengths[d_id] += int(sp[1] - sp[0])
  ```
  In Stage 5 (`family_finder`):
  ```python
  unions = domain_lengths[final_r] + domain_lengths[final_c] - summed_inters
  valid = (unions > 0) & (summed_inters > 0)
  similarities = summed_inters[valid] / unions[valid]
  ```
- **Impact**:
  1. **Severe Sensitivity and Conservation Bias**: Suppose Domain $A$ is a broadly conserved, high-copy housekeeping domain present across 100 target sequences, with a total cumulative target length of $\text{domain\_lengths}[A] = 10,000\text{ bp}$. Suppose Domain $B$ is an authentic functional homolog present on only 1 target sequence with length $\text{domain\_lengths}[B] = 100\text{ bp}$. On that shared target sequence, Domain $A$ and Domain $B$ share a complete $100\text{ bp}$ overlap ($\text{summed\_inters} = 100\text{ bp}$).
  2. **False Dissimilarity**: The global Jaccard similarity between Domain $A$ and Domain $B$ is calculated as:
     $$J(A, B) = \frac{100}{10,000 + 100 - 100} = \frac{100}{10,000} = 0.01$$
     Because $0.01 < \text{min\_fam\_sim}$ (default $0.25$), the edge between $A$ and $B$ is discarded.
  3. **Biological Invalidity**: Domain $B$ is incorrectly segregated from its true biological family simply because Domain $A$ was detected in more homologs across the database. Global length summation penalizes unbalanced target hit frequencies and violates the biological premise of homology detection.
- **Proposed Fix**:
  Normalize target overlaps locally per target sequence using average reciprocal overlap or containment, or calculate family similarity based on the set of shared target sequences and alignment profiles rather than raw global nucleotide summation:
  $$S(A, B) = \frac{\sum_{t \in T_A \cap T_B} \text{overlap}_t(A, B)}{\sum_{t \in T_A \cap T_B} \max(\text{len}_t(A), \text{len}_t(B))} \times \frac{|T_A \cap T_B|}{|T_A \cup T_B|}$$

```python
# FAULTY (domain_finder.py - Global nucleotide union penalizes high-support domains):
unions = domain_lengths[final_r] + domain_lengths[final_c] - summed_inters
similarities = summed_inters[valid] / unions[valid]

# REVISED (domain_finder.py - Length-weighted sequence-normalized Jaccard):
# Track domain length strictly on co-occurring targets or use sequence containment
co_occur_union = co_occur_lengths_r + co_occur_lengths_c - summed_inters
jaccard_overlap = np.where(co_occur_union > 0, summed_inters / co_occur_union, 0.0)
target_jaccard = shared_target_counts / union_target_counts
similarities = jaccard_overlap * target_jaccard
```

- **Validation**:
  1. Benchmark clustering accuracy on simulated domain families where family member copy counts vary from 1 copy to 100 copies.
  2. Verify that low-copy homologs are correctly clustered into the major family rather than discarded as solitary noise.

---

## Transitive Domain Chaining and False Clustering in DPC Cascading Assignment

- **Evidence**:
  In `clustering.py:930-945`:
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
- **Impact**:
  1. **Transitive Boundary Collapse (Domain Chaining)**: DPC connects non-peak points to their nearest higher-density neighbor along any edge with $s\_\text{max\_higher} > 0.0$. In long multi-domain RNA molecules or chimeric transcripts, Domain $A$ overlaps with Domain $B$ by 35% Jaccard similarity, and Domain $B$ overlaps with Domain $C$ by 35% Jaccard similarity. Domains $A$ and $C$ share 0% overlap.
  2. **Unintended Family Merging**: If the density gradient flows from $A \to B \to C$ (where $C$ is the peak), pointer doubling cascades $A$'s assignment into $C$'s cluster. Consequently, entirely non-overlapping, biologically distinct domains $A$ and $C$ are merged into the same domain family.
  3. **Lack of Compactness Guarantees**: Single-linkage transitive propagation without a minimum cluster similarity threshold or diameter check produces artificially large, heterogeneous clusters that fail structural RNA family validation.
- **Proposed Fix**:
  Enforce a path-similarity threshold or direct similarity validation against the assigned cluster center (peak) before finalizing cluster membership:
  $$s(i, \text{peak}(i)) \ge \alpha \cdot \text{min\_sim}$$
  Points failing direct peak affinity should be reassigned to secondary peaks or classified as noise.

```python
# FAULTY (clustering.py - Unconstrained pointer doubling merges completely disjoint domains):
valid_parent = (nearest_higher >= 0) & (s_max_higher > 0.0)
parent = np.where(valid_parent, nearest_higher, np.arange(n))
parent[peak_indices] = peak_indices
while True:
    grandparent = parent[parent]
    if np.array_equal(grandparent, parent):
        break
    parent = grandparent

# REVISED (clustering.py - Constrained lineage validation against cluster center):
assigned_peaks = parent.copy()
# Validate that point has non-zero or adequate direct similarity to its terminal peak
direct_sim_to_peak = mat[np.arange(n), assigned_peaks].toarray().ravel()
unreachable = (assigned_peaks != np.arange(n)) & (direct_sim_to_peak < (min_sim * 0.5))
assigned_peaks[unreachable] = -1
```

- **Validation**:
  1. Construct a linear graph chain $D_1 - D_2 - D_3 - D_4$ where adjacent nodes have similarity 0.35 but non-adjacent nodes have similarity 0.0.
  2. Confirm that $D_1$ is not lumped into the same cluster as $D_4$.

---

## Point-Specific Halo Filtering Inconsistency and Cluster Mean Bias

- **Evidence**:
  In `clustering.py:832-845` (`halo_filter`):
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
  ```
  In `domain_finder.py:685` and `domain_finder.py:700`, `point_specific` defaults to `False` unless specified in `attributes['point_specific']`.
- **Impact**:
  1. **Cluster-Wide Decapitation in Default Mode**: In default mode (`point_specific=False`), traditional halo filtering computes $\rho_b(c) = \max_{i \in \text{border}(c)} \rho_i$, and prunes all cluster members with $\rho_j < \rho_b(c)$. If a single high-density point in a large cluster shares a weak boundary alignment with an adjacent domain, $\rho_b(c)$ becomes equal to that high density. Consequently, 80–90% of authentic internal cluster members with lower density are pruned as noise.
  2. **Median Distortion Under Incomplete Clustering**: In `point_specific=True` mode, calculating `np.median(rho[members])` includes peripheral points that were chained transitively into the cluster. Furthermore, as halo points are removed, cluster density statistics are not updated dynamically, making border detection order-dependent.
  3. **Discrepancy with Documented Directives**: Memory and project guidelines state that Rodriguez-Laio halo filtering is biologically inappropriate for structured ncRNAs due to ragged boundaries and non-uniform alignment depth. Yet `halo_method="similarity_weighted"` remains active by default with `point_specific=False`, systematically stripping valid biological variants.
- **Proposed Fix**:
  Set `point_specific=True` as the default in `domain_finder.py`, or disable halo filtering by default (`halo_method=None`) for ncRNA domain discovery. When halo filtering is enabled, use local density quantiles (e.g., 20th percentile) rather than global maximum border density $\rho_b$.

```python
# FAULTY (domain_finder.py - Traditional halo enabled by default with cluster-wide decapitation):
point_specific_val = bool(attributes.get("point_specific", False))
halo_method = attributes.get("halo_method", "similarity_weighted")

# REVISED (domain_finder.py - Robust point-specific halo default preserving sparse variants):
point_specific_val = bool(attributes.get("point_specific", True))
halo_method = attributes.get("halo_method", None)  # Safe default for ncRNA
```

- **Validation**:
  1. Evaluate family recovery across variable cluster sizes (2 to 250 members).
  2. Confirm that authentic biological homologs in the lower density tail of a cluster are not pruned.

---

## Overlooked Tandem Duplications Due to Unconditional Same-Sequence Filter

- **Evidence**:
  In `aligners.py:153-154`:
  ```python
  if q_seq_str == t_seq_str:
      continue
  ```
  In `PREVIOUS_UPDATE.md:240-252` and `PREVIOUS_UPDATE.md:313`, this is justified as follows:
  > "Filtering same-sequence hits is an intentional scientific choice: self-sequence duplications do not constitute independent comparative evidence of a conserved domain family across sequences... It should not be modified."
- **Impact**:
  1. **Biological Exclusion of Tandem and Multi-Copy ncRNA Domains**: Many structural non-coding RNAs (e.g., tandem riboswitches, CRISPR repeats, box C/D snoRNA guide duplexes, and multi-hairpin microRNA precursors) contain authentic duplicated functional domains within the same transcript or chromosome.
  2. **Self-Hit Inconsistency**: The parser unconditionally skips all hits where `q_seq_str == t_seq_str`, regardless of coordinates. However, in `domain_finder.py:241-246`, Stage 2 artificially re-injects trivial self-hits:
     ```python
     self_targets[local_id] = [query_id, peak_span[0], peak_span[1], local_id]
     ```
     This re-injects identical coordinate self-hits while permanently suppressing true biological tandem duplications ($q\_\text{start} \ne t\_\text{start}$).
  3. **Violation of Alignment Tool Semantics**: MMseqs2 outputs legitimate local alignments between distinct loci on the same sequence. Discarding all same-sequence hits suppresses tandem domain discovery.
- **Proposed Fix**:
  Only filter trivial self-alignments where the query and target coordinates are identical ($q\_\text{start} == t\_\text{start}$ and $q\_\text{end} == t\_\text{end}$). Permit non-identical local alignments on the same sequence ($q\_\text{seq} == t\_\text{seq}$ with non-overlapping or distinct coordinates) to be retained for tandem domain discovery.

```python
# FAULTY (aligners.py - Discards all same-sequence hits including tandem duplications):
if q_seq_str == t_seq_str:
    continue

# REVISED (aligners.py - Filter only trivial identical coordinate hits):
if q_seq_str == t_seq_str and q_start == t_start and q_end == t_end:
    continue
```

- **Validation**:
  1. Parse a BLAST-TAB dataset with known dual-riboswitch tandem domains on the same query.
  2. Verify that tandem domain instances are detected at both distinct loci.

---

## Premature Interval Merging and Strand Consensus Inaccuracies in GFF3 Export

- **Evidence**:
  In `domain_finder.py:44-55` (`determine_strand`):
  ```python
  def determine_strand(strands: np.ndarray) -> str:
      if strands.size == 0:
          return "+"
      neg_count = np.sum(strands == -1)
      pos_count = np.sum(strands == 1)
      return "-" if neg_count > pos_count else "+"
  ```
  In `domain_finder.py:620-635` (`evaluate_clusters`):
  ```python
  sorted_domains = sorted(domains, key=lambda d: d[1])
  spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int32)
  merged_spans = merge_overlapping_spans(spans_arr)

  for m_start, m_end in merged_spans:
      overlapping_strands = [
          d[3] for d in sorted_domains
          if max(d[1], m_start) < min(d[2], m_end)
      ]
      neg = overlapping_strands.count("-")
      pos = overlapping_strands.count("+")
      consensus_strand = "-" if neg > pos else "+"
      component_domains.append((seq_id, int(m_start), int(m_end), consensus_strand))
  ```
- **Impact**:
  1. **Adjacent Half-Open Interval Collapsing**: In `merge_overlapping_spans`, intervals $[10, 20)$ and $[20, 30)$ are merged because $\text{brk} = \text{np.flatnonzero}(s[1:] > e\_\text{cum}[:-1])$ requires a strictly greater start coordinate ($s > e$) to split. For 0-based half-open intervals $[s, e)$, two abutting domains $[10, 20)$ and $[20, 30)$ do not share any nucleotides ($[10, 20) \cap [20, 30) = \emptyset$). Merging abutting domains collapses two distinct biological tandem domains into an artificial hybrid domain $[10, 30)$.
  2. **Antisense Domain Strand Mixing**: When overlapping same-family domains occur on opposite strands (e.g., sense-antisense overlapping non-coding transcripts), `evaluate_clusters` performs majority voting on the strand string (`+` vs `-`). This forces an arbitrary single strand on the merged domain and erases the dual-strand annotation.
  3. **Discrepancy in Coordinate Conversion**: Coordinates in GFF3 must be 1-based closed $[s+1, e]$. If abutting intervals are improperly merged, the resulting GFF3 annotations misrepresent biological domain boundaries.
- **Proposed Fix**:
  Use $s \ge e\_\text{cum}$ to prevent abutting non-overlapping half-open intervals from merging. Partition same-family domains by strand prior to merging so that sense and antisense annotations remain distinct.

```python
# FAULTY (domain_finder.py - Abutting half-open intervals are collapsed into one):
brk = np.flatnonzero(s[1:] > e_cum[:-1])

# REVISED (domain_finder.py - Strict half-open non-overlap preservation):
# In 0-based half-open [s, e), if s[1] == e[0], they touch but do not overlap
brk = np.flatnonzero(s[1:] >= e_cum[:-1])
```

- **Validation**:
  1. Run `merge_overlapping_spans` on $[10, 20)$ and $[20, 30)$.
  2. Verify that two separate spans $[10, 20)$ and $[20, 30)$ are returned rather than $[10, 30)$.

---

## Resource Starvation and Static Worker Spawning in Target Processing

- **Evidence**:
  In `domain_finder.py:786-810`:
  ```python
  if n_target_seqs <= cpu * 4:
      target_batch_size = max(1, int(np.ceil(n_target_seqs / max(1, cpu * 2))))
  else:
      target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))
  ```
  And in `executor.py:270-290` (`Parallel`):
  ```python
  if chunk_size == 'auto':
      if total_items > 0 and n_jobs > 0:
          chunk_size = max(1, min(64, total_items // (n_jobs * 4)))
  ```
- **Impact**:
  1. **Worker Starvation on Skewed Targets**: Batch sizing is computed strictly based on target sequence count, ignoring interval density. If batch 0 contains 10 target sequences with $50,000$ hits each, and batch 1 contains 10 target sequences with $1$ hit each, worker 0 runs for 15 minutes while all other worker processes sit idle.
  2. **Unused Parallel Abstraction**: `domain_finder.py` uses raw `mpire.WorkerPool` directly rather than the unified `executor.Parallel` class. This bypasses the centralized error handling, signal handling, and adaptive chunking provided in `executor.py`.
- **Proposed Fix**:
  Calculate batch partitions weighted by cumulative interval count per batch ($\sum \text{spans}$) rather than sequence count alone. Unify worker pool calls to use `executor.Parallel`.

```python
# FAULTY (domain_finder.py - Unweighted sequence-count batching):
target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))

# REVISED (domain_finder.py - Interval-workload balanced batching):
target_counts = seq_ends - seq_starts
cum_hits = np.cumsum(target_counts)
target_batch_edges = 50_000  # Target hits per batch
batch_cutoffs = np.searchsorted(cum_hits, np.arange(target_batch_edges, cum_hits[-1], target_batch_edges))
```

- **Validation**:
  1. Benchmark pipeline runtime on skewed target datasets with uneven hit distributions.
  2. Confirm uniform CPU core utilization across all allocated workers.

---

## Signal Handling and Recursion Limits in Executor Infrastructure

- **Evidence**:
  In `executor.py:169`:
  ```python
  sys.setrecursionlimit(5000)
  ```
  In `executor.py:180-205` (`timeout_signal`):
  ```python
  old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
  signal.alarm(seconds)
  ```
  In `executor.py:530-555` (`run_external`):
  Background drain threads read intermediate pipe standard error streams using daemon threads.
- **Impact**:
  1. **Thread-Unsafe Signal Handlers**: `signal.signal(signal.SIGALRM, ...)` can only be set from the main Python thread. When worker threads or multi-threaded caller processes invoke `timeout_signal`, Python raises `ValueError: signal only works in main thread of the main interpreter`.
  2. **Arbitrary Recursion Limit Mutation**: Mutating `sys.setrecursionlimit(5000)` inside `checkpoint` without restoring it leaves the global interpreter in an altered state that can cause uncatchable stack overflow C-level segmentation faults on deeply nested structures.
  3. **Zombie Pipe Threads**: In `run_external`, if a child process fails to terminate promptly, daemon drain threads block indefinitely on `pipe_proc.stderr.read()`, leaking file descriptors.
- **Proposed Fix**:
  Use `subprocess.run(..., timeout=timeout)` or thread-safe timer events instead of `SIGALRM`. Restore `sys.getrecursionlimit()` in a `try...finally` block.

```python
# FAULTY (executor.py - Unsafe global signal and recursion mutations):
old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
signal.alarm(seconds)
...
sys.setrecursionlimit(5000)

# REVISED (executor.py - Thread-safe timeout management):
# Use subprocess timeout and restore recursion limit safely
orig_limit = sys.getrecursionlimit()
try:
    sys.setrecursionlimit(max(orig_limit, 5000))
    ...
finally:
    sys.setrecursionlimit(orig_limit)
```

- **Validation**:
  1. Execute `timeout_signal` and `run_external` inside worker threads.
  2. Verify that thread safety exceptions and file descriptor leaks are eliminated.

---

# Phased Implementation Plan for Future Revisions

This phased implementation plan provides concise, mathematically sound, step-by-step guidance for future engineering agents implementing code updates.

## Phase: Span Comparison and Edge Generator Refactoring
1. **Target Files:** `clustering.py`, `domain_finder.py`
   - **Target Functions:** `compare_sorted_spans`, `overlaps_on_target`
   - **Specific Instructions:**
     - Enforce lexicographical input sorting (`order = np.lexsort((ends, starts))`) in `compare_sorted_spans`.
     - Implement in-loop edge chunk yielding within `compare_sorted_spans` so that when accumulated intersections reach `max_edges = 250_000`, coordinate chunks are flushed directly to `ShardedEdgeWriter` without waiting for sequence completion.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not accumulate edges across dense sequences in Python lists prior to chunking.
     - Do not alter 0-based half-open coordinate arithmetic $[start, end)$ during intersection calculations.

## Phase: Domain Family Similarity and Multi-Sequence Normalization
1. **Target Files:** `domain_finder.py`
   - **Target Function:** `family_finder`
   - **Specific Instructions:**
     - Refactor the global Jaccard similarity calculation to normalize overlap against the co-occurring target sequence lengths or calculate sequence-weighted containment.
     - Eliminate global nucleotide length summation across unrelated target sequences to prevent false segregation of high-copy versus low-copy homologs.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not sum raw Jaccard values across batches; sum raw integer intersections and unions per co-occurring sequence pair.
     - Enforce 64-bit integer accumulators for global nucleotide intersection keys.

## Phase: DPC Lineage Constraints and Halo Filtering Defaults
1. **Target Files:** `clustering.py`, `domain_finder.py`
   - **Target Functions:** `dpc`, `halo_filter`, `find_domains`
   - **Specific Instructions:**
     - Constrain pointer-doubling cascading assignment in `dpc` by requiring a minimum direct or path similarity to the cluster peak, preventing transitive chaining of non-overlapping domains.
     - Set `point_specific=True` by default in `find_domains`, and set default `halo_method=None` for ncRNA domain discovery to prevent cluster-wide member decapitation.
   - **Anti-Patterns & Risks to Avoid:**
     - Never assign noise points (-1) to valid clusters or include them in median density calculations.
     - Avoid floating-point perturbations for tie-breaking; preserve exact lexicographical sort ordering.

## Phase: Tandem Duplication Recovery and Strand Consensus Preservation
1. **Target Files:** `aligners.py`, `domain_finder.py`
   - **Target Functions:** `parse_mmseqs_tsv`, `evaluate_clusters`, `merge_overlapping_spans`
   - **Specific Instructions:**
     - In `aligners.py:parse_mmseqs_tsv`, restrict same-sequence filtering strictly to identical coordinates ($q\_\text{start} == t\_\text{start} \text{ and } q\_\text{end} == t\_\text{end}$), admitting distinct tandem alignments.
     - In `merge_overlapping_spans`, change splitting threshold to $s \ge e\_\text{cum}$ to prevent merging abutting half-open intervals.
     - In `evaluate_clusters`, partition domain instances by strand before merging to preserve distinct sense and antisense annotations.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not drop valid tandem duplications across identical chromosomes or transcripts.
     - Do not merge opposing-strand domains into an arbitrary majority strand.

## Phase: Workload-Balanced Parallel Execution and Concurrency Safety
1. **Target Files:** `domain_finder.py`, `executor.py`
   - **Target Functions:** `find_domains`, `Parallel`, `run_external`
   - **Specific Instructions:**
     - Partition Stage 4 target batches based on cumulative hit count rather than sequence count alone.
     - Replace thread-unsafe `SIGALRM` handlers with thread-safe timeout primitives.
     - Ensure recursion limit changes in `checkpoint` are safely restored in `try...finally` blocks.
   - **Anti-Patterns & Risks to Avoid:**
     - Do not allow background thread pipe readers to hang indefinitely on dead child processes.
     - Do not hardcode shared scratch directories across concurrent worker runs.
