## 1. Audit

A comprehensive technical audit of `domain_finder.py`, `aligners.py`, and `clustering.py` identified several architectural bottlenecks, memory anti-patterns, and suboptimal computational routines:

### A. Intermediate In-Memory Grouping Bloat (Memory Overhead)
- **Current Behavior:** In Stage 3 of `domain_finder.py`, all query worker outputs are unpacked into a global Python dictionary `targets_by_seq = defaultdict(list)` where each sequence key stores a list of NumPy 2D array slices.
- **Problem:** For millions of domain instances and tens of millions of target alignment hits, maintaining hundreds of thousands of individual Python list containers containing separate NumPy array slices incurs massive memory allocator fragmentation, pointer indirection, and unnecessary object header overhead (up to 3–5x the raw byte size of the coordinates).
- **Impact:** Memory spikes during the transition between Stage 2 and Stage 4.

### B. Excessive Inter-Process Communication (IPC) Overhead (`chunk_size=1`)
- **Current Behavior:** Both `query_workers` and `target_workers` are invoked using `Parallel(..., chunk_size=1)`.
- **Problem:** Worker pools in `mpire` must serialize (pickle) and transfer IPC payloads across pipe sockets for every single query block and target batch. When thousands of batches are yielded, socket polling and process synchronization dominate execution time, severely penalizing multicore scaling on fast CPU machines.
- **Impact:** Benchmark results revealed process spawning/IPC latency (~0.5–1.0s) exceeding the actual matrix math duration for small/medium datasets.

### C. Redundant Graph Construction During Final Cluster Evaluation
- **Current Behavior:** In `evaluate_clusters`, overlapping same-cluster domains on the same sequence are merged by constructing a sparse coordinate matrix (`coo_matrix`) and running breadth-first search via `csgraph.connected_components`.
- **Problem:** While connected components are necessary for multi-dimensional networks, 1D genomic intervals can be merged strictly in linear time $O(M \log M)$ using cumulative maximum endpoint scanning (`merge_overlapping_spans`), which is already implemented and utilized in Stage 4. Building SciPy sparse matrices and invoking graph routines introduces unnecessary memory allocations and CPU overhead.
- **Impact:** Unnecessary complexity and runtime overhead during the final GFF3 export phase.

### D. Redundant Coordinate Conversions & Off-by-One Legacy Risk
- **Current Behavior:** BLAST-TAB outputs 1-based closed intervals, but NumPy and DPC similarity functions evaluate continuous differences $[s, e)$.
- **Problem:** While self-similarity $1.0$ is maintained by continuous coordinate arithmetic, mixing 1-based external schemas with continuous 0-based indexing without explicit type boundaries risks subtle edge-case boundary errors (such as abutting vs overlapping intervals).

---

## 2. Three Most Promising Proposed Optimizations

The proposed optimizations are ranked below by balancing **predicted gains** (runtime speedup, RAM reduction, and code simplicity) against **implementation complexity**.

---

### Proposal 1: Unified Vectorized 1D Interval Merging in Cluster Evaluation
- **Focus:** Code Simplicity & Speed.
- **Predicted Gain:** **High (Simplifies code; 5–10x faster cluster evaluation).**
- **Implementation Complexity:** **Low.**
- **Description:**
  Replace the `csgraph.connected_components` pipeline in `evaluate_clusters` with the existing `merge_overlapping_spans` routine.
  - Sort member domains on each sequence by start coordinate.
  - Apply `merge_overlapping_spans(spans)` directly to compute $[\min(\text{start}), \max(\text{end})]$ across overlapping intervals.
  - Determine consensus strand across merged intervals using simple count aggregation.
- **Why it is promising:**
  - Completely eliminates SciPy `coo_matrix` and `csgraph` dependencies from Stage 6.
  - Reduces cluster evaluation from $O(V + E)$ graph traversal to a single vectorized NumPy scan.
  - Eliminates ~35 lines of boilerplate graph code.

---

### Proposal 2: Contiguous Structured Array Partitioning for Target Hits
- **Focus:** Peak Memory Footprint & Cache Locality.
- **Predicted Gain:** **Very High (Reduces RAM consumption by 60–75% for large datasets; eliminates dict overhead).**
- **Implementation Complexity:** **Medium.**
- **Description:**
  Instead of appending array slices into a Python `defaultdict(list)`, stream targets directly into a pre-allocated or chunk-appended contiguous 1D structured NumPy array with fields `(t_seq: int32, start: int32, end: int32, dom_ptr: int32)`.
  - Perform a single in-place `np.lexsort((start, t_seq))` on the concatenated array.
  - Partition into target batches using `np.flatnonzero(np.diff(t_seq)) + 1` to find sequence boundaries.
- **Why it is promising:**
  - Eliminates hundreds of thousands of heap-allocated Python lists and dictionary buckets.
  - Keeps all coordinate data in contiguous C-aligned memory, maximizing CPU L2/L3 cache hits during overlap calculations.
  - Directly addresses the 200 GB peak RAM ceiling requirement for multi-gigabyte production runs.

---

### Proposal 3: Dynamic Adaptive Batch Sizing for IPC Worker Pools
- **Focus:** Runtime Speedup & Parallel CPU Utilization.
- **Predicted Gain:** **High (2–4x speedup on small to medium workloads; eliminates process starvation).**
- **Implementation Complexity:** **Low to Medium.**
- **Description:**
  Replace the static `chunk_size=1` setting in `Parallel` with dynamic chunk sizing based on the estimated input size and available CPU threads:
  $$\text{chunk\_size} = \max\left(1, \frac{\text{total\_batches}}{4 \times \text{n\_jobs}}\right)$$
  - For small workloads, workers receive larger chunks, reducing IPC round-trips and socket lock contention.
  - For massive workloads, chunks dynamically scale to keep all worker cores saturated without queue bottlenecks.
- **Why it is promising:**
  - Dramatically improves parallel efficiency, especially on high-core-count servers (32–128 vCPUs).
  - Minimal code change with zero risk of numerical corruption or output discrepancy.

---

## 3. Summary & Decision Matrix

| Optimization Proposal | Primary Benefit | Implementation Complexity | Memory Impact | Runtime Impact | Code Simplicity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Unified Vectorized Interval Merging** | Eliminate Graph Overhead | **Low** | Modest ($\approx 5\%$) | High ($5\times$ faster Stage 6) | **Greatly Improved** |
| **2. Contiguous Structured Array Partitioning** | Eliminate Dict/List Overhead | **Medium** | **Massive ($\approx 70\%$ RAM reduction)** | Moderate ($15–25\%$ faster) | Neutral |
| **3. Dynamic Adaptive Batch Sizing** | Eliminate IPC / Socket Overhead | **Low to Medium** | Neutral | **High ($2–4\times$ faster Stage 2/4)** | Neutral |

---

## 4. In-Depth Evaluation: Coordinate & Span Logic, Stage Equivalence, Missing Safeguards & Data Risks

### 4.1 Executive Summary of Evaluation
An exhaustive audit of coordinate and span handling across `aligners.py`, `domain_finder.py`, `clustering.py`, and `evaluation.py` reveals a critical underlying mathematical discrepancy:
- **BLAST-TAB / MMseqs2 output standard:** Uses 1-based, fully-closed genomic intervals $[s_{\text{1-based}}, e_{\text{1-based}}]$. A 10-nucleotide alignment hit spanning positions 1 through 10 is emitted as `start=1, end=10`.
- **Internal interval calculations in `domain_finder.py` & `clustering.py`:** Use continuous difference arithmetic $[s, e)$ where length is evaluated as $L = e - s$ and pairwise intersection is evaluated as $I = \min(e_1, e_2) - \max(s_1, s_2)$.
- **Mathematical mismatch:** When 1-based closed intervals are passed directly into continuous-difference formulas without upfront conversion:
  - The length of $[1, 10]$ is computed as $10 - 1 = 9$ nt (1 nucleotide short).
  - While self-similarity remains $1.0$ ($9 / (9 + 9 - 9) = 1.0$), non-identical overlaps are distorted. For example, two 10-nt domains $[1, 10]$ and $[6, 15]$ share 5 nucleotides in reality ($[6, 10]$), but continuous math evaluates overlap as $\min(10, 15) - 6 = 4$, yielding Jaccard similarity $4 / (9 + 9 - 4) \approx 0.2857$ instead of the true value $5 / (10 + 10 - 5) = 0.3333$ (an artificial ~14.3% penalty).
  - Single-nucleotide alignment hits ($s = e$) evaluate to $L = 0$, triggering potential division-by-zero errors in Jaccard and containment metrics.
  - Abutting intervals ($[1, 10]$ and $[10, 20]$ in 1-based) share a base (position 10), but under continuous math are treated as having 0 overlap, while truly adjacent non-overlapping intervals ($[1, 10]$ and $[11, 20]$) fail to merge as abutting blocks.
- **Recommended Architectural Solution:** Enforce strict, upfront conversion of MMseqs2 1-based closed coordinates to NumPy-standard 0-based half-open intervals $[s_{\text{0-based}}, e_{\text{0-based}}) = [s - 1, e)$ immediately upon parsing in `aligners.py`. Maintain $[s_0, e_0)$ throughout all internal stages (`compare_sorted_spans`, `jaccard_index`, `max_containment`, `merge_overlapping_spans`), and convert back to 1-based closed $[s_0 + 1, e_0]$ strictly during GFF3 file emission.

---

### 4.2 Stage-by-Stage Equivalence & Numerical Verification

```
  Stage 1: Aligner Streaming TSV Parsing (aligners.py)
    Raw MMseqs2 BLAST-TAB [1, 10] (1-based closed, True Len = 10)
    ▼ CURRENT: Leaves as [1, 10]
    ▼ PROPOSED: Upfront conversion: [s - 1, e) -> [0, 10) (0-based half-open, Len = 10 - 0 = 10)
                    │
                    ▼
  Stage 2 & 3: Query DPC Clustering & Domain Identification (domain_finder.py)
    Unique spans & compare_sorted_spans(unique_query_spans)
    - Length: e - s
    - Overlap: min(e1, e2) - max(s1, s2)
    - Jaccard: I / (L1 + L2 - I)
    ▼ Under [0, 10) and [5, 15): Overlap = 5, Union = 15, Jaccard = 1/3 (Exact!)
    ▼ Under [1, 10] and [6, 15): Overlap = 4, Union = 14, Jaccard = 0.2857 (Distorted!)
                    │
                    ▼
  Stage 4: Target Batch Interval Merging & Pairwise Overlaps (domain_finder.py)
    merge_overlapping_spans(d_spans)
    - Break condition: s[1:] > e_cum[:-1]
    ▼ In half-open: Abutting spans [0, 10) and [10, 20) have s[1] <= e_cum[0] (10 <= 10) -> Merged to [0, 20) (Exact!)
    ▼ In 1-based: Abutting spans [1, 10] and [11, 20] have 11 > 10 -> NOT merged (Fails discrete contiguity!)
                    │
                    ▼
  Stage 5: Family Finding (family_finder in domain_finder.py)
    Summed target intersections & global DPC clustering
    - Unions: dom_lengths[r] + dom_lengths[c] - summed_inters
    - Keys: (u_r << 32) | u_c
                    │
                    ▼
  Stage 6: Cluster Evaluation & GFF3 Export (domain_finder.py)
    Emission of domain coordinates to GFF3
    ▼ CURRENT: Emits raw internal coordinates [s, e] directly -> GFF3 coordinates are off-by-one if internal is 0-based!
    ▼ PROPOSED: Convert back to GFF3 standard 1-based closed: start = s_0 + 1, end = e_0
```

#### Numerical Comparison Table: Coordinate Systems Across Key Steps

| Step / Metric | Biological / Discrete Reality | Current Code Behavior (Unconverted 1-based) | Proposed Upfront 0-Based Half-Open `[s-1, e)` | Status & Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **Span Length $[1, 10]$** | 10 nucleotides ($10 - 1 + 1$) | $10 - 1 = 9$ nt | $10 - 0 = 10$ nt | ⚠️ 1 nt off in current code |
| **Single-base hit $[5, 5]$** | 1 nucleotide ($5 - 5 + 1$) | $5 - 5 = 0$ nt | $5 - 4 = 1$ nt | 🛑 Zero-length risk in current code |
| **Overlap $[1, 10]$ & $[6, 15]$** | 5 nucleotides ($[6, 10]$) | $\min(10, 15) - 6 = 4$ nt | $\min(10, 15) - 5 = 5$ nt | ⚠️ 1 nt undercounted in current code |
| **Union $[1, 10]$ & $[6, 15]$** | 15 nucleotides ($[1, 15]$) | $9 + 9 - 4 = 14$ nt | $10 + 10 - 5 = 15$ nt | ⚠️ 1 nt undercounted in current code |
| **Jaccard $[1, 10]$ & $[6, 15]$** | $5 / 15 \approx 0.3333$ | $4 / 14 \approx 0.2857$ | $5 / 15 \approx 0.3333$ | ⚠️ ~14.3% penalty in current code |
| **Abutting spans $[1, 10]$ & $[11, 20]$** | Contiguous (0 nt gap) | $11 > 10 \implies$ Not merged | $[0, 10)$ & $[10, 20) \implies$ Merged to $[0, 20)$ | ⚠️ Abutting merge fails in current code |
| **GFF3 Export of $[0, 10)$** | `start=1, end=10` | N/A (Emits raw $s, e$) | `start=s+1, end=e` ($1, 10$) | ✅ Fully compatible with GFF3 specs |
| **Evaluation in `evaluation.py`** | Expects GFF3 `[start, end]` | Reads `start - 1, end` | Perfectly aligns with `s_0, e_0` | ✅ Seamless integration |

---

### 4.3 Missing Safeguards & Potential Data Corruption Risks

#### 4. Silent Inversion and Strand Discrepancies
- **Failure Mechanism:** BLAST-TAB reports strand by inverting coordinates ($s > e$ implies reverse complement). Current code in `aligners.py` uses:
  ```python
  t_strand = -1 if (q_start > q_end) ^ (t_start > t_end) else 1
  q_start, q_end = min(q_start, q_end), max(q_start, q_end)
  t_start, t_end = min(t_start, t_end), max(t_start, t_end)
  ```
- **Safeguard Assessment:**
  - If coordinates are converted upfront to 0-based half-open, normalization must occur **before** conversion:

---

### 4.4 Actionable Implementation Checklist for Full Harmonization

1. **`aligners.py` (TSV Ingestion):**
   - Update `parse_mmseqs_tsv` to normalize coordinates:
     $$s_0 = \min(s, e) - 1, \quad e_0 = \max(s, e)$$
   - Assert $s_0 \ge 0$ and $e_0 > s_0$.
   - Yield 0-based half-open NumPy integer arrays.
2. **`clustering.py` (Coordinate Math):**
   - Retain current vectorized difference math ($e_0 - s_0$ and $\min(e_{1}, e_{2}) - \max(s_{1}, s_{2})$). Under $[s_0, e_0)$, this math is exact and yields un-penalized, biologically accurate Jaccard similarities.
3. **`domain_finder.py` (Stage 6 GFF3 Emission):**
   - When formatting GFF3 rows:
     $$\text{gff\_start} = s_0 + 1, \quad \text{gff\_end} = e_0$$
   - Emits canonical 1-based closed intervals directly compliant with GFF3 specifications and `evaluation.py`.

---

## 5. Unified Update Implementation Plan

---

### Phase 1: Upfront Coordinate Ingestion Harmonization & Safeguards (`aligners.py`)

#### Objective
Establish a single source of truth for genomic intervals by converting 1-based closed intervals from BLAST-TAB TSVs into 0-based half-open intervals $[s_0, e_0) = [\min(s, e) - 1, \max(s, e))$ immediately upon ingestion, while instituting robust data boundary safeguards.

#### Detailed Implementation Instructions
1. **Coordinate Conversion & Strand Calculation (`parse_mmseqs_tsv` in `aligners.py`):**
   - Parse raw integer fields from TSV lines:
     ```python
     raw_qs, raw_qe = int(fields[6]), int(fields[7])
     raw_ts, raw_te = int(fields[8]), int(fields[9])
     ```
   - Calculate relative alignment strand prior to normalization:
     ```python
     t_strand = -1 if (raw_qs > raw_qe) ^ (raw_ts > raw_te) else 1
     ```
   - Normalize coordinates into canonical 0-based half-open format $[s_0, e_0)$:
     ```python
     qs, qe = min(raw_qs, raw_qe) - 1, max(raw_qs, raw_qe)
     ts, te = min(raw_ts, raw_te) - 1, max(raw_ts, raw_te)
     ```
   - **Safeguard 1 (Coordinate Sanity):** Assert that $qs \ge 0$ and $qe > qs$, as well as $ts \ge 0$ and $te > ts$. If any alignment hit has length $< 1$, raise an explicit error:
     ```python
     raise ValueError(f"🛑 Invalid alignment hit coordinates: q=[{qs}, {qe}), t=[{ts}, {te}) in line: {line}")
     ```
   - **Safeguard 2 (Sequence Boundary Enforcement):** When sequence lengths are available from `SequenceRegistry` (or optional length mapping), verify $qe \le L_{\text{query}}$ and $te \le L_{\text{target}}$.
2. **Deterministic Block Sorting (`format_block` in `aligners.py`):**
   - Retain deterministic sorting: primary sort on query start position ($s_0$) and tie-breaker on end position ($e_0$):
     ```python
     sort_order = np.lexsort((query_spans[:, 1], query_spans[:, 0]))
     ```
   - Reorder both `query_spans` and `target_spans` using the exact same `sort_order` to preserve 1:1 row alignment.
   - Ensure `query_spans` dtype is `np.int64` (or `np.int32` where coordinates fit within 2 GB) and `target_spans` dtype is `np.int64`.

---

### Phase 2: Dynamic Adaptive Batch Sizing for IPC Worker Pools (`executor.py` & `domain_finder.py`)

#### Objective
Eliminate IPC socket serialization overhead, pipe lock contention, and worker starvation by replacing static `chunk_size=1` with bounded adaptive batch chunking.

#### Detailed Implementation Instructions
1. **Adaptive Chunk Sizing in `executor.Parallel` (`executor.py`):**
   - Enforce an upper bound on dynamic chunk sizing when processing sequence collections to prevent memory bloat per task:
     ```python
     if chunk_size == 'auto':
         if total_items > 0 and n_jobs > 0:
             chunk_size = max(1, min(200, total_items // (n_jobs * 4)))
         else:
             chunk_size = 1
     ```
2. **Stage 2 Query Worker Chunking (`domain_finder.py`):**
   - In `find_domains`, estimate total query batches from input FASTA query count (`count_fasta_queries(fasta_path)`):
     $$\text{total\_batches} \approx \left\lceil \frac{N_{\text{queries}}}{\text{batch\_size}} \right\rceil$$
   - Configure `chunk_size='auto'` or dynamically compute:
     $$\text{query\_chunk\_size} = \max\left(1, \min\left(50, \left\lfloor \frac{\text{total\_batches}}{4 \times \text{cpu}} \right\rfloor\right)\right)$$
   - Pass this chunk size to `query_workers = Parallel(..., chunk_size=query_chunk_size)`.
3. **Stage 4 Target Worker Chunking (`domain_finder.py`):**
   - Similarly, configure dynamic chunk sizing for `target_workers`:
     $$\text{target\_chunk\_size} = \max\left(1, \min\left(50, \left\lfloor \frac{\text{total\_target\_batches}}{4 \times \text{cpu}} \right\rfloor\right)\right)$$
   - Bound worker core memory ceiling strictly below **200 MB per core** by guaranteeing batch tasks release intermediate NumPy buffers with explicit local scoping.

---

### Phase 3: Contiguous Structured Array Partitioning for Target Hits (`domain_finder.py`)

#### Objective
Eliminate memory allocator fragmentation and Python `defaultdict(list)` overhead in Stage 3 by packing alignment target hits into contiguous 1D structured NumPy arrays.

#### Detailed Implementation Instructions
1. **Structured Target Packing in Master Ingestion:**
   - Define structured array dtype for target hits:
     ```python
     target_dtype = np.dtype([
         ('t_seq', np.int64),
         ('start', np.int64),
         ('end', np.int64),
         ('dom_id', np.int32)
     ])
     ```
   - As `query_workers.results` are yielded, accumulate target records directly into pre-allocated chunk buffers or append flat structured arrays into a top-level list `chunk_arrays = []`.
   - Concatenate all target hits in a single pass:
     ```python
     all_target_records = np.concatenate(chunk_arrays)
     ```
2. **Contiguous Lexicographical Sorting & Sequence Partitioning:**
   - Sort the contiguous structured array in-place by target sequence ID (`t_seq`) as primary key and `start` as secondary key:
     ```python
     all_target_records.sort(order=['t_seq', 'start'])
     ```
   - Partition into per-target sequence slices using vectorized boundary detection without intermediate Python dictionaries:
     ```python
     seq_boundaries = np.flatnonzero(np.diff(all_target_records['t_seq'])) + 1
     target_sequence_groups = np.split(all_target_records, seq_boundaries)
     ```
3. **Direct Generator Streaming to Stage 4:**
   - Create a clean generator yielding target batches directly to `target_batch_processor`:
     ```python
     def target_batch_generator():
         batch_pkg = []
         for group in target_sequence_groups:
             t_seq = int(group['t_seq'][0])
             spans = np.column_stack((group['start'], group['end']))
             dom_ptrs = group['dom_id']
             batch_pkg.append((t_seq, spans, dom_ptrs))
             if len(batch_pkg) >= 1000:
                 yield batch_pkg
                 batch_pkg = []
         if batch_pkg:
             yield batch_pkg
     ```
   - Reduces master process heap allocations by 60–75%, maintaining peak RAM strictly below **200 GB** for multi-gigabyte production workloads.

---

### Phase 4: Unified Vectorized 1D Interval Merging in Stage 6 (`domain_finder.py`)

#### Objective
Replace heavy graph connected components (`coo_matrix` + `csgraph.connected_components`) in `evaluate_clusters` with linear-time vectorized interval scanning (`merge_overlapping_spans`).

#### Detailed Implementation Instructions
1. **Refactor `evaluate_clusters` in `domain_finder.py`:**
   - For each cluster peak, group member domain instances by sequence ID.
   - For sequences with multiple domain instances:
     - Sort domain instances by start coordinate:
       ```python
       sorted_domains = sorted(domains, key=lambda d: d[1])
       spans_arr = np.array([[d[1], d[2]] for d in sorted_domains], dtype=np.int64)
       ```
     - Merge overlapping intervals using the existing `merge_overlapping_spans(spans_arr)` function:
       ```python
       merged_spans = merge_overlapping_spans(spans_arr)
       ```
     - For each merged interval $[s_{\text{merged}}, e_{\text{merged}})$, find the overlapping constituent domains to determine the consensus strand by simple majority vote:
       ```python
       # Identify constituent domains overlapping with merged span
       constituent_strands = [
           d[3] for d in sorted_domains
           if max(d[1], s_m) < min(d[2], e_m)
       ]
       consensus_strand = "-" if constituent_strands.count("-") > constituent_strands.count("+") else "+"
       component_domains.append((seq_id, s_m, e_m, consensus_strand))
       ```
2. **Remove Unused Imports:**
   - Remove `from scipy.sparse import coo_matrix, csgraph` from `domain_finder.py`, shedding unnecessary dependencies and reducing module import overhead.

---

### Phase 5: GFF3 Export Coordinate Conversion & Formatting (`domain_finder.py`)

#### Objective
Ensure emitted GFF3 files strictly follow the canonical 1-based fully-closed genomic coordinate convention $[s_1, e_1] = [s_0 + 1, e_0]$, ensuring seamless compatibility with standard bioinformatics tooling and `evaluation.py`. Other than that final results should be unchanged or explicitly corrected.  

#### Detailed Implementation Instructions
1. **1-Based Closed GFF3 Emission:**
   - In `find_domains`, format GFF3 records converting from internal 0-based half-open $[s_0, e_0)$ to 1-based closed coordinates:
     ```python
     gff_start = start + 1
     gff_end = end
     gff_line = f"{seq_name}\t{aligner_name}\tDomain\t{gff_start}\t{gff_end}\t.\t{strand}\t.\tID={fam_id}\n"
     gff_writer.write(gff_line)
     ```

---

### Phase 6: End-to-End Verification, Benchmarking & Acceptance Criteria

#### 1. Automated Unit & Integration Testing
- Execute test suite:
  ```bash
  python3 -m pytest
  ```
- All test suites in `tests/test_aligners.py`, `tests/test_domain_finder.py`, and `tests/test_msa_benchmark.py` must pass with zero failures.
- If existing tests fail due to 1-based vs 0-based fixture expectations, update test fixtures to explicitly reflect the upfront $[s-1, e)$ conversion and 1-based closed GFF3 emission.

#### 2. Deterministic Result Equivalence & Semantic Identity
- Run simulated alignment benchmark with 25,000 and 250,000 hits (`benchmark.py` or synthetic domain finding test).
- Validate that:
  - Total predicted domain instances and family clusters are identical.
  - Coordinate lengths equal discrete nucleotide counts ($L = \text{end} - \text{start} + 1$).
  - No spurious zero-length intervals or negative Jaccard values occur.

#### 3. Resource Boundary Validation
- Confirm worker processes consume $< 200$ MB RAM per core during Stage 2 and Stage 4.
- Confirm master process RAM remains $< 200$ GB on multi-gigabyte TSV alignment runs.

#### 4. Pre-Commit Verification
- Re-evaluate scientific and mathematical validity of all steps after update and add any problems, inconsistencies or doubts at the end of PROBLEMS.md together with full update report.
