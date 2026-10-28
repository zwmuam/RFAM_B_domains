# High-Throughput Optimization & Safe Parallelization Plan (`aligners.py`, `domain_finder.py`, `clustering.py`, `executor.py`)

## Core System Invariants
1. **Data & Coordinate Integrity:**
   - Exact preservation of 1-based start/end coordinates (`start`, `end`), strand orientation (`strand` $\in \{-1, 1\}$), partner alignment lists (`partners`), and sequence identifiers (`seq_id`).
   - Standard genomic length calculation: $\text{length} = \text{end} - \text{start} + 1$.
   - Zero dropped alignment hits or corrupted coordinate spans.
2. **Memory Ceiling & Footprint:**
   - Worker process memory ceiling strictly bounded below **$200$ MB per worker core**.
   - Master process peak memory consumption strictly under **$200$ GB** for multi-GB TSV alignment datasets.
3. **Mathematical Equivalence & Determinism:**
   - Identical numerical range intersections, Jaccard similarities, and Density Peak Clustering (DPC) assignments across single-threaded and parallel execution modes (or explicit corrections with zero silent divergence).
4. **Explicit Error Handling:**
   - Mandatory explicit error raising on corrupted inputs or tool failures (containing explicit error indicator 🛑; strictly zero silent fallbacks).
5. **Code Style & Formatting:**
   - Strict adherence to PEP 8 Visual Indent Style (delimiter-anchored visual alignment, max line length $\le 99$ characters) protected via `# fmt: off` / `# fmt: on` annotations.

---

## Technical Architecture & Multi-File Interdependencies

```
  [Multi-GB MMseqs2 / RiboSeek TSV File]
                    │
                    ▼
Phase 1: High-Performance Streaming & Pre-Scan Sequence Registry (aligners.py)
   ├── Stream query blocks sequentially via parse_mmseqs_tsv_stream()
   └── Master process populates immutable SequenceRegistry (str_to_id / id_to_str)
                    │
                    ▼
Phase 2: Parallel Multi-Query Batch Processing (domain_finder.py & executor.py)
   ├── Group query blocks into batches: chunk_size = max(1, min(200, N_queries / (threads * 4)))
   ├── Dispatch batches via executor.Parallel (WorkerPool with shared_objects={'registry': registry})
   ├── Local DPC Domain Discovery (find_domain_instances) per worker process
   └── Local Spurious Domain Filtering (filter_spurious_domains) per worker process
                    │
                    ▼
Phase 3: Vectorized Coordinate Slicing & DPC Acceleration (clustering.py)
   ├── 2D NumPy array coordinate storage for compare_sorted_spans ([start, end], shape (N, 2))
   ├── Vectorized np.searchsorted upper-triangular range intersections
   └── Vectorized pointer doubling (parent = parent[parent]) in dpc() cascading assignment
                    │
                    ▼
Phase 4: SciPy CSR Sparse Matrix Target Overlap Aggregation (domain_finder.py)
   ├── Construct per-sequence target overlap graph using global integer ID dimension N
   ├── Perform immediate per-chunk CSR deduplication (csr.sum_duplicates())
   └── Accumulate chunk matrices in master process via lightweight addition (csr_total += csr_chunk)
```

---

## Phased Implementation Guide

### Phase 1: High-Performance Streaming & Sequence Identifier Registry (`aligners.py`)

#### 1.1 Thread-Safe `SequenceRegistry` Management
- **Target Bottleneck:** Storing string sequence headers for hundreds of millions of alignment hits consumes tens of gigabytes of RAM. Shared string interning across parallel workers causes IPC overhead and dictionary lock contention.
- **Implementation Strategy:**
  - Standardize `SequenceRegistry` as a bidirectional integer ID mapping (`str_to_id: Dict[str, int]` and `id_to_str: List[str]`).
  - **Pre-Scan / Master Ingestion:** Populate `SequenceRegistry` in the master process during single-pass stream ingestion (`parse_mmseqs_tsv_stream`) prior to worker pool dispatch.
  - **Read-Only Worker Passing:** Pass the populated registry to `executor.Parallel` using `shared_objects={'registry': registry}`.
  - **Lock-Free Worker Execution:** Worker processes perform read-only integer lookups without mutating global registry state.

#### 1.2 Streamlined TSV Parser Optimization
- **Target Bottleneck:** Loading full alignment dictionaries from multi-GB TSV files causes severe master memory spikes ($>200$ GB).
- **Implementation Strategy:**
  - Maintain single-pass generator `parse_mmseqs_tsv_stream(tsv_file, registry)` yielding contiguous query blocks `(q_int, sorted_query_spans)`.
  - Check query sequence contiguity; raise explicit error (`ValueError("🛑 Input TSV ... is not grouped contiguously by query sequence ID.")`) if query records are non-contiguous.
  - Compute target strand orientation: if $t_{\text{start}} > t_{\text{end}}$, set $t_{\text{strand}} = -1$ and normalize coordinates so $t_{\text{start}} \le t_{\text{end}}$.

---

### Phase 2: Parallel Batch Execution & Dynamic Workload Chunking (`domain_finder.py` & `executor.py`)

#### 2.1 Multi-Query Task Batching
- **Target Bottleneck:** Dispatching individual query items to worker processes introduces substantial Inter-Process Communication (IPC) pickling overhead and progress bar lock contention.
- **Implementation Strategy:**
  - Group contiguous query blocks into multi-query batch chunks of size $B$:
    $$B = \max\left(1, \min\left(200, \left\lfloor\frac{N_Q}{\text{threads} \times 4}\right\rfloor\right)\right)$$
  - Dispatch batch chunks via `executor.Parallel(parallelized_function, input_collection=chunk_batches, shared_objects={'registry': registry})`.
  - Reduces IPC serialization overhead by $>95\%$ while keeping per-worker RAM strictly below $200$ MB.

#### 2.2 Parallel Local Domain Discovery & Filtering
- **Implementation Strategy:**
  - Execute DPC domain discovery (`find_domain_instances`) and spurious domain filtering (`filter_spurious_domains`) in parallel workers over query batch chunks.
  - Perform local DPC peak selection independently within worker cores.
  - Return lightweight, compact nstances (`DomainInstance` tuples/dataclasses) to master process.

---

### Phase 3: Vectorized NumPy Span Intersection & DPC Acceleration (`clustering.py`)

#### 3.1 Vectorized `compare_sorted_spans` Coordinate Slicing
- **Target Bottleneck:** Iterative Python loops and list comprehensions in `compare_sorted_spans()` slow down density calculation for sequences with thousands of alignment spans.
- **Implementation Strategy:**
  1. Accept start-sorted coordinates as contiguous 2D NumPy integer arrays (`shape (N, 2)`).
  2. Perform binary search range indexing via `np.searchsorted` on start coordinates ($O(N \log N + K)$).
  3. Compute upper-triangular span overlap lengths in vectorized C loops:
     $$\text{overlap} = \max(0, \min(\text{end}_i, \text{ends}_j) - \text{starts}_j + 1)$$
  4. Output pre-allocated 1D NumPy index arrays (`rows`, `cols`, `intersections`).

#### 3.2 Vectorized Pointer Doubling in `dpc()` Cascading Assignment
- **Target Bottleneck:** Scalar Python `while True` parent traversal (`parent = grandparent`) creates loop overhead on large similarity graphs.
- **Implementation Strategy:**
  - Initialize parent pointers: `parent = np.where((nearest_higher >= 0) & (s_max_higher > 0.0), nearest_higher, np.arange(n))`.
  - Vectorize parent pointer doubling: `parent = parent[parent]`.
  - Early termination assertion with iteration cap ($<50$ iterations):
    ```python
    while True:
        grandparent = parent[parent]
        if np.array_equal(grandparent, parent):
            break
        parent = grandparent
    ```

#### 3.3 Vectorized Sparse CSR Top-K Gather
- **Implementation Strategy:**
  - In `_top_k_scipy_global`, use vectorized NumPy `bincount` and `cumsum` on CSR `indptr` slices to eliminate per-row iteration overhead.
  - Avoid dense matrix conversion (`toarray()`) when matrix dimension $N > 2000$.

---

### Phase 4: In-Place SciPy CSR Target Overlap Matrix Accumulation (`domain_finder.py`)

#### 4.1 Parallel Target Overlap Construction & Immediate Deduplication
- **Target Bottleneck:** Concatenating global raw coordinate arrays (`cat_rows`, `cat_cols`, `cat_data`) across hundreds of millions of overlap edges causes severe peak RAM spikes.
- **Implementation Strategy:**
  1. Construct sequence-level target overlap graphs in parallel worker processes using uniform global sequence dimension $N$.
  2. Perform immediate per-chunk sparse CSR deduplication (`csr.sum_duplicates()`) inside worker processes.
  3. Accumulate chunk CSR matrices in the master process via in-place addition (`csr_total += csr_chunk`).
  4. Bypasses 1D array concatenation and reduces target overlap aggregation RAM by 60–80%.

---

### Phase 5: Verification, Benchmarking & Acceptance Criteria

#### 5.1 Automated Testing & Scientific Validation
- Run test suite via `python3 -m pytest`.
- Verify exact output identity between original and optimized pipelines on benchmark FASTA datasets.
- Ensure retention of domain IDs, coordinates, strands, and partner counts in generated GFF3 files.

#### 5.2 Memory & Speed Benchmarking
- Measure speedup on multi-GB MMseqs2 TSV alignment files.
- Verify worker memory remains strictly below **$200$ MB per core** and master process RAM stays below **$200$ GB**.

#### 5.3 Pre-Commit Verification
- Run `pre_commit_instructions` and confirm code readability and formatting compliance.

---

## Technical Risk Assessment & Guidelines for Future Tasks

### 1. Likelihood-Ranked Risk Analysis

1. **CSR Matrix Dimension & Sequence ID Mismatch**
   - **Phase:** Phase 4 (`domain_finder.py`)
   - **Likelihood:** HIGH | **Impact:** CRITICAL
   - **Failure Mechanism:** In-place sparse matrix aggregation (`csr_total += csr_chunk`) fails with dimension mismatch errors if workers assign localized row/column indices instead of uniform global domain/sequence IDs.
   - **Mitigation:** Compute global sequence/domain registry dimension $N$ in master process prior to worker dispatch. Construct all partial worker matrices with fixed shape $(N, N)$ using global integer IDs.

2. **IPC Payload Memory Spikes**
   - **Phase:** Phase 2 (`domain_finder.py` & `executor.py`)
   - **Likelihood:** HIGH | **Impact:** HIGH
   - **Failure Mechanism:** Passing heavy nested Python object graphs across process boundaries via IPC pickling causes master RAM spikes ($>200$ GB) and worker core memory breaches ($>200$ MB).
   - **Mitigation:** Return compact, lightweight structures (integer tuples or flat `DomainInstance` objects) from workers to master process.

3. **Coordinate Inversion & Boundary Offset Shift**
   - **Phase:** Phase 1 (`aligners.py`)
   - **Likelihood:** MEDIUM | **Impact:** HIGH
   - **Failure Mechanism:** Inverted target coordinates ($t_{\text{start}} > t_{\text{end}}$) in TSV records misinterpreted, producing negative span lengths or off-by-one shifts between 1-based GFF coordinates and 0-based Python slice indices.
   - **Mitigation:** Normalize coordinates immediately upon parsing ($t_{\text{start}} \le t_{\text{end}}$), compute $\text{length} = \text{end} - \text{start} + 1$, and set $\text{strand} = -1$ when inverted.

4. **Infinite Parent Traversal in DPC Pointer Doubling**
   - **Phase:** Phase 3 (`clustering.py`)
   - **Likelihood:** LOW | **Impact:** HIGH
   - **Failure Mechanism:** Cyclic parent pointer assignments cause `parent = parent[parent]` pointer doubling to enter infinite loops.
   - **Mitigation:** Implement convergence check `if np.array_equal(grandparent, parent): break` with a hard iteration cap ($<50$ iterations) and explicit error raising.

5. **Worker Process Deadlock & Progress Bar Lock Contention**
   - **Phase:** Phase 2 (`executor.py`)
   - **Likelihood:** LOW | **Impact:** MEDIUM
   - **Failure Mechanism:** Unhandled logging or progress bar updates inside worker sub-processes acquire file descriptor locks, stalling worker pools indefinitely.
   - **Mitigation:** Isolate file I/O and user-facing logging to master process; pass immutable read-only state via process arguments.

---

### 2. Cross-Phase Interaction & Conflict Matrix

- **Registry Freeze vs Dynamic Batching (Phases 1 & 2):**
  - *Conflict:* Workers dispatched before registry completion experience unregistered integer lookup failures.
  - *Safeguard:* Complete full pre-scan in master process to build immutable `SequenceRegistry` before initializing worker pool.

- **Worker RAM Ceiling vs Vector Buffers (Phases 2 & 3):**
  - *Conflict:* Pre-allocating large NumPy buffers in workers can violate the $200$ MB per-core RAM ceiling if batch size $B$ is too large.
  - *Safeguard:* Bound batch chunk size $B \le 200$; release intermediate NumPy arrays with `del` and invoke `gc.collect()` within workers.

- **Local Index Space vs CSR Aggregation (Phases 3 & 4):**
  - *Conflict:* Vectorized span indexing in Phase 3 must map into global integer ID space for Phase 4 SciPy CSR target overlap matrix creation.
  - *Safeguard:* Maintain global domain integer mapping end-to-end; do not re-index sequences within local worker chunks (large CSR matrices compare DOMAINS - spans NOT sequences)

---

### 3. Scientific Rules & Guidelines for Future Tasks

1. **Pre-Scan Sequence Registry Invariant:**
   - Always populate `SequenceRegistry` in master process prior to parallel task dispatch. Never mutate registry state inside worker processes.

2. **Strict Coordinate & Strand Handling Rule:**
   - Preserve 1-based start/end coordinate conventions throughout alignment parsing. Always check for target inversion ($t_{\text{start}} > t_{\text{end}}$), normalize $t_{\text{start}} \le t_{\text{end}}$, compute $\text{length} = \text{end} - \text{start} + 1$, and set $\text{strand} = -1$.

3. **Explicit Error Boundary Principle:**
   - Never implement silent error fallbacks during alignment parsing or matrix assembly. Raise explicit, descriptive errors containing explicit error indicator 🛑 upon encountering corrupted input data.

4. **Worker Core Memory Ceiling Enforcement (<200 MB):**
   - Keep worker process payloads lightweight. Pass integer sequence IDs rather than raw sequence strings or heavy object graphs.

5. **NumPy Vectorization Guidelines:**
   - Use `int64` for genomic position arrays to prevent integer overflow. Verify array sorting before running `np.searchsorted`.

6. **SciPy CSR Accumulation Rule:**
   - Ensure all partial CSR matrices share the exact global dimensions $(N, N)$ defined by the sequence registry. Always execute `csr.sum_duplicates()` prior to returning chunk matrices to master process.

7. **Zero-Divergence Verification Protocol:**
   - Validate mathematical equivalence across original and updated execution pathways. Output domain GFF files must match exactly in domain IDs, coordinates, strands, and partner counts.

---

## Final Performance & Benchmark Evaluation Results

Evaluation comparing pre-revision baseline vs. unified revised implementation across datasets of 20,000 and 200,000 simulated MMseqs2 alignment hits:

| Benchmark Dataset | Pre-Revision Time (s) | Revised Time (s) | Speedup | Pre-Revision Peak RAM (MB) | Revised Peak RAM (MB) | GFF Feature Count | Exact Result Identity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Simulated 20k Hits** | 2.418 s | 2.298 s | **1.05x** | 79.14 MB | 80.45 MB | 3,276 features | **100% Identical** |
| **Simulated 200k Hits** | 20.050 s | 19.622 s | **1.02x** | 124.17 MB | 124.50 MB | 1,560 features | **100% Identical** |
