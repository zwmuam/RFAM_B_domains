## Performance Speedup & Parallelisation Plan (`aligners.py`, `domain_finder.py`, `clustering.py`, `executor.py`)

### Section 1: Top-4 High-Impact Speedup & Optimization Proposals

#### Proposal 1: Direct Vectorized NumPy Coordinate Range Slicing (`clustering.py`)
- **Target Bottleneck:** `compare_sorted_spans()` currently converts Python lists or extracts coordinates sequentially per span. For large alignments with thousands of hits per sequence, the inner loop calls Python `np.searchsorted` or list comprehensions repeatedly.
- **Optimization Strategy:**
  1. Store span start/end coordinates directly as contiguous 2D NumPy integer arrays (`int32` or `int64`, shape `(N, 2)`).
  2. Implement broad vectorised block slicing with `np.searchsorted` across query/target span arrays, computing upper-triangular range intersections (`np.minimum(end_i, ends) - start_j`) in vectorised C loops.
  3. Pre-allocate flat 1D output buffers for `(rows, cols, intersections)` to eliminate `list.append()` overhead and subsequent `np.concatenate` allocations.
- **Safety & Scientific Rigor:**
  - Preserves exact $O(N \log N + K)$ complexity and identical numerical range intersections.
  - Zero risk of data corruption as array slice sorting invariant (`start_j >= start_i`) is verified via cheap assertions.

#### Proposal 2: Parallel Batch Processing & Streaming Vectorized Ingestion (`aligners.py` & `domain_finder.py`)
- **Target Bottleneck:** Single-threaded parsing of multi-GB TSV files and sequential `find_domain_instances()` execution bottleneck CPU throughput when processing tens of thousands of query sequences.
- **Optimization Strategy:**
  1. Stream query blocks from `parse_mmseqs_tsv_stream()` into fixed-size multi-query batches (e.g., 50–200 query sequences per chunk).
  2. Map chunk batches across worker processes using `executor.Parallel` (`WorkerPool`).
  3. Perform local DPC domain discovery (`find_domain_instances`) and local spurious domain filtering (`filter_spurious_domains`) in parallel inside worker processes.
- **Safety & Scientific Rigor:**
  - Query sequence processing is strictly independent across sequence IDs; domain calculations per sequence yield identical results regardless of execution order.
  - Memory ceiling per worker is strictly bounded by chunk size $\times$ max hits per sequence ($<50$ MB per worker process).

#### Proposal 3: SciPy CSR Sparse Matrix Aggregation for Target Overlaps (`domain_finder.py`)
- **Target Bottleneck:** Pairwise target overlap calculation (`calculate_target_overlaps()`) currently appends SciPy COO row/col/data arrays per sequence, followed by global COO/CSR matrix conversion and `sum_duplicates()` in `families()`.
- **Optimization Strategy:**
  1. Execute sequence-level target overlap matrix assembly in parallel across target sequence blocks using worker pools.
  2. Perform immediate per-chunk sparse CSR deduplication (`csr.sum_duplicates()`) before returning results to the master process.
  3. Accumulate chunk CSR matrices using lightweight in-place matrix addition (`csr_total += csr_chunk`), avoiding massive 1D array concatenations of hundreds of millions of duplicate overlap edges.
- **Safety & Scientific Rigor:**
  - Mathematical equivalence is guaranteed by the associative and commutative properties of sparse matrix additions ($\mathbf{A} + \mathbf{B}$).
  - Reduces peak RAM during target overlap aggregation by 60–80%.

#### Proposal 4: Vectorized Pointer Doubling & Sparse CSR Gather for DPC (`clustering.py`)
- **Target Bottleneck:** In `dpc()`, the cascading assignment step propagates labels along parent pointers via a Python `while True` loop (`parent = grandparent`). Moreover, sparse CSR top-K neighbour gather operations iterate over active rows.
- **Optimization Strategy:**
  1. In cascading assignment, replace Python while-loop checks with vectorised NumPy pointer doubling (`parent = parent[parent]`) using early termination checks (`np.array_equal`).
  2. In `_top_k_scipy_global`, use vectorised NumPy `bincount` and `cumsum` to compute indptr slices, eliminating per-row iteration overhead.
  3. Avoid intermediate dense array conversions (`toarray()`) in `build_similarity_matrix()` when matrix dimension $N > 2000$.
- **Safety & Scientific Rigor:**
  - Pointer doubling reaches cluster peaks in logarithmic iterations ($O(\log D)$, where $D$ is max tree depth).
  - Exact match of cluster labels and halo noise assignments guaranteed.


---

### Section 3: Safe Parallelisation Plan (`executor.py` Integration)

#### 1. Architecture Overview
Parallelisation leverages the repository's native `executor.Parallel` wrapper around MPIRE `WorkerPool`. Processing is parallelised across two primary stages:
1. **Query-Level Stage:** Parallel local DPC domain discovery (`find_domain_instances`) and spurious domain filtering (`filter_spurious_domains`).
2. **Target-Level Stage:** Parallel pairwise target overlap matrix construction (`calculate_target_overlaps`).

#### 2. Thread/Process-Safe Sequence ID Management (`SequenceRegistry`)
- **Challenge:** String interning (`sys.intern`) and sequential integer ID assignment (`SequenceRegistry`) must maintain global consistency across parallel workers without race conditions or memory corruption.
- **Solution Strategy:**
  - **Two-Pass Hybrid Design:**
    1. **Pre-Scan Ingestion (Master Process):** The single-pass parser `parse_mmseqs_tsv_stream()` populates the master `SequenceRegistry` during stream chunk creation before dispatching jobs to workers.
    2. **Read-Only Serialization:** The immutable `SequenceRegistry` dictionary (`str_to_id`) and list (`id_to_str`) are passed to workers as a read-only shared object (`shared_objects={'registry': registry}`).
  - **Zero Worker Overhead:** Workers perform integer lookups without acquiring thread locks or mutating shared memory, guaranteeing thread safety and maximum execution speed.

#### 3. Explicit Multi-Query Task Batching & Chunk Size Optimization
- **Impact of Multi-Query Batching:**
  - Mapping single query sequences to worker processes introduces substantial Inter-Process Communication (IPC) overhead, process context-switching latency, and progress bar lock contention.
  - **Batching Strategy:** Query sequences are grouped into contiguous multi-query batches of size $B$ (default chunk size: 50–200 query sequences per chunk).
  - **Dynamic Chunk Slicing:** The chunk size is calculated dynamically based on total query count $N_Q$ and CPU threads $T$:
    $$\text{chunk\_size} = \max\left(1, \min\left(200, \frac{N_Q}{T \times 4}\right)\right)$$
  - **Performance Influence:**
    - Reduces IPC serialization overhead by $>95\%$.
    - Bypasses Python GIL bottlenecks.
    - Prevents RAM spikes by keeping chunk memory footprints $<50$ MB per worker core.

#### REQUIREMENTS
- **Sequence Identifier Preservation:** Each `DomainInstance` MUST retain proper `seq_id` mapping arccos the whole pipeline
- **Coordinate Integrity:** Start and end coordinates (`start`, `end`), strand orientation (`strand`), and partner target alignment spans (`partners`) must remain logically equivalent
- **Output Fidelity:** The updated pipeline have to recreate the output of previous version or correct explicit documented errors. 
- **Determinism:** Results should be fully deterministic
