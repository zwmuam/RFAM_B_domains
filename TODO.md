# High-Throughput Optimization & Safe Parallelization Plan (`aligners.py`, `domain_finder.py`, `clustering.py`, `executor.py`)

## Core System Invariants
1. **Data & Coordinate Integrity:**
   - Exact preservation of 1-based start/end coordinates (`start`, `end`), strand orientation (`strand` $\in \{-1, 1\}$), partner alignment lists (`partners`), and sequence identifiers (`seq_id`).
   - Zero dropped alignment hits or corrupted coordinate spans.
2. **Memory Ceiling & Footprint:**
   - Worker process memory ceiling strictly bounded ($<50$ MB per worker core).
   - Master process peak memory consumption strictly under $200$ GB for multi-GB TSV alignment datasets.
3. **Mathematical Equivalence & Determinism:**
   - Identical numerical range intersections, Jaccard similarities, and Density Peak Clustering (DPC) assignments across single-threaded and parallel execution modes.
4. **Explicit Error Handling:**
   - Mandatory explicit error raising (containing `🛑`) on missing bioinformatic executables or non-contiguously grouped TSV files (no silent fallbacks).
5. **Code Style & Formatting:**
   - Strict adherence to PEP 8 Visual Indent Style (delimiter-anchored visual alignment, max line length $\le 99$ characters) protected via `# fmt: off` / `# fmt: on` annotations.

---

## Technical Architecture & Multi-File Interdependencies

```
  [Multi-GB MMseqs2 / RiboSeek TSV File]
                    │
                    ▼
Phase 1: Streaming Ingestion & Pre-Scan Sequence Registry (aligners.py)
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
   ├── Construct per-sequence target overlap graph in worker processes
   ├── Perform immediate per-chunk CSR deduplication (csr.sum_duplicates())
   └── Accumulate chunk matrices in master process via lightweight in-place addition (csr_total += csr_chunk)
```

---

## Phased Implementation Roadmap for Future Jules Task

### Phase 1: High-Performance Streaming & Thread-Safe Sequence Identifier Registry (`aligners.py`)

#### 1.1 Thread-Safe Two-Pass `SequenceRegistry` Management
- **Target Bottleneck:** String allocations across hundreds of millions of alignment hits consume tens of gigabytes of RAM. Shared string interning (`sys.intern`) across parallel processes risks race conditions or lock contention.
- **Implementation Strategy:**
  - Standardize `SequenceRegistry` as a bi-directional integer ID mapping (`str_to_id: Dict[str, int]` and `id_to_str: List[str]`).
  - **Pre-Scan Ingestion:** Master process populates `SequenceRegistry` during single-pass stream chunk creation (`parse_mmseqs_tsv_stream`) before worker pool dispatch.
  - **Read-Only Worker Passing:** Pass immutable registry dictionaries/lists to `executor.Parallel` using `shared_objects={'registry': registry}`.
  - **Zero Worker Overhead:** Workers perform lock-free integer dictionary lookups without mutating global state.

#### 1.2 Optimized TSV Streaming Ingestion
- **Target Bottleneck:** Full dictionary loading of multi-GB TSV files causes severe RAM spikes.
- **Implementation Strategy:**
  - Maintain single-pass generator `parse_mmseqs_tsv_stream(tsv_file, registry)` yielding contiguous query blocks `(q_int, sorted_query_spans)`.
  - Assert contiguous query ID grouping in input TSV; raise explicit error (`🛑 Input TSV ... is not grouped contiguously`) if query records are interspersed.
  - Compute strand orientation as `-1` when target start > target end, normalizing target coordinates so `start <= end`.

---

### Phase 2: Parallel Batch Execution & Dynamic Workload Chunking (`domain_finder.py` & `executor.py`)

#### 2.1 Multi-Query Task Batching
- **Target Bottleneck:** Dispatching individual query sequences to worker processes causes substantial Inter-Process Communication (IPC) overhead, process context-switching latency, and progress bar lock contention.
- **Implementation Strategy:**
  - Group query blocks into multi-query batch chunks of size $B$:
    $$\text{chunk\_size} = \max\left(1, \min\left(200, \left\lfloor\frac{N_Q}{\text{threads} \times 4}\right\rfloor\right)\right)$$
  - Pass batch chunks through `executor.Parallel(parallelized_function, input_collection=chunk_batches, shared_objects={'registry': registry})`.
  - Reduces IPC serialization overhead by $>95\%$ and bypasses Python GIL bottlenecks.

#### 2.2 Parallel Local Domain Discovery & Filtering
- **Implementation Strategy:**
  - Execute `find_domain_instances()` and `filter_spurious_domains()` in parallel worker processes across query chunks.
  - Process local DPC domain instance selection independently within worker cores.
  - Return compact lists of `DomainInstance` objects to the master process to maintain worker RAM footprint below $50$ MB per core.

---

### Phase 3: Vectorized NumPy Span Intersection & DPC Acceleration (`clustering.py`)

#### 3.1 Vectorized `compare_sorted_spans` Coordinate Slicing
- **Target Bottleneck:** Iterative Python `for` loops and list comprehensions in `compare_sorted_spans()` create major bottlenecks when processing sequences with thousands of alignment spans.
- **Implementation Strategy:**
  1. Store span coordinates directly as contiguous 2D NumPy integer arrays (`int32` or `int64`, shape `(N, 2)`).
  2. Implement broad vectorized range search using `np.searchsorted` across start coordinates, maintaining $O(N \log N + K)$ complexity.
  3. Compute upper-triangular range intersections (`np.minimum(end_i, ends) - start_j`) inside vectorized C loops.
  4. Pre-allocate flat 1D NumPy output buffers (`rows`, `cols`, `intersections`) to eliminate Python `list.append` and intermediate `np.concatenate` allocations.

#### 3.2 Vectorized Pointer Doubling in `dpc()` Cascading Assignment
- **Target Bottleneck:** Sequential Python `while True` parent traversal (`parent = grandparent`) in `dpc()` creates scalar loop overhead on large similarity graphs.
- **Implementation Strategy:**
  - Replace scalar loop checks with vectorized NumPy pointer doubling: `parent = parent[parent]`.
  - Use early termination assertion: `if np.array_equal(grandparent, parent): break`.
  - Reaches cluster peaks in logarithmic iterations ($O(\log D)$, where $D$ is max tree depth).

#### 3.3 Vectorized Sparse CSR Top-K Gather
- **Implementation Strategy:**
  - In `_top_k_scipy_global`, use vectorized NumPy `bincount` and `cumsum` for indptr slices, eliminating per-row iteration overhead.
  - Avoid intermediate dense matrix conversions (`toarray()`) when matrix dimension $N > 2000$.

---

### Phase 4: In-Place SciPy CSR Target Overlap Matrix Accumulation (`domain_finder.py`)

#### 4.1 Parallel Target Overlap Construction & Immediate Deduplication
- **Target Bottleneck:** Global concatenation of hundreds of millions of duplicate overlap coordinate arrays (`cat_rows`, `cat_cols`, `cat_data`) creates massive RAM spikes during family clustering in `families()`.
- **Implementation Strategy:**
  1. Execute sequence-level target overlap matrix assembly (`calculate_target_overlaps`) in parallel across target sequence blocks using worker pools.
  2. Perform immediate per-chunk sparse CSR deduplication (`csr.sum_duplicates()`) inside worker processes.
  3. Accumulate chunk CSR matrices in the master process using lightweight in-place matrix addition (`csr_total += csr_chunk`), bypassing massive 1D array concatenations.
  4. Reduces peak RAM during target overlap aggregation by 60–80%.

---

### Phase 5: Verification, Benchmarking & Acceptance Criteria

#### 5.1 Automated Testing & Scientific Validation
- Run unit and integration tests using `python3 -m pytest`.
- Verify exact identity of output GFF files produced before and after optimization on benchmark FASTA datasets.
- Confirm exact retention of domain IDs, coordinates, strands, and partner alignment counts.

#### 5.2 Performance & Memory Benchmarking
- Validate overall speedup on multi-GB MMseqs2 TSV alignment files.
- Verify peak RAM consumption strictly stays below $200$ GB overall and $<50$ MB per worker process core.

#### 5.3 Pre-Commit Verification
- Run `pre_commit_instructions` and confirm code readability and formatting compliance.
