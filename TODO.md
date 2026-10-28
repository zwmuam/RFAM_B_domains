# High-Throughput Optimization & Safe Parallelization Plan (`aligners.py`, `domain_finder.py`, `clustering.py`, `executor.py`)

## Core System Invariants
1. **Data & Coordinate Integrity:**
   - Exact preservation of 1-based start/end coordinates (`start`, `end`), strand orientation (`strand` $\in \{-1, 1\}$), partner alignment lists (`partners`), and sequence identifiers (`seq_id`).
   - Zero dropped alignment hits or corrupted coordinate spans.
2. **Memory Ceiling & Footprint:**
   - Worker process memory ceiling strictly bounded ($<200$ MB per worker core).
   - Master process peak memory consumption strictly under $200$ GB for multi-GB TSV alignment datasets.
3. **Mathematical Equivalence & Determinism:**
   - Identical numerical range intersections, Jaccard similarities, and Density Peak Clustering (DPC) assignments across single-threaded and parallel execution modes (or explicit corrections)
4. **Explicit Error Handling:**
   - Mandatory explicit error raising on errors (no silent fallbacks).
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
  - Assert contiguous query ID grouping in input TSV; raise explicit error (`Input TSV ... is not grouped contiguously`) if query records are interspersed.
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

---

## Comprehensive Risk Assessment & Guidelines for Future Tasks

### 1. Likelihood-Ranked Risk Analysis for Multi-GB MMseqs2 TSV Datasets

1. **Non-Contiguous TSV Query Stream Interleaving (Risk ID: R-01)**
   - **Phase:** Phase 1 (`aligners.py`)
   - **Likelihood:** HIGH | **Impact:** CRITICAL
   - **Failure Mechanism:** TSV files generated across parallel search runs may interleave records
     for the same query sequence across non-adjacent lines. Stream parsers yielding query blocks
     on query ID transitions will split single queries into multiple fragmented chunks, corrupting
     Density Peak Clustering (DPC) local neighborhood counts and dropping partner alignment hits.
   - **Preventive Mitigation:** Enforce single-pass pre-scan tracking seen query IDs in master
     process. Raise explicit error (`raise ValueError("Input TSV is not grouped...")`) if
     a query sequence ID reappears after yielding.

2. **CSR Matrix Dimension & Sequence ID Mismatch (Risk ID: R-02)**
   - **Phase:** Phase 4 (`domain_finder.py`)
   - **Likelihood:** HIGH | **Impact:** CRITICAL
   - **Failure Mechanism:** In-place sparse matrix aggregation (`csr_total += csr_chunk`) fails
     with dimension mismatch errors or silently misaligns target sequence overlap graph nodes if
     worker processes dynamically assign local row/column integer IDs instead of uniform global
     sequence IDs.
   - **Preventive Mitigation:** Compute total global target sequence registry dimension N in master
     process prior to worker dispatch. Construct all worker chunk matrices with fixed shape
     (N, N) using global sequence ID integers.

3. **Inter-Process Communication Payload Memory Overhead (Risk ID: R-03)**
   - **Phase:** Phase 2 (`domain_finder.py` & `executor.py`)
   - **Likelihood:** HIGH | **Impact:** HIGH
   - **Failure Mechanism:** Passing heavy nested Python object graphs (e.g. `Span` instances with
     full `partners` lists) across process boundaries via IPC pickling causes severe master
     process RAM spikes (>200 GB) and worker core memory breaches (>50 MB).
   - **Preventive Mitigation:** Return compact lightweight structures (e.g. integer tuples or
     flat `DomainInstance` dataclasses) from worker processes to master process.

4. **Genomic Coordinate Integer Overflow (Risk ID: R-04)**
   - **Phase:** Phase 3 (`clustering.py`)
   - **Likelihood:** MEDIUM | **Impact:** CRITICAL
   - **Failure Mechanism:** Standard signed 32-bit integer arrays (`int32`) overflow on
     chromosome span calculations exceeding 2.14 billion bp, wrapping into negative coordinate
     values and causing non-deterministic span comparison failures or memory segmentation faults.
   - **Preventive Mitigation:** Cast all genomic start/end coordinate arrays explicitly to
     `int64` dtypes during 2D NumPy array construction.

5. **Coordinate Inversion & 1-Based Boundary Offset Shift (Risk ID: R-05)**
   - **Phase:** Phase 1 (`aligners.py`)
   - **Likelihood:** MEDIUM | **Impact:** HIGH
   - **Failure Mechanism:** Inverted target coordinates (`target_start > target_end`) in TSV
     records misinterpreted during parsing, producing negative span lengths (`end - start < 0`) or
     off-by-one shifts between 1-based GFF coordinates and 0-based Python slice indices.
   - **Preventive Mitigation:** Normalize coordinates immediately upon parsing (`start <= end`),
     explicitly calculate `length = end - start + 1`, and set `strand = -1` when inverted.

6. **Repetitive Element Density Matrix Explosion (Risk ID: R-06)**
   - **Phase:** Phase 4 (`domain_finder.py`)
   - **Likelihood:** MEDIUM | **Impact:** HIGH
   - **Failure Mechanism:** Highly repetitive non-coding RNA elements generate dense alignment
     graphs with millions of pairwise target overlaps, turning sparse CSR matrices into dense
     representations and exhausting available memory during `csr.sum_duplicates()`.
   - **Preventive Mitigation:** Apply E-value and ungapped score candidate pre-filtering prior
     to sparse graph assembly; store overlap weights using single-precision `float32`.

7. **Infinite Parent Traversal in DPC Pointer Doubling (Risk ID: R-07)**
   - **Phase:** Phase 3 (`clustering.py`)
   - **Likelihood:** LOW | **Impact:** HIGH
   - **Failure Mechanism:** Cyclic or non-monotonic parent pointer assignments in DPC tree
     cascading cause `parent = parent[parent]` pointer doubling loops to enter infinite recursion.
   - **Preventive Mitigation:** Implement convergence assertions (`if np.array_equal(parent,
     grandparent): break`) with a hard iteration cap (<50 iterations) and explicit error raising.

8. **Worker Pool Process Deadlock & Progress Bar Lock Contention (Risk ID: R-08)**
   - **Phase:** Phase 2 (`executor.py`)
   - **Likelihood:** LOW | **Impact:** MEDIUM
   - **Failure Mechanism:** Unhandled stdout/stderr logging or progress bar updates inside worker
     sub-processes acquire GIL or file descriptor locks, stalling worker pools indefinitely.
   - **Preventive Mitigation:** Isolate all file I/O and user-facing logging to master process;
     pass immutable read-only state via process arguments.

---

### 2. Detailed Optimization Phase Risk Evaluation & Failure Modes

#### Phase 1: High-Performance Streaming & Sequence Identifier Registry (`aligners.py`)
- **Primary Risk:** Multi-GB MMseqs2 BLAST-TAB TSV files generated across multi-threaded runs can
  contain non-contiguously grouped query records. Parsing non-contiguous streams yields partial
  query blocks, corrupting Density Peak Clustering (DPC) local neighborhood counts and splitting
  alignment partner lists.
- **Data Integrity Risk:** Converting 1-based TSV genomic coordinates into internal
  representations risks 0-based boundary drift. Target strand inversion
  (`target_start > target_end`) must set `strand = -1` while normalizing start <= end.
- **Memory Risk:** Mutating `SequenceRegistry` inside worker processes breaks lock-free isolation,
  leading to non-deterministic string-to-integer mappings across workers.

#### Phase 2: Parallel Batch Execution & Dynamic Chunking (`domain_finder.py` & `executor.py`)
- **Primary Risk:** Sub-optimal chunk size B = max(1, min(200, floor(N_Q / (4 * threads)))) can
  cause extreme performance degradation. Overly small B incurs excessive Inter-Process
  Communication (IPC) overhead, whereas excessively large B exceeds the <50 MB RAM ceiling per
  worker core.
- **Process Memory Risk:** Transferring large nested Python dictionaries or full `Span` object
  graphs through IPC pickling causes severe master process memory spikes, breaching the 200 GB
  memory ceiling.

#### Phase 3: Vectorized NumPy Span Intersection & DPC Acceleration (`clustering.py`)
- **Primary Risk:** Multi-GB coordinate array slicing via `np.searchsorted` requires strict
  start-coordinate sorting. Unsorted or partially sorted inputs yield corrupted upper-triangular
  range intersections, distorting DPC local density (rho) and delta values.
- **Mathematical Equivalence Risk:** Vectorized pointer doubling (`parent = parent[parent]`) must
  produce identical cluster assignments to sequential scalar loops. Incomplete convergence
  criteria can orphan valid cluster members or create disconnected graph components.

#### Phase 4: In-Place SciPy CSR Target Overlap Matrix Accumulation (`domain_finder.py`)
- **Primary Risk:** Sparse matrix aggregation (`csr_total += csr_chunk`) requires identical
  global shape dimensions (N, N). If worker processes build sparse matrices using localized
  sequence indices instead of global registry integer IDs, in-place addition corrupts
  matrix layout.
- **Efficiency Risk:** Un-deduplicated coordinate lists concatenated globally cause peak RAM
  spikes during `families()` clustering. Intermediate CSR `sum_duplicates()` within worker
  processes is mandatory.

---

### 3. Cross-Phase Interaction & Conflict Matrix

- **Registry Freeze vs Dynamic Batching (Phases 1 & 2):**
  - *Conflict:* Phase 2 worker dispatch before Phase 1 registry completion causes unregistered
    integer lookup failures in workers.
  - *Safeguard:* Complete full single-pass pre-scan in master process to build immutable
    `SequenceRegistry` before initializing worker pool.

- **Worker RAM Ceiling vs Vector Buffers (Phases 2 & 3):**
  - *Conflict:* Phase 3 flat buffer pre-allocation in worker processes can violate Phase 2
    <50 MB RAM ceiling if batch size B is too large.
  - *Safeguard:* Bound batch chunk size B <= 200; release intermediate NumPy arrays with `del`
    and invoke `gc.collect()` within workers.

- **Local Index Space vs CSR Aggregation (Phases 3 & 4):**
  - *Conflict:* Vectorized span indexing in Phase 3 must map into global integer ID space for
    Phase 4 SciPy CSR target overlap matrix creation.
  - *Safeguard:* Maintain global sequence ID integer mapping end-to-end; do not re-index
    sequences within local worker chunks.

---

### 4. Scientific Rules & Preventive Guidelines for Future Tasks

1. **Pre-Scan Sequence Registry Invariant:**
   - Always populate `SequenceRegistry` in the master process prior to parallel task dispatch.
     Never mutate the registry inside worker processes.

2. **Strict Coordinate & Strand Handling Rule:**
   - Preserve 1-based start/end coordinate conventions throughout alignment parsing. Always check
     for target inversion (`t_start > t_end`), normalize t_start <= t_end, and explicitly assign
     `strand = -1`.

3. **Explicit Error Boundary Principle:**
   - Never implement silent error suppression or fallback mechanisms during alignment parsing or
     matrix assembly. Raise explicit, descriptive errors containing error markers upon
     encountering invalid or non-contiguous input data.

4. **Worker Core Memory Ceiling Enforcement (<50 MB):**
   - Keep worker process payloads lightweight. Pass integer sequence IDs rather than raw sequence
     strings or heavy object graphs.

5. **NumPy Vectorization Safety Guidelines:**
   - Use `int64` for all genomic position arrays to prevent integer overflow. Verify array
     sorting before running `np.searchsorted`.

6. **SciPy CSR Accumulation Rule:**
   - Ensure all partial CSR matrices share the exact global dimensions (N, N) defined by the
     immutable sequence registry. Always execute `csr.sum_duplicates()` prior to returning chunk
     matrices to the master process.

7. **Zero-Divergence Verification Protocol:**
   - Validate mathematical equivalence across single-threaded and parallel execution pathways.
     Output domain GFF files must match exactly in domain IDs, coordinates, strands, and partner
     counts.
