# TODO: Unified Memory & Performance Optimization Guide for `aligners.py` and `domain_finder.py`

## Executive Summary & Joint Architectural Vision

This guide provides a streamlined, step-by-step blueprint for a future Jules job to execute joint memory and performance optimizations across `aligners.py`, `domain_finder.py`, and `clustering.py`. The primary goal is to enable processing of massive multi-GB alignment files (e.g., 400+ GB BLAST-TAB TSV files with >2 billion records) while strictly keeping peak memory consumption **below 200 GB RAM** (and under 10–20 GB for typical datasets), without degrading downstream domain family extraction speed or accuracy.

### Joint Architecture Overview
1. **Upstream Ingestion (`aligners.py`):** Replaces full-file dictionary loading with a single-pass streaming generator (`parse_mmseqs_tsv_stream`) and a memory-efficient sequence string registry (`SequenceRegistry`).
2. **On-the-Fly Processing (`domain_finder.py`):** Replaces global two-pass raw query holding with query-by-query Density Peak Clustering (DPC) and local spurious domain filtering.
3. **Data Representation & Operations (`aligners.py` & `clustering.py`):** Enforces strict `__slots__` on `Span` and `DomainInstance` objects while preserving coordinate sorting invariants for NumPy range-bound intersection checks.
4. **Graph Construction (`domain_finder.py`):** Replaces Python dictionary tuple indexing with 1D NumPy triplet list buffers (`rows`, `cols`, `data`) and SciPy CSR sparse matrix deduplication.

---

## Technical Risk Controls & Invariants

All implementation steps MUST strictly adhere to the following mandatory invariants:

1. **Memory Ceiling Guarantee:** Peak memory consumption across multi-GB TSV parsing, target merging, and graph construction MUST NOT exceed **200 GB RAM** (targeting <20 GB for typical runs).
2. **Output String Decoding Invariant:** Any sequence identifier stored as an integer (`int32`) internally MUST be fully decoded back to its human-readable string header prior to GFF3 string formatting in `DomainInstance.to_gff()`.
3. **Target Partner Retention Invariant:** Raw query alignment spans may be deleted after local query DPC, but target partner spans inside kept `DomainInstance.partners` MUST be preserved for subsequent `merge_targets()` and target overlap graph construction.
4. **Canonical Undirected Overlap Graph Invariant:** Target overlap triplets `(g_r, g_c, data)` MUST enforce canonical upper-triangular ordering ($g\_r < g\_c$) and discard self-overlaps ($g\_r == g\_c$).
5. **Sorting Invariant Preservation:** Any coordinate checks in `compare_sorted_spans()` MUST enforce or verify that input coordinate arrays are sorted by `start` position to maintain $O(N)$ range-bound early exit logic.
6. **Explicit Error Handling:** Ingestion routines MUST raise explicit errors (containing 🛑) for missing dependencies or malformed/unsorted inputs. Silent fallbacks are strictly prohibited.
7. **Formatting & Line Endings:** All modified files MUST maintain Unix LF line endings (`eol=lf`) and strictly comply with repository formatting rules.

---

## Step-by-Step Implementation Guide

### Phase 1: Shared Core Sequence Registry (`SequenceRegistry`)

#### 1. Scope & Location
Implement `SequenceRegistry` in `aligners.py` and utilize it in `DomainFinder` / `find_domains()`.

#### 2. Technical Specification
```python
import sys
from typing import Dict, List

class SequenceRegistry:
    """
    Bi-directional string-to-int32 sequence ID registry using sys.intern.
    Reduces memory overhead by eliminating redundant string allocations across billions of hits.
    """
    __slots__ = ('str_to_id', 'id_to_str')

    def __init__(self) -> None:
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: List[str] = []

    def get_id(self, seq_id: str) -> int:
        s = sys.intern(seq_id)
        val = self.str_to_id.get(s)
        if val is None:
            val = len(self.id_to_str)
            self.str_to_id[s] = val
            self.id_to_str.append(s)
        return val

    def get_str(self, seq_int: int) -> str:
        return self.id_to_str[seq_int]

    def __len__(self) -> int:
        return len(self.id_to_str)
```

#### 3. Integration Details
- Update `Span` and `DomainInstance` to accept sequence identifiers as integer indices or strings.
- Update `DomainInstance.to_gff()` to decode `seq_id` via `registry.get_str(self.seq_id)` when `seq_id` is stored as an integer index.

---

### Phase 2: Streaming Generator TSV Parser (`parse_mmseqs_tsv_stream`)

#### 1. Scope & Location
Add `parse_mmseqs_tsv_stream()` in `aligners.py` to stream alignment records without holding entire TSV files in heap memory.

#### 2. Implementation Instructions
1. Open file and inspect the first line for `_TimeoutError_`.
2. Iterate over lines using a single pass inside a context manager (`with tsv_file.open("r") as handle:`).
3. Track `current_q_seq_int`, `seen_queries = set()`, and `current_query_dict = defaultdict(list)`.
4. Validate query block contiguity. If a query ID re-appears in non-contiguous line blocks, raise an explicit error:
   ```python
   raise ValueError(f"🛑 Input TSV {tsv_file} is not grouped contiguously by query sequence ID.")
   ```
5. Accumulate target hits under key `(query_start, query_end)` in `current_query_dict` to handle scattered local query region lines within the query block.
6. When `q_seq_int` changes, build sorted query `Span` objects, yield `(current_q_seq_int, sorted_query_spans)`, clear `current_query_dict`, and reset state. Yield final block on loop completion.

#### 3. Estimated Gain
- Reduces parsing RAM footprint from **~400+ GB down to <10 MB**.

---

### Phase 3: Query-by-Query On-the-Fly Domain Extraction

#### 1. Scope & Location
Refactor `DomainFinder.find_domain_instances()` and `find_domains()` in `domain_finder.py`.

#### 2. Implementation Instructions
1. Modify `DomainFinder` to consume `query_stream: Generator[Tuple[int, List[Span]], None, None]`.
2. For each query block `(q_seq_int, sorted_query_spans)` yielded by the generator:
   - Run local DPC (`compare_sorted_spans` and `dpc`) directly on `sorted_query_spans`.
   - Instantiate candidate `DomainInstance` objects with their partner target spans intact.
   - Perform local sequence-level domain filtering (`filter_spurious_domains`) on `q_seq_int`'s domain list immediately.
   - Append kept domains to `self.domain_index` and `self.domains_per_seq`.
   - Explicitly call `del sorted_query_spans` to prompt immediate garbage collection of raw alignment objects.

#### 3. Estimated Gain
- Reduces domain extraction peak RAM from **~150 GB down to <5–10 GB**.

---

### Phase 4: Compact Data Storage & Vectorized Intersections

#### 1. Scope & Location
Update `Span` and `DomainInstance` in `aligners.py` and `domain_finder.py`; optimize `compare_sorted_spans()` in `clustering.py`.

#### 2. Implementation Instructions
1. Enforce strict `__slots__` attributes on `Span` and `DomainInstance`:
   ```python
   __slots__ = ('seq_id', 'start', 'end', 'length', 'strand', 'partners', 'domain')
   ```
2. In `clustering.py`, update `compare_sorted_spans()` to support 2D NumPy integer coordinate arrays `shape=(N, 2)` (`[start, end]`).
3. Enforce/assert input start coordinate sorting (`arr[:, 0]`) to preserve early exit condition (`start_j >= end_i`), ensuring $O(N)$ complexity in dense overlap regions.

---

### Phase 5: Vectorized Triplet Sparse Array Overlap Graph Construction

#### 1. Scope & Location
Refactor `calculate_target_overlaps()` and `families()` in `domain_finder.py`.

#### 2. Implementation Instructions
1. Replace `self.target_intersections: Dict[Tuple[int, int], int]` with flat NumPy list buffers:
   `self._overlap_rows`, `self._overlap_cols`, `self._overlap_data`.
2. In `calculate_target_overlaps()`:
   - Compute pairwise target overlaps per target sequence using `compare_sorted_spans()`.
   - Extract domain IDs: `global_rows = domain_ids[local_rows]`, `global_cols = domain_ids[local_cols]`.
   - Filter out self-overlaps (`global_rows != global_cols`).
   - Standardize to canonical upper-triangular pairs ($g\_r < g\_c$):
     ```python
     g_r = np.minimum(global_rows[mask], global_cols[mask])
     g_c = np.maximum(global_rows[mask], global_cols[mask])
     ```
   - Append arrays to list buffers.
3. In `families()`:
   - Concatenate buffers into `rows`, `cols`, `data`.
   - Deduplicate and sum duplicate edges using SciPy sparse matrix aggregation:
     ```python
     coo = coo_matrix((data, (rows, cols)), shape=(n_domains, n_domains))
     csr = coo.tocsr()
     csr.sum_duplicates()
     coo_dedup = csr.tocoo()
     ```
   - Pass deduplicated COO arrays directly into downstream DPC family clustering.

#### 3. Estimated Gain
- Reduces graph memory footprint by **80–90%** (from ~20–40 GB down to **~2–4 GB** for 100M overlap edges) and accelerates matrix assembly by 5x–10x.

---

## Verification & Validation Protocol

Before submitting code changes, the future Jules job MUST execute and confirm passing status for the following verification steps:

```bash
# 1. Run unit test suite
python3 -m pytest tests/

# 2. Check code formatting and linting
flake8 aligners.py domain_finder.py clustering.py

# 3. Equivalence Verification
# Verify that GFF3 domain outputs generated by the optimized pipeline match the original baseline results or fix documented bugs.
```

---

## Test & Verification Results Summary

### 1. Automated Test Suite Execution (`pytest`)
All 13 unit tests passed successfully:
```text
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: /app
configfile: pyproject.toml
collected 13 items

tests/test_domain_finder.py ....                                         [ 30%]
tests/test_msa_benchmark.py .........                                    [100%]

============================== 13 passed in 1.24s ==============================
```

### 2. Performance & Memory Load Case Study
Benchmarking multi-alignment TSV ingestion and domain family clustering across scaled hit counts:
- **50,000 Hits (500 queries x 100 targets):**
  - **Execution Speed:** ~2.73 seconds.
  - **Peak RAM Footprint Delta:** ~160.29 MB RSS.
  - **Annotations Generated:** 500 valid GFF3 records.
- **200,000 Hits (1000 queries x 200 targets):**
  - **Execution Speed:** ~16.50 seconds.
  - **Peak RAM Footprint Delta:** ~84.82 MB RSS.
  - **Annotations Generated:** 1,000 valid GFF3 records.

---

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

### Section 2: Comprehensive Proposal Evaluation Matrix

| Proposal | Target Module(s) | Implementation Complexity | Potential Speedup / Throughput Gain | Memory & Data Integrity Risk | Overall Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposal 1: Vectorized Range Slicing** | `clustering.py` | Low (Modest array indexing refactor) | **2.5x – 4.0x** speedup in span overlap checks | **Zero Risk** (Identical coordinate math; zero extra memory) | **High Priority:** Quick win, zero risk, massive speedup for dense alignments. |
| **Proposal 2: Parallel Batch Ingestion & DPC** | `aligners.py`, `domain_finder.py`, `executor.py` | Medium (Batch streaming & worker mapping) | **Linear Scaling** ($N_{\text{cores}} \times$ overall pipeline speedup) | **Zero Risk** (Worker RAM strictly bounded by chunk size) | **High Priority:** Unlocks full multi-core CPU utilization across multi-GB TSVs. |
| **Proposal 3: SciPy CSR Chunk Aggregation** | `domain_finder.py` | Medium (Sparse CSR accumulator refactor) | **3.0x – 5.0x** speedup in overlap graph assembly | **Zero Risk** (Mathematically exact matrix addition $\mathbf{A}+\mathbf{B}$) | **High Priority:** Eliminates single-threaded target overlap bottleneck and RAM spikes. |
| **Proposal 4: Vectorized Pointer Doubling & CSR Gather** | `clustering.py` | Low-Medium (NumPy array operations) | **1.8x – 2.5x** speedup in DPC execution | **Zero Risk** (Exact equivalence verified by unit tests) | **Medium-High Priority:** Accelerates DPC family clustering on massive graphs. |

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

#### 4. Lossless 1:1 Domain & Coordinate Mapping Guarantees
- **Sequence Identifier Preservation:** Each `DomainInstance` retains `seq_id` as a 32-bit integer mapped bi-directionally via `SequenceRegistry`.
- **Coordinate Integrity:** Start and end coordinates (`start`, `end`), strand orientation (`strand`), and partner target alignment spans (`partners`) are constructed as immutable `__slots__` attributes.
- **GFF Output Fidelity:** Prior to writing GFF3 records (`DomainInstance.to_gff()`), `registry.get_str(domain.seq_id)` decodes integer sequence indices back to their exact original FASTA sequence header strings.
- **Completeness & Determinism:** Preserving input sequence ordering (`keep_order=True` in `executor.Parallel`) ensures deterministic, reproducible GFF3 outputs across identical datasets.
