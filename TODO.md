# TODO: Memory Optimization Roadmap for `aligners.py` and `domain_finder.py`

This document details realistic options for optimizing memory usage when loading and processing large MMseqs2 / RiboSeek alignment outputs (e.g., ~400 GB TSV files) without sacrificing downstream accuracy or computational efficiency.

Solutions are ordered by **feasibility**, balancing performance gains against implementation complexity.

---

## Tier 1: High Feasibility & High Impact (Low-to-Medium Complexity)

### 1. Streaming Generator TSV Parsing (`parse_mmseqs_tsv_stream`)
- **Problem:** `parse_mmseqs_tsv` loads the entire 400 GB TSV file into a deeply nested Python dictionary (`dict[str, dict[tuple, list[tuple]]]`) and converts every alignment into Python `Span` instances before downstream processing.
- **MMseqs2 Alignment Guarantee:** As documented in the [MMseqs2 Wiki](https://github.com/soedinglab/mmseqs2/wiki), `easy-search` outputs alignment hits grouped contiguously by query sequence ID (`query_id`).
- **Proposed Solution:** Implement a streaming generator `parse_mmseqs_tsv_stream(tsv_file)` using `itertools.groupby` or single-pass line streaming. Yield hits query-by-query `(q_seq_id, list_of_spans)`.
- **Memory Savings:** Reduces RAM usage for TSV parsing from **~400+ GB down to <1 MB** (storing only one query's alignments in memory at any given time).
- **Implementation Complexity:** Very Low (minor refactoring to a Python generator function).

### 2. String Interning (`sys.intern`) & Integer Sequence Indexing
- **Problem:** Sequence identifiers (`q_seq_id`, `t_seq_id`) are repeatedly instantiated as redundant string objects across millions of alignment lines, consuming significant Python heap overhead.
- **Proposed Solution:**
  1. Apply `sys.intern(seq_id)` during line parsing.
  2. Map sequence string IDs to contiguous 32-bit integers (`int32`) via a global string dictionary during parsing and map back to string IDs during GFF export.
- **Memory Savings:** **60–80% memory reduction** for string-heavy objects.
- **Implementation Complexity:** Very Low (1–3 lines of code).

### 3. Query-by-Query On-the-Fly Domain Extraction
- **Problem:** `DomainFinder.find_domain_instances` currently waits for all query alignment spans to be in memory before starting domain instance identification.
- **Proposed Solution:** Combine streaming TSV parsing with domain identification. Process `find_domain_instances()` and `filter_spurious_domains()` inside the query generator loop. Once domain instances for sequence $Q_i$ are extracted, discard the raw alignment spans so Python's garbage collector frees them immediately.
- **Memory Savings:** Drastically reduces active heap size from hundreds of millions of raw alignment spans to only filtered domain instances (**~100x reduction in peak RAM**).
- **Implementation Complexity:** Low-Medium.

### 4. Compact Structural Storage for `Span` Objects (`__slots__` Optimization & NumPy Arrays)
- **Problem:** Instantiating millions of `Span` objects creates significant Python class overhead (attribute dictionaries, object reference overhead, list objects for `partners`).
- **Proposed Solution:**
  1. Replace object reference chains for initial alignment parsing with 2D NumPy integer arrays `np.ndarray` (columns: `[start, end, strand, target_seq_int, target_start, target_end, target_strand]`).
  2. Instantiate full `Span` / `DomainInstance` objects only for candidate domain peaks after Density Peak Clustering (DPC).
- **Memory Savings:** **70–85% reduction** in alignment representation footprint.
- **Implementation Complexity:** Medium.

---

## Tier 2: Medium Feasibility & High Performance Gains (Medium Complexity)

### 5. Out-of-Core Columnar Ingestion with DuckDB / Polars / PyArrow
- **Problem:** Pure Python line-by-line reading (`for line in handle:`) of a 400 GB text file is bound by single-threaded string parsing and Python interpreter overhead.
- **Proposed Solution:**
  1. Utilize **DuckDB** or **Polars** to read the TSV file out-of-core using multi-threaded C++/Rust SIMD engines.
  2. Example DuckDB stream query:
     ```sql
     SELECT column_1 as target_id, column_2 as query_id,
            column_7 as target_start, column_8 as target_end,
            column_9 as query_start, column_10 as query_end
     FROM read_csv_auto('hits.tsv', delim='\t', header=False)
     WHERE column_1 != column_2
     ```
  3. Stream query results in chunks using `fetch_record_batch()` or `iter_pyarrow_batches()`.
- **Memory Savings:** Bounded memory footprint (configurable memory limit, e.g., 8 GB RAM) regardless of TSV size.
- **Performance Impact:** **5–10x faster ingestion speed**.
- **Implementation Complexity:** Medium.

### 6. Disk-Backed Intermediate Storage for Target Overlaps
- **Problem:** `DomainFinder.target_intersections` stores pairwise domain overlaps in an in-memory dictionary `Dict[Tuple[int, int], int]` and `self.targets_per_seq` accumulates target spans for every domain.
- **Proposed Solution:** Store intermediate target spans and overlap counts in a temporary SQLite or DuckDB table on disk (`tmp_dir / "intersections.duckdb"`). Query overlap sums during family clustering using aggregation queries.
- **Memory Savings:** Bounded peak RAM footprint (< 2 GB RAM) for large-scale genome alignment graphs.
- **Implementation Complexity:** Medium.

### 7. Vectorized Triplet Arrays for Overlap Graph Construction
- **Problem:** Constructing `self.target_intersections[canonical_key] += intersect` creates millions of 2-tuples and dictionary lookup operations.
- **Proposed Solution:** Accumulate `global_rows`, `global_cols`, and `intersections` into contiguous NumPy 1D arrays (`int32`) and compress duplicate edges using `coo_matrix((data, (rows, cols)))` or `np.bincount` / COO aggregation.
- **Memory Savings:** Eliminates dict/tuple hashing overhead (**saving ~100–200 bytes per graph edge**).
- **Implementation Complexity:** Medium.

---

## Tier 3: Advanced Architectural Optimizations (Higher Complexity)

### 8. Numba-Accelerated Range Comparisons (`compare_sorted_spans`)
- **Problem:** `compare_sorted_spans` in `clustering.py` uses nested Python loops (`for i in range(n): for j in range(i+1, n):`) and appends to Python lists (`low_indices`, `high_indices`, `intersection_lengths`).
- **Proposed Solution:** Compile `compare_sorted_spans` with `@numba.njit` operating directly on contiguous 1D NumPy arrays of start and end coordinates `starts: np.ndarray`, `ends: np.ndarray`.
- **Memory Savings & Speedup:** Eliminates intermediate Python list allocations, producing a **10–20x speedup** and **~80% reduction** in temporary memory overhead during range overlap computations.
- **Implementation Complexity:** Medium-High.

### 9. PyArrow Memory-Mapped RecordBatches
- **Problem:** In-memory object copies across processing stages (`Span` -> `DomainInstance` -> DPC matrix inputs) duplicate coordinate data in RAM.
- **Proposed Solution:** Store coordinate arrays in PyArrow RecordBatches backed by memory-mapped files (`pyarrow.memory_map`).
- **Memory Savings:** Zero-copy slicing across pipeline steps and multi-processing workers.
- **Implementation Complexity:** High.
