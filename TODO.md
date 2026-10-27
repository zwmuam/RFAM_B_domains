# TODO: Memory Optimization Roadmap for `aligners.py` and `domain_finder.py`

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
