# TODO: Memory Optimization Roadmap for `aligners.py` and `domain_finder.py`

## Tier 1: High Feasibility & High Impact (Low-to-Medium Complexity)

### 1. Streaming Generator TSV Parsing (`parse_mmseqs_tsv_stream`)
- **Problem:** `parse_mmseqs_tsv` in `aligners.py` currently loads an entire TSV alignment file into memory as an unsorted Python dictionary (`dict[str, dict[tuple, list[tuple]]]`), builds full `Span` objects for all target alignments, and returns a massive in-memory dictionary `sorted_queries`. For a 400 GB TSV file containing ~2–4 billion alignment lines, holding all parsed Python dictionaries and `Span` instances in RAM requires ~300–600+ GB RAM, causing severe memory exhaustion (OOM).
- **MMseqs2 Alignment Guarantee & Technical Re-evaluation:**
  - Standard MMseqs2 `easy-search` and RiboSeek output alignment records strictly grouped contiguously by query sequence ID (`query_id`, `fields[1]` in BLAST-TAB tab-separated format).
  - *Critical Edge Case:* Although lines are grouped by `query_id`, target hits for a single local query region `(query_start, query_end)` may be scattered across non-adjacent lines within that query block. Additionally, input TSVs might occasionally be pre-sorted or unsorted if produced by non-standard aligner runs.
  - *Streaming Strategy:* Instead of loading all queries into memory, implement a single-pass generator `parse_mmseqs_tsv_stream(tsv_file: Path)`. As lines stream in, accumulate lines belonging to the current `q_seq_id`. Once `q_seq_id` changes, aggregate local query spans, sort coordinates by `start`, construct query `Span` objects with partner targets, yield `(q_seq_id, query_spans)`, and immediately clear local memory.
- **Detailed Implementation Instructions:**
  1. In `aligners.py`, create `parse_mmseqs_tsv_stream(tsv_file: Path) -> Generator[Tuple[str, List[Span]], None, None]`.
  2. Implement single-pass line streaming using a `for line in handle:` loop, tracking `current_q_seq_id` and an in-memory accumulator `current_query_dict = defaultdict(list)`.
  3. When `fields[1]` differs from `current_q_seq_id`:
     - Process `current_query_dict` to sort query coordinates by start position.
     - Build `Span` objects for the query and its partner target spans.
     - `yield (current_q_seq_id, query_spans)`.
     - Reset `current_query_dict` and update `current_q_seq_id`.
  4. Yield the final buffered query block after loop termination.
  5. In `domain_finder.py`, update `find_domains()` and `DomainFinder` to consume `parse_mmseqs_tsv_stream` directly instead of expecting a full dictionary.
- **Estimated Gain:**
  - **Memory Impact:** Reduces TSV parsing RAM overhead from **~400+ GB down to < 10 MB** (peak heap size limited to the alignments of a single query sequence).
  - **Speed Impact:** Eliminates global dictionary reallocation and multi-pass dictionary sorting, improving parsing throughput by ~15–25%.
- **Necessary Workload:**
  - `aligners.py`: ~35–45 lines changed/added (`parse_mmseqs_tsv_stream`).
  - `domain_finder.py`: ~15–25 lines changed to support generator-based ingestion.

---

### 2. String Interning (`sys.intern`) & Integer Sequence Indexing
- **Problem:** Alignment files contain hundreds of millions of repeating sequence identifiers (`q_seq_id`, `t_seq_id`). In standard Python, each string instance occupies ~50–80 bytes plus 8 bytes per pointer reference. Across 1 billion alignment records, storing duplicate sequence strings consumes ~60–80 GB of redundant heap memory.
- **Technical Re-evaluation:**
  - `sys.intern()` ensures that identical sequence string values share a single singleton object in Python's global internal string table.
  - An even more compact and cache-friendly approach is **Integer Sequence Indexing**: maintaining a two-way string lookup dictionary (`SequenceRegistry` or `id_map: dict[str, int]` and `rev_map: list[str]`). Sequence IDs are converted to 32-bit contiguous integers (`np.int32`) during parsing and stored in integer form throughout domain extraction, target merging, and overlap graph construction. Re-conversion to string identifiers happens strictly during GFF export.
- **Detailed Implementation Instructions:**
  1. Define a global or scoped `SequenceRegistry` helper class in `aligners.py` or `domain_finder.py`:
     ```python
     class SequenceRegistry:
         def __init__(self):
             self.str_to_id: Dict[str, int] = {}
             self.id_to_str: List[str] = []

         def get_id(self, seq_id: str) -> int:
             # Fast string lookup with interning fallback
             s = sys.intern(seq_id)
             val = self.str_to_id.get(s)
             if val is None:
                 val = len(self.id_to_str)
                 self.str_to_id[s] = val
                 self.id_to_str.append(s)
             return val

         def get_str(self, seq_int: int) -> str:
             return self.id_to_str[seq_int]
     ```
  2. Modify `Span` / numpy alignment arrays to store `seq_id` as an `int` internally.
  3. Update `DomainInstance.to_gff()` to decode `seq_int` back to `seq_str` via `registry.get_str(self.seq_id)`.
- **Estimated Gain:**
  - **Memory Impact:** **70–85% reduction** in sequence ID storage footprint (~50–70 GB RAM saved on multi-GB files).
  - **Speed Impact:** Integer comparisons and hashing (`int32`) are significantly faster than string comparisons in dictionary keys (`targets_per_seq`, `domains_per_seq`).
- **Necessary Workload:**
  - `aligners.py`: ~15–20 lines changed/added.
  - `domain_finder.py`: ~20–30 lines changed.

---

### 3. Query-by-Query On-the-Fly Domain Extraction
- **Problem:** In `domain_finder.py`, `DomainFinder.find_domain_instances()` currently iterates over `self.sorted_queries.values()`, storing all query `Span` objects in memory before running Density Peak Clustering (DPC). It then accumulates all extracted `DomainInstance` objects in `self.domains_per_seq`. For massive multi-GB alignments, keeping all raw query alignment spans and extracted domains simultaneously in RAM consumes massive memory (~100–150 GB).
- **Technical Re-evaluation:**
  - Query domain instance extraction (`find_domain_instances` + initial local filtering) operates strictly on alignments belonging to a *single* query sequence $Q_i$. The alignments of sequence $Q_i$ are completely independent from sequence $Q_j$ during local DPC peak detection.
  - *On-the-Fly Pipeline:* Combine `parse_mmseqs_tsv_stream()` with query-level processing. For each query $(Q_i, \text{spans}_{Q_i})$ yielded by the generator:
    1. Run local DPC (`compare_sorted_spans` + `dpc`) on $\text{spans}_{Q_i}$.
    2. Extract candidate `DomainInstance` objects for $Q_i$.
    3. Filter spurious domains locally for $Q_i$ via `max_containment`.
    4. Store only the finalized, kept `DomainInstance` objects in `self.domain_index` / `self.domains_per_seq`.
    5. Discard raw alignment spans for $Q_i$ so Python's garbage collector frees them immediately.
- **Detailed Implementation Instructions:**
  1. Refactor `DomainFinder.find_domain_instances()` to accept a generator `query_stream: Generator[Tuple[int, List[Span]], None, None]`.
  2. Within the generator processing loop:
     - Process query spans for sequence $Q_i$.
     - Run `compare_sorted_spans` and `dpc` to derive `peak_to_cluster`.
     - Construct `DomainInstance` objects for $Q_i$.
     - Perform local `filter_spurious_domains` on $Q_i$'s domain list.
     - Append kept domains to `self.domains_per_seq` and assign global integer domain IDs.
     - Explicitly `del sorted_query_spans` to prompt GC clean-up.
  3. Remove global two-pass storage of raw query spans in `DomainFinder`.
- **Estimated Gain:**
  - **Memory Impact:** **~100x reduction in peak RAM during domain extraction**. Reduces peak RAM from ~150 GB down to **< 5–10 GB** (holding only the extracted, filtered domain instances rather than billions of raw alignment hits).
  - **Speed Impact:** Improved cache locality and reduced GC overhead.
- **Necessary Workload:**
  - `domain_finder.py`: ~45–60 lines refactored in `DomainFinder`.

---

### 4. Compact Structural Storage for `Span` Objects (`__slots__` Optimization & NumPy Arrays)
- **Problem:** Instantiating millions of Python `Span` and `DomainInstance` objects creates substantial Python runtime object overhead (8 bytes per pointer, 48+ bytes base object size, GC tracking headers). A query with 10,000 alignment hits creates millions of `Span` instances, leading to heap fragmentation and ~10–20 GB overhead even for moderate alignment sets.
- **Technical Re-evaluation:**
  - Initial alignment parsing only requires 7 integer fields: `[start, end, strand, target_seq_int, target_start, target_end, target_strand]`.
  - Storing raw alignment hits as contiguous 2D NumPy integer arrays (`np.ndarray` of dtype `int32` or `int64`) reduces memory footprint to exactly 28–56 bytes per alignment hit without Python object headers.
  - Full `Span` and `DomainInstance` Python objects need only be instantiated for final representative domain peaks and partner targets after DPC clustering.
- **Detailed Implementation Instructions:**
  1. In `aligners.py`, update `Span` to strictly enforce `__slots__`:
     `__slots__ = ('seq_id', 'start', 'end', 'length', 'partners', 'strand', 'domain')`.
  2. Implement an optional structured NumPy parsing mode `parse_query_alignments_numpy(line_block)`:
     - Pack alignment spans into a 2D NumPy array: `shape=(N, 7)`, `dtype=np.int32`.
     - Columns: `0: q_start`, `1: q_end`, `2: t_seq_int`, `3: t_start`, `4: t_end`, `5: t_strand`, `6: query_span_id`.
  3. Vectorize coordinate comparison functions in `clustering.py` (`compare_sorted_spans`) to operate directly on 2D NumPy array columns instead of list indexing on Python `Span` objects:
     ```python
     # Vectorized coordinate intersection calculation
     start_i, end_i = query_arr[:, 0], query_arr[:, 1]
     # Vectorized overlap logic using numpy broadcast or searchsorted
     ```
- **Estimated Gain:**
  - **Memory Impact:** **70–85% footprint reduction** for raw alignment representation (~15–20 GB RAM saved during local query processing).
  - **Speed Impact:** 2x–5x speedup in local coordinate intersection checks (`compare_sorted_spans`) via NumPy vectorization.
- **Necessary Workload:**
  - `aligners.py`: ~20–30 lines.
  - `domain_finder.py`: ~30–40 lines.
  - `clustering.py`: ~25–35 lines (`compare_sorted_spans` numpy vectorized overload).

### 5. Vectorized Triplet Arrays for Overlap Graph Construction
- **Problem:** In `DomainFinder.calculate_target_overlaps()`, edge accumulation executes `self.target_intersections[canonical_key] += intersect` inside nested loops over every overlapping target span pair. For tens of millions of target overlaps, constructing `(r, c)` tuple keys and performing dictionary lookups creates massive overhead in Python dictionary hash tables (~100–200 bytes per edge).
- **Technical Re-evaluation:**
  - Pairwise target overlaps can be accumulated directly into contiguous 1D NumPy arrays (`global_rows`, `global_cols`, `intersections`) during target processing.
  - To compress duplicate domain overlap edges `(d_i, d_j)` across different target sequences:
    1. Append all pairwise intersections to dynamic NumPy buffer arrays or `list[np.ndarray]`.
    2. Concatenate into global `rows`, `cols`, `data` arrays.
    3. Use SciPy sparse COO/CSR matrix aggregation (`coo_matrix((data, (rows, cols)), shape=(N, N)).sum_duplicates()`) or `np.bincount` to sum duplicate edge weights instantly in C/Fortran vector code.
  - This avoids Python dictionary tuple allocations entirely.
- **Detailed Implementation Instructions:**
  1. In `DomainFinder`, replace `self.target_intersections: Dict[Tuple[int, int], int]` with flat numpy list buffers:
     `self._overlap_rows = []`, `self._overlap_cols = []`, `self._overlap_data = []`.
  2. In `calculate_target_overlaps()`:
     - For each target sequence, extract `global_rows`, `global_cols`, and `intersections` as 1D NumPy `int32` / `int64` arrays.
     - Append array chunks to `self._overlap_rows`, `self._overlap_cols`, `self._overlap_data`.
  3. In `families()`:
     - Concatenate buffer chunks: `rows = np.concatenate(self._overlap_rows)`, etc.
     - Sum duplicate domain-pair edges using SciPy sparse matrix aggregation:
       ```python
       coo = coo_matrix((data, (rows, cols)), shape=(n_domains, n_domains))
       coo.sum_duplicates()
       # Extract canonical upper-triangular non-zero entries
       ```
- **Estimated Gain:**
  - **Memory Impact:** **80–90% reduction in edge storage memory** (reducing graph footprint from ~20–40 GB down to **~2–4 GB** for 100M overlap edges).
  - **Speed Impact:** 5x–10x faster overlap accumulation compared to Python `defaultdict(int)` tuple indexing.
- **Necessary Workload:**
  - `domain_finder.py`: ~35–50 lines changed in `calculate_target_overlaps` and `families`.
