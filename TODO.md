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

---

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

---

## Technical Risk Evaluation by Proposal

These problems must be solved or avoided entirely

### 1. Streaming Generator TSV Parsing (`parse_mmseqs_tsv_stream`)

#### Technical Risks & Edge Cases
- **Unsorted TSV Input Risk:** Standard MMseqs2 and RiboSeek outputs are grouped contiguously by query sequence ID (`fields[1]`). However, if an input TSV is pre-filtered, concatenated from multiple split search runs, or produced by a third-party aligner without query-grouping guarantees, a single-pass streaming parser will encounter duplicate query sequence IDs across disconnected line blocks. This should be immediately caught as an error to break faulty iteration over corrupted file. Otherwise downstream analyses become meaningless. 
- **Scattered Local Query Region Records:** Within a single query sequence block, target hits belonging to identical local query coordinates `(query_start, query_end)` are not guaranteed to be on adjacent lines. The accumulator must maintain `current_query_dict` across the entire sequence block before yielding.
- **Generator State Leakage & Exception Safety:** If downstream domain extraction raises an unhandled exception or terminates early (e.g., during pipeline cancellation or timeout), file handles and partially consumed generator iterators may remain open, leading to file locks or resource leakage. Ensure closing after exception.

---

### 2. String Interning (`sys.intern`) & Integer Sequence Indexing

#### Technical Risks & Edge Cases
- **Registry State Scope & Lifecycle Management:** If `SequenceRegistry` is implemented as a global singleton, string-to-int mappings persist across multiple pipeline executions in long-running processes (e.g., benchmark loops or Web API workers). This leads to unbounded memory growth and potential ID collision across unrelated sequence sets. Ensure locality or global termination.
- **Multiprocessing / Serialization Failure:** In multi-process benchmark or alignment pipelines (e.g., `multiprocessing` or `concurrent.futures`), integer sequence IDs mapped via process-local registries become completely invalid when passed across process boundaries. Re-encoding or pickling without the registry will cause catastrophic sequence misassignment in GFF export. Ensure the index reproducibility, especially in MPIRE parallel framework (not implemented yet)
- **Asymmetric String Decoding in GFF Export:** If `DomainInstance.seq_id` stores an integer but downstream tools or formatting functions (`DomainInstance.to_gff()`, `DomainInstance.to_dict()`) expect a string identifier without accessing the `SequenceRegistry`, raw integer strings (e.g., `"1042"`) will be written to GFF headers, corrupting downstream bioinformatic pipelines (e.g., `bedtools`, `gffread`). Ensure downstream compatibility.

---

### 3. Query-by-Query On-the-Fly Domain Extraction

#### Technical Risks & Edge Cases
- **Premature Partner Target Span Deletion:** In `domain_finder.py`, `DomainFinder` executes `merge_targets()` and `calculate_target_overlaps()` *after* all query domains are extracted. These downstream steps rely on target partner alignment spans stored inside `DomainInstance.partners`.
  - *Critical Distinction:* A target sequence $T_j$ appears as a partner across alignments belonging to *different* query sequences $Q_i$. If raw alignment spans or target partner references are garbage-collected immediately after processing sequence $Q_i$, `merge_targets()` will fail or receive empty target lists, completely destroying the target overlap graph!
- **Global Integer Domain ID Indexing Inconsistency:** In the original `filter_spurious_domains`, global domain IDs (`self.domain_index[next(dc)]`) are assigned after sequence-level domain filtering. In streaming query-by-query processing, integer domain IDs must be assigned sequentially and atomically during extraction to ensure `self.domain_index` and `self.domains_per_seq` remain perfectly synchronized.

#### Efficiency Loss & Unforeseen Consequences
- **Garbage Collection (GC) Thrashing:** Creating and explicitly deleting millions of small temporary dictionary and list objects per query sequence triggers frequent Python GC cycles, potentially degrading throughput if Python's generational collector scans large surviving heap regions repeatedly.

---

### 4. Compact Structural Storage for `Span` Objects (`__slots__` & NumPy Arrays)

#### Technical Risks & Edge Cases
- **Sorting Invariant Violation in Vectorized `compare_sorted_spans`:** The current `compare_sorted_spans` in `clustering.py` strictly relies on input spans being **sorted by start coordinate** (`start_i <= start_j`). This enables the critical early-exit optimization:
  ```python
  if start_j >= end_i:
      break  # no further intersections possible
  ```
  If alignments are converted into 2D NumPy arrays without enforcing start-coordinate sorting, vectorized array operations (e.g., pairwise broadcasting or outer subtraction) will either produce incorrect overlap matrices or revert to $O(N^2)$ dense matrix checks, causing severe speed regressions for high-depth alignment regions ($N > 10,000$). Ensure this never happens.
- **`__slots__` Attribute Restriction:** Enforcing `__slots__` on `Span` and `DomainInstance` prevents dynamic assignment of arbitrary monkey-patched attributes (e.g., `span.temp_score = 0.5`). Any legacy or experimental code attempting dynamic attribute assignment will raise `AttributeError` - this is intended, but check downstream functions and correct
- **Strand Sign Inversion & Zero-Coordinate Bug:** Converting strand representations between integer values ($+1, -1, 0$) and GFF string characters (`'+'`, `'-'`, `'.'`) within fixed-width integer NumPy arrays (`int8` or `int32`) risks sign inversion if bitwise shifts or unsigned casts are mistakenly applied. Ensure it never happens.

---

### 5. Vectorized Triplet Arrays for Overlap Graph Construction

#### Technical Risks & Edge Cases
- **Canonical Upper-Triangular Edge Duplication:** In `DomainFinder.calculate_target_overlaps()`, overlap edges between domains $d_i$ and $d_j$ must be undirected and canonicalized ($r < c$). Ensure that existing graph meets these requirements, find safe bypass or drop this optimisation (Vectorized Triplet Arrays) entirely.

---

# Warning Set for Future Jules Tasks
adhere to the following mandatory guidelines, invariants, and verification protocols.

---

## Mandatory Engineering Rules & Guidelines

1. **Strict Line Ending & Formatter Compliance (`.gitattributes` & PEP 8):**
   - All text and Python source files MUST maintain Unix LF line endings (`eol=lf`).
   - Code changes MUST strictly comply with PEP 8 Visual Indent Style (delimiter-anchored alignment, max line length $\le 99$ characters, no post-bracket newlines) protected by `# fmt: off` / `# fmt: on` annotations.

2. **Explicit Error Handling:**
   - Prefer raising clear, explicit errors containing. Silent fallbacks or mock data generation during benchmark execution are explicitly prohibited.

3. **Memory Ceiling Guarantee (< 200 GB RAM):**
   - Peak memory consumption across all multi-GB TSV parsing, target merging, and graph construction operations MUST NOT exceed **200 GB RAM** for datasets <1TB

4. **Preserve Sequence ID Decode Invariants for Downstream GFF Output:**
   - Any integer sequence indexing (`SequenceRegistry`) MUST be fully encapsulated. All methods producing user-facing or file output (`DomainInstance.to_gff()`, `DomainInstance.to_dict()`, log messages) MUST return valid decoded sequence string identifiers.

5. **Canonical Undirected Overlap Graph Representation:**
   - Overlap edge accumulation MUST strictly enforce canonical ordering $r < c$ for every domain pair $(d_r, d_c)$ before sparse matrix construction.
   - Self-overlaps ($r == c$) MUST be asserted against or dropped immediately upon calculation.

6. **Sorting Invariant Preservation in Local Intersection Computation:**
   - Any optimized, NumPy-vectorized, or C-extension implementation of `compare_sorted_spans` MUST verify that input coordinate arrays are sorted by `start` position before executing range-bound intersection checks.

7. **Query-Grouping Pre-condition Verification in Streaming TSVs:**
   - `parse_mmseqs_tsv_stream` MUST assert or check that input lines are grouped contiguously by query ID. If an unsorted TSV is detected mid-stream, it MUST raise an explicit `ValueError` instructing the user to sort the TSV by query sequence ID.

---

## Required Verification & Test Procedures

Before submitting any optimization code changes, the following test suite commands and verification steps MUST be executed and confirmed passing:

```bash
# 1. Run full pytest suite across alignment and domain finder modules Create New tests as EXISTING ARE OUTDATED OR MISSING

# 2. Check code formatting against pyproject.toml / formatting guidelines

# 3. Verify equivalence of the results of original and updated results. Results should either match ones obtained prior to the code revision or be explicit correction of evident error.
python3 -m flake8 aligners.py domain_finder.py clustering.py
```

---

# Unified Joint Memory Optimization Roadmap & Implementation Guide

## Executive Summary & System-Wide Architectural Blueprint

This guide details the unified, joint implementation plan for memory-optimizing `aligners.py` and `domain_finder.py`. The primary objective is to enable processing of massive multi-GB alignment files (e.g. 400+ GB BLAST-TAB search outputs containing over 2 billion records) while strictly keeping peak memory consumption **below 200 GB RAM** (and under 10–20 GB for typical multi-GB inputs), without degrading execution speed or downstream domain family extraction accuracy.

The joint architecture addresses the memory bottleneck holistically:
1. **Upstream Alignment Ingestion (`aligners.py`):** Replaces full-file dictionary loading with single-pass query streaming (`parse_mmseqs_tsv_stream`) and zero-overhead sequence string interning/indexing (`SequenceRegistry`).
2. **Downstream Domain Extraction (`domain_finder.py`):** Replaces global two-pass query holding with on-the-fly, query-by-query Density Peak Clustering (DPC) and local spurious domain filtering.
3. **Data Representation & Array Operations (`clustering.py` & `aligners.py`):** Leverages compact NumPy integer arrays for local coordinate intersection checks (`compare_sorted_spans`) and enforces strict `__slots__` attributes on `Span` and `DomainInstance` classes.
4. **Graph Construction & Edge Accumulation (`domain_finder.py`):** Replaces Python dictionary tuple indexing with contiguous 1D NumPy triplet buffers (`rows`, `cols`, `data`) and SciPy CSR sparse matrix sum-deduplication.

---

## Joint Implementation Architecture & Phase Flowchart

```
                          [ 400+ GB BLAST-TAB TSV File ]
                                        │
                                        ▼
                  Phase 1 & 2: parse_mmseqs_tsv_stream()
             ┌──────────────────────────────────────────────────┐
             │ - Single-pass file streaming handle               │
             │ - Unsorted Query ID Guard (assert/raise ValueError)│
             │ - Fast Sequence ID -> int32 conversion           │
             │ - Yields (q_seq_int, query_spans_list/array)     │
             └─────────────────────────┬────────────────────────┘
                                       │ Stream query block
                                       ▼
                  Phase 3 & 4: On-the-Fly Domain Extraction
             ┌──────────────────────────────────────────────────┐
             │ - Local DPC on q_i (compare_sorted_spans)        │
             │ - Local spurious domain filter (max_containment) │
             │ - Assign global atomic integer domain_id         │
             │ - Save kept DomainInstance + partner targets     │
             │ - Discard raw query_spans_list (prompt GC)       │
             └─────────────────────────┬────────────────────────┘
                                       │ Accumulated Domain Instances
                                       ▼
                        Phase 3.2: merge_targets()
             ┌──────────────────────────────────────────────────┐
             │ - Group partner target spans by target sequence  │
             │ - Connected component reduction per target seq    │
             └─────────────────────────┬────────────────────────┘
                                       │ Merged Target Spans
                                       ▼
                  Phase 5: Vectorized Sparse Edge Accumulation
             ┌──────────────────────────────────────────────────┐
             │ - Local pairwise target intersections            │
             │ - Append (global_r, global_c, intersect) to      │
             │   1D NumPy triplet buffers                       │
             │ - SciPy COO -> CSR sum_duplicates() aggregation  │
             └─────────────────────────┬────────────────────────┘
                                       │ Sparse Overlap Graph Matrix
                                       ▼
                           Phase 5.2: families()
             ┌──────────────────────────────────────────────────┐
             │ - DPC clustering on domain family overlap matrix │
             │ - GFF Export: SequenceRegistry.get_str(seq_int)  │
             └──────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Implementation Instructions

### Phase 1: Shared Core Sequence Registry (`SequenceRegistry`)

#### 1. Scope & Location
Implement `SequenceRegistry` in `aligners.py` and import/instantiate it inside `DomainFinder` or `find_domains()`.

#### 2. Technical Specification & Code
```python
class SequenceRegistry:
    """
    Bi-directional string-to-int32 sequence ID registry utilizing string interning.
    Ensures single-instance storage of repeating sequence headers across multi-billion line TSVs.
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

#### 3. Risk Controls & Invariants
- **Scope Locality:** Instantiate a single `SequenceRegistry` per pipeline execution inside `find_domains()`. Pass this instance into `parse_mmseqs_tsv_stream` and `DomainFinder`. Avoid global singleton variables to prevent memory leakage across benchmark loops.
- **Output String Decoding Invariant:** `DomainInstance.to_gff()` must accept or reference `registry` to decode integer sequence IDs back to human-readable sequence string names before writing to GFF format.

---

### Phase 2: Single-Pass Streaming Generator TSV Parser (`parse_mmseqs_tsv_stream`)

#### 1. Scope & Location
Add `parse_mmseqs_tsv_stream` to `aligners.py`.

#### 2. Technical Specification & Implementation Instructions
1. Check for timeout or error indicator lines (`_TimeoutError_`) on file opening.
2. Maintain single-pass line iteration via `for line in handle:` within a context manager (`with tsv_file.open("r") as handle:`).
3. Track `current_q_seq_int: Optional[int] = None`, `seen_queries: set[int] = set()`, and `current_query_dict = defaultdict(list)`.
4. Validate Query Contiguity (Unsorted Input Guard):
   ```python
   if q_seq_int != current_q_seq_int:
       if q_seq_int in seen_queries:
           raise ValueError(
               f"🛑 Input TSV file {tsv_file} is not grouped contiguously by query sequence. "
               f"Query ID '{q_seq_str}' appeared in non-adjacent line blocks. "
               f"Please sort the alignment TSV by query ID before processing."
           )
       if current_q_seq_int is not None:
           # Process and yield current query spans
           yield current_q_seq_int, _build_sorted_query_spans(current_q_seq_int, current_query_dict, registry)
           current_query_dict.clear()
       seen_queries.add(q_seq_int)
       current_q_seq_int = q_seq_int
   ```
5. Handle local region coordinate aggregation: Store alignments under key `(query_start, query_end)` in `current_query_dict` to accommodate scattered lines belonging to the same local region within a single query block.
6. Yield the final query block after loop termination.

#### 3. Expected Memory Impact
Reduces TSV parsing RAM overhead from **~400+ GB down to < 10 MB** (peak heap size limited to alignments of a single query sequence).

---

### Phase 3: Query-by-Query On-the-Fly Domain Extraction Pipeline

#### 1. Scope & Location
Refactor `DomainFinder.find_domain_instances()` and `find_domains()` in `domain_finder.py`.

#### 2. Detailed Implementation Instructions
1. Modify `DomainFinder.__init__` to accept a generator or stream rather than a fully loaded in-memory dictionary.
2. Refactor `find_domain_instances(self, query_stream, ...)`:
   ```python
   for q_seq_int, sorted_query_spans in query_stream:
       n_spans = len(sorted_query_spans)
       if n_spans == 0:
           continue
       elif n_spans == 1:
           peak_span = sorted_query_spans[0]
           self.create_new_domain(peak_span=peak_span, targets=peak_span.partners)
       else:
           rows, cols, intersections = compare_sorted_spans(sorted_query_spans)
           if rows.size == 0:
               peak_to_cluster = {p: np.array([p]) for p in range(n_spans)}
           else:
               span_lengths = np.array([s.length for s in sorted_query_spans])
               weights = np.array([s.n_partners() for s in sorted_query_spans])
               similarities = jaccard_index(intersections, span_lengths[rows], span_lengths[cols])
               peak_to_cluster, n_noise = dpc(rows=rows, cols=cols, sims=similarities, n=n_spans, ...)
               self.n_halo_alns += n_noise

           for peak_id, member_ids in peak_to_cluster.items():
               peak_span = sorted_query_spans[peak_id]
               member_targets = []
               for m_id in member_ids:
                   member_targets.extend(sorted_query_spans[int(m_id)].partners)
               self.create_new_domain(peak_span=peak_span, targets=member_targets)

       # Perform sequence-level local spurious domain filtering immediately
       if q_seq_int in self.domains_per_seq:
           self._filter_sequence_domains_local(q_seq_int, min_containment, reassign_targets)

       # Explicitly delete temporary raw query alignment spans to prompt garbage collection
       del sorted_query_spans
   ```

#### 3. Target Partner Retention Guarantee
- **Invariant:** `DomainInstance` objects must retain partner target `Span` objects in `DomainInstance.partners`.
- Raw query alignments are safely garbage-collected because target partner information is preserved inside the kept `DomainInstance` objects for subsequent `merge_targets()` execution.

---

### Phase 4: Compact Structural Storage & Vectorized Coordinate Intersections

#### 1. Scope & Location
Update `Span` in `aligners.py` and `compare_sorted_spans` in `clustering.py`.

#### 2. Detailed Implementation Instructions
1. Enforce strict `__slots__` in `Span` and `DomainInstance`:
   ```python
   class Span:
       __slots__ = ('seq_id', 'start', 'end', 'length', 'strand', 'partners', 'domain')
   ```
2. Vectorize coordinate intersection checks in `clustering.py`:
   - Support operating directly on 2D `np.int32` coordinate arrays `shape=(N, 2)` `[start, end]`.
   - **Sorting Invariant Preservation:** Assert or enforce that input coordinate arrays are sorted by `start` coordinate (`arr[:, 0]`). Maintain the range-bound early exit condition (`start_j >= end_i`) to preserve $O(N)$ efficiency in dense region overlap detection.

---

### Phase 5: Vectorized Triplet Sparse Array Overlap Graph Construction

#### 1. Scope & Location
Refactor `calculate_target_overlaps()` and `families()` in `domain_finder.py`.

#### 2. Detailed Implementation Instructions
1. Replace `self.target_intersections: Dict[Tuple[int, int], int]` with dynamic NumPy list buffers:
   ```python
   self._overlap_rows: List[np.ndarray] = []
   self._overlap_cols: List[np.ndarray] = []
   self._overlap_data: List[np.ndarray] = []
   ```
2. In `calculate_target_overlaps()`:
   ```python
   for seq_id, targets in self.targets_per_seq.items():
       n_targets = len(targets)
       if n_targets > 1:
           local_rows, local_cols, intersections = compare_sorted_spans(targets)
           if local_rows.size > 0:
               domain_ids = np.array([t.domain for t in targets], dtype=np.int32)
               global_rows = domain_ids[local_rows]
               global_cols = domain_ids[local_cols]

               # Enforce canonical undirected upper-triangular ordering (r < c)
               mask = global_rows != global_cols
               g_r = np.minimum(global_rows[mask], global_cols[mask])
               g_c = np.maximum(global_rows[mask], global_cols[mask])
               data = intersections[mask]

               self._overlap_rows.append(g_r)
               self._overlap_cols.append(g_c)
               self._overlap_data.append(data)
   ```
3. In `families()`:
   ```python
   if self._overlap_rows:
       rows = np.concatenate(self._overlap_rows)
       cols = np.concatenate(self._overlap_cols)
       data = np.concatenate(self._overlap_data)

       # Aggregate duplicate edges using SciPy sparse matrix
       coo = coo_matrix((data, (rows, cols)), shape=(n_domains, n_domains))
       csr = coo.tocsr()
       csr.sum_duplicates()
       coo_dedup = csr.tocoo()

       rows = coo_dedup.row
       cols = coo_dedup.col
       intersections = coo_dedup.data
   ```

#### 3. Expected Memory Gain
Reduces overlap graph memory footprint by **80–90%** (from ~20–40 GB down to **~2–4 GB** for 100M overlap edges) and accelerates matrix assembly by 5x–10x.

---

## Verification & Validation Protocol

Upon implementing the roadmap changes, execute the following commands to confirm formatting, error handling, and numerical equivalence:

```bash
# 1. Run full test suite across alignment and domain finder modules
python3 -m pytest tests/

# 2. Verify code formatting and linting compliance
flake8 aligners.py domain_finder.py clustering.py

# 3. Test multi-GB execution memory ceiling
# Confirm peak RAM remains strictly < 200 GB for multi-GB alignment files.
```
