# Consolidated Scientific and Technical Audit of the RNA Domain-Finding Pipeline

This document is the authoritative, self-contained audit and implementation guide for the non-coding RNA (ncRNA) domain-finding pipeline. It synthesizes and replaces the preliminary analyses and working drafts in the repository. Every architectural component, algorithmic mechanism, and proposed optimization across `domain_finder.py`, `clustering.py`, `aligners.py`, and `executor.py` has been re-evaluated directly against the working code, mathematical invariants, and reproducible experiments.

A future engineering agent (Jules) must read this complete document before making any changes to the codebase, adhere strictly to the baseline invariants, and consult the specific implementation plan and open questions detailed below.

---

# Scope, Baseline Revision, and Evidence Limits

- **Pipeline Scope**: The audited workflow takes sequence similarity searches (MMseqs2 or Riboseek BLAST-TAB) of unannotated RNA sequences, discovers local domain instances on query sequences, projects and indexes target spans across genomes or transcriptomes, calculates pairwise interval intersections, clusters domain instances into global families using Density Peak Clustering (DPC), and exports genomic annotations in GFF3 format for downstream structure-aware multiple sequence alignment (MAFFT/MUSCLE), secondary structure consensus prediction (RNAalifold), and Infernal Covariance Model calculation.
- **Audited Repository Revision**: Commit `662253232d43905602bc58aacc3c58c5981c7ea1` (clean working tree). Verified SHA-256 prefixes of core files: `domain_finder.py` (3cad7c5c), `clustering.py` (1bcdd9b6), `aligners.py` (1d99793f), `executor.py` (d17fc277), `gff_sequence_extractor.py` (08512e23), `domain_finder_precomputed.py` (11702371), `README.md` (7b24aa65).
- **Audit Environment**: Linux x86_64, Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, mpire 2.10.2, pandas 3.1.1, loguru 0.7.3, psutil 7.2.2.
- **Evidence Classification**:
  - *Verified*: Formally confirmed through code inspection and reproducible automated execution.
  - *Plausible Risk / Hypothesis*: Mechanistically sound hypothesis consistent with the code structure, requiring validation on empirical biological datasets.
  - *Incorrect / Unsupported / Obsolete*: Prior claims shown to be mathematically invalid, factually inaccurate, or already resolved in the current codebase.
- **Data Limitations**: Historical benchmark numbers reported in previous drafts (such as 105,441 and 121,265 hits across 2,338 and 9,999 sequences, claimed speedups, and biological duplication percentages) lack raw input data, seeds, or execution logs in the tracked repository. They are treated here as previously reported claims rather than verified empirical truths. Experiments conducted during this audit rely on reproducible synthetic tests and controlled mathematical fixtures.

---

# Consolidated Assessment of Problems and Recommendations

## Interval Comparison and Input Sorting Invariants

- **Status**: Core interval arithmetic is verified correct for sorted inputs. The sortedness precondition is real and unenforced in code. The prior proposal to sort silently inside the function is incorrect.
- **Evidence and Code Fact**: In `clustering.py:compare_sorted_spans`, interval intersections are evaluated using forward binary searches:
  ```python
  starts = spans[:, 0]
  ends = spans[:, 1]
  for i in range(n):
      end_i = ends[i]
      r = max(i + 1, int(np.searchsorted(starts, end_i, side="left")))
  ```
  Prior reports asserted that a long enclosing interval causes `searchsorted` to truncate prematurely and miss downstream overlaps. This assertion is mathematically incorrect for half-open intervals sorted by start coordinate. If `starts` is non-decreasing, any overlapping downstream interval `j > i` must have `starts[j] < ends[i]`. Because iteration `i` inspects all `j` from `i + 1` up to `r = searchsorted(starts, ends[i])`, every downstream interval overlapping `i` is examined. There is no need for interval `j` to search backward to `i`, because `i` already paired with `j`.
- **Verified Defect**: The function assumes `spans` is sorted by start coordinate without validating this assumption. On unsorted inputs such as `[[50, 60], [0, 100], [10, 20]]`, the function computes overlap lengths using `min(ends[i], ends[j]) - starts[j]`, which is valid only when `starts[i] <= starts[j]`. For this unsorted input, it reports an overlap length of 60 between interval 0 `[50, 60)` and interval 1 `[0, 100)` (true overlap is 10) and emits a spurious overlap between interval 0 and interval 2 `[10, 20)`. Existing callers in `domain_finder.py` currently pass sorted inputs (`np.unique` in Stage 2; `np.argsort(target_spans[:, 0])` in Stage 4), so this is a latent vulnerability rather than an active pipeline failure.
- **Flawed Solution Rejected**: A prior proposal suggested silently sorting intervals inside `compare_sorted_spans` via `order = np.lexsort((ends, starts))`. This is unsafe because `compare_sorted_spans` returns row and column index arrays into the input array. If the function sorts internally, the returned indices refer to the reordered array, whereas callers (`overlaps_on_target`) immediately use those indices to index into external arrays (`aligned_domain_pointers[local_rows]`), causing domain pointers and span lengths to become misaligned.
- **Cost Model**: `compare_sorted_spans` is not an O(N) or fully compiled C operation; it is an output-sensitive Python loop over `N` spans executing binary searches and slice concatenations. In dense regions where all intervals overlap, running time and memory scale quadratically: a synthetic target with 3,000 mutually overlapping spans generates 4,498,500 pairs, 54 MB of output arrays, and 114 MB of temporary peak memory in 0.2 seconds.
- **Recommendation**: Retain the half-open interval overlap arithmetic. Add an explicit O(N) validation check at the start of `compare_sorted_spans`: `if np.any(starts[1:] < starts[:-1]): raise ValueError("spans must be sorted by start coordinate")`. Do not sort silently.

## Target-Span Storage, Memory Bounds, and Dereplication

- **Status**: The flat 16-byte structured array is verified and compact. Global coordinate dereplication is unviable and rejected. The claimed 500 MB worker memory bound is unsupported by the code.
- **Physical Layout and Breakeven Reality**: At Stage 3, target hits are stored in a contiguous 1D structured array with dtype `[('t_seq', int32), ('start', int32), ('end', int32), ('dom_id', int32)]`, requiring exactly 16 bytes per hit. The proposed dereplication scheme would separate unique coordinates `(t_seq, start, end)` (12 bytes) from domain associations using an inverted index (4 bytes for domain ID plus 4 bytes for coordinate indptr). For `U` unique coordinates and `N` total hits with duplication rate `d = 1 - U/N`, the storage cost is `16 * (1 - d) * N + 4 * N = (20 - 16 * d) * N` bytes. The mathematical breakeven requires `d > 25.0%`. If coordinate duplication is below 25%, dereplication increases physical storage.
- **Domain Graph Ingestion Reality**: Downstream Stage 5 requires a graph between domain instances, not coordinates. If a unique coordinate group contains `k` distinct domains, those domains induce `k * (k - 1) / 2` pairwise within-group edges. If two overlapping coordinate groups contain `k1` and `k2` domains, they induce `k1 * k2` pairwise cross-group edges. Expanding these Cartesian products in Python creates severe interpreter bottlenecks and memory bloat, slowing down overlap computation by 10-fold to 14-fold without reducing the number of edges delivered to Stage 5.
- **Worker Memory Reality**: `ShardedEdgeWriter` flushes edge buffers to disk when accumulated edges reach 250,000 records. However, `target_batch_processor` calls `writer.add(...)` only after `overlaps_on_target` has completely materialized all edges for a target sequence. Furthermore, `writer.add` appends incoming arrays before checking the threshold. In a direct test, supplying 5 edges to a writer with a threshold of 2 produced a single shard of 5 edges. If a single target sequence generates millions of overlaps, worker memory spikes quadratically before any shard flush occurs. The claim in `README.md` that worker memory is strictly capped below 500 MB is unsupported.
- **Recommendation**: Retain the flat 16-byte representation. Do not implement cross-domain coordinate dereplication. To prevent worker memory spikes on pathological dense targets, implement chunked overlap generation that streams bounded pair arrays directly to `ShardedEdgeWriter`.

## Scaling Bottlenecks Across Pipeline Stages

- **Status**: Stage 5 edge reassembly and key reduction is a verified scaling bottleneck on large families.
- **Raw Edges vs Unique Pairs**: In a complete all-against-all homology search of a conserved family with `F` members, each target sequence carries up to `F` overlapping spans. Stage 4 emits raw edge contributions `Q = F * F * (F - 1) / 2`, while the family graph contains only `E = F * (F - 1) / 2` unique domain pairs. The ratio `Q / E = F`. On mock data with `F = 80`, Stage 4 generates 252,800 raw edge records for just 3,160 unique domain pairs.
- **Stage 5 Memory Scaling**: In `family_finder`, the parent process reconstructs all edge shards into three contiguous `int32` arrays (`all_rows`, `all_cols`, `all_inters`), allocates 64-bit composite keys `(u_r << 32) | u_c`, sorts them with `np.argsort`, and reduces duplicate edges. The total memory consumed during this step scales with the number of raw edges `Q` (approximately 70 bytes per raw edge across temporary arrays), not the unique graph size `E`. For large datasets with hundreds of millions of pairwise overlaps, Stage 5 parent memory will become the primary memory ceiling of the entire pipeline.
- **Intra-Target Pair Collapse**: Collapsing duplicate domain pairs within individual target sequences before writing shards helps only when the same two domains overlap at multiple disjoint loci on that sequence (such as repeated motifs or tandem exons). On a single shared locus, each pair occurs only once per target, so intra-target deduplication does not reduce `Q / E = F`.
- **Recommendation**: Maintain exact integer summation of intersection lengths throughout Stage 4 and Stage 5. If profiling on large biological datasets reveals that Stage 5 parent memory dominates, prototype an external sorted-run merge or chunked key reduction that aggregates identical domain pairs incrementally before allocating the full CSR similarity matrix.

## Density Peak Clustering Backends and Numerical Behavior

- **Status**: Current call-site backend choices (NumPy dense in Stage 2, SciPy CSR in Stage 5) are appropriate. Claims of guaranteed cross-backend numerical equivalence are disproved by a reproducible counterexample. Automatic backend switching without a defined tie-breaking policy is rejected.
- **The Counterexample**: A 10-node undirected graph with edge similarities:
  ```text
  (0, 1, 0.4), (0, 7, 0.6), (0, 8, 0.2), (0, 9, 0.6), (2, 9, 0.2),
  (3, 9, 0.4), (4, 6, 0.8), (5, 6, 0.4), (6, 9, 0.6)
  ```
  Evaluated with `halo_method=None`, `min_sim=0.35`, `min_dens=1.0`:
  - Dense NumPy DPC returns two multi-node clusters `{0, 1, 3, 7, 9}` and `{4, 5, 6}`, with `{2}` and `{8}` as singletons.
  - Sparse SciPy DPC returns a single merged cluster `{0, 1, 3, 4, 5, 6, 7, 9}`, with `{2}` and `{8}` as singletons.
- **Root Cause**: Nodes 0, 6, and 9 have identical theoretical local density (2.8). In dense summation, all three nodes evaluate to 2.8000000000000003. In sparse CSR summation, nodes 0 and 6 evaluate to 2.8 while node 9 evaluates to 2.8000000000000003 (a difference of 4.4e-16). Because DPC ranks points by density with index tie-breaking to determine which node is higher, this sub-epsilon numerical divergence flips the nearest-higher assignment: dense DPC treats node 0 as the higher peak, whereas sparse DPC treats node 9 as the peak and links node 6 directly to it via their 0.6 edge.
- **Crossover Reality**: Previous documentation contained contradictory crossover thresholds (claiming SciPy is faster starting at N=150–200, while nearby benchmark tables showed NumPy faster at N=200 and 1% density). Furthermore, `build_similarity_matrix` dense initialization is not an O(1) operation; it allocates an N x N buffer and executes indexed updates.
- **Recommendation**: Preserve the existing static backend choices: `backend="numpy"` in Stage 2 (where query spans average 10–15 intervals) and `backend="scipy"` in Stage 5 (where global domain instances number in the tens of thousands and graph density is below 0.5%). Do not implement dynamic backend dispatch or `backend="auto"` until an explicit numerical tolerance policy for density ties is established and evaluated.

## Chaining, Halo Filtering, and Similarity Metrics

- **Status**: Transitive assignment along positive edges is standard DPC behavior. Proposed direct-to-peak lineage validation changes the clustering model and is rejected. Traditional cluster-wide halo filtering can cause excessive member pruning, but defaults should remain intact pending labeled biological benchmarks.
- **DPC Pointer Doubling**: In `clustering.py:dpc`, non-peak points inherit cluster labels from their nearest higher-density neighbor via pointer doubling. This allows chaining: domain A connects to B (similarity 0.40), and B connects to C (similarity 0.40), while A and C share zero similarity. If C is the peak, both A and B are assigned to C. This is an intended property of density-based clustering that accommodates continuous evolutionary divergence. Requiring direct similarity between every member and the cluster peak would alter the fundamental clustering objective and reject legitimate members of heterogeneous families.
- **Lineage Validation Flaw**: The pseudo-code patch proposed in earlier drafts called `.toarray()` on dense NumPy arrays (raising AttributeError) and used index -1 for unassigned points, triggering silent wrap-around to the last column. It is mathematically and syntactically defective.
- **Halo Filtering Mechanics**: In traditional cluster-wide halo mode (`point_specific=False`), the boundary density cutoff is set to the maximum density among all border points of a cluster. If a large, authentic family contains a single high-density border point sharing a weak hit with an adjacent locus, that single point elevates the cluster threshold, causing legitimate internal members with lower density to be pruned as noise (-1). In `point_specific=True` mode, border points are pruned only if their individual density falls strictly below the cluster median. However, claims that traditional halo universally prunes 50% to 90% of biological members lack traceable provenance.
- **Global Jaccard vs Containment**: In Stage 5, Jaccard similarity is computed as `summed_inters / (length_a + length_b - summed_inters)`, where lengths are accumulated across all target sequences. For an asymmetric pair where domain A covers 5,000 nt across 50 targets and domain B covers 100 nt on 1 target, full overlap yields Jaccard 0.02. This is an accurate calculation of total footprint overlap, not an arithmetic error. Replacing Jaccard with global containment `summed_inters / min(length_a, length_b)` would assign similarity 1.0 to any single accidental alignment between a large multi-copy element and an unrelated single-copy sequence, merging distinct families.
- **Recommendation**: Retain current DPC pointer doubling, halo filtering defaults, and global Jaccard similarity. Evaluate point-specific halo filtering and co-occurrence normalized similarities only as configurable, opt-in experimental modes on curated benchmark datasets.

## Same-Sequence Alignment Policy

- **Status**: Skipping same-sequence hits is an intentional comparative genomics filter. Relaxing it introduces non-tandem self-matches and changes biological family interpretation.
- **Code Fact**: In `aligners.py:parse_mmseqs_tsv`, all records where query sequence ID equals target sequence ID are dropped immediately:
  ```python
  if q_seq_str == t_seq_str:
      continue
  ```
- **Biological and Algorithmic Implications**: In comparative RNA domain discovery, families are defined by independent evolutionary conservation across distinct sequence contexts. While tandem duplications on the same sequence are biologically real, cross-sequence alignments to single-copy homologs often capture those tandem instances without requiring same-sequence hits.
- **Consequences of Coordinate-Only Filtering**: A prior proposal recommended skipping only identical coordinates (`q_seq == t_seq and q_start == t_start and q_end == t_end`). When tested on synthetic sequences, this relaxed filter successfully recovered isolated tandem copies. However, it also admitted inverted-repeat self-alignments (stem-loops aligning to themselves in reverse orientation) and small shifted self-hits, turning them into spurious single-sequence families with negative strand assignments.
- **Recommendation**: Retain the current same-sequence exclusion filter as the baseline policy. If recovery of single-sequence tandem duplications is desired in the future, it must be introduced as an explicit, opt-in mode with dedicated filtering for inverted repeats, self-overlaps, and reciprocal duplicates.

## Touching Intervals, Strand Consensus, and GFF3 Export

- **Status**: Merging touching intervals is correct for coverage arithmetic in Stage 4, but coalesces adjacent same-family features in Stage 6 export. Strand voting reflects relative alignment orientation rather than verified biological transcription strand.
- **Touching Intervals**: In `domain_finder.py:merge_overlapping_spans`, the breakpoint condition `s[1:] > e_cum[:-1]` deliberately merges abutting half-open intervals such as `[10, 20)` and `[20, 30)` into `[10, 30)`. In Stage 4, this is completely harmless: abutting intervals share zero nucleotides, and their union covers the exact same bases, producing identical coverage lengths and pairwise intersections. In Stage 6 (`evaluate_clusters`), however, reusing this function merges adjacent copies of the same family on the same sequence into a single GFF feature. Whether two adjacent domains represent one chimeric locus or two tandem features is an annotation policy decision, not an interval arithmetic error. Changing `>` to `>=` globally would break interval union calculations.
- **Strand Voting**: `determine_strand` votes on relative alignment orientations (-1 if query or target is reverse-complemented) and resolves ties to `+`. This records relative alignment orientation, not absolute biological transcription strand. In small symmetric families (such as two forward sequences and two reverse sequences), every sequence sees concordant hits from its own orientation and discordant hits from the opposite orientation, causing all members to vote `-`. Downstream `gff_sequence_extractor.py` uses this strand to orient extracted FASTA sequences, so orientation errors propagate to downstream alignment tools unless those tools use `--adjustdirection`.
- **GFF3 Feature Identity**: In `find_domains`, the GFF writer assigns the family peak index as the `ID` attribute for every feature in that family. In strict GFF3 specifications, shared `ID` attributes are reserved for multipart features (such as discontinuous exons of a single transcript). Downstream `gff_sequence_extractor.py` relies on this convention by using `ID` as the family grouping key.
- **Recommendation**: Maintain current interval merging in Stage 4. If separation of adjacent family copies in GFF export is required, implement a dedicated export-level boundary rule rather than altering `merge_overlapping_spans`. Retain strand voting as relative orientation, but document that true genomic sense/antisense annotation requires an external biological anchor.

## Workload Balancing and Executor Infrastructure

- **Status**: Temporary directory isolation is verified in the current code. Stage 4 target-sequence batching is susceptible to hit skew. Broad criticism of executor signal handling does not reflect active pipeline defects.
- **Temporary Directory Safety**: `domain_finder.py:find_domains` creates an isolated directory using `tempfile.TemporaryDirectory(prefix="domain_finder_stage4_", dir=base_tmp)` and cleans it in a `finally` block. This guarantees process isolation during concurrent runs. However, if the caller supplies a custom `tmp_dir` path that does not already exist on disk, `TemporaryDirectory` raises `FileNotFoundError`. The parent directory should be created with `mkdir(parents=True, exist_ok=True)` before creating the temporary directory.
- **Target Batch Skew**: Stage 4 partitions targets by sequence count (`target_batch_size = max(10, min(1000, n_target_seqs // (cpu * 4)))`). In datasets where a few target sequences contain tens of thousands of hits while the remainder contain few, worker tasks become unbalanced. Because interval overlap work can scale quadratically with local hit density, a single dense target sequence will dominate task execution time.
- **Executor Helpers**: `executor.py` contains `timeout_signal` (which uses POSIX `SIGALRM`) and `checkpoint` (which mutates `sys.setrecursionlimit`). Neither helper is called within the domain-finding execution path (`find_domains`, `clustering.py`, `aligners.py`). They do not represent active defects in Stage 4 execution.
- **Recommendation**: Ensure `tmp_dir` is created on disk before initializing `TemporaryDirectory`. Instrument per-target hit counts and overlap work before modifying Stage 4 batching. Do not alter `executor.py` signal handling as part of domain-finding maintenance.

---

# Proposals Deferred or Rejected

To prevent future implementation runs from pursuing unproductive or harmful changes, the following proposals from earlier drafts are explicitly deferred or rejected:

- **Cross-Domain Coordinate Dereplication**: Rejected. Coordinate duplication in representative RNA datasets is well below the 25% breakeven threshold. Reconstructing domain pairs via Cartesian product expansion slows down overlap computation by 10-fold to 14-fold and provides zero reduction in Stage 5 memory or edge count.
- **Automatic DPC Backend Switching (`backend="auto"`)**: Deferred. Floating-point summation order in density calculations breaks numerical parity on tied and near-tied nodes, altering cluster assignments. Static allocation (`numpy` for Stage 2, `scipy` for Stage 5) is safe and optimal for the respective graph sizes.
- **Lineage Validation / Direct-to-Peak DPC Constraint**: Rejected. Transitive pointer doubling along positive edges is standard DPC behavior that accommodates natural sequence divergence. The proposed patch is syntactically invalid and changes the biological clustering model without validation.
- **Global `>` to `>=` Replacement in `merge_overlapping_spans`**: Rejected. Merging abutting intervals is correct for union-of-covered-bases arithmetic in Stage 4. Any separation of touching loci for annotation must be implemented exclusively at Stage 6 GFF export.
- **Replacing Jaccard with Containment Metric**: Rejected. Global containment assigns a similarity of 1.0 to any single-target overlap between a broadly conserved RNA and an unrelated sequence, causing spurious cross-family collapse.
- **Unconditional Relaxation of Same-Sequence Alignment Filter**: Rejected. Admitting all non-identical same-sequence hits captures inverted-repeat stem-loops and shifted alignments, generating spurious single-sequence clusters.
- **Refactoring `executor.py` Signal Handlers**: Rejected. The signal and recursion helpers have no active call sites in the domain-finding pipeline.

---

# Practical Implementation and Verification Plan

The following implementation steps are justified by verified code behavior. Each step specifies the exact files, functions, intended modifications, invariants, risks, and required tests. Steps must be implemented and tested sequentially.

## Step 1: Enforce Start-Sorting Precondition in Span Comparison
- **Target File and Function**: `clustering.py:compare_sorted_spans`
- **Intended Modification**: Add an explicit, fast O(N) check verifying that start coordinates are non-decreasing:
  ```python
  if n > 1 and np.any(starts[1:] < starts[:-1]):
      raise ValueError("Input spans must be sorted by start coordinate in compare_sorted_spans")
  ```
  Update the function docstring to state the sortedness precondition and the quadratic worst-case complexity for dense overlapping intervals.
- **Invariants to Preserve**: Returned index arrays (`lows`, `highs`) and intersection lengths (`inters`) must remain `int32`. Empty arrays must be returned when `n < 2`. Callers (`domain_instances_on_query`, `overlaps_on_target`) must remain unchanged. Do not sort intervals internally.
- **Reason**: Unsorted inputs produce incorrect overlap lengths and spurious edges without raising an error. Enforcing the precondition prevents silent data corruption.
- **Risks**: A minor O(N) validation overhead per call (fractions of a microsecond).
- **Required Tests and Measurements**:
  - Test sorted inputs with ties, nesting, and abutting intervals against a quadratic reference oracle.
  - Test unsorted inputs to confirm `ValueError` is raised.
  - Run the full test suite to confirm existing callers satisfy the precondition.

## Step 2: Ensure Temporary Directory Parent Exists
- **Target File and Function**: `domain_finder.py:find_domains`
- **Intended Modification**: Before initializing `tempfile.TemporaryDirectory`, ensure the caller-supplied parent directory exists:
  ```python
  if tmp_dir is not None:
      base_tmp = Path(tmp_dir)
      base_tmp.mkdir(parents=True, exist_ok=True)
  else:
      base_tmp = None
  ```
- **Invariants to Preserve**: Unique per-run directory naming (`prefix="domain_finder_stage4_"`), process isolation, and cleanup in the `finally` block.
- **Reason**: Passing a non-existent directory path as `tmp_dir` (as shown in `README.md` examples) currently triggers an unhandled `FileNotFoundError`.
- **Risks**: Negligible.
- **Required Tests and Measurements**:
  - Invoke `find_domains` passing a non-existent temporary path (e.g., `./tmp_test_scratch/sub`). Verify directory creation, successful execution, and clean removal.
  - Execute concurrent pipeline runs sharing the same base directory to verify isolation.

## Step 3: Implement Bounded Overlap Streaming for Dense Target Sequences
- **Target Files and Functions**: `clustering.py:compare_sorted_spans`, `domain_finder.py:overlaps_on_target`, `domain_finder.py:target_batch_processor`, `domain_finder.py:ShardedEdgeWriter`
- **Intended Modification**: Refactor interval overlap generation so that pairs from dense target sequences can be emitted in bounded chunks (e.g., max 250,000 edges per chunk) directly to `ShardedEdgeWriter`, rather than accumulating all pairs in memory across the entire target. Maintain an O(1) fast path for typical sparse targets where total candidate pairs fall below the buffer threshold.
- **Invariants to Preserve**: The concatenated set of emitted edges `(rows, cols, intersections)` must match the unchunked output exactly in values, ordering, and `int32` dtypes. Self-domain pairs must remain filtered out. Shard naming and disk formats must remain identical.
- **Reason**: A single dense target with thousands of overlapping hits can generate millions of pairwise edges, causing worker resident set size (RSS) to spike quadratically before sharding occurs.
- **Risks**: Increased Python loop overhead on small targets if the fast path is bypassed. Ensure targets with few spans bypass chunking logic entirely.
- **Required Tests and Measurements**:
  - Compare emitted edge multisets and summed intersection lengths between chunked and unchunked implementations across random and dense interval sets.
  - Stress-test a synthetic target with 5,000 mutually overlapping spans. Measure worker peak RSS to confirm memory remains bounded by the chunk threshold.
  - Verify that end-to-end GFF3 output on benchmark datasets is bitwise identical.

## Step 4: Correct Pipeline Documentation and Align Examples
- **Target File**: `README.md`
- **Intended Modification**:
  - Correct the `find_domains` code example to reflect the actual function signature: `find_domains(fasta_path, attributes, output_gff, tmp_dir, result_dir, cpu)`.
  - Correct the MMseqs2 `--format-output` column order in documentation to match the parser's expected standard BLAST-TAB field order (qstart, qend, tstart, tend at columns 6–9).
  - Clarify that `ShardedEdgeWriter` is an incremental flush mechanism rather than a strict 500 MB worker RSS ceiling.
  - Document that the MMseqs2 parser excludes all same-sequence hits as a comparative genomics policy.
  - Accurately describe DPC halo filtering defaults (`halo_method="similarity_weighted"`, `point_specific=False`).
- **Invariants to Preserve**: All application code remains unchanged.
- **Reason**: Eliminates discrepancies between documentation, examples, and the verified codebase.
- **Required Tests and Measurements**:
  - Execute all Python and shell code snippets from `README.md` in a clean environment to verify syntax and runtime validity.

---

# Open Questions and Experimental Protocols

For architectural proposals that remain uncertain, the following structured experiments must be conducted before any code changes are introduced:

## Experiment: Edge Multiplicity (Q / E) and Memory Profiling on Real Datasets
- **Objective**: Measure the empirical ratio of raw edge contributions `Q` to unique domain pairs `E`, and determine whether Stage 4 worker generation or Stage 5 parent reduction dominates peak memory on biological data.
- **Protocol**: Instrument `domain_finder.py` to record: total target hits `N`, unique coordinates `U`, raw overlap edges `Q`, unique domain pairs `E`, total shard bytes written, parent process RSS, and per-worker peak RSS across Stages 2, 4, and 5. Run on representative biological datasets (e.g., bacterial transcriptomes, eukaryotic ncRNA pools).
- **Decision Criterion**: If `Q / E` is small (< 2.0) and Stage 5 memory remains under 1 GB, retain the current in-memory reduction. If `Q / E` is large (> 10.0) and Stage 5 memory exceeds system limits, implement an external sorted-run merge or chunked key reduction.

## Experiment: Density Tie Policy and Cross-Backend Parity
- **Objective**: Establish whether dense NumPy and sparse SciPy DPC can achieve reliable numerical parity on real RNA similarity graphs.
- **Protocol**: On similarity matrices derived from biological runs, identify all pairs of nodes with identical local density within floating-point epsilon (1e-12). Compare cluster assignments between NumPy and SciPy under the current strict lexicographical rank vs a tolerance-based rank that groups tied densities into identical bins.
- **Decision Criterion**: If tolerance-based ranking eliminates partition differences without altering biological cluster validity, document the policy and consider dynamic backend dispatch. If differences persist due to graph path topology, retain separate static backends.

## Experiment: Biological Validation of Halo and Similarity Policies
- **Objective**: Evaluate whether point-specific halo filtering or co-occurrence normalized similarities improve ncRNA seed alignment quality.
- **Protocol**: Assemble a benchmark of curated, diverse Rfam reference families with known secondary structures and genomic coordinates (including multi-copy tRNAs, riboswitches, snoRNAs, and repeat-embedded elements). Run the pipeline comparing:
  - Baseline Jaccard vs co-occurrence normalized similarity.
  - Traditional cluster-wide halo vs point-specific halo vs disabled halo (`halo_method=None`).
  Evaluate boundary accuracy (F1-score at 80% reciprocal overlap), family purity, fragmentation, and noise classification rates.
- **Decision Criterion**: Adopt alternative halo or similarity metrics only if curated reference recall improves significantly without causing false merges between distinct structural families. Retain current defaults as baseline.

---

# Stop and Report Directives for Future Implementation

A future coding agent must stop execution and report to the user if any of the following conditions are encountered:
1. The working tree diverges from the verified SHA-256 file hashes recorded in this document, or core function signatures in `domain_finder.py`, `clustering.py`, or `aligners.py` have changed.
2. A proposed optimization produces any divergence in canonical domain pair-to-sum maps, per-domain coverage lengths, or exported GFF3 coordinates on regression test suites.
3. Measurement noise or hardware variability prevents reliable verification of memory or runtime improvements across at least 5 benchmark repetitions.
4. An instruction requires altering biological clustering defaults, coordinate conventions (0-based half-open), integer data types (`int32` coordinates/indices, `int64` keys/sums), or multiprocessing start methods (`forkserver`) without explicit user authorization.
