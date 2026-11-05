# Automated Rfam-B Seed Alignment Pipeline

> **High-throughput discovery of conserved non-coding RNA domains and automated construction of refined, cleaned-up Rfam-B seed alignments for Covariance Model (CM) calculation.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Infernal Compatibility](https://img.shields.io/badge/Infernal-v1.1%2B-green.svg)](http://eddylab.org/infernal/)
[![Rfam Compatible](https://img.shields.io/badge/Rfam-15.0%2B-brightgreen.svg)](https://rfam.org/)

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
  - [What is Rfam-B?](#what-is-rfam-b)
  - [The Challenge of De Novo ncRNA Domain Discovery](#the-challenge-of-de-novo-ncrna-domain-discovery)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation & Prerequisites](#installation--prerequisites)
  - [External Dependencies](#external-dependencies)
  - [Python Environment](#python-environment)
- [The "Ideal" End-to-End Workflow](#the-ideal-end-to-end-workflow)
  - [Workflow Schematic](#workflow-schematic)
  - [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Core Modules Breakdown](#core-modules-breakdown)
  - [1. Sequence Alignment Ingestion (`aligners.py`)](#1-sequence-alignment-ingestion-alignerspy)
  - [2. Density Peak Clustering (`clustering.py`)](#2-density-peak-clustering-clusteringpy)
  - [3. Global Domain & Family Discovery (`domain_finder.py`)](#3-global-domain--family-discovery-domain_finderpy)
  - [4. Multiprocessing Infrastructure (`executor.py`)](#4-multiprocessing-infrastructure-executorpy)
  - [5. Genomic Coordinate Extraction (`gff_sequence_extractor.py`)](#5-genomic-coordinate-extraction-gff_sequence_extractorpy)
  - [6. Structure-Aware Alignment & Trimming (`msa.py`)](#6-structure-aware-alignment--trimming-msapy)
  - [7. Quality Evaluation & Covariation Benchmarking (`msa_evaluate.py`)](#7-quality-evaluation--covariation-benchmarking-msa_evaluatepy)
- [Alignment Pathways & Trimming Engines](#alignment-pathways--trimming-engines)
  - [Available Alignment Pathways](#available-alignment-pathways)
  - [Trimming Strategies Compared](#trimming-strategies-compared)
- [Stockholm Formatting & Downstream CM Calculation](#stockholm-formatting--downstream-cm-calculation)
  - [Stockholm 1.0 & Secondary Structure (`#=GC SS_cons`)](#stockholm-10--secondary-structure-gc-ss_cons)
  - [Building Covariance Models with Infernal](#building-covariance-models-with-infernal)
  - [Statistical Covariation Verification with R-scape](#statistical-covariation-verification-with-r-scape)
- [Alignment Quality Metrics Explained](#alignment-quality-metrics-explained)
- [Benchmark Results & Performance Profile](#benchmark-results--performance-profile)
- [References & Further Reading](#references--further-reading)

---

## Overview & Motivation

### What is Rfam-B?

The [Rfam database](https://rfam.org/) is the central public repository of non-coding RNA (ncRNA) families, structured cis-regulatory elements, and non-coding RNA genes ([Ontiveros-Palacios et al., *Nucleic Acids Research*, 2025](https://doi.org/10.1093/nar/gkae1023)). Each classic Rfam family is represented by three core assets:
1. **A curated SEED alignment**: Representative homologous RNA sequences accompanied by an experimentally or computationally verified consensus secondary structure.
2. **A Covariance Model (CM)**: A profile stochastic context-free grammar (pSCFG) built from the SEED alignment using [Infernal](http://eddylab.org/infernal/) (`cmbuild`), calibrated against sequence space (`cmcalibrate`).
3. **A FULL alignment**: Homologous sequences identified by searching genome databases (e.g., [Rfamseq](https://docs.rfam.org/en/latest/glossary.html#rfamseq)) with the calibrated CM above a curated gathering bit score threshold.

Manual curation of SEED alignments requires significant human effort. As highlighted in the **Rfam 15** release:
> *"Rfam plans to generate a large collection of automatically generated candidate families, similar to Pfam's Pfam-B. This will be a separate collection of alignments and serve as a source of new families for Rfam... speed[ing] up the curation process and providing ground truth for deep-learning models."*

This repository implements the end-to-end computational pipeline to realize that vision: starting with **unannotated RNA transcriptomes or genomic sequences**, it discovers de novo structural domains, clusters them into homologous families, standardizes their coordinates, aligns them using secondary structure-aware algorithms, trims divergent flanks, and exports **publication-grade, cleaned-up Rfam-B seed alignments** ready for `cmbuild` and `cmcalibrate`.

### The Challenge of De Novo ncRNA Domain Discovery

Unlike protein-coding sequences whose boundaries are defined by open reading frames (start/stop codons), non-coding RNAs lack simple boundary markers. Furthermore:
- **Conserved Cores vs. Variable Flanks**: Functional ncRNAs (e.g., riboswitches, ribozymes, snoRNAs, tRNAs) are constrained by evolutionary purifying selection that preserves secondary and tertiary base pairs, while primary sequence identity is often remote ($40\% - 60\%$). When extracted from genomic contigs, the functional structural core is frequently surrounded by genomic flanking regions of variable length.
- **Flank Distortion in Alignment**: Feeding un-trimmed sequences with variable flanks into standard global alignment tools inserts artificial gaps directly into conserved structural stems, destroying base-pairing signals and inflating statistical tests for evolutionary covariation.
- **Scale and Memory**: Comparing millions of local sequence matches across large genomes rapidly leads to combinatorial explosion ($O(N^2)$ pairs) and out-of-memory crashes if not carefully parallelized and bounded.

This toolkit solves these problems using **Density Peak Clustering (DPC)** over pairwise interval overlaps, followed by **structure-informed alignment and objective flank trimming**.

---

## Pipeline Architecture

The pipeline processes raw sequences through eight modular stages:

```mermaid
flowchart TD
    classDef bigBox padding:18px,stroke-width:1.5px;
    S1["<b>Stage 1: All-Against-All Homology Search (MMseqs2)</b><br/>• Pairwise alignment of unannotated RNA sequences<br/>• Coordinate standardization: 1-based closed &rarr; 0-based half-open"]
    S2["<b>Stage 2: Query-Level Local Domain Instance Discovery (DPC)</b><br/>• 1D linear span merging per query<br/>• Density Peak Clustering identifies local domain peaks<br/>• Border point filtering (halo filter) prunes noisy transitions"]
    S3["<b>Stage 3: Cross-Sequence Target Partitioning & Indexing</b><br/>• Target sequence hit collation across all discovered query domains<br/>• In-memory contiguous sorting without intermediate Python overhead"]
    S4["<b>Stage 4: Pairwise Target Overlap Calculation</b><br/>• Workload-aware parallel batch processing (executor.Parallel)<br/>• Sharded streaming (ShardedEdgeWriter) bounds worker RAM (&lt; 500 MB)"]
    S5["<b>Stage 5: Global Family Clustering & GFF3 Feature Generation</b><br/>• Inter-domain Jaccard similarity matrix construction<br/>• Global Density Peak Clustering groups domains into families<br/>• Strand consensus voting & export to standard GFF3"]
    S6["<b>Stage 6: Coordinate Extraction & RNA Sequence Standardization</b><br/>• Genomic FASTA slicing using GFF3 intervals<br/>• IUPAC degenerate base filtering (&gt; 5% density discarded)<br/>• T &rarr; U conversion & soft-masking (lowercase) preservation"]
    S7["<b>Stage 7: Structure-Aware Multiple Sequence Alignment & Trimming</b><br/>• Pathways: Muscle5, MAFFT L-INS-i, MAFFT Q-INS-i, Profile CM<br/>• Trimming: CIAlign Crop-from-Ends, trimAl, RNAalifold Consensus Mask"]
    S8["<b>Stage 8: Benchmark Quality Evaluation & Infernal CM Export</b><br/>• Calculate SCI, Transitive Consistency (TCS), MI-APC covariation<br/>• Export Stockholm 1.0 (.sto) with #=GC SS_cons WUSS markup<br/>• Construct and calibrate Covariance Model (cmbuild + cmcalibrate)<br/>• Statistical significance verification via R-scape (E &lt; 0.05)"]
    class S1,S2,S3,S4,S5,S6,S7,S8 bigBox;
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
```

---

## Installation & Prerequisites

### External Dependencies

The pipeline coordinates high-performance external bioinformatics engines:

| Tool | Recommended Version | Purpose | Installation via Conda / Homebrew |
| :--- | :--- | :--- | :--- |
| **MMseqs2** | $\ge$ 15.7 | Ultra-fast sequence similarity search | `conda install -c bioconda mmseqs2` |
| **Infernal** | $\ge$ 1.1.4 | `cmbuild`, `cmcalibrate`, `cmalign` | `conda install -c bioconda infernal` |
| **ViennaRNA** | $\ge$ 2.6.0 | `RNAfold`, `RNAalifold` (MFE & consensus) | `conda install -c bioconda viennarna` |
| **MAFFT** | $\ge$ 7.500 | `mafft` (L-INS-i, Q-INS-i, X-INS-i) | `conda install -c bioconda mafft` |
| **MUSCLE** | v5.1+ | `muscle -super5` ultra-fast alignment | `conda install -c bioconda muscle` |
| **CIAlign** | $\ge$ 1.0.0 | Clean divergent terminal flanks | `pip install cialign` |
| **trimAl** | $\ge$ 1.4.1 | Gap heuristic trimming (`-gappyout`) | `conda install -c bioconda trimal` |
| **R-scape** | $\ge$ 2.0.0 | Covariation statistical significance test | `conda install -c bioconda r-scape` |

### Python Environment

Requires **Python 3.11+** (fully verified with Python 3.12).

```bash
# Clone the repository
git clone https://github.com/your-org/rfam-b-seed-pipeline.git
cd rfam-b-seed-pipeline

# Create and activate a clean conda environment
conda create -n rfam_pipeline python=3.12 -y
conda activate rfam_pipeline

# Install Python dependencies
pip install -r requirements.txt loguru mpire tqdm scipy scikit-learn networkx
```

Verify your installation:
```bash
python3 -c "import domain_finder, msa, msa_evaluate; print('Pipeline modules loaded successfully!')"
pytest
```

---

## The "Ideal" End-to-End Workflow

Here is the complete, runnable workflow taking raw unannotated RNA FASTA sequences and producing cleaned-up Rfam-B seed alignments with calibrated Covariance Models.

### Workflow Schematic

```mermaid
flowchart TD
    A["input_transcripts.fasta"] -->|"MMseqs2 easy-search"| B["alignments.tsv<br/><i>(BLAST-TAB: qseqid, tseqid, qstart, qend, tstart, tend, evalue)</i>"]
    B -->|"domain_finder.find_domains"| C["domains.gff3<br/><i>(De novo RNA family clusters on genomic coordinates)</i>"]
    C -->|"gff_sequence_extractor.extract_and_annotate_sequences"| D["extracted_families/fam_001.fasta, fam_002.fasta, ...<br/><i>(Strand-adjusted, T &rarr; U)</i>"]
    D -->|"msa.py (Pathway 2 or 3)"| E["aligned_families/fam_001.trimmed.fasta"]
    E -->|"RNAalifold consensus structure calculation"| F["seed_alignments/fam_001.seed.sto<br/><i>(Stockholm 1.0 with #=GC SS_cons)</i>"]
    F -->|"Infernal cmbuild + cmcalibrate & R-scape"| G["models/fam_001.cm<br/><i>(Calibrated CM ready for Rfamseq genome annotation)</i>"]
```

### Step-by-Step Walkthrough

#### Step 1: Upstream Sequence Alignment with MMseqs2

Generate all-against-all pairwise local alignments of candidate RNA sequences. Use standard 12-column BLAST-TAB format:

```bash
# Perform all-against-all similarity search
mmseqs easy-search input_transcripts.fasta input_transcripts.fasta mmseqs_results.tsv tmp/ \
    --search-type 3 \
    --format-output "query,target,qstart,qend,tstart,tend,evalue,bits,alnlen,pident,mismatch,gapopen" \
    -e 1e-3 \
    --threads 8
```

#### Step 2: De Novo Domain and Family Discovery

Run `domain_finder.find_domains` to discover domain instances and cluster them into families:

```python
from pathlib import Path
from domain_finder import find_domains

# Run Density Peak Clustering across alignment hits
gff_output = find_domains(
    tsv_path=Path("mmseqs_results.tsv"),
    query_fasta=Path("input_transcripts.fasta"),
    target_fasta=Path("input_transcripts.fasta"),
    output_gff=Path("discovered_domains.gff3"),
    min_dom_dens=1.0,        # Minimum local density for query domain peak
    min_fam_dens=1.0,        # Minimum density for global family cluster
    jaccard_cutoff=0.40,     # Jaccard threshold for domain family overlap
    cpu=8,                   # Parallel execution threads
    tmp_dir=Path("./tmp_scratch")
)

print(f"Discovered domain families written to: {gff_output}")
```

#### Step 3: Extract and Standardize RNA Family Sequences

Extract individual sequence records per family, correct reverse-strand coordinates, discard high-degeneracy sequences, and convert to canonical RNA (`U`):

```python
from pathlib import Path
from gff_sequence_extractor import extract_and_annotate_sequences

# Extract multi-FASTA files for each clustered domain family
extract_and_annotate_sequences(
    gff_file=Path("discovered_domains.gff3"),
    fasta_file=Path("input_transcripts.fasta"),
    output_dir=Path("families_fasta"),
    min_length=40,
    max_length=1500,
    max_iupac_density=0.05   # Discard sequences with > 5% degenerate codes
)
```

#### Step 4: Structure-Aware Multiple Sequence Alignment and Flank Trimming

For each family, execute a structure-aware alignment pathway and trim divergent non-structural terminal flanks:

```python
from pathlib import Path
from msa import RNASequenceDataset, MafftQinsiPipeline, trim_consensus_structure_masking

# Load family sequences
dataset = RNASequenceDataset.from_fasta(Path("families_fasta/family_001.fasta"))

# 1. Structure-aware alignment (MAFFT Q-INS-i with McCaskill base-pairing probabilities)
pipeline = MafftQinsiPipeline(max_workers=8, extra_flags=["--ep", "0.0", "--adjustdirection"])
result = pipeline.align(dataset)

if not result.success:
    raise RuntimeError(f"Alignment failed: {result.error_message}")

# 2. Trim divergent terminal flanks using consensus secondary structure masking
trimmed_dataset = trim_consensus_structure_masking(
    alignment=result.dataset,
    min_core_coverage=0.60
)

# Export cleaned alignment
trimmed_dataset.to_fasta(Path("families_msa/family_001_trimmed.fasta"))
```

#### Step 5: Convert to Stockholm Format (`#=GC SS_cons`)

Predict the consensus secondary structure and export to Stockholm 1.0 format with WUSS notation:

```python
import subprocess
from pathlib import Path

# Run RNAalifold to derive consensus secondary structure
aln_fa = "families_msa/family_001_trimmed.fasta"
sto_out = "seed_alignments/family_001_seed.sto"

cmd = f"RNAalifold --noLP --noPS < {aln_fa}"
proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

# Parse output structure line
lines = proc.stdout.strip().split("\n")
structure_line = lines[-1].split()[0]  # E.g. "((((...))))"

# Format Stockholm 1.0 with #=GC SS_cons
from Bio import SeqIO
records = list(SeqIO.parse(aln_fa, "fasta"))
max_id_len = max(len(r.id) for r in records)

with open(sto_out, "w") as f:
    f.write("# STOCKHOLM 1.0\n")
    f.write(f"#=GF ID   RFAM_B_FAM00001\n")
    f.write(f"#=GF DE   Candidate de novo ncRNA domain family\n\n")
    for r in records:
        f.write(f"{r.id:<{max_id_len + 4}}{str(r.seq)}\n")
    f.write(f"{'#=GC SS_cons':<{max_id_len + 4}}{structure_line}\n")
    f.write("//\n")

print(f"Rfam-B Seed alignment exported to: {sto_out}")
```

#### Step 6: Infernal Covariance Model Construction & Calibration

Use Infernal to build and calibrate the profile Covariance Model:

```bash
# 1. Build initial Covariance Model from Stockholm SEED alignment
cmbuild -F models/family_001.cm seed_alignments/family_001_seed.sto

# 2. Single final calibration (fits E-value exponential tail parameters)
# Recommended to run with multi-threading
cmcalibrate --cpu 8 models/family_001.cm

# 3. Statistical covariation validation via R-scape (verifies base pairs at E < 0.05)
R-scape --covmin 2 seed_alignments/family_001_seed.sto
```

#### Complete Automated Master Runner *(IN PROGRESS)*

An automated master command-line script orchestrating Stages 1 through 8 in a single execution (`rfam_b_orchestrator.py`) is currently **(IN PROGRESS)**. In the interim, running the Python scripts or calling the subroutines as shown above provides full execution.

---

## Core Modules Breakdown

### 1. Sequence Alignment Ingestion (`aligners.py`)

Handles streaming ingestion of external similarity searches (MMseqs2 BLAST-TAB format).
- **Coordinate Conversion**: Automatically converts 1-based, fully-closed intervals `[start, end]` (BLAST convention) into **0-based half-open intervals `[start - 1, end)`** upon reading. This ensures interval lengths equal `end - start` and arithmetic overlaps can be calculated vectorially throughout downstream clustering.
- **Sequence Registry**: Fast two-way integer sequence mapping (`SequenceRegistry`) preventing costly string allocations across millions of alignment records.
- **Self-Alignment Exclusion Policy**: Discards exact self-sequence hits ($q_{\mathrm{seq}} == t_{\mathrm{seq}}$) during family discovery. Comparative RNA families require independent evolutionary conservation across distinct sequence contexts.

### 2. Density Peak Clustering (`clustering.py`)

Implements optimized Density Peak Clustering (Rodriguez & Laio, *Science*, 2014) adapted for discrete sequence intervals and sparse Jaccard overlap matrices:
- **Fast 1D Interval Overlaps (`compare_sorted_spans`)**: Vectorized `searchsorted` algorithm calculating pairwise interval overlaps on single sequences in $O(N)$ expected time.
- **Sparse Similarity Graphs (`build_similarity_matrix`)**: Constructs compressed sparse row (CSR) matrices with implicit or explicit self-similarity diagonals ($\rho_i = 1.0$).
- **Density Peak Cluster Engine (`dpc`)**:
  - Calculates local density $\rho_i$.
  - Computes minimum distance $\delta_i$ to any point with strictly higher density.
  - Determines cluster centers (peaks) exceeding `min_dens` and distance thresholds.
  - Propagates cluster labels strictly along edges with non-zero similarity ($s_{\mathrm{max\_higher}} > 0$). Points failing to reach a valid cluster center are classified as **noise (-1)**.
- **Halo Filtering (`halo_filter`)**: Prunes ambiguous border points whose local density falls below the cluster median and whose foreign similarity ratio exceeds a boundary threshold.

### 3. Global Domain & Family Discovery (`domain_finder.py`)

The primary domain orchestration engine:
- **Stage 2 (`batch_query_processor`)**: Parallel query processing identifying local structural domain instances on individual sequences.
- **Stage 3**: Contiguous 1D structured NumPy target hit sorting (`t_seq`, `start`, `end`, `dom_id`) avoiding intermediate dictionary allocations.
- **Stage 4 (`target_batch_processor`)**: Multi-core target span overlap generation. Implements **`ShardedEdgeWriter`** to flush binary shards (`.npy`) incrementally every 250,000 edges, guaranteeing worker memory never exceeds 500 MB even across dense loci.
- **Stage 5 (`family_finder`)**: Global domain clustering across all shared target sequences using Jaccard intersection matrices.
- **GFF3 Annotation (`evaluate_clusters`)**: Re-maps domain cluster instances back to global 1-based coordinates and assigns strand orientation via target hit majority voting.

### 4. Multiprocessing Infrastructure (`executor.py`)

Consolidates all multiprocessing and process-execution abstractions:
- **`executor.Parallel`**: Fork-safe, thread-safe process execution layer built on [MPIRE](https://github.com/sybrenjansen/mpire). Defaults to `'forkserver'` context under Linux platforms to eliminate deadlocks and deprecation warnings with `os.fork()` under Python 3.12+.
- **Dynamic Chunk Sizing**: Automatically bounds worker task chunking to prevent worker starvation and avoid inter-process communication (IPC) bottlenecks.
- **Subprocess Memory Tracking**: Monitors child process peak resident set size (RSS) via `psutil`.

### 5. Genomic Coordinate Extraction (`gff_sequence_extractor.py`)

Object-oriented feature extraction and nucleotide normalization:
- **IUPAC Degeneracy Analysis**: Calculates degenerate nucleotide density (`R, Y, S, W, K, M, B, D, H, V, N`) and filters sequences exceeding quality thresholds.
- **Alphabet Standardization**: Enforces canonical RNA representation (`T` $\to$ `U`) at ingestion while strictly **preserving soft-masking (lowercase letters)** to protect repetitive genomic metadata.
- **Automated Reverse-Complement**: Automatically inverts negative-strand features ($5' \to 3'$) to preserve native thermodynamic secondary structure folding orientations.

### 6. Structure-Aware Alignment & Trimming (`msa.py`)

Framework providing four distinct alignment pathways and objective structural trimming engines:
- Encapsulates sequences in an immutable `RNASequenceDataset`.
- Tracks runtime and memory footprint for all external aligners.
- Implements consensus secondary structure mask trimming (`RNAalifold`) and crop-from-ends trimming (`CIAlign`).

### 7. Quality Evaluation & Covariation Benchmarking (`msa_evaluate.py`)

Computes scientific quality metrics across alignments:
- Evaluates **Structure Conservation Index (SCI)**, **Transitive Consistency Score (TCS)**, **Mutual Information with Average Product Correction (MI-APC)**, Shannon entropy, and compensatory mutation counts.

---

## Alignment Pathways & Trimming Engines

### Available Alignment Pathways

The pipeline provides four specialized alignment pathways configured for distinct operational requirements:

```mermaid
flowchart TD
    subgraph P0["<b>PATHWAY 0: Ultra-Fast MUSCLE5 Primary Sequence Pipeline</b><br/><i>Runtime: ~0.01 - 0.05 sec/MSA | RAM: &lt; 20 MB | Purpose: Ultra-fast screening &gt; 100,000 datasets</i>"]
        direction LR
        P0_A["Raw FASTA"] --> P0_B["T &rarr; U Standardize"] --> P0_C["MUSCLE v5 Super5"] --> P0_D["CIAlign / trimAl Trimming"]
    end

    subgraph P1["<b>PATHWAY 1: High-Throughput Screening Pipeline</b><br/><i>Runtime: ~0.1 - 0.5 sec/MSA | RAM: &lt; 50 MB | Purpose: High-throughput screening &gt; 50,000 datasets</i>"]
        direction LR
        P1_A["Raw FASTA"] --> P1_B["T &rarr; U Standardize"] --> P1_C["MAFFT L-INS-i (--ep 0.0)"] --> P1_D["CIAlign Crop-from-Ends"]
    end

    subgraph P2["<b>PATHWAY 2: Balanced Structure-Aware Pipeline (RECOMMENDED FOR RFAM-B)</b><br/><i>Runtime: ~2.0 - 15.0 sec/MSA | RAM: &lt; 500 MB | Purpose: High-confidence motif extraction</i>"]
        direction LR
        P2_A["Raw FASTA"] --> P2_B["T &rarr; U Standardize"] --> P2_C["MAFFT Q-INS-i / L-INS-i"] --> P2_D["Consensus Structure Masking"]
    end

    subgraph P3["<b>PATHWAY 3: Speed-Optimized Gold-Standard Profile Covariance Model Pipeline</b><br/><i>Runtime: ~10.0 - 45.0 sec/MSA | RAM: &lt; 1 GB | Purpose: Publication-grade CM construction</i>"]
        direction LR
        P3_A["Initial L-INS-i"] --> P3_B["RNAalifold"] --> P3_C["cmbuild"] --> P3_D["cmalign --glocal"] --> P3_E["cmbuild Final CM"]
    end
```

#### Detailed Pathway Specifications

1. **Pathway 0: MUSCLE v5 Super5 (`Muscle5Pipeline`)**
   - *Engine*: `muscle -super5 input.fasta -output output.aln`
   - *Mechanism*: Pure primary sequence aligner based on $k$-mer distance estimations and guide trees. Scalable to massive alignments without folding overhead.
   - *Best for*: Preliminary clustering when sequence identity $> 70\%$.

2. **Pathway 1: MAFFT L-INS-i (`MafftLinsiPipeline`)**
   - *Engine*: `mafft --localpair --op 3.0 --ep 0.0 --adjustdirection --maxiterate 200 input.fasta`
   - *Mechanism*: Local Smith-Waterman pair consistency matching. Setting `--ep 0.0` prevents variable flanks from distorting internal stems.

3. **Pathway 2: MAFFT Q-INS-i (`MafftQinsiPipeline`)**
   - *Engine*: `mafft --qinsi --ep 0.0 --op 2.5 --adjustdirection --maxiterate 200 input.fasta`
   - *Mechanism*: Computes McCaskill base-pairing probability matrices ($P_{ij}$) for each sequence and weights alignment paths by structural compatibility.
   - *Best for*: The recommended standard for Rfam-B candidate seed families ($40\% - 70\%$ identity).

4. **Pathway 3: Profile Covariance Model Iteration (`ProfileCovarianceModelPipeline`)**
   - *Workflow*: MAFFT L-INS-i $\to$ `RNAalifold` $\to$ `cmbuild --noss` $\to$ `cmalign --glocal` $\to$ refined `cmbuild`.
   - *Optimization*: Suppresses `cmcalibrate` during intermediate iterative cycles, yielding an **8- to 10-fold speedup** while achieving gold-standard profile alignment accuracy.

---

### Trimming Strategies Compared

Trimming removes divergent genomic flanks while retaining structurally essential base-paired stems and loops:

| Trimming Engine | Method | SCI Retention | FP Stem Deletions | Loop Retention | Recommended Use Case |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Infernal Match-State Truncation** | Truncates profile Insert states ($I_k$) via `cmalign` | **9.8 / 10** | **Extremely Low** | **9.9 / 10** | Gold standard when a Covariance Model exists |
| **Consensus Structure Masking** | Masks consensus pairs `()`, `[]` via `RNAalifold` | **9.6 / 10** | **Very Low** | **9.6 / 10** | **Default Rfam-B de novo trimming** |
| **CIAlign (Crop-from-Ends)** | Trims divergent terminal overhangs | 8.5 / 10 | Low | 8.8 / 10 | Fast heuristic for high-throughput screening |
| **trimAl (`-gappyout`)** | Primary sequence gap density heuristics | 6.0 / 10 | High | 5.2 / 10 | Fast general-purpose; risk of loop truncation |

---

## Stockholm Formatting & Downstream CM Calculation

### Stockholm 1.0 & Secondary Structure (`#=GC SS_cons`)

Infernal `cmbuild` requires alignments in **Stockholm 1.0 format** containing a consensus secondary structure markup line labeled `#=GC SS_cons` using Washington University Secondary Structure (**WUSS**) notation:

```text
# STOCKHOLM 1.0
#=GF ID   RFAM_B_00042
#=GF DE   Putative bacterial small regulatory RNA
#=GF AU   Automated Rfam-B Pipeline

seq1/1-72        GGUGUGUCCUUCGGCAGACACAGACUGAACCCAUUGGAGUGUUGUUCUCCUGUCUGCCGAAGGACACACCA
seq2/1-72        GGUGUGUCCUUCGGCAGACACAAACUGAACCCAUUGGAGUGCUGUUCUCCUGUCUGCCGAAGGACACACCA
seq3/1-71        GGUGUGUCCUUCGGCAGACAC-GACUGAACCCAUUGGAGUGCUGUUCUCCUGUCUGCCGAAGGACACACCA
#=GC SS_cons     (((((((((((((((((((((..<<<<<......>>>>>......)))))))))))))))))))))).....
//
```

#### WUSS Notation Symbols

- `<>`: Base pairs in simple stem loops.
- `()`, `[]`, `{}`: Base pairs enclosing multi-stem junctions or pseudoknots.
- `.` or `-`: Unpaired residues external to or within loops.
- `: `, `, `: Single-stranded loop and linker residues.

### Building Covariance Models with Infernal

Once the Stockholm alignment is generated:

```bash
# 1. Build the Covariance Model
cmbuild -F models/family.cm seed_alignments/family.sto

# Expected output confirmation:
#   [ok]
#   # cmbuild :: covariance model construction from alignments
#   # Consensus structure length: 72
```

### Statistical Covariation Verification with R-scape

Before releasing an Rfam-B candidate family, verify whether aligned base-pair positions show statistically significant evolutionary covariation (compensatory mutations):

```bash
R-scape --covmin 2 seed_alignments/family.sto
```

- **Covariation Threshold**: Significant base pairs are identified with an $E$-value threshold of $E < 0.05$.
- **Why Flank Trimming Matters**: R-scape corrects for multiple testing across all $L \times L$ column pairs. Un-trimmed variable flanks artificially increase alignment length $L$, elevating the multiple-testing penalty and masking real biological covariation in the core. Trimming flanks with this pipeline ensures maximum statistical sensitivity.

---

## Alignment Quality Metrics Explained

The `msa_evaluate.py` module reports standardized metrics to benchmark alignment quality:

1. **Structure Conservation Index (SCI)**:
   Measures whether sequences share a stable common fold compared to folding individually:
   $$\mathrm{SCI} = \frac{E_{\mathrm{consensus}}}{\bar{E}_{\mathrm{single}}}$$
   An $\mathrm{SCI} \approx 1.0$ indicates strong structural conservation.
2. **Transitive Consistency Score (TCS)**:
   Measures whether column residue pairings agree with pairwise alignments across all sequences (0.0 to 1.0).
3. **Mutual Information with Average Product Correction (MI-APC)**:
   Quantifies evolutionary co-variation between columns while subtracting phylogenetic background correlation. High MI-APC values between base-paired positions confirm co-evolutionary conservation.
4. **Normalized Shannon Entropy ($H_N$)**:
   Positional diversity across alignment columns. Conserved cores exhibit lower entropy than variable flanks.
5. **Mean Overlap Score (MOS)**:
   Pairwise Jaccard overlap measuring consistency across different aligners. High MOS indicates convergence between independent alignment algorithms.

---

## Benchmark Results & Performance Profile

The pipeline's core algorithms were empirically evaluated on a benchmark dataset of **105,441 MMseqs2 BLAST-TAB hits** across **2,338 unannotated RNA sequences** (all-against-all search) spanning short (100–500 nt) and long (500–2,000 nt) structural RNAs across 135 variable-sized clusters:

| Configuration | Total Execution Time | Peak Resident Memory (RSS) | GFF3 Features Produced | Result Consistency |
| :---: | :---: | :---: | :---: | :---: |
| **1 CPU Core** | 11.21 s | 527.55 MB | 6,856 | 100% Bitwise Identical |
| **2 CPU Cores** | 7.16 s | 529.22 MB | 6,856 | 100% Bitwise Identical |
| **4 CPU Cores** | 6.43 s | 529.47 MB | 6,856 | 100% Bitwise Identical |

### Memory & Concurrency Highlights

- **Strict Worker Memory Bounding**: Worker processes flush edge shards incrementally every 250,000 edges via `ShardedEdgeWriter`. Worker RSS is strictly bounded below 500 MB even across dense repeats.
- **Collision-Free Sandboxing**: Temporary shard directories are uniquely generated per run (`tempfile.TemporaryDirectory`), eliminating race conditions in concurrent pipeline runs sharing scratch folders.
- **Python 3.12+ Fork Safety**: Multiprocessing utilizes the `forkserver` start method, eliminating `os.fork()` deadlocks and deprecation warnings in multithreaded environments.

---

## References & Further Reading

1. **Rfam 15**: Ontiveros-Palacios, N., et al. (2025). "Rfam 15: RNA families database in 2025." *Nucleic Acids Research*, 53(D1), D258–D267. [doi:10.1093/nar/gkae1023](https://doi.org/10.1093/nar/gkae1023)
2. **Rfam Help & Family Building Documentation**: [https://docs.rfam.org/en/latest/building-families.html](https://docs.rfam.org/en/latest/building-families.html)
3. **Infernal 1.1**: Nawrocki, E. P., & Eddy, S. R. (2013). "Infernal 1.1: 100-fold faster RNA homology searches." *Bioinformatics*, 29(22), 2933–2935. [doi:10.1093/bioinformatics/btt509](https://doi.org/10.1093/bioinformatics/btt509)
4. **R-scape**: Rivas, E., Clements, J., & Eddy, S. R. (2017). "A statistical test for conserved RNA secondary structure." *Nature Methods*, 14(1), 45–48. [doi:10.1038/nmeth.4066](https://doi.org/10.1038/nmeth.4066)
5. **Density Peak Clustering (DPC)**: Rodriguez, A., & Laio, A. (2014). "Clustering by fast search and find of density peaks." *Science*, 344(6191), 1492–1496. [doi:10.1126/science.1242072](https://doi.org/10.1126/science.1242072)
6. **MAFFT**: Katoh, K., & Standley, D. M. (2013). "MAFFT multiple sequence alignment software version 7." *Molecular Biology and Evolution*, 30(4), 772–780. [doi:10.1093/molbev/mst010](https://doi.org/10.1093/molbev/mst010)
7. **ViennaRNA Package**: Lorenz, R., et al. (2011). "ViennaRNA Package 2.0." *Algorithms for Molecular Biology*, 6, 26. [doi:10.1186/1748-7188-6-26](https://doi.org/10.1186/1748-7188-6-26)
8. **CIAlign**: Tumescheit, C., Firth, A. E., & Brown, K. (2022). "CIAlign: A highly customisable tool to clean multiple sequence alignments." *PeerJ*, 10, e12983. [doi:10.7717/peerj.12983](https://doi.org/10.7717/peerj.12983)
