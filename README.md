# RFAM_B_domains: Non-Coding RNA Multiple Sequence Alignment & Benchmarking Framework

A modern, production-grade Python 3.10+ framework for benchmarking non-coding RNA (ncRNA) Multiple Sequence Alignment (MSA) tools.

The framework integrates genomic feature extraction, alignment execution across structural and sequence-based aligners, rigorous bioinformatic metric evaluations, automated Excel reporting, and publication-quality visualization routines.

---

## Table of Contents
1. [Overview & Workflow](#overview--workflow)
2. [Installation & Environment Setup](#installation--environment-setup)
3. [Usage](#usage)
4. [Visualization & Analysis](#visualization--analysis)
5. [Theoretical Breakdown of Alignment Methods](#theoretical-breakdown-of-alignment-methods)
6. [Theoretical Breakdown of Evaluation Metrics](#theoretical-breakdown-of-evaluation-metrics)
7. [References & Citations](#references--citations)

---

## Overview & Workflow

The framework operates via a 4-step benchmarking workflow:

1. **Extraction & Coordinate Mapping (`gff_sequence_extractor.py`)**:
   Extracts target genomic intervals defined in primary GFF feature annotations (e.g., predicted clusters) from an input FASTA file (`input.fasta`). Overlapping secondary annotations (e.g., reference Rfam annotations) are coordinate-shifted and strand-adjusted to match the extracted sequence frames. Output files (`.fasta` and `.gff`) are saved per feature ID.
2. **Multiple Sequence Alignment (`msa.py`)**:
   Aligns extracted sequence datasets across all configured aligners (`Muscle5`, `MAFFT Q-INS-i`, `MAFFT L-INS-i`, `MAFFT X-INS-i`, `R-Coffee`, `Structural Encoding`). Exports aligned sequences in FASTA (`.fna`) and Stockholm (`.sto`) formats for manual inspection.
3. **Scientific Quality Evaluation (`msa_evaluate.py`)**:
   Evaluates each alignment output across structural conservation, covariation, consistency, entropy, overlap, sequence identity, and resource utilization metrics (wall-clock runtime and peak memory footprint via `psutil`). Aligner failures or missing CLI dependencies are caught and assigned penalized worst-case metric values (maximum entropy $H_N = 1.0$, all other metrics $= 0.0$).
4. **Data Export & Visualization (`benchmark.py`, `visualize_benchmark.py`)**:
   Aggregates per-alignment evaluation metrics into a pandas DataFrame and exports a multi-sheet Excel workbook (`benchmark_results.xlsx`). Generates violin distribution plots, performance heatmaps, and runtime vs. memory scatter plots.

---

## Installation & Environment Setup

### Prerequisites
* Linux / macOS system environment
* Conda or Mamba package manager

### 1. Environment Setup via Conda / Mamba
Clone the repository and create the Conda environment using `msa_benchmark.yml` or `environment.yml`:

```bash
git clone https://github.com/your-org/RFAM_B_domains.git
cd RFAM_B_domains

# Create and activate environment via msa_benchmark.yml or environment.yml
conda env create -f msa_benchmark.yml
conda activate msa_benchmark
```

The Conda environment automatically installs both Python dependencies (`numpy`, `pandas`, `openpyxl`, `biopython`, `psutil`, `matplotlib`, `seaborn`, `pytest`) and external CLI tools (`mafft`, `muscle`, `t-coffee`, `viennarna`).

### 2. Manual Installation (Pip)
If managing Python dependencies via `pip` in an existing environment:

```bash
pip install -r requirements.txt
```

### 3. External Bioinformatic Tool Dependencies
Ensure the following CLI tools are available in your system `PATH`:
* **MAFFT** ($\ge 7.520$): `mafft`, `mafft-qinsi`, `mafft-linsi`, `mafft-xinsi`
* **MUSCLE** ($\ge 5.1.0$): `muscle`
* **T-Coffee** ($\ge 13.45$): `t_coffee` (required for R-Coffee alignment and TCS evaluation)
* **ViennaRNA Package** ($\ge 2.5.1$): `RNAfold`, `RNAalifold` (required for SCI and structure covariation calculation)

---

## Usage

### 1. Running the Complete Benchmark Suite

To run the automated benchmark workflow:

```bash
python benchmark.py
```

#### Programmatic Usage Example (`benchmark.py`):

```python
from pathlib import Path
from benchmark import run_benchmark_workflow

df = run_benchmark_workflow(
    input_fasta=Path("input.fasta"),
    cluster_gff=Path("cluster.gff"),
    reference_gff=Path("reference.gff"),
    output_dir=Path("benchmark_results"),
    flank_length=0
)
```

### 2. Running Individual Modules Programmatically

#### Sequence Extraction (`gff_sequence_extractor.py`)
```python
from pathlib import Path
from gff_sequence_extractor import GFFSequenceExtractor

extractor = GFFSequenceExtractor(
    fasta_path=Path("input.fasta"),
    primary_gff_path=Path("cluster.gff"),
    secondary_gff_path=Path("reference.gff"),
    flank_length=10
)
seqs, adjusted_gff_df = extractor.extract_and_adjust()
exported_files = extractor.export_data(
    output_dir=Path("extracted_data"),
    extracted_sequences=seqs,
    adjusted_gff_df=adjusted_gff_df
)
```

#### Running MSA Pipelines (`msa.py`)
```python
from pathlib import Path
from msa import RNASequenceDataset, MafftQinsiPipeline

dataset = RNASequenceDataset.from_fasta(Path("extracted_data/cluster_1.fasta"))
pipeline = MafftQinsiPipeline()
result = pipeline.align(dataset, output_path=Path("cluster_1_qinsi.fna"))

if result.is_successful:
    print(f"Alignment successful in {result.execution_time_seconds:.2f}s, Peak RSS: {result.memory_peak_mb:.1f} MB")
else:
    print(f"Alignment failed: {result.error_message}")
```

---

## Visualization & Analysis

To generate publication-grade figures (violin plots, heatmaps, scatter plots) from an evaluation Excel file (e.g. `tests/mock_results.xlsx` or `benchmark_results.xlsx`):

```bash
python visualize_benchmark.py
```

Outputs are saved in `tests/figures/`:
* `metric_distributions_violin.png`: Violin plot distributions for SCI, TCS, MI-APC, and $H_N$.
* `pipeline_performance_heatmap.png`: Heatmap matrix comparing mean metrics across pipelines.
* `runtime_memory_scatter.png`: Scatter plot comparing Wall-Clock Execution Time vs Peak Memory Usage colored by SCI.

---

## Theoretical Breakdown of Alignment Methods

Aligning non-coding RNAs requires methods that consider both primary sequence identity and secondary structure conservation.

### 1. MUSCLE v5 (`muscle5`)
* **Theoretical Basis**: Introduces Progressive Perturbed Pairwise (PPP) alignments and ensemble representations. Generates an ensemble of perturbed alignments to capture alignment uncertainty.
* **Mechanism**: Uses k-mer distance estimations combined with profile-profile comparisons.

### 2. MAFFT Q-INS-i (`mafft_qinsi`)
* **Theoretical Basis**: Incorporates McCaskill secondary structure base-pairing probability matrices calculated from individual sequences into pairwise alignment algorithms.
* **Mechanism**: Uses the McCaskill algorithm (McCaskill, 1990) to construct base-pairing probability matrices $P_{ij}$ for each sequence, aligning sequences by maximizing structural probability consensus across secondary base pairs.

### 3. MAFFT L-INS-i (`mafft_linsi`)
* **Theoretical Basis**: Local pairwise alignment algorithm with maximum consistency refinement.
* **Mechanism**: Constructs local alignment libraries using position-specific gap penalties.

### 4. MAFFT X-INS-i (`mafft_xinsi`)
* **Theoretical Basis**: Structural alignment method utilizing pairwise structural alignment algorithms (e.g., MXSCARNA).
* **Mechanism**: Calculates full stem-candidate pairing matrices across sequence pairs before progressive alignment.

### 5. R-Coffee (`rcoffee`)
* **Theoretical Basis**: Extension of T-Coffee that incorporates secondary structure folding predictions into consistency libraries.
* **Mechanism**: Constructs an RNA structure library to maximize consistency between alignment columns and predicted base pairs.

### 6. Structural Encoding (`structural_encoding`)
* **Theoretical Basis**: Structure-informed alignment pipeline leveraging explicit structural probability matrices (MAFFT Q-INS-i / globalpair).
* **Mechanism**: Operates directly on standardized canonical RNA bases (U-encoding) without case-erasure artifacts.

---

## Theoretical Breakdown of Evaluation Metrics

### 1. Structure Conservation Index (SCI)
* **Formulation**:
  $$\text{SCI} = \frac{E_{\text{consensus}}}{\bar{E}_{\text{single}}}$$
  where $E_{\text{consensus}}$ is the Minimum Free Energy (MFE) of the consensus structure predicted by `RNAalifold`, and $\bar{E}_{\text{single}}$ is the arithmetic mean MFE of individual ungapped sequences predicted by `RNAfold`.

### 2. Transitive Consistency Score (TCS)
* **Definition**: Evaluates the consistency of column residue pairs across all pairwise alignment paths in a multiple sequence alignment ($0 - 100\%$).

### 3. Mutual Information with APC Covariation (MI-APC)
* **Formulation**:
  $$\text{MI-APC}(i, j) = \max(0, \text{MI}(i, j) - \text{APC}(i, j))$$
  Quantifies co-evolutionary base-pairing signal between alignment columns $i$ and $j$, adjusted for background phylogenetic noise.

### 4. Resource Footprint Metrics
* **Execution Time**: Wall-clock execution time in seconds required for alignment execution.
* **Peak Memory Footprint**: Peak Resident Set Size (RSS) memory usage in megabytes (MB) tracked via `psutil` during subprocess execution.

---

## References & Citations

1. **Katoh, K., & Standley, D. M. (2013)**. MAFFT multiple sequence alignment software version 7: improvements in performance and usability. *Molecular Biology and Evolution*, 30(4), 772-780.
2. **Edgar, R. C. (2022)**. MUSCLE v5 enables high-throughput ensemble alignments with accuracy estimation. *Nature Communications*, 13(1), 6068.
3. **Moretti, S., et al. (2008)**. R-Coffee: a tool for multiple alignment of non-coding RNA sequences using predicted secondary structures. *Nucleic Acids Research*, 36(Web Server issue), W10-W13.
4. **Bernhart, S. H., et al. (2008)**. RNAalifold: computing consensus structures for RNA alignments. *BMC Bioinformatics*, 9(1), 474.
5. **McCaskill, J. S. (1990)**. The equilibrium partition function and base pairing probabilities of RNA secondary structure. *Biopolymers*, 29(6‐7), 1105-1119.
6. **Dunn, S. D., Wahl, L. M., & Gloor, G. B. (2008)**. Mutual information without the overhead: adjusting mutual information for coevolution analysis. *Bioinformatics*, 24(3), 333-340.
7. **Notredame, C., Higgins, D. G., & Heringa, J. (2000)**. T-Coffee: A novel method for fast and accurate multiple sequence alignments. *Journal of Molecular Biology*, 302(1), 205-217.
