# RFAM_B_domains: Non-Coding RNA Multiple Sequence Alignment & Benchmarking Framework

A modern, production-grade Python 3.10+ framework for benchmarking non-coding RNA (ncRNA) Multiple Sequence Alignment (MSA) tools.

The framework integrates genomic feature extraction, alignment execution across structural and sequence-based aligners, rigorous bioinformatic metric evaluations, and automated Excel reporting.

---

## Table of Contents
1. [Overview & Workflow](#overview--workflow)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Theoretical Breakdown of Alignment Methods](#theoretical-breakdown-of-alignment-methods)
5. [Theoretical Breakdown of Evaluation Metrics](#theoretical-breakdown-of-evaluation-metrics)
6. [References & Citations](#references--citations)

---

## Overview & Workflow

The framework operates via a 4-step benchmarking workflow:

1. **Extraction & Coordinate Mapping (`gff_sequence_extractor.py`)**:
   Extracts target genomic intervals defined in primary GFF feature annotations (e.g., predicted clusters) from an input FASTA file (`input.fasta`). Overlapping secondary annotations (e.g., reference Rfam annotations) are coordinate-shifted and strand-adjusted to match the extracted sequence frames. Output files (`.fasta` and `.gff`) are saved per feature ID.
2. **Multiple Sequence Alignment (`msa.py`)**:
   Aligns extracted sequence datasets across all configured aligners (`Muscle5`, `MAFFT Q-INS-i`, `MAFFT L-INS-i`, `MAFFT X-INS-i`, `R-Coffee`, `Structural Encoding`). Exports aligned sequences in FASTA (`.fna`) and Stockholm (`.sto`) formats for manual inspection.
3. **Scientific Quality Evaluation (`msa_evaluate.py`)**:
   Evaluates each alignment output across structural conservation, covariation, consistency, entropy, overlap, and sequence identity metrics. Aligner failures or missing CLI dependencies are caught and assigned penalized worst-case metric values (maximum entropy $H_N = 1.0$, all other metrics $= 0.0$).
4. **Data Export & Summary (`benchmark.py`)**:
   Aggregates per-alignment evaluation metrics into a pandas DataFrame and exports a multi-sheet Excel workbook (`benchmark_results.xlsx`) containing raw metrics and pipeline summary averages.

---

## Installation

### Prerequisites
* Linux / macOS system environment
* Conda or Mamba package manager

### 1. Environment Setup via Conda
Clone the repository and create the Conda environment using `msa_benchmark.yml`:

```bash
git clone https://github.com/your-org/RFAM_B_domains.git
cd RFAM_B_domains

# Create and activate environment
conda env create -f msa_benchmark.yml
conda activate msa_benchmark
```

### 2. Manual Package Installation (Pip)
If managing dependencies via `pip`:

```bash
pip install numpy pandas openpyxl biopython psutil pytest
```

### 3. External Bioinformatic Tool Dependencies
To enable full aligner and structure evaluation execution, ensure the following CLI tools are available in your system `PATH`:
* **MAFFT** ($\ge 7.520$): `mafft`, `mafft-qinsi`, `mafft-linsi`, `mafft-xinsi`
* **MUSCLE** ($\ge 5.1.0$): `muscle`
* **T-Coffee**: `t_coffee` (required for R-Coffee and TCS evaluation)
* **ViennaRNA Package** ($\ge 2.5.1$): `RNAfold`, `RNAalifold` (required for SCI and structure covariation calculation)

---

## Usage

### 1. Running the Complete Benchmark Suite

To run the automated benchmark workflow, configure input file paths in `benchmark.py` and run:

```bash
python benchmark.py
```

#### Code Example (`benchmark.py` interface):

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

### 2. Using Individual Modules Programmatically

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
    print(f"Alignment successful in {result.execution_time_seconds:.2f}s")
else:
    print(f"Alignment failed: {result.error_message}")
```

#### Evaluating Alignment Metrics (`msa_evaluate.py`)
```python
from pathlib import Path
from msa_evaluate import AlignmentEvaluator

evaluator = AlignmentEvaluator()
aln_path = Path("cluster_1_qinsi.fna")

sci = evaluator.calculate_structure_conservation_index(aln_path)
tcs = evaluator.calculate_transitive_consistency_score(aln_path)
cov = evaluator.calculate_structural_covariation_score(aln_path)
hn = evaluator.calculate_normalized_shannon_entropy(aln_path)

print(f"SCI: {sci:.4f}, TCS: {tcs:.2f}, H_N: {hn:.4f}")
```

---

## Theoretical Breakdown of Alignment Methods

Aligning non-coding RNAs requires methods that consider both primary sequence identity and secondary structure conservation.

### 1. MUSCLE v5 (`muscle5`)
* **Theoretical Basis**: Introduces Progressive Perturbed Pairwise (PPP) alignments and ensemble representations. Rather than relying on a single deterministic guide tree, MUSCLE v5 generates an ensemble of perturbed alignments to capture alignment uncertainty.
* **Mechanism**: Uses k-mer distance estimations combined with profile-profile comparisons, making it effective for sequence-diverse RNA families.

### 2. MAFFT Q-INS-i (`mafft_qinsi`)
* **Theoretical Basis**: Incorporates McCaskill secondary structure base-pairing probability matrices calculated from individual sequences into pairwise alignment algorithms.
* **Mechanism**: Uses the McCaskill algorithm (McCaskill, 1990) to construct base-pairing probability matrices $P_{ij}$ for each sequence, aligning sequences by maximizing structural probability consensus across secondary base pairs.

### 3. MAFFT L-INS-i (`mafft_linsi`)
* **Theoretical Basis**: Local pairwise alignment algorithm with maximum consistency refinement.
* **Mechanism**: Constructs local alignment libraries using a Needleman-Wunsch variant with position-specific gap penalties, optimized for sequences sharing conserved local domains flanked by variable regions.

### 4. MAFFT X-INS-i (`mafft_xinsi`)
* **Theoretical Basis**: Structural alignment method utilizing pairwise structural alignment algorithms (e.g., MXSCARNA).
* **Mechanism**: Calculates full stem-candidate pairing matrices across sequence pairs before progressive alignment, making it suitable for complex RNA secondary structures with low primary sequence conservation.

### 5. R-Coffee (`rcoffee`)
* **Theoretical Basis**: Extension of T-Coffee that incorporates secondary structure predictions into consistency libraries.
* **Mechanism**: Uses RNAfold or pairwise thermodynamic folding estimates to construct an RNA structure library. The objective function maximizes the consistency between alignment columns and predicted base pairs.

### 6. Structural Encoding (`structural_encoding`)
* **Theoretical Basis**: Structure-informed alignment pipeline leveraging explicit structural probability matrices (MAFFT Q-INS-i / globalpair).
* **Mechanism**: Operates directly on standardized canonical RNA bases (U-encoding) without case-erasure artifacts, preserving thermodynamic probability weighting throughout multi-pass iterative refinement.

---

## Theoretical Breakdown of Evaluation Metrics

The framework evaluates multiple orthogonal dimensions of RNA alignment quality:

### 1. Structure Conservation Index (SCI)
* **Definition**: Measures the degree to which individual RNA secondary structures are conserved in the consensus structure.
* **Formulation**:
  $$\text{SCI} = \frac{E_{\text{consensus}}}{\bar{E}_{\text{single}}}$$
  where $E_{\text{consensus}}$ is the Minimum Free Energy (MFE) of the consensus alignment predicted by `RNAalifold`, and $\bar{E}_{\text{single}}$ is the mean MFE of individual ungapped sequences predicted by `RNAfold`.
* **Interpretation**: $\text{SCI} \approx 1.0$ indicates strong structural conservation. $\text{SCI} > 1.0$ indicates compensatory mutations stabilizing the consensus structure. Unsuccessful or unstructured alignments yield $\text{SCI} = 0.0$.

### 2. Transitive Consistency Score (TCS)
* **Definition**: Evaluates the consistency of column residue pairs across all pairwise alignment paths in a multiple sequence alignment.
* **Mechanism**: Computed via T-Coffee `-evaluate` or vectorized pairwise residue column matching. Measures local column reliability ($0 - 100\%$).

### 3. Mutual Information with APC Covariation (MI-APC)
* **Definition**: Quantifies co-evolutionary base-pairing signal between alignment columns $i$ and $j$, adjusted for background phylogenetic noise.
* **Formulation**:
  $$\text{MI}(i, j) = \sum_{x, y \in \{A,C,G,U\}} P(x_i, y_j) \log_2 \frac{P(x_i, y_j)}{P(x_i) P(y_j)}$$
  $$\text{APC}(i, j) = \frac{\overline{\text{MI}}_i \cdot \overline{\text{MI}}_j}{\overline{\text{MI}}_{\text{overall}}}$$
  $$\text{MI-APC}(i, j) = \max(0, \text{MI}(i, j) - \text{APC}(i, j))$$
* **Consensus Base-Pair Covariation**: Average MI-APC score computed specifically across base-paired positions defined in the consensus secondary structure (`RNAalifold`).

### 4. Compensatory Mutation Count
* **Definition**: Counts the number of consensus base-paired positions $(i, j)$ exhibiting at least two distinct canonical base pairs (e.g., A-U and G-C) across sequences with a positive covariation score ($\text{MI-APC} > 0.01$).

### 5. Normalized Shannon Entropy ($H_N$)
* **Definition**: Measures positional variability and column disorder across non-gap IUPAC nucleotide positions.
* **Formulation**:
  $$H(c) = -\sum_{x \in \mathcal{A}} P(x) \log_2 P(x)$$
  $$H_N = \frac{1}{L} \sum_{c=1}^{L} \frac{H(c)}{\log_2 |\mathcal{A}|}$$
  where $|\mathcal{A}|$ is the IUPAC alphabet size.
* **Interpretation**: $H_N = 0.0$ represents completely conserved columns. $H_N = 1.0$ represents maximum positional disorder (used as the penalty value for failed alignments).

### 6. Mean Overlap Score (MOS)
* **Definition**: Computes the mean Jaccard overlap index of aligned non-gap residue pairs between distinct alignment pipelines for a dataset.
* **Formulation**:
  $$\text{MOS} = \frac{1}{\binom{K}{2}} \sum_{a < b} \frac{|S_a \cap S_b|}{|S_a \cup S_b|}$$
  where $S_a$ is the set of aligned residue pairs $(seq\_id, pos_i, col_k)$ produced by pipeline $a$.

### 7. Pairwise Sequence Similarity Statistics
* **Definition**: Calculates mean, median, minimum, and maximum percent sequence identity across all unique sequence pairs in the alignment matrix:
  $$\text{Identity} = \frac{\text{Matches}}{\text{Aligned Non-Gap Columns}} \times 100\%$$

---

## References & Citations

1. **Katoh, K., & Standley, D. M. (2013)**. MAFFT multiple sequence alignment software version 7: improvements in performance and usability. *Molecular Biology and Evolution*, 30(4), 772-780.
2. **Edgar, R. C. (2022)**. MUSCLE v5 enables high-throughput ensemble alignments with accuracy estimation. *Nature Communications*, 13(1), 6068.
3. **Moretti, S., et al. (2008)**. R-Coffee: a tool for multiple alignment of non-coding RNA sequences using predicted secondary structures. *Nucleic Acids Research*, 36(Web Server issue), W10-W13.
4. **Bernhart, S. H., et al. (2008)**. RNAalifold: computing consensus structures for RNA alignments. *BMC Bioinformatics*, 9(1), 474.
5. **McCaskill, J. S. (1990)**. The equilibrium partition function and base pairing probabilities of RNA secondary structure. *Biopolymers*, 29(6‐7), 1105-1119.
6. **Dunn, S. D., Wahl, L. M., & Gloor, G. B. (2008)**. Mutual information without the overhead: adjusting mutual information for coevolution analysis. *Bioinformatics*, 24(3), 333-340.
7. **Notredame, C., Higgins, D. G., & Heringa, J. (2000)**. T-Coffee: A novel method for fast and accurate multiple sequence alignments. *Journal of Molecular Biology*, 302(1), 205-217.
8. **Washietl, S., Hofacker, I. L., & Stadler, P. F. (2005)**. Fast and reliable prediction of noncoding RNAs. *Proceedings of the National Academy of Sciences*, 102(7), 2454-2459.
