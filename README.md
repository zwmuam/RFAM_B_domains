# RFAM_B_domains: Non-Coding RNA Multiple Sequence Alignment & Benchmarking Framework

A modern framework for benchmarking non-coding RNA (ncRNA) Multiple Sequence Alignment (MSA) tools and trimming strategies.

---

## Table of Contents
1. [Overview & Workflow](#overview--workflow)
2. [Installation & Environment Setup](#installation--environment-setup)
3. [Implemented Pipeline Pathways & Trimming Modes](#implemented-pipeline-pathways--trimming-modes)
4. [Usage](#usage)
5. [Visualization & Analysis](#visualization--analysis)
6. [Theoretical Breakdown of Evaluation Metrics](#theoretical-breakdown-of-evaluation-metrics)
7. [References & Citations](#references--citations)

---

## Overview & Workflow

The framework operates via an enhanced 5-step benchmarking workflow:

1. **Extraction & Coordinate Mapping (`gff_sequence_extractor.py`)**:
   Extracts target genomic intervals defined in primary GFF annotations from an input FASTA file. Converts $T \to U$ while preserving lowercase character soft-masking metadata. Sequences exceeding the maximum IUPAC degenerate nucleotide threshold ($> 5\%$) are excluded. Overlapping secondary annotations are coordinate-shifted and strand-adjusted onto extracted sequence frames.
2. **Multiple Sequence Alignment (`msa.py`)**:
   Aligns extracted sequence datasets across all configured aligners:
   - **Pathway 0**: Ultra-Fast MUSCLE v5 Super5 Primary Sequence Pipeline
   - **Pathway 1**: High-Throughput MAFFT L-INS-i Pipeline
   - **Pathway 2**: Balanced Structure-Aware MAFFT Q-INS-i Pipeline
   - **Pathway 3**: Speed-Optimized Gold-Standard Profile Covariance Model Pipeline (`cmbuild` + `cmalign --glocal`)
   - MAFFT X-INS-i, R-Coffee, and Structural Encoding

3. **Alignment Trimming (`msa.py`)**:
   Applies trimming strategies post-alignment:
   - **CIAlign Crop-from-ends**: Trims unaligned terminal overhangs while preserving internal structural core columns.
   - **trimAl gappyout**: Heuristic gap density trimming.
   - **Consensus Secondary Structure Masking**: `RNAalifold` consensus structure masking protecting paired base columns `()` `[]` and loop boundaries.
4. **Scientific Quality Evaluation Pre- and Post-Trimming (`msa_evaluate.py`)**:
   Evaluates each alignment output both **pre-trimming** and **post-trimming** across structural conservation (SCI), covariation (MI-APC), consistency (TCS), entropy ($H_N$), overlap (MOS), sequence identity, runtime, and memory footprint. Aligner or trimming failures are penalised with worst-case metric values ($H_N = 1.0$, all other metrics $= 0.0$).
5. **Data Export & Reporting (`benchmark.py`)**:
   Aggregates pre-trim and post-trim evaluation metrics into a pandas DataFrame and exports a multi-sheet Excel workbook (`benchmark_results.xlsx`). *Note*: `benchmark.py` strictly excludes `argparse` and uses hardcoded path defaults in `__main__`.

---

## Installation & Environment Setup

### Conda Environment (`environment_msa.yml`)
Create and activate the MSA benchmarking Conda environment containing Python libraries (`pandas`, `numpy`, `biopython`, `psutil`) and bioinformatics packages (`mafft`, `muscle`, `t-coffee`, `viennarna`, `infernal`, `trimal`, `cialign`):

```bash
conda env create -f environment_msa.yml
conda activate msa_benchmark_msa
```

---

## Implemented Pipeline Pathways & Trimming Modes

| Pathway / Method | Alignment Engine | Trimming Engine | Objective & Purpose | Error Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Pathway 0** | MUSCLE v5 Super5 | CIAlign Crop-from-ends / trimAl | Ultra-fast primary sequence screening ($> 100,000$ alignments) |
| **Pathway 1** | MAFFT L-INS-i | CIAlign Crop-from-ends | High-throughput local pair consistency matching |
| **Pathway 2** | MAFFT Q-INS-i / L-INS-i | Consensus Structure Masking | Structure-aware base-pairing probability alignment |
| **Pathway 3** | Infernal `cmbuild` + `cmalign --glocal` | Profile Match-State Truncation | Speed-optimized profile Covariance Model alignment |

---

## Usage

To execute the benchmark suite:

```bash
python benchmark.py
```

To run unit and integration tests:

```bash
python -m pytest
```

---

## Technical Test Summary

Refer to `TECHNICAL_TEST_SUMMARY.md` for full details regarding external tools tested, simulated, or unavailable during runtime execution.
