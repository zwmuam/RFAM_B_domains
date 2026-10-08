# Technical Test Summary

## Executive Overview
This document details the test execution and tool availability status for the Non-Coding RNA Multiple Sequence Alignment & Benchmarking Framework.

All 12 automated unit and integration tests in `tests/test_msa_benchmark.py` and `test_visualizations.py` were executed and passed successfully (100% pass rate).

---

## Tool Availability & Execution Matrix

| Bioinformatic Tool / Module | Required Binary / Method | Status in Test Environment | Test Strategy / Policy | Deviation Symbol |
| :--- | :--- | :--- | :--- | :--- |
| **MAFFT** | `mafft`, `mafft-linsi`, `mafft-qinsi`, `mafft-xinsi` | Not installed in sandbox system PATH | Explicit error handling tested (`is_successful=False`, error msg containing 🛑). No silent fallback. | 🛑 |
| **MUSCLE** | `muscle` | Not installed in sandbox system PATH | Explicit error handling tested (`is_successful=False`, error msg containing 🛑). No silent fallback. | 🛑 |
| **T-Coffee** | `t_coffee` | Not installed in sandbox system PATH | Explicit error handling tested for R-Coffee. TCS evaluation uses vectorized fallback matrix calculation. | 🛑 |
| **ViennaRNA** | `RNAfold`, `RNAalifold` | Not installed in sandbox system PATH | SCI & structural covariation fallback to zero/mean APC when binaries absent. | 🛑 |
| **Infernal** | `cmbuild`, `cmalign` | Not installed in sandbox system PATH | ProfileCovarianceModelPipeline tests explicit error reporting with 🛑 symbol. | 🛑 |
| **CIAlign** | `cialign` | Not installed in sandbox system PATH | Pure-Python crop-from-ends terminal overhang trimming fallback implemented & verified in tests. | 🛑 |
| **trimAl** | `trimal` | Not installed in sandbox system PATH | Pure-Python gap fraction column filtering fallback implemented & verified in tests. | 🛑 |

---

## Core Algorithmic & Quality Control Verification
The following core capabilities were fully verified via deterministic Python test cases:
1. **Alphabet Standardization & Soft-Masking**: Correct conversion of $T \to U$ while preserving lowercase soft-masking metadata (`acgtACGT` $\to$ `acguACGU`).
2. **Reverse Complement Orientation Handling**: Strand direction adjustment preserving character case and RNA base representation.
3. **IUPAC Degenerate Filtering**: Filtering of sequence records with $> 5\%$ ambiguous nucleotide density.
4. **Trimming Logic**: Crop-from-ends and gappyout terminal/internal column trimming algorithms.
5. **Benchmark Metric Evaluation**: Vectorized TCS, SCI, MI-APC, $H_N$, and MOS calculations pre-trimming and post-trimming.
6. **No-Argparse Constraint**: Hardcoded defaults in `benchmark.py` `__main__` block without CLI parser dependencies.
