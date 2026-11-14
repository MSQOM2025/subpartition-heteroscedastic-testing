# Subpartition-Based Null Statistics for Heteroscedastic Multi-Group Testing

Reproducibility code for the paper:

> **Subpartition-Based Null Statistics for Heteroscedastic Multi-Group Testing: Validity, Pooling, and Stratification**

This repository contains all simulation scripts, unit self-tests, and the real-data audit used to generate the numerical results and tables reported in the manuscript.

---

## Repository Contents

| Script | Purpose | Paper Output |
| :--- | :--- | :--- |
| `simulation_subpartition_sp1.py` | Unit self-tests and 500-replicate Gaussian audits (`n1 = n2 = 20`) for Scenarios A, B, C, and E | Tables 1 and 2; Section 5.1--5.3 |
| `simulation_subpartition_sp2_nested.py` | Nested resolution audit (`G = 1000, K = 20, B in {20, 100, 500, 2000}`) for Scenario D | Table 3; Section 5.4 |
| `simulation_extension_sp3.py` | Small-sample null calibration (`n1 = n2 = 8`) and `B = 2000` signal comparison (`n1 = n2 = 10`) across Gaussian, Student-$t_3$, and contaminated normal errors vs. Welch $t$-test and cross-group permutation | Tables 4 and 5 (Section 5.5) |
| `real_data_audit.py` | Empirical audit on the balanced Golub et al. (1999) leukemia cohort (`G = 3051`, `n1 = n2 = 10`, `K = 10`) | Section 6 (real-data table) |

---

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scipy`

Install dependencies in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install numpy pandas scipy
