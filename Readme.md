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

```

---

## Reproducing the Results

### 1. Unit Self-Tests and Scenarios A, B, C, and E (`sp1`)

Run internal mathematical verification checks (sign-reversal identity, affine invariance, shift invariance under alternatives, BH equivalence against SciPy, and exact $F_{1,36}$ null calibration):

```bash
python simulation_subpartition_sp1.py --mode test
```

Run a quick smoke check (`reps = 20`) or the full 500-replicate study (`reps = 500`):

```bash
python simulation_subpartition_sp1.py --mode smoke --outdir output_sp1
python simulation_subpartition_sp1.py --mode full --outdir output_sp1
```

Outputs are written to `output_sp1/` (`self_tests.json`, `config_*.json`, `replicates_*.csv`, and `summary_*.csv`), reproducing Tables 1 and 2 (Sections 5.1--5.3).

### 2. Nested BH Resolution Audit (`sp2`)

Note: By default in `__main__`, `run_nested_audit(reps=5)` is set for a fast sanity check. To reproduce the 500-replicate audit in Table 3 (Section 5.4), set `reps=500` in `simulation_subpartition_sp2_nested.py` and run:

```bash
python simulation_subpartition_sp2_nested.py
```

This writes `nested_audit_replicates_500.csv` and `nested_audit_summary_500.csv`.

### 3. Small-Sample, Non-Gaussian, and Welch Comparison (`sp3`)

Run the 20-replicate smoke test (~3.5 minutes) or the 200-replicate full study (~29 minutes on a standard desktop CPU) to reproduce Tables 4 and 5 (Section 5.5):

```bash
python simulation_extension_sp3.py --mode smoke --outdir output_sp3
python simulation_extension_sp3.py --mode full --outdir output_sp3
```

Outputs are written to `output_sp3/` (`sp3_null_audit_*.csv` and `sp3_signal_audit_*.csv`).

### 4. Golub et al. (1999) Leukemia Data Audit

`real_data_audit.py` automatically downloads the preprocessed 3,051-gene Golub training matrix (`golub.csv`), selects the balanced cohort of the first 10 ALL samples (columns 1--10) and first 10 AML samples (columns 28--37) with `seed = 2026`, and outputs `real_data_audit_summary.csv` reproducing Table 6 (Section 6; requires internet access on first run):

```bash
python real_data_audit.py
```

---

## Random Seeds and Determinism

All scripts use deterministic `numpy.random.default_rng` streams initialized with `seed = 2026` (via SHA-256 stream splitting in `sp1` and `sp3`) so that every replicate and table is reproducible across runs.

## License

This code is released for academic and non-commercial research use (MIT License recommended).
