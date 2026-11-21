# Subpartition-Based Null Statistics for Heteroscedastic Multi-Group Testing

Reproducibility code for the paper:

> **Subpartition-Based Null Statistics for Heteroscedastic Multi-Group Testing: Validity, Pooling, and Stratification**

This repository contains all simulation scripts, unit self-tests, supplemental audits, and real-data scripts used to generate the numerical results and tables reported in the manuscript.

---

## Repository Contents

| Script | Purpose | Paper Output |
| :--- | :--- | :--- |
| `simulation_subpartition_sp1.py` | Unit self-tests and 500-replicate Gaussian audits (`n1 = n2 = 20`) for Scenarios A, B, C, and E | Tables 1 and 2; Sections 5.1--5.3 |
| `simulation_subpartition_sp2_nested.py` | Nested resolution audit (`G = 1000`, `K = 20`, `B in {20, 100, 500, 2000}`) for Scenario D | Table 3; Section 5.4 |
| `simulation_extension_sp3.py` | Small-sample null calibration (`n1 = n2 = 8`) and `B = 2000` signal comparison (`n1 = n2 = 10`) across Gaussian, Student-$t_3$, and contaminated normal errors vs. Welch $t$-test and cross-group permutation | Tables 4 and 5; Section 5.5 |
| `simulation_boot_sp5.py` | Per-feature and pooled parametric Gaussian bootstrap benchmarks on the exact 200-replicate data streams of `sp3` | Parametric bootstrap rows in Tables 4 and 5; Section 5.5 |
| `audit_supplement_sp4.py` | Exact Scenario A KS $p$-values (`10k` and `50k` features), auxiliary vs. in-sample stratification audit (`n_aux in {10, 50}`), and 30-subcohort Golub sensitivity audit | Table 1 KS $p$-values (Section 5.1), Table 6 (Section 5.6), and Table 8 (Section 6.2) |
| `real_data_audit.py` | Empirical audit on the balanced Golub et al. (1999) leukemia cohort (`G = 3051`, `n1 = n2 = 10`, `K = 10`) | Table 7; Sections 6.1--6.2 |
| `paired_diff.py` | Consistency assertions, zero-discovery counts, and paired differences across the 30 overlapping Golub subcohorts | Table 9; Section 6.2 |

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

### 3. Small-Sample, Non-Gaussian, Welch, and Parametric Bootstrap Comparisons (`sp3` & `sp5`)

Run the 20-replicate smoke test (~3.5 minutes) or the 200-replicate full study (~29 minutes on a standard desktop CPU) for cross-group permutation, Welch $t$-test, and subpartition pooling:

```bash
python simulation_extension_sp3.py --mode smoke --outdir output_sp3
python simulation_extension_sp3.py --mode full --outdir output_sp3
```

Outputs are written to `output_sp3/` (`sp3_null_audit_*.csv` and `sp3_signal_audit_*.csv`).

To generate the matching per-feature and pooled parametric Gaussian bootstrap benchmarks on the exact same 200-replicate data streams (~4--5 minutes), run:

```bash
python simulation_boot_sp5.py --outdir output_sp5
```

Outputs are written to `output_sp5/` (`sp5_boot_null_full.csv` and `sp5_boot_signal_full.csv`), completing Tables 4 and 5 (Section 5.5).

### 4. Supplemental Audits: Scenario A KS, Auxiliary vs. In-Sample, and Golub Sensitivity (`sp4`)

Run the smoke check or the full 200-replicate / 30-subcohort supplemental audit (`audit_supplement_sp4.py` caches `golub.csv` in `output_sp4/` after downloading it on the first run):

```bash
python audit_supplement_sp4.py --mode smoke --outdir output_sp4
python audit_supplement_sp4.py --mode full --outdir output_sp4
```

This writes `sp4_scenario_a_ks.csv` (Table 1 KS $p$-values and the 50,000-feature check in Section 5.1), `sp4_aux_vs_insample_null_full.csv` and `sp4_aux_vs_insample_sig_full.csv` (Table 6, Section 5.6), and `sp4_golub_raw_full.csv` and `sp4_golub_summary_full.csv` (Table 8, Section 6.2).

After `sp4` finishes, run `paired_diff.py` to verify data-integrity assertions and compute the zero-discovery counts and paired differences for Table 9 (Section 6.2):

```bash
python paired_diff.py
```

### 5. Primary Golub et al. (1999) Leukemia Data Audit

`real_data_audit.py` automatically downloads the preprocessed 3,051-gene Golub training matrix (`golub.csv`), selects the balanced cohort of the first 10 ALL samples (columns 1--10) and first 10 AML samples (columns 28--37) with `seed = 2026`, and outputs `real_data_audit_summary.csv` reproducing Table 7 (Sections 6.1--6.2; requires internet access on first run):

```bash
python real_data_audit.py
```

---

## Random Seeds and Determinism

All scripts use deterministic `numpy.random.default_rng` streams initialized with `seed = 2026` (via SHA-256 stream splitting in `sp1`, `sp3`, `sp4`, and `sp5`) so that every replicate and table is reproducible across runs.

---

## License

This code is provided for academic and non-commercial research purposes.

If you use this code in your research, please cite the associated manuscript:

**"Subpartition-Based Null Statistics for Heteroscedastic Multi-Group Testing: Validity, Pooling, and Stratification."**

*Currently under peer review.*

Commercial use requires prior permission from the author(s).
