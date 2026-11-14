# -*- coding: utf-8 -*-
"""real_data_audit.py

Real Genomic Data Audit: Golub et al. (1999) Leukemia Dataset.
Audits heteroscedasticity, naive vs. stratified pooling, and the BH resolution barrier.
"""

import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd


def pair_stat(x, y):
    """Vectorized SP-1 Welch dispersion statistic."""
    h = x.shape[-1] // 2
    ma, mb = x[..., :h].mean(-1), x[..., h:].mean(-1)
    na, nb = y[..., :h].mean(-1), y[..., h:].mean(-1)
    vx = (x[..., :h].var(-1, ddof=1) + x[..., h:].var(-1, ddof=1)) / (4 * h)
    vy = (y[..., :h].var(-1, ddof=1) + y[..., h:].var(-1, ddof=1)) / (4 * h)
    den = vx + vy
    t = ((ma + mb - na - nb) / 2) ** 2 / den
    u = ((ma - mb - na + nb) / 2) ** 2 / den
    return t, u


def labels_for(values, count):
    """Quantile-based equal-frequency binning."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), int)
    labels[order] = np.arange(len(values)) * count // len(values)
    return labels


def empirical_p(reference, observed):
    """Right-tail empirical p-value."""
    ref = np.sort(np.asarray(reference).ravel())
    obs = np.asarray(observed)
    return (1 + len(ref) - np.searchsorted(ref, obs, side="left")) / (
        len(ref) + 1
    )


def stratified_pool_p(t, bank, labels):
    """Within-stratum pooled empirical p-values."""
    out = np.full(len(t), np.nan)
    for ell in np.unique(labels):
        idx = labels == ell
        out[idx] = empirical_p(bank[idx], t[idx])
    return out


def bh_rejections(p, q):
    """Benjamini-Hochberg step-up rejections count."""
    order = np.argsort(p)
    hit = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    return int(hit.size and (hit[-1] + 1))


def load_golub_data():
    """Download and prepare the balanced Golub et al. (1999) benchmark."""
    print("Downloading benchmark dataset (Golub et al., 1999)...", flush=True)
    url = "https://github.com/pipeton8/6.439-stats-comp-applications/raw/main/Assignments/databases/PS1_data.zip"

    req = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(req.read()))
    df = pd.read_csv(z.open("PS1_data/golub_data/golub.csv"), index_col=0)

    # 3051 genes x 38 samples (27 ALL, 11 AML)
    # Subset to balanced groups: 10 ALL vs 10 AML (matching n=20)
    all_samples = df.iloc[:, :10].values  # First 10 ALL samples
    aml_samples = df.iloc[:, 27:37].values  # 10 AML samples

    gene_names = df.index.values
    return all_samples, aml_samples, gene_names


def run_real_data_audit():
    seed = 2026
    rng = np.random.default_rng(seed)

    x, y, gene_names = load_golub_data()
    G, n = x.shape  # G=3051, n=10

    print(
        f"Data loaded successfully: G = {G} genes, n1 = {n} (ALL), n2 = {n} (AML)."
    )

    # ۱. ارزیابی ناهمگونی تجربی واریانس‌ها
    vx = x.var(axis=1, ddof=1)
    vy = y.var(axis=1, ddof=1)
    r_hat = vx / vy

    percentiles = [5, 25, 50, 75, 95]
    perc_vals = np.percentile(r_hat, percentiles)
    print("\n--- Empirical Variance Ratio (r_hat = s1^2 / s2^2) Percentiles ---")
    for p, val in zip(percentiles, perc_vals):
        print(f"  {p}th percentile: {val:.3f}")

    # ۲. محاسبه آماره مشاهده‌شده با افراز متوازن ثابت
    t_obs = pair_stat(x, y)[0]

    # ۳. لایه‌بندی بر مبنای نسبت واریانس تخمینی (K = 10)
    K = 10
    labels = labels_for(np.log(r_hat), K)

    # ۴. ساخت بانک بزرگ جایگشت‌های درون‌گروهی تا B_max = 2000
    B_max = 2000
    b_grid = [20, 100, 500, 2000]
    alphas = [0.05, 0.01]

    print(
        f"\nConstructing maximal null reference bank (B_max = {B_max})...",
        flush=True,
    )
    start_time = time.time()

    bank_full = np.empty((G, B_max), dtype=float)
    base_idx_x = np.tile(np.arange(n), (G, 1))
    base_idx_y = np.tile(np.arange(n), (G, 1))

    for b in range(B_max):
        ix = rng.permuted(base_idx_x, axis=1)
        iy = rng.permuted(base_idx_y, axis=1)
        bank_full[:, b] = pair_stat(
            np.take_along_axis(x, ix, 1), np.take_along_axis(y, iy, 1)
        )[1]

    print(f"Null bank constructed in {time.time() - start_time:.1f} seconds.")

    # ۵. ارزیابی تجمیع خام (Naive Pooled - Gao 2006) در برابر تجمیع لایه‌ای (Stratified)
    results = []

    for B in b_grid:
        bank_sub = bank_full[:, :B]

        # روش ۱: تجمیع خام (Gao 2006)
        pvals_naive = empirical_p(bank_sub, t_obs)

        # روش ۲: تجمیع لایه‌بندی‌شده (روش پیشنهادی ما)
        pvals_strat = stratified_pool_p(t_obs, bank_sub, labels)

        ref_per_stratum = (G // K) * B
        floor_strat = 1.0 / (ref_per_stratum + 1.0)
        floor_naive = 1.0 / (G * B + 1.0)

        for q in alphas:
            r_naive = bh_rejections(pvals_naive, q)
            r_strat = bh_rejections(pvals_strat, q)

            min_rej_strat = int(np.ceil(G * floor_strat / q))

            results.append(
                {
                    "B": B,
                    "alpha": q,
                    "naive_bh_rejections": r_naive,
                    "strat_bh_rejections": r_strat,
                    "min_p_strat": float(np.min(pvals_strat)),
                    "floor_val_strat": float(floor_strat),
                    "n_at_floor_strat": int(
                        np.sum(pvals_strat <= floor_strat + 1e-12)
                    ),
                    "barrier_k_strat": min_rej_strat,
                }
            )

    res_df = pd.DataFrame(results)

    print("\n================== REAL DATA AUDIT RESULTS ==================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(res_df.to_string(index=False))

    res_df.to_csv("real_data_audit_summary.csv", index=False)
    print("\nWrote summary to real_data_audit_summary.csv")


if __name__ == "__main__":
    run_real_data_audit()