# -*- coding: utf-8 -*-
"""simulation_subpartition_sp2_nested.py

Nested resolution audit for stratified subpartition pooling.
Auditing the finite-sample Benjamini-Hochberg resolution constraint.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd


def pair_stat(x, y):
    """SP-1 Vectorized pair statistic: last axis is sample size n."""
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
    """Equal-frequency quantile binning."""
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
    """Stratified pooling of null statistics within each stratum."""
    out = np.full(len(t), np.nan)
    for ell in np.unique(labels):
        idx = labels == ell
        out[idx] = empirical_p(bank[idx], t[idx])
    return out


def bh_mask(p, q):
    """Benjamini-Hochberg step-up procedure."""
    order = np.argsort(p)
    hit = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    out = np.zeros(len(p), bool)
    if hit.size:
        out[order[: hit[-1] + 1]] = True
    return out


def run_nested_audit(
    seed=2026,
    reps=5,  # ۵ تکرار برای تست اولیه
    n=20,
    G=1000,
    B_max=2000,
    b_grid=(20, 100, 500, 2000),
    n_strata=20,
    delta=2.0,
    pi0=0.8,
    alphas=(0.05, 0.01),
):
    master_rng = np.random.default_rng(seed)
    ratios_pool = np.array([1.0, 4.0, 16.0, 64.0, 256.0])

    # تخصیص دقیق ۲۰۰ ژن به هر سطح نسبت واریانس
    base_ratios = np.repeat(ratios_pool, G // len(ratios_pool))

    rows = []

    # استفاده از tile جهت تضمین حافظه پیوسته و قابل نوشتن (Writable C-contiguous)
    base_idx_x = np.tile(np.arange(n), (G, 1))
    base_idx_y = np.tile(np.arange(n), (G, 1))

    start_time = time.time()

    for rep in range(reps):
        rng_data = np.random.default_rng(master_rng.integers(1e9))

        # توزیع تصادفی نسبت‌ها و انتخاب مستقل ویژگی‌های آلترناتیو
        perm_features = rng_data.permutation(G)
        ratios = base_ratios[perm_features]

        is_null = np.ones(G, bool)
        n_alt = int(round((1.0 - pi0) * G))
        alt_indices = rng_data.choice(G, size=n_alt, replace=False)
        is_null[alt_indices] = False

        # تولید داده دو گروه
        x = rng_data.normal(size=(G, n)) * np.sqrt(ratios[:, None])
        y = rng_data.normal(size=(G, n)) + (~is_null)[:, None] * delta

        # ۱. محاسبه آماره مشاهده‌شده
        t = pair_stat(x, y)[0]

        # ۲. برآورد نسبت واریانس و لایه‌بندی
        estimate = np.log(x.var(1, ddof=1) / y.var(1, ddof=1))
        labels = labels_for(estimate, n_strata)

        # ۳. ساخت بانک پوچ تا عمق B_max با جایگشت‌های مستقل
        rng_perm = np.random.default_rng(master_rng.integers(1e9))
        bank_full = np.empty((G, B_max), dtype=float)

        for b in range(B_max):
            ix = rng_perm.permuted(base_idx_x, axis=1)
            iy = rng_perm.permuted(base_idx_y, axis=1)
            bank_full[:, b] = pair_stat(
                np.take_along_axis(x, ix, 1), np.take_along_axis(y, iy, 1)
            )[1]

        # ۴. ارزیابی تو در تو (Nested) روی مقادیر شبکه B
        for B in b_grid:
            bank_sub = bank_full[:, :B]
            pvals = stratified_pool_p(t, bank_sub, labels)

            # مشخصات رزولوشن درون هر لایه
            ref_size_per_stratum = (G // n_strata) * B
            floor_val = 1.0 / (ref_size_per_stratum + 1.0)

            for q in alphas:
                mask = bh_mask(pvals, q)
                R = int(mask.sum())
                tp = int(np.sum(mask & ~is_null))
                fp = R - tp

                fdp = fp / R if R > 0 else 0.0
                power = tp / int(np.sum(~is_null))
                raw_power = float(np.mean(pvals[~is_null] <= q))

                min_p = float(np.min(pvals))
                n_at_floor = int(np.sum(pvals <= floor_val + 1e-12))
                n_pvals_leq_alpha = int(np.sum(pvals <= q))
                min_rejections_if_any = int(np.ceil(G * floor_val / q))

                rows.append(
                    {
                        "rep": rep,
                        "B": B,
                        "alpha": q,
                        "bh_rejections": R,
                        "bh_power": power,
                        "raw_power": raw_power,
                        "fdp": fdp,
                        "min_p": min_p,
                        "floor_val": floor_val,
                        "n_at_floor": n_at_floor,
                        "n_pvals_leq_alpha": n_pvals_leq_alpha,
                        "min_rejections_if_any": min_rejections_if_any,
                    }
                )

        elapsed = time.time() - start_time
        print(f"Replicate {rep + 1}/{reps} complete ({elapsed:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["B", "alpha"])
        .agg(
            mean_bh_rejections=("bh_rejections", "mean"),
            mean_bh_power=("bh_power", "mean"),
            mean_raw_power=("raw_power", "mean"),
            mean_fdp=("fdp", "mean"),
            mean_min_p=("min_p", "mean"),
            mean_n_at_floor=("n_at_floor", "mean"),
            mean_n_pvals_leq_alpha=("n_pvals_leq_alpha", "mean"),
            floor_val=("floor_val", "first"),
            min_rejections_if_any=("min_rejections_if_any", "first"),
        )
        .reset_index()
    )

    return df, summary


if __name__ == "__main__":
    df, summary = run_nested_audit(reps=5)
    # ذخیره فایل‌ها برای درج مستقیم در مقاله
    df.to_csv("nested_audit_replicates_500.csv", index=False)
    summary.to_csv("nested_audit_summary_500.csv", index=False)
    print("\n================== FINAL SUMMARY (500 Reps) ==================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(summary.to_string(index=False))