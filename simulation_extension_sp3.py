# -*- coding: utf-8 -*-
"""simulation_extension_sp3.py

Extension audit addressing referee gaps:
1) Small-n (n1=n2=8) null calibration across Gaussian, Student-t3, and
   contaminated normal errors comparing Cross-Group Perm, Welch t-test,
   Naive Pooled Subpartition, and Stratified Subpartition (using |log r_hat|).
2) Signal power & FDP comparison at B=2000 across Welch, Naive (K=1),
   and Stratified (K=4, 10, 20) for delta in {1.0, 2.0, 4.0} under
   Gaussian and Student-t3 errors (n1=n2=10).
"""

import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def make_rng(seed, *parts):
    text = repr((seed, parts)).encode()
    entropy = int.from_bytes(hashlib.sha256(text).digest()[:16], "little")
    return np.random.default_rng(entropy)


def draw_errors(rng, dist, shape):
    if dist == "normal":
        return rng.normal(size=shape)
    if dist == "t3":
        # Unit-variance symmetric heavy-tailed distribution
        return rng.standard_t(3, size=shape) / np.sqrt(3.0)
    if dist == "contam":
        # 85% N(0,1) + 15% N(0,25), variance = 4.6, normalized to unit variance
        mix = rng.random(size=shape) < 0.15
        raw = rng.normal(size=shape) * np.where(mix, 5.0, 1.0)
        return raw / np.sqrt(4.6)
    raise ValueError(f"Unknown dist: {dist}")


def pair_stat(x, y):
    """Vectorized subpartition statistic on last axis (supports batch dims)."""
    h = x.shape[-1] // 2
    ma, mb = x[..., :h].mean(-1), x[..., h:].mean(-1)
    na, nb = y[..., :h].mean(-1), y[..., h:].mean(-1)
    vx = (x[..., :h].var(-1, ddof=1) + x[..., h:].var(-1, ddof=1)) / (4.0 * h)
    vy = (y[..., :h].var(-1, ddof=1) + y[..., h:].var(-1, ddof=1)) / (4.0 * h)
    den = np.maximum(vx + vy, 1e-300)
    t_obs = ((ma + mb - na - nb) / 2.0) ** 2 / den
    t_null = ((ma - mb - na + nb) / 2.0) ** 2 / den
    return t_obs, t_null


def welch_pvalues(x, y):
    """Vectorized two-sided Welch t-test p-values across features."""
    n1, n2 = x.shape[-1], y.shape[-1]
    mx, my = x.mean(-1), y.mean(-1)
    vx = x.var(-1, ddof=1) / n1
    vy = y.var(-1, ddof=1) / n2
    se2 = np.maximum(vx + vy, 1e-300)
    t_stat = (mx - my) / np.sqrt(se2)
    df = se2 ** 2 / np.maximum((vx ** 2) / (n1 - 1) + (vy ** 2) / (n2 - 1), 1e-300)
    return 2.0 * student_t.sf(np.abs(t_stat), df=df)


def labels_for(values, count):
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), int)
    labels[order] = np.arange(len(values)) * count // len(values)
    return labels


def empirical_p(reference, observed):
    ref = np.sort(np.asarray(reference).ravel())
    obs = np.asarray(observed)
    return (1 + len(ref) - np.searchsorted(ref, obs, side="left")) / (len(ref) + 1)


def pool_p(t_obs, bank, labels=None):
    if labels is None:
        return empirical_p(bank, t_obs)
    out = np.full(len(t_obs), np.nan)
    for ell in np.unique(labels):
        idx = labels == ell
        out[idx] = empirical_p(bank[idx], t_obs[idx])
    return out


def bh_mask(p, q):
    order = np.argsort(p)
    hit = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    out = np.zeros(len(p), bool)
    if hit.size:
        out[order[: hit[-1] + 1]] = True
    return out


def build_subpartition_bank(x, y, B, rng, chunk=100):
    G, n = x.shape
    bank = np.empty((G, B), dtype=float)
    base = np.arange(n)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        ix = rng.permuted(np.broadcast_to(base, (m, G, n)), axis=-1)
        iy = rng.permuted(np.broadcast_to(base, (m, G, n)), axis=-1)
        xb = np.take_along_axis(np.broadcast_to(x, (m, G, n)), ix, axis=-1)
        yb = np.take_along_axis(np.broadcast_to(y, (m, G, n)), iy, axis=-1)
        bank[:, done : done + m] = pair_stat(xb, yb)[1].T
        done += m
    return bank


def cross_group_perm_pvalues(x, y, B, rng, chunk=100):
    """Standard pooled cross-group label permutation using ANOVA-type F/mean-diff."""
    G, n = x.shape
    z = np.concatenate([x, y], axis=1)
    # Standard pooled-variance two-sample t^2 statistic (homoscedastic permutation target)
    mx, my = x.mean(-1), y.mean(-1)
    sp2 = ((n - 1) * x.var(-1, ddof=1) + (n - 1) * y.var(-1, ddof=1)) / (2 * n - 2)
    t_obs = (mx - my) ** 2 / np.maximum(2.0 * sp2 / n, 1e-300)

    bank = np.empty((G, B), dtype=float)
    base = np.arange(2 * n)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        iz = rng.permuted(np.broadcast_to(base, (m, G, 2 * n)), axis=-1)
        zb = np.take_along_axis(np.broadcast_to(z, (m, G, 2 * n)), iz, axis=-1)
        xb, yb = zb[..., :n], zb[..., n:]
        sp2_b = ((n - 1) * xb.var(-1, ddof=1) + (n - 1) * yb.var(-1, ddof=1)) / (2 * n - 2)
        bank[:, done : done + m] = ((xb.mean(-1) - yb.mean(-1)) ** 2 / np.maximum(2.0 * sp2_b / n, 1e-300)).T
        done += m
    return empirical_p(bank, t_obs)


def run_null_audit(seed=2026, reps=200, G=1000, n=8, B=1000, K=10):
    """Audit 1: small-n null calibration across Gaussian, t3, contaminated normal."""
    ratios_grid = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
    ratios = np.resize(ratios_grid, G)
    alphas = (0.05, 0.01)
    dists = ("normal", "t3", "contam")
    rows = []

    t0 = time.time()
    for dist in dists:
        for rep in range(reps):
            rng_d = make_rng(seed, "null_audit", dist, rep)
            x = draw_errors(rng_d, dist, (G, n)) * np.sqrt(ratios[:, None])
            y = draw_errors(rng_d, dist, (G, n))

            t_obs = pair_stat(x, y)[0]
            rng_p = make_rng(seed, "null_perm", dist, rep)
            bank = build_subpartition_bank(x, y, B, rng_p)
            abs_log_rhat = np.abs(np.log(x.var(1, ddof=1) / y.var(1, ddof=1)))
            labels = labels_for(abs_log_rhat, K)

            p_methods = {
                "cross_group_perm": cross_group_perm_pvalues(x, y, B, make_rng(seed, "cg", dist, rep)),
                "welch_t": welch_pvalues(x, y),
                "naive_subpart": pool_p(t_obs, bank, None),
                f"strat_subpart_K{K}": pool_p(t_obs, bank, labels),
            }

            for mname, pvals in p_methods.items():
                for q in alphas:
                    for r in ratios_grid:
                        mask = ratios == r
                        rows.append({
                            "dist": dist, "n": n, "method": mname, "alpha": q,
                            "ratio": r, "rep": rep, "size": float(np.mean(pvals[mask] <= q))
                        })
        print(f"[Null audit n={n}] {dist} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dist", "method", "alpha", "ratio"])["size"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["mcse"] = summary["std"] / np.sqrt(summary["count"])
    return df, summary


def run_signal_audit(seed=2026, reps=200, G=1000, n=10, B=2000, pi0=0.8):
    """Audit 2: BH power & FDP comparison at B=2000 across Welch, Naive, and K=4,10,20."""
    ratios_grid = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
    base_ratios = np.repeat(ratios_grid, G // len(ratios_grid))
    alphas = (0.05, 0.01)
    dists = ("normal", "t3")
    deltas = (1.0, 2.0, 4.0)
    strata_list = (4, 10, 20)
    rows = []

    t0 = time.time()
    for dist in dists:
        for delta in deltas:
            for rep in range(reps):
                rng_d = make_rng(seed, "sig_data", dist, delta, rep)
                ratios = base_ratios[rng_d.permutation(G)]
                is_null = np.ones(G, bool)
                alt_idx = rng_d.choice(G, size=int(round((1.0 - pi0) * G)), replace=False)
                is_null[alt_idx] = False

                x = draw_errors(rng_d, dist, (G, n)) * np.sqrt(ratios[:, None])
                y = draw_errors(rng_d, dist, (G, n)) + (~is_null)[:, None] * delta

                t_obs = pair_stat(x, y)[0]
                rng_p = make_rng(seed, "sig_perm", dist, delta, rep)
                bank = build_subpartition_bank(x, y, B, rng_p)
                abs_log_rhat = np.abs(np.log(x.var(1, ddof=1) / y.var(1, ddof=1)))

                p_methods = {
                    "welch_t": welch_pvalues(x, y),
                    "naive_K1": pool_p(t_obs, bank, None),
                }
                for K in strata_list:
                    p_methods[f"strat_K{K}"] = pool_p(t_obs, bank, labels_for(abs_log_rhat, K))

                for mname, pvals in p_methods.items():
                    for q in alphas:
                        rej = bh_mask(pvals, q)
                        R = int(rej.sum())
                        tp = int(np.sum(rej & ~is_null))
                        fp = R - tp
                        rows.append({
                            "dist": dist, "n": n, "delta": delta, "method": mname,
                            "alpha": q, "rep": rep,
                            "bh_rejections": R,
                            "bh_power": tp / int(np.sum(~is_null)),
                            "raw_power": float(np.mean(pvals[~is_null] <= q)),
                            "fdp": fp / R if R > 0 else 0.0,
                        })
            print(f"[Signal audit n={n}] {dist}, delta={delta} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dist", "delta", "alpha", "method"])
        .agg(
            bh_rejections=("bh_rejections", "mean"),
            bh_power=("bh_power", "mean"),
            raw_power=("raw_power", "mean"),
            fdr=("fdp", "mean"),
            fdr_mcse=("fdp", lambda s: np.std(s, ddof=1) / np.sqrt(len(s))),
        )
        .reset_index()
    )
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--outdir", default="output_sp3")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    reps = 20 if args.mode == "smoke" else 200

    _, null_sum = run_null_audit(reps=reps, n=8, B=1000, K=10)
    null_sum.to_csv(out / f"sp3_null_audit_{args.mode}.csv", index=False)

    _, sig_sum = run_signal_audit(reps=reps, n=10, B=2000)
    sig_sum.to_csv(out / f"sp3_signal_audit_{args.mode}.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print("\n=== NULL AUDIT SUMMARY ===")
    print(null_sum.to_string(index=False))
    print("\n=== SIGNAL AUDIT SUMMARY ===")
    print(sig_sum.to_string(index=False))


if __name__ == "__main__":
    main()