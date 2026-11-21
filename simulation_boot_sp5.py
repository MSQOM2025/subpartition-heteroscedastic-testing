# -*- coding: utf-8 -*-
"""simulation_boot_sp5.py

Evaluates Parametric Gaussian Bootstrap (both per-feature and pooled across features)
on the exact same 200-replicate data streams as simulation_extension_sp3.py.
"""

import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erfc


def make_rng(seed, *parts):
    text = repr((seed, parts)).encode()
    entropy = int.from_bytes(hashlib.sha256(text).digest()[:16], "little")
    return np.random.default_rng(entropy)


def draw_errors(rng, dist, shape):
    if dist == "normal":
        return rng.normal(size=shape)
    if dist == "t3":
        return rng.standard_t(3, size=shape) / np.sqrt(3.0)
    if dist == "contam":
        mix = rng.random(size=shape) < 0.15
        raw = rng.normal(size=shape) * np.where(mix, 5.0, 1.0)
        return raw / np.sqrt(4.6)
    raise ValueError(f"Unknown dist: {dist}")


def welch_stat_and_ratio(x, y):
    n = x.shape[-1]
    vx = np.maximum(x.var(-1, ddof=1), 1e-300)
    vy = np.maximum(y.var(-1, ddof=1), 1e-300)
    w_obs = (x.mean(-1) - y.mean(-1)) ** 2 / ((vx + vy) / n)
    r_hat = vx / vy
    return w_obs, r_hat


def boot_gauss_per_feature_p(w_obs, r_hat, n, rng, M=2000):
    """Rao-Blackwellized per-feature Gaussian parametric bootstrap p-values for Welch W_g."""
    u1 = rng.chisquare(n - 1, size=(1, M))
    u2 = rng.chisquare(n - 1, size=(1, M))
    scale = (r_hat[:, None] * u1 + u2) / ((n - 1) * (1.0 + r_hat[:, None]))
    arg = np.sqrt(np.maximum(w_obs[:, None] * scale / 2.0, 0.0))
    return np.mean(erfc(arg), axis=1)


def boot_gauss_pooled_p(w_obs, r_hat, n, rng, B=1000):
    """Cross-feature pooled Gaussian parametric bootstrap p-values."""
    G = len(w_obs)
    z = rng.normal(size=(G, B))
    u1 = rng.chisquare(n - 1, size=(G, B))
    u2 = rng.chisquare(n - 1, size=(G, B))
    bank = (n - 1) * (1.0 + r_hat[:, None]) * (z ** 2) / (r_hat[:, None] * u1 + u2)
    ref = np.sort(bank.ravel())
    return (1 + len(ref) - np.searchsorted(ref, w_obs, side="left")) / (len(ref) + 1)


def bh_mask(p, q):
    order = np.argsort(p)
    hit = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    out = np.zeros(len(p), bool)
    if hit.size:
        out[order[: hit[-1] + 1]] = True
    return out


def run_null_boot(seed=2026, reps=200, G=1000, n=8, B=1000):
    ratios_grid = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
    ratios = np.resize(ratios_grid, G)
    rows = []
    t0 = time.time()

    for dist in ("normal", "t3", "contam"):
        for rep in range(reps):
            rng_d = make_rng(seed, "null_audit", dist, rep)
            x = draw_errors(rng_d, dist, (G, n)) * np.sqrt(ratios[:, None])
            y = draw_errors(rng_d, dist, (G, n))
            w_obs, r_hat = welch_stat_and_ratio(x, y)

            p_pf = boot_gauss_per_feature_p(w_obs, r_hat, n, make_rng(seed, "boot_pf", dist, rep))
            p_pool = boot_gauss_pooled_p(w_obs, r_hat, n, make_rng(seed, "boot_pool", dist, rep), B=B)

            for mname, pvals in [("gauss_boot_pf", p_pf), ("gauss_boot_pool", p_pool)]:
                for q in (0.05, 0.01):
                    for r in ratios_grid:
                        mask = ratios == r
                        rows.append({
                            "dist": dist, "method": mname, "alpha": q, "ratio": r,
                            "rep": rep, "size": float(np.mean(pvals[mask] <= q))
                        })
        print(f"[Boot Null n={n}] {dist} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby(["dist", "method", "alpha", "ratio"])["size"].agg(["mean", "std", "count"]).reset_index()
    summary["mcse"] = summary["std"] / np.sqrt(summary["count"])
    return summary


def run_signal_boot(seed=2026, reps=200, G=1000, n=10, B=2000, pi0=0.8):
    ratios_grid = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
    base_ratios = np.repeat(ratios_grid, G // len(ratios_grid))
    rows = []
    t0 = time.time()

    for dist in ("normal", "t3"):
        for delta in (1.0, 2.0, 4.0):
            for rep in range(reps):
                rng_d = make_rng(seed, "sig_data", dist, delta, rep)
                ratios = base_ratios[rng_d.permutation(G)]
                is_null = np.ones(G, bool)
                alt_idx = rng_d.choice(G, size=int(round((1.0 - pi0) * G)), replace=False)
                is_null[alt_idx] = False

                x = draw_errors(rng_d, dist, (G, n)) * np.sqrt(ratios[:, None])
                y = draw_errors(rng_d, dist, (G, n)) + (~is_null)[:, None] * delta
                w_obs, r_hat = welch_stat_and_ratio(x, y)

                p_pf = boot_gauss_per_feature_p(w_obs, r_hat, n, make_rng(seed, "sig_boot_pf", dist, delta, rep))
                p_pool = boot_gauss_pooled_p(w_obs, r_hat, n, make_rng(seed, "sig_boot_pool", dist, delta, rep), B=B)

                for mname, pvals in [("gauss_boot_pf", p_pf), ("gauss_boot_pool", p_pool)]:
                    for q in (0.05, 0.01):
                        rej = bh_mask(pvals, q)
                        R = int(rej.sum())
                        tp = int(np.sum(rej & ~is_null))
                        fp = R - tp
                        rows.append({
                            "dist": dist, "delta": delta, "method": mname, "alpha": q, "rep": rep,
                            "bh_power": tp / int(np.sum(~is_null)),
                            "fdp": fp / R if R > 0 else 0.0,
                        })
            print(f"[Boot Signal n={n}] {dist}, delta={delta} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    return df.groupby(["dist", "delta", "alpha", "method"]).agg(
        bh_power=("bh_power", "mean"),
        fdr=("fdp", "mean"),
        fdr_mcse=("fdp", lambda s: np.std(s, ddof=1) / np.sqrt(len(s))),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output_sp5")
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    null_sum = run_null_boot(reps=200)
    sig_sum = run_signal_boot(reps=200)
    null_sum.to_csv(out / "sp5_boot_null_full.csv", index=False)
    sig_sum.to_csv(out / "sp5_boot_signal_full.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    print("\n=== PARAMETRIC BOOTSTRAP NULL SUMMARY ===")
    print(null_sum.to_string(index=False))
    print("\n=== PARAMETRIC BOOTSTRAP SIGNAL SUMMARY ===")
    print(sig_sum.to_string(index=False))


if __name__ == "__main__":
    main()