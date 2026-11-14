# -*- coding: utf-8 -*-
"""Subpartition audit simulation, version SP-1.
Normal errors, two groups, even equal group sizes.
A tests paired distributional equality using independent datasets.
B/C/E use complete-null data; D uses a genuine null/alternative mixture.
Repeated splits recompute weights for each split and use those same weights
for its X and D statistics. No cross-group permutation is used.
"""
import argparse
import hashlib
import json
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp, f, false_discovery_control

VERSION = 'SP-1'

@dataclass
class Config:
    seed: int = 2026
    n: int = 20
    G: int = 1000
    B: int = 20
    reps: int = 20
    reference_size: int = 20000
    a_size: int = 10000
    pi0: float = 0.8
    ratios: tuple = (1., 4., 16., 64., 256.)
    alphas: tuple = (.05, .01)
    strata: tuple = (4, 10, 20)
    effects: tuple = (.5, 1., 2., 4.)

    def rng(self, *parts):
        text = repr((self.seed, parts)).encode()
        entropy = int.from_bytes(hashlib.sha256(text).digest()[:16], 'little')
        return np.random.default_rng(entropy)

    def validate(self):
        if self.n < 4 or self.n % 2:
            raise ValueError('n must be even and >= 4')
        if min(self.G, self.B, self.reps, self.reference_size, self.a_size) < 2:
            raise ValueError('Counts must be >= 2')
        if self.G < max(self.strata) or not 0 < self.pi0 < 1:
            raise ValueError('Invalid G or pi0')
        if any(r <= 0 for r in self.ratios):
            raise ValueError('Ratios must be positive')


def pair_stat(x, y):
    """Last axis: first h and last h observations form a fixed split."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.shape[-1] < 4 or x.shape[-1] % 2:
        raise ValueError('Equal shapes and even sample sizes >= 4 required')
    h = x.shape[-1] // 2
    ma, mb = x[..., :h].mean(-1), x[..., h:].mean(-1)
    na, nb = y[..., :h].mean(-1), y[..., h:].mean(-1)
    vx = (x[..., :h].var(-1, ddof=1) + x[..., h:].var(-1, ddof=1))/(4*h)
    vy = (y[..., :h].var(-1, ddof=1) + y[..., h:].var(-1, ddof=1))/(4*h)
    den = vx + vy
    if np.any(~np.isfinite(den)) or np.any(den <= 0):
        raise ValueError('Nonpositive or nonfinite variance')
    t = ((ma+mb-na-nb)/2)**2/den
    u = ((ma-mb-na+nb)/2)**2/den
    return t, u


def empirical_p(reference, observed):
    ref = np.sort(np.asarray(reference).ravel())
    obs = np.asarray(observed)
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(obs)):
        raise ValueError('Nonfinite statistic')
    return (1 + len(ref)-np.searchsorted(ref, obs, side='left'))/(len(ref)+1)


def bh(p, q):
    p = np.asarray(p)
    order = np.argsort(p)
    hit = np.flatnonzero(p[order] <= q*np.arange(1, len(p)+1)/len(p))
    out = np.zeros(len(p), bool)
    if hit.size:
        out[order[:hit[-1]+1]] = True
    return out


def fdp(rejected, is_null):
    return float(np.sum(rejected & is_null)/max(int(rejected.sum()), 1))


def reference(cfg, ratio, tag):
    """Independent datasets: marginal null distribution, not re-splits of one dataset."""
    rng = cfg.rng('reference', tag, ratio)
    x = rng.normal(size=(cfg.reference_size, cfg.n))*np.sqrt(ratio)
    y = rng.normal(size=x.shape)
    return pair_stat(x, y)[0]


def generate(cfg, rep, scenario, delta=0.):
    rng = cfg.rng('data', scenario, rep)
    r = np.resize(np.array(cfg.ratios), cfg.G)
    is_null = np.ones(cfg.G, bool)
    if scenario == 'D':
        is_null[int(cfg.pi0*cfg.G):] = False
    x = rng.normal(size=(cfg.G, cfg.n))*np.sqrt(r[:, None])
    y = rng.normal(size=x.shape) + (~is_null)[:, None]*delta
    return x, y, r, is_null


def observed_and_bank(cfg, x, y, rep, tag):
    t = pair_stat(x, y)[0]
    rng = cfg.rng('splits', tag, rep)
    bank = np.empty((cfg.G, cfg.B))
    bank[:, 0] = pair_stat(x, y)[1]
    for b in range(1, cfg.B):
        ix = rng.permuted(np.broadcast_to(np.arange(cfg.n), x.shape), axis=1)
        iy = rng.permuted(np.broadcast_to(np.arange(cfg.n), y.shape), axis=1)
        bank[:, b] = pair_stat(np.take_along_axis(x, ix, 1),
                               np.take_along_axis(y, iy, 1))[1]
    return t, bank


def labels_for(values, count):
    order = np.argsort(values, kind='stable')
    labels = np.empty(len(values), int)
    labels[order] = np.arange(len(values))*count//len(values)
    return labels


def pool_p(t, bank, labels=None):
    if labels is None:
        return empirical_p(bank, t)
    out = np.full(len(t), np.nan)
    for ell in np.unique(labels):
        idx = labels == ell
        out[idx] = empirical_p(bank[idx], t[idx])
    if not np.all(np.isfinite(out)):
        raise RuntimeError('Unassigned p-values')
    return out


def summarize_rows(raw):
    df = pd.DataFrame(raw)
    keys = ['scenario', 'n', 'method', 'ratio', 'stratum', 'n_strata', 'alpha', 'delta', 'metric']
    result = df.groupby(keys, dropna=False)['value'].agg(['mean', 'std', 'count']).reset_index()
    result['mcse'] = result['std']/np.sqrt(result['count'])
    return df, result


def run(cfg, scenarios):
    rows = []
    def add(s, method, metric, value, **kw):
        row = dict(scenario=s, n=cfg.n, method=method, metric=metric,
                   ratio=np.nan, stratum=-1, n_strata=0,
                   alpha=np.nan, delta=0., rep=-1, value=float(value))
        row.update(kw)
        rows.append(row)

    if 'A' in scenarios:
        for ratio in cfg.ratios:
            rng = cfg.rng('A', ratio)
            x = rng.normal(size=(cfg.a_size, cfg.n))*np.sqrt(ratio)
            y = rng.normal(size=x.shape)
            t, u = pair_stat(x, y)
            cut = cfg.a_size//2
            add('A', 'fixed_split_pair', 'paired_mean_difference', np.mean(t-u), ratio=ratio)
            add('A', 'independent_halves', 'ks_distance', ks_2samp(t[:cut], u[cut:]).statistic, ratio=ratio)
            ref = reference(cfg, ratio, 'A')
            for source, values in [('observed', t), ('null', u)]:
                p = empirical_p(ref, values)
                for q in cfg.alphas:
                    for j, val in enumerate(p <= q):
                        add('A', source, 'oracle_reference_size', val,
                            ratio=ratio, alpha=q, rep=j)
        print('A completed', flush=True)

    refs = {r: reference(cfg, r, 'B') for r in cfg.ratios} if 'B' in scenarios else {}
    for s in ('B', 'C', 'D', 'E'):
        if s not in scenarios:
            continue
        for rep in range(cfg.reps):
            deltas = cfg.effects if s == 'D' else (0.,)
            for delta in deltas:
                x, y, ratios, truth = generate(cfg, rep, s, delta)
                if s == 'E':
                    x = x/np.sqrt(ratios[:, None])
                    ratios = np.ones(cfg.G)
                t, bank = observed_and_bank(cfg, x, y, rep, s)
                settings = [('pooled', 0, None)]
                if s in ('C', 'D'):
                    estimate = np.log(x.var(1, ddof=1)/y.var(1, ddof=1))
                    for count in cfg.strata:
                        settings.append(('estimated_strata', count, labels_for(estimate, count)))
                    settings.append(('oracle_ratio_strata', len(cfg.ratios),
                                     np.searchsorted(np.array(cfg.ratios), ratios)))
                for method, count, labels in settings:
                    p = pool_p(t, bank, labels)
                    for q in cfg.alphas:
                        rejected = bh(p, q)
                        add(s, method, 'fdr', fdp(rejected, truth), alpha=q,
                            rep=rep, n_strata=count, delta=delta)
                        add(s, method, 'bh_rejections', rejected.sum(), alpha=q,
                            rep=rep, n_strata=count, delta=delta)
                        for r in np.unique(ratios):
                            mask = truth & (ratios == r)
                            add(s, method, 'raw_size', np.mean(p[mask] <= q), ratio=r,
                                alpha=q, rep=rep, n_strata=count, delta=delta)
                        if labels is not None:
                            for ell in np.unique(labels):
                                mask = truth & (labels == ell)
                                if mask.any():
                                    add(s, method, 'raw_size_by_stratum', np.mean(p[mask] <= q),
                                        stratum=int(ell), alpha=q, rep=rep, n_strata=count, delta=delta)
                        if s == 'D':
                            add(s, method, 'bh_power', np.mean(rejected[~truth]), alpha=q,
                                rep=rep, n_strata=count, delta=delta)
                            add(s, method, 'raw_power', np.mean(p[~truth] <= q), alpha=q,
                                rep=rep, n_strata=count, delta=delta)
                if s == 'B':
                    for r in cfg.ratios:
                        add(s, 'pooled', 'ks_to_independent_null',
                            ks_2samp(bank.ravel(), refs[r], method='asymp').statistic,
                            ratio=r, rep=rep)
            if rep == 0 or (rep+1) % 10 == 0:
                print(f'{s}: replicate {rep+1}/{cfg.reps}', flush=True)
    return summarize_rows(rows)


def self_test():
    rng = np.random.default_rng(49)
    x = rng.normal(size=(300, 20))
    y = rng.normal(size=x.shape)*3
    t, u = pair_stat(x, y)
    h = 10
    xf, yf = x.copy(), y.copy()
    xf[:, h:] *= -1
    yf[:, h:] *= -1
    tf, uf = pair_stat(xf, yf)
    np.testing.assert_allclose(tf, u, atol=1e-12)
    np.testing.assert_allclose(uf, t, atol=1e-12)
    for scale, shift in [(3., 7.), (-2., -4.)]:
        ts, us = pair_stat(scale*x+shift, scale*y+shift)
        np.testing.assert_allclose(ts, t, atol=1e-12)
        np.testing.assert_allclose(us, u, atol=1e-12)
    _, ud = pair_stat(x+5., y-2.)
    np.testing.assert_allclose(ud, u, atol=1e-12)
    ref = np.array([1., 2., 2., 4.])
    obs = np.array([0., 2., 5.])
    np.testing.assert_allclose(empirical_p(ref, obs),
        [(1+np.sum(ref >= v))/(len(ref)+1) for v in obs])
    p = rng.uniform(size=1000)
    p[:30] = np.linspace(.00001, .001, 30)
    for q in (.01, .05):
        np.testing.assert_array_equal(bh(p, q), false_discovery_control(p, method='bh') <= q)
    assert Config(seed=1).rng('a').random() != Config(seed=2).rng('a').random()
    # Independent normal datasets validate scale and degrees of freedom.
    x = rng.normal(size=(20000, 20)); y = rng.normal(size=x.shape)
    t, u = pair_stat(x, y)
    for q in (.05, .01):
        threshold = f.isf(q, 1, 36)
        assert abs(np.mean(t > threshold)-q) < .008
        assert abs(np.mean(u > threshold)-q) < .008
    return {'status': 'passed', 'tests': ['sign_reversal_identity', 'affine_invariance',
        'null_shift_invariance', 'tail_ties', 'BH_vs_SciPy', 'master_seed',
        'normal_F_1_36_calibration'], 'normal_size_0.05': float(np.mean(t > f.isf(.05,1,36)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('test', 'smoke', 'full'), default='test')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--B', type=int, default=20)
    p.add_argument('--G', type=int, default=1000)
    p.add_argument('--reps', type=int)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--scenarios', default='ABCDE')
    p.add_argument('--outdir', default='output_sp1')
    args = p.parse_args()
    if not set(args.scenarios) <= set('ABCDE') or not args.scenarios:
        p.error('scenarios must be a nonempty subset of ABCDE')
    report = self_test()
    print('Self-tests:', report, flush=True)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out/'self_tests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.mode == 'test':
        return
    cfg = Config(seed=args.seed, n=args.n, B=args.B, G=args.G,
                 reps=args.reps or (20 if args.mode == 'smoke' else 500))
    cfg.validate()
    metadata = dict(version=VERSION, config=asdict(cfg), python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scenarios=args.scenarios,
                    warning='Empirical audit, not a proof of exact p-values or FDR control')
    stem = f'{args.mode}_n{cfg.n}_B{cfg.B}'
    (out/f'config_{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(VERSION, asdict(cfg), flush=True)
    raw, summary = run(cfg, args.scenarios)
    raw.to_csv(out/f'replicates_{stem}.csv', index=False)
    summary.to_csv(out/f'summary_{stem}.csv', index=False)
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
