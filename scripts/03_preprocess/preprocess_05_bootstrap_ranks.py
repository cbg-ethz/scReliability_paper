#!/usr/bin/env python3
"""
Step 5: Bootstrap ranking stability for Fig 2a.

Resamples perturbations (with replacement, stratified by dataset × triage category)
to quantify how stable the "all → genuine" rank change is under perturbation-level
sampling variability. Computes both pooled and per-dataset bootstrap distributions
for the Wei et al. and Ahlmann-Eltze et al. benchmarks.

Reads:  intermediate/wei_genetic_merged_2d.csv
        intermediate/ae_panel_a_merged_2d.csv

Writes: intermediate/bootstrap_ranks_pooled_wei.csv
        intermediate/bootstrap_ranks_pooled_ae.csv
        intermediate/bootstrap_ranks_perdataset_wei.csv
        intermediate/bootstrap_ranks_perdataset_ae.csv
        intermediate/rank_stability_summary_wei.csv
        intermediate/rank_stability_summary_ae.csv

Design:
    • Paired bootstrap: within each iteration, sample perturbation keys once,
      then look up all methods' scores at those keys (preserves cross-method
      correlation structure).
    • Stratified: resample within (dataset × triage) to preserve class structure
      and avoid degenerate bootstrap samples with 0 genuine in small datasets.
    • Excess over baseline: same dataset-weighted mean aggregation as
      config.ds_weighted_excess (per-dataset mean, then mean across datasets).
    • Per-dataset: within-dataset resample, rank change is dataset-local.

Usage:
    python preprocess_05_bootstrap_ranks.py [--n_boot 1000] [--seed 42]
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from config import (INTERMEDIATE_DIR, intermediate_path)
import numpy as np
import pandas as pd

# Matches the Wei skip set used by Figure 2a and config.BENCHMARK_SETTINGS: baseMLP and baseReg stay in
# the ranking as competing baselines, only baseControl / trainMean / scFoundation are excluded.
SKIP_WEI = {'baseControl', 'trainMean', 'scFoundation'}


# ══════════════════════════════════════════════════════════════════
# Core bootstrap routine
# ══════════════════════════════════════════════════════════════════
def _wide_pivot(df, metric, pert_col):
    """Pivot to wide form: (dataset, perturbation) × method → metric.

    Also returns per-row dataset and triage arrays for stratification.
    """
    wide = df.pivot_table(
        index=['dataset', pert_col],
        columns='method',
        values=metric,
        aggfunc='mean',  # collapse any duplicate (dataset, pert, method) rows
    )
    # Triage per (dataset, perturbation)
    triage_lookup = df.groupby(['dataset', pert_col])['triage_2d'].first()
    wide = wide.join(triage_lookup.rename('__triage__'), how='left')
    wide['__dataset__'] = wide.index.get_level_values('dataset')
    return wide


def _strata_positions(wide):
    """Map (dataset, triage) → integer positions in `wide` rows."""
    wide_reset = wide.reset_index(drop=True)
    groups = wide_reset.groupby(['__dataset__', '__triage__']).indices
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _dataset_weighted_excess(excess_mat, datasets, methods, ds_order):
    """Per-iter aggregation matching config.ds_weighted_excess.

    excess_mat : (N, M) array of per-row (metric - baseline) excesses
    datasets   : (N,) string array of dataset per row
    methods    : list of M method names
    ds_order   : list of datasets to include (weights each dataset equally)

    Returns
    -------
    pd.Series indexed by method (mean of per-dataset means).
    """
    df = pd.DataFrame(excess_mat, columns=methods)
    df['__dataset__'] = datasets
    df = df[df['__dataset__'].isin(ds_order)]
    if len(df) == 0:
        return pd.Series({m: np.nan for m in methods})
    per_ds = df.groupby('__dataset__')[methods].mean()
    # Equal-weight across the datasets that are actually present in this sample
    return per_ds.mean(axis=0)


def bootstrap_ranks(df, metric, bl_method, pert_col, skip, n_boot=1000,
                    seed=42, scope='pooled', min_genuine=5):
    """Run stratified paired bootstrap, returning per-iter ranks.

    Parameters
    ----------
    df : DataFrame (long form; one row per (dataset, method, perturbation))
    metric : str (column name of the score to rank)
    bl_method : str (baseline method name)
    pert_col : str ('perturbation')
    skip : set[str] (methods to exclude from ranking)
    n_boot : int
    seed : int
    scope : 'pooled' or 'per_dataset'
    min_genuine : int (minimum genuine count required to include a dataset in
                       per-dataset analysis)

    Returns
    -------
    DataFrame with columns [dataset_subset, iter, stratum, method, rank, excess]
    (`dataset_subset` = 'pooled' in pooled scope, or the dataset name in per_dataset)
    """
    wide = _wide_pivot(df, metric, pert_col)

    # Methods to rank (exclude baseline and skip set)
    all_methods_in_df = [m for m in wide.columns if m not in ('__triage__', '__dataset__')]
    methods = sorted([m for m in all_methods_in_df if m not in skip and m != bl_method])
    if bl_method not in wide.columns:
        raise ValueError(f"Baseline '{bl_method}' not in data")

    # Build numeric matrices for speed
    method_cols = methods + [bl_method]
    mat = wide[method_cols].to_numpy(dtype=np.float64)
    ds_arr = wide['__dataset__'].to_numpy()
    tr_arr = wide['__triage__'].to_numpy()
    bl_idx = len(methods)  # last column is baseline

    rng = np.random.default_rng(seed)
    records = []

    if scope == 'pooled':
        # Strata: (dataset, triage) across all datasets
        strata = {}
        for (ds, tr), idx in pd.DataFrame({'ds': ds_arr, 'tr': tr_arr}).groupby(['ds','tr']).indices.items():
            strata[(ds, tr)] = np.asarray(idx, dtype=np.int64)

        all_datasets = sorted(set(ds_arr))

        for b in range(n_boot):
            sampled = np.concatenate([
                rng.choice(strata[(ds, tr)], size=strata[(ds, tr)].size, replace=True)
                for (ds, tr) in strata
            ])
            sample_mat = mat[sampled]
            sample_ds = ds_arr[sampled]
            sample_tr = tr_arr[sampled]

            # Excess over baseline (element-wise)
            excess = sample_mat[:, :bl_idx] - sample_mat[:, bl_idx:bl_idx+1]

            # All stratum
            nonan_mask = ~np.isnan(excess).all(axis=1)
            exc_all = _dataset_weighted_excess(
                excess[nonan_mask], sample_ds[nonan_mask], methods, all_datasets)
            rank_all = exc_all.rank(ascending=False, method='min')

            # Genuine stratum
            gen_mask = (sample_tr == 'Specific') & nonan_mask
            if gen_mask.sum() > 0:
                exc_gen = _dataset_weighted_excess(
                    excess[gen_mask], sample_ds[gen_mask], methods, all_datasets)
                rank_gen = exc_gen.rank(ascending=False, method='min')
            else:
                exc_gen = pd.Series({m: np.nan for m in methods})
                rank_gen = pd.Series({m: np.nan for m in methods})

            for m in methods:
                records.append({'dataset_subset': 'pooled', 'iter': b, 'stratum': 'all',
                                'method': m, 'excess': float(exc_all[m]),
                                'rank': float(rank_all[m])})
                records.append({'dataset_subset': 'pooled', 'iter': b, 'stratum': 'genuine',
                                'method': m, 'excess': float(exc_gen[m]),
                                'rank': float(rank_gen[m])})

    elif scope == 'per_dataset':
        # Per-dataset strata: (triage,) within each dataset
        unique_ds = sorted(set(ds_arr))
        for ds in unique_ds:
            ds_mask = (ds_arr == ds)
            ds_positions = np.where(ds_mask)[0]
            ds_tr = tr_arr[ds_mask]

            # Skip datasets with too few genuine
            n_gen = int((ds_tr == 'Specific').sum())
            if n_gen < min_genuine:
                continue

            strata = {}
            for tr in np.unique(ds_tr):
                idx = ds_positions[ds_tr == tr]
                strata[tr] = idx

            for b in range(n_boot):
                sampled = np.concatenate([
                    rng.choice(strata[tr], size=strata[tr].size, replace=True)
                    for tr in strata
                ])
                sample_mat = mat[sampled]
                sample_tr = tr_arr[sampled]

                excess = sample_mat[:, :bl_idx] - sample_mat[:, bl_idx:bl_idx+1]

                # All stratum (within this dataset → just the overall mean)
                with np.errstate(invalid='ignore'):
                    exc_all_vals = np.nanmean(excess, axis=0)
                exc_all = pd.Series(dict(zip(methods, exc_all_vals)))
                rank_all = exc_all.rank(ascending=False, method='min')

                # Genuine stratum
                gen_mask = (sample_tr == 'Specific')
                if gen_mask.sum() > 0:
                    with np.errstate(invalid='ignore'):
                        exc_gen_vals = np.nanmean(excess[gen_mask], axis=0)
                    exc_gen = pd.Series(dict(zip(methods, exc_gen_vals)))
                    rank_gen = exc_gen.rank(ascending=False, method='min')
                else:
                    exc_gen = pd.Series({m: np.nan for m in methods})
                    rank_gen = pd.Series({m: np.nan for m in methods})

                for m in methods:
                    records.append({'dataset_subset': ds, 'iter': b, 'stratum': 'all',
                                    'method': m, 'excess': float(exc_all[m]),
                                    'rank': float(rank_all[m])})
                    records.append({'dataset_subset': ds, 'iter': b, 'stratum': 'genuine',
                                    'method': m, 'excess': float(exc_gen[m]),
                                    'rank': float(rank_gen[m])})
    else:
        raise ValueError(f"scope must be 'pooled' or 'per_dataset', got {scope}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════
def summarize(df_boot):
    """Compute bootstrap 95% CIs per (dataset_subset, method).

    Returns a long-form summary with: dataset_subset, method,
    rank_all_med/lo/hi, rank_gen_med/lo/hi, delta_rank_med/lo/hi,
    stable_shift (bool: CI for delta_rank excludes 0),
    excess_all_med/lo/hi, excess_gen_med/lo/hi.
    """
    rows = []
    for (subset, method), grp in df_boot.groupby(['dataset_subset', 'method']):
        g_all = grp[grp['stratum'] == 'all']
        g_gen = grp[grp['stratum'] == 'genuine']
        # Align by iter (paired)
        merged = g_all[['iter','rank','excess']].merge(
            g_gen[['iter','rank','excess']], on='iter', suffixes=('_all','_gen'))
        if len(merged) == 0:
            continue
        delta = (merged['rank_all'] - merged['rank_gen']).to_numpy()
        ra = merged['rank_all'].to_numpy()
        rg = merged['rank_gen'].to_numpy()
        ea = merged['excess_all'].to_numpy()
        eg = merged['excess_gen'].to_numpy()

        def q(a, p):
            a = a[~np.isnan(a)]
            return float(np.percentile(a, p)) if len(a) else np.nan

        dlo, dhi = q(delta, 2.5), q(delta, 97.5)
        rows.append({
            'dataset_subset': subset,
            'method': method,
            'rank_all_med': q(ra, 50), 'rank_all_lo': q(ra, 2.5), 'rank_all_hi': q(ra, 97.5),
            'rank_gen_med': q(rg, 50), 'rank_gen_lo': q(rg, 2.5), 'rank_gen_hi': q(rg, 97.5),
            'delta_rank_med': q(delta, 50), 'delta_rank_lo': dlo, 'delta_rank_hi': dhi,
            'stable_shift': bool((dlo > 0) or (dhi < 0)) if not (np.isnan(dlo) or np.isnan(dhi)) else False,
            'excess_all_med': q(ea, 50), 'excess_all_lo': q(ea, 2.5), 'excess_all_hi': q(ea, 97.5),
            'excess_gen_med': q(eg, 50), 'excess_gen_lo': q(eg, 2.5), 'excess_gen_hi': q(eg, 97.5),
            'n_iter_valid': int((~np.isnan(delta)).sum()),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_genuine', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(str(INTERMEDIATE_DIR), exist_ok=True)

    # ── Load data (mirror fig2_benchmarks.py preprocessing) ──
    print(f"[{time.strftime('%H:%M:%S')}] Loading intermediate tables...")
    wei_gen = pd.read_csv(intermediate_path('wei_genetic_merged_2d.csv'))
    ae_a = pd.read_csv(intermediate_path('ae_panel_a_merged_2d.csv'), encoding='latin-1')
    ae_a_test = ae_a[ae_a['train'] == 'test'].copy()
    ae_a_avg = ae_a_test.groupby(
        ['method', 'dataset', 'perturbation', 'triage_2d']
    ).agg(r2_delta=('r2_delta', 'mean')).reset_index()

    print(f"  Wei:  {len(wei_gen):,} rows, {wei_gen['dataset'].nunique()} datasets, "
          f"{wei_gen['method'].nunique()} methods")
    print(f"  AE:   {len(ae_a_avg):,} rows (seed-avg test), "
          f"{ae_a_avg['dataset'].nunique()} datasets, {ae_a_avg['method'].nunique()} methods")

    # ── Bootstrap: Wei ──
    print(f"\n[{time.strftime('%H:%M:%S')}] Wei pooled bootstrap (n_boot={args.n_boot})...")
    t0 = time.time()
    wei_pooled = bootstrap_ranks(
        wei_gen, metric='pcc', bl_method='trainMean', pert_col='perturbation',
        skip=SKIP_WEI, n_boot=args.n_boot, seed=args.seed, scope='pooled')
    print(f"  done in {time.time()-t0:.1f}s, {len(wei_pooled):,} records")
    wei_pooled.to_csv(intermediate_path('bootstrap_ranks_pooled_wei.csv'), index=False)

    print(f"[{time.strftime('%H:%M:%S')}] Wei per-dataset bootstrap...")
    t0 = time.time()
    wei_perds = bootstrap_ranks(
        wei_gen, metric='pcc', bl_method='trainMean', pert_col='perturbation',
        skip=SKIP_WEI, n_boot=args.n_boot, seed=args.seed, scope='per_dataset',
        min_genuine=args.min_genuine)
    print(f"  done in {time.time()-t0:.1f}s, {len(wei_perds):,} records, "
          f"datasets: {sorted(wei_perds['dataset_subset'].unique())}")
    wei_perds.to_csv(intermediate_path('bootstrap_ranks_perdataset_wei.csv'), index=False)

    # ── Bootstrap: AE ──
    print(f"\n[{time.strftime('%H:%M:%S')}] AE pooled bootstrap...")
    t0 = time.time()
    ae_pooled = bootstrap_ranks(
        ae_a_avg, metric='r2_delta', bl_method='mean', pert_col='perturbation',
        skip=set(), n_boot=args.n_boot, seed=args.seed, scope='pooled')
    print(f"  done in {time.time()-t0:.1f}s, {len(ae_pooled):,} records")
    ae_pooled.to_csv(intermediate_path('bootstrap_ranks_pooled_ae.csv'), index=False)

    print(f"[{time.strftime('%H:%M:%S')}] AE per-dataset bootstrap...")
    t0 = time.time()
    ae_perds = bootstrap_ranks(
        ae_a_avg, metric='r2_delta', bl_method='mean', pert_col='perturbation',
        skip=set(), n_boot=args.n_boot, seed=args.seed, scope='per_dataset',
        min_genuine=args.min_genuine)
    print(f"  done in {time.time()-t0:.1f}s, {len(ae_perds):,} records, "
          f"datasets: {sorted(ae_perds['dataset_subset'].unique())}")
    ae_perds.to_csv(intermediate_path('bootstrap_ranks_perdataset_ae.csv'), index=False)

    # ── Summaries ──
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing summaries...")
    wei_all = pd.concat([wei_pooled, wei_perds], ignore_index=True)
    ae_all = pd.concat([ae_pooled, ae_perds], ignore_index=True)
    wei_summary = summarize(wei_all)
    ae_summary = summarize(ae_all)
    wei_summary.to_csv(intermediate_path('rank_stability_summary_wei.csv'), index=False)
    ae_summary.to_csv(intermediate_path('rank_stability_summary_ae.csv'), index=False)

    # ── Quick text summary ──
    print("\n" + "="*70)
    print("RANK STABILITY SUMMARY")
    print("="*70)
    for label, summ in [("Wei", wei_summary), ("AE", ae_summary)]:
        print(f"\n── {label} ──")
        for subset in summ['dataset_subset'].unique():
            ss = summ[summ['dataset_subset'] == subset].sort_values('rank_gen_med')
            n_stable = int(ss['stable_shift'].sum())
            n_tot = len(ss)
            print(f"  {subset}: {n_stable}/{n_tot} methods with stable ΔRank (95% CI excludes 0)")
            top = ss.head(3)
            for _, r in top.iterrows():
                print(f"    {r['method']:>15s}  "
                      f"rank_all={r['rank_all_med']:.0f} [{r['rank_all_lo']:.0f},{r['rank_all_hi']:.0f}]  "
                      f"rank_gen={r['rank_gen_med']:.0f} [{r['rank_gen_lo']:.0f},{r['rank_gen_hi']:.0f}]  "
                      f"Δ={r['delta_rank_med']:+.1f} [{r['delta_rank_lo']:+.1f},{r['delta_rank_hi']:+.1f}]"
                      f"{'  *' if r['stable_shift'] else ''}")

    print("\n✓ Step 5 complete.")


if __name__ == '__main__':
    main()
