#!/usr/bin/env python3
"""
Step 3: Compute derived tables from merged data.

Reads:  INTERMEDIATE_DIR/wei_genetic_merged_2d.csv
        INTERMEDIATE_DIR/ae_panel_c_merged_2d.csv
Writes: INTERMEDIATE_DIR/rankings_genetic_top100.csv
        INTERMEDIATE_DIR/rankings_genetic_top5000.csv
        INTERMEDIATE_DIR/shared_fraction_ae_panel_c.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
import pandas as pd
import numpy as np

os.makedirs(str(INTERMEDIATE_DIR), exist_ok=True)

# ---- Rankings table ----
def compute_rankings(merged_path, out_name, label):
    """Compute method rankings from merged Wei data."""
    if not os.path.exists(merged_path):
        print(f"⚠ Not found: {merged_path}")
        return

    df = pd.read_csv(merged_path)
    skip = SKIP_GEN
    df = df[~df['method'].isin(skip)]
    print(f"\n[{label}] {len(df)} rows after removing baselines")

    # Mean PCC per method across all/genuine/trivial
    rows = []
    for method in sorted(df['method'].unique()):
        msub = df[df['method'] == method]
        row = {'method': method}

        # All conditions
        row['mean_all'] = msub['pcc'].mean()
        row['median_all'] = msub['pcc'].median()
        row['n_all'] = len(msub)

        # Genuine
        gen = msub[msub['triage_2d'] == 'Specific']
        row['mean_genuine'] = gen['pcc'].mean() if len(gen) > 0 else np.nan
        row['median_genuine'] = gen['pcc'].median() if len(gen) > 0 else np.nan
        row['n_genuine'] = len(gen)

        # Trivial (Shared)
        triv = msub[msub['triage_2d'] == 'Shared']
        row['mean_falsely'] = triv['pcc'].mean() if len(triv) > 0 else np.nan
        row['n_falsely'] = len(triv)

        # Mean CE on genuine
        gen_ce = gen['CE'].dropna()
        row['mean_CE_genuine'] = gen_ce.mean() if len(gen_ce) > 0 else np.nan

        rows.append(row)

    rdf = pd.DataFrame(rows)

    # Rankings
    rdf['rank_all']     = rdf['mean_all'].rank(ascending=False).astype(int)
    rdf['rank_genuine'] = rdf['mean_genuine'].rank(ascending=False).astype(int)
    rdf['rank_delta']   = rdf['rank_all'] - rdf['rank_genuine']

    # PCC drop: mean_genuine - mean_all
    rdf['pcc_drop'] = rdf['mean_genuine'] - rdf['mean_all']

    # Sort by rank_genuine
    rdf = rdf.sort_values('rank_genuine')

    out = intermediate_path(out_name)
    rdf.to_csv(out, index=False)
    print(f"✓ Saved {out}")
    print(rdf[['method', 'mean_all', 'mean_genuine', 'rank_all', 'rank_genuine', 'rank_delta']].to_string())

print("=== Rankings (top 100 DEGs) ===")
compute_rankings(intermediate_path('wei_genetic_merged_2d.csv'),
                 'rankings_genetic_top100.csv', 'genetic_top100')

if os.path.exists(intermediate_path('wei_genetic_5k_merged_2d.csv')):
    print("\n=== Rankings (top 5000) ===")
    compute_rankings(intermediate_path('wei_genetic_5k_merged_2d.csv'),
                     'rankings_genetic_top5000.csv', 'genetic_top5000')


# ---- Shared fraction (AE Panel C) ----
ae_c_path = intermediate_path('ae_panel_c_merged_2d.csv')
if os.path.exists(ae_c_path):
    print("\n=== Shared fraction (AE Panel C) ===")
    ae_c = pd.read_csv(ae_c_path)

    # arch_group
    ae_c['arch_group'] = ae_c['method'].map(AE_ARCH_GROUP)

    # trainMean baseline: per (dataset, perturbation, seed, train), get 'mean' method r2_delta
    mean_bl = ae_c[ae_c['method'] == 'mean'][
        ['dataset', 'perturbation', 'seed', 'train', 'r2_delta']
    ].rename(columns={'r2_delta': 'trainMean_score'})

    # If 'mean' method not present, use lpm_selftrained as baseline
    if len(mean_bl) == 0:
        print("  Note: 'mean' method not in Panel C, using lpm_selftrained as baseline")
        mean_bl = ae_c[ae_c['method'] == 'lpm_selftrained'][
            ['dataset', 'perturbation', 'seed', 'train', 'r2_delta']
        ].rename(columns={'r2_delta': 'trainMean_score'})

    sf = ae_c.merge(mean_bl, on=['dataset', 'perturbation', 'seed', 'train'], how='left')

    # excess = r2_delta - trainMean_score
    sf['excess'] = sf['r2_delta'] - sf['trainMean_score']

    # shared_fraction = r2_delta / trainMean_score (where trainMean > 0)
    sf['shared_fraction'] = np.where(
        sf['trainMean_score'].abs() > 0.001,
        sf['r2_delta'] / sf['trainMean_score'],
        np.nan)

    # pcc = r2_delta (alias used by some figure scripts)
    sf['pcc'] = sf['r2_delta']

    out = intermediate_path('shared_fraction_ae_panel_c.csv')
    sf.to_csv(out, index=False)
    print(f"✓ Saved {out} ({len(sf)} rows)")
    print(f"  Methods: {sorted(sf['method'].unique())}")
else:
    print("⚠ AE Panel C not found, skipping shared fraction.")

print("\n✓ Step 3 complete.")
