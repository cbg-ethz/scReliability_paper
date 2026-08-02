#!/usr/bin/env python3
"""
Step 1: Combine per-dataset systema+reliability CSVs into combined triage tables.

Reads:  GENETIC_SYSTEMA_DIR/systema_reliability_2d_*.csv
        CELLULAR_SYSTEMA_DIR/systema_cc_reliability_2d_*.csv
Writes: INTERMEDIATE_DIR/combined_genetic_2d.csv
        INTERMEDIATE_DIR/combined_cellular_2d.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
import pandas as pd
import numpy as np


def rederive_triage(df, kind="genetic"):
    """Re-derive triage_2d from ρ + cos_sim using project-wide thresholds
    in config.py. Overrides whatever the systema output wrote, so threshold
    changes don't require re-running the slow compute_reliability pipeline."""
    rel_candidates = ["sb_from_median_all_genes", "sb_from_mean_all_genes",
                      "reliability", "sb_from_median"]
    rho_col = next((c for c in rel_candidates if c in df.columns), None)
    if rho_col is None or "cos_sim" not in df.columns:
        print(f"  ⚠ Cannot re-derive triage ({kind}): "
              f"need ρ + cos_sim columns. Have: {list(df.columns)[:10]}...")
        return df
    df = df.copy()
    df["triage_2d"] = derive_triage_2d(df, rho_col=rho_col)
    print(f"  Re-derived triage_2d using ρ≥{RELIABILITY_THRESH}, "
          f"cos θ≥{COSSIM_HIGH_THRESH:.4f} (1/√2) on column '{rho_col}'")
    return df

def load_merged_files(directory, prefix, suffix=".csv"):
    """Load all systema+reliability CSVs from a directory."""
    dfs = []
    directory = Path(directory)
    if not directory.exists():
        print(f"⚠ Directory not found: {directory}")
        return pd.DataFrame()
    for f in sorted(directory.glob(f"{prefix}*{suffix}")):
        df = pd.read_csv(f)
        if 'dataset' not in df.columns:
            df['dataset'] = f.stem.replace(prefix, '')
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df)} rows")
    if not dfs:
        print(f"  No files found with prefix '{prefix}' in {directory}")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows across {combined['dataset'].nunique()} datasets")
    return combined

# ── Main ──────────────────────────────────────────────────────────
os.makedirs(str(INTERMEDIATE_DIR), exist_ok=True)

print("=== Genetic Perturbation ===")
df_genetic = load_merged_files(GENETIC_SYSTEMA_DIR, prefix="systema_reliability_2d_")

print("\n=== Cellular Context ===")
df_cellular = load_merged_files(CELLULAR_SYSTEMA_DIR, prefix="systema_cc_reliability_2d_")

# Re-derive triage using project-wide thresholds (config.py)
if len(df_genetic) > 0:
    df_genetic = rederive_triage(df_genetic, kind="genetic")
if len(df_cellular) > 0:
    df_cellular = rederive_triage(df_cellular, kind="cellular")

# Save
if len(df_genetic) > 0:
    out = intermediate_path('combined_genetic_2d.csv')
    df_genetic.to_csv(out, index=False)
    print(f"\n✓ Saved {out} ({len(df_genetic)} rows, {df_genetic['dataset'].nunique()} datasets)")
    # Summary
    if 'triage_2d' in df_genetic.columns:
        print("  Triage breakdown:")
        for cat, n in df_genetic['triage_2d'].value_counts().items():
            print(f"    {cat}: {n} ({100*n/len(df_genetic):.1f}%)")

if len(df_cellular) > 0:
    out = intermediate_path('combined_cellular_2d.csv')
    df_cellular.to_csv(out, index=False)
    print(f"\n✓ Saved {out} ({len(df_cellular)} rows, {df_cellular['dataset'].nunique()} datasets)")
    if 'triage_2d' in df_cellular.columns:
        print("  Triage breakdown:")
        for cat, n in df_cellular['triage_2d'].value_counts().items():
            print(f"    {cat}: {n} ({100*n/len(df_cellular):.1f}%)")

# Pooled quality summary — regenerated from the freshly re-derived triage
if len(df_genetic) > 0 and len(df_cellular) > 0:
    print("\n── Pooled quality summary ──")
    all_df = pd.concat([df_genetic.assign(_scope='perturb_seq'),
                        df_cellular.assign(_scope='scrna')], ignore_index=True)
    # sci_plex = any dataset starting with "sciplex"; moved from perturb_seq
    all_df.loc[all_df['dataset'].astype(str).str.lower().str.startswith('sciplex'),
               '_scope'] = 'sci_plex'

    def _row(name, sub):
        vc = sub['triage_2d'].value_counts()
        n_unrel = int(vc.get('Unreliable', 0))
        n_triv  = int(vc.get('Shared', 0))
        n_gen   = int(vc.get('Specific', 0))
        n_tot = len(sub)
        return {'scope': name, 'n_total': n_tot,
                'n_unreliable': n_unrel, 'n_shared': n_triv, 'n_specific': n_gen,
                'frac_unreliable': n_unrel / n_tot if n_tot else 0.0,
                'frac_shared':     n_triv  / n_tot if n_tot else 0.0,
                'frac_specific':   n_gen   / n_tot if n_tot else 0.0}

    rows = [
        _row('all',         all_df),
        _row('perturb_seq', all_df[all_df['_scope'] == 'perturb_seq']),
        _row('sci_plex',    all_df[all_df['_scope'] == 'sci_plex']),
        _row('scrna',       all_df[all_df['_scope'] == 'scrna']),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(intermediate_path('pooled_quality_summary.csv'),
                   index=False, float_format='%.4f')
    print(summary.to_string(index=False))
    print(f"✓ Saved {intermediate_path('pooled_quality_summary.csv')}")

print("\n✓ Step 1 complete.")
