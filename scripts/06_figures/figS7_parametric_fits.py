#!/usr/bin/env python3
"""Supplementary Figure 7 — Parametric reliability model fits across 29
datasets.

Per-dataset overlay of empirical reliability vs. cells-per-perturbation
curves, with the median per-pert τ² fit overlaid.
"""
import sys, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *
from _paths import save_fig

setup_style()


def load_curves(rdir, prefix, n_col, ctx, pert_id_col):
    dfs = []
    for f in sorted(Path(rdir).glob(f"{prefix}*.csv")):
        if "summary" in f.name or "metadata" in f.name:
            continue
        ds = f.stem.replace(prefix, "")
        df = pd.read_csv(f)
        df["dataset"] = ds
        df["context_type"] = ctx
        df["n_cells"] = df[n_col]
        # One curve per (perturbation, condition). A dataset can profile the same perturbation in
        # several conditions (Frangieh has three), and keying on the perturbation alone would draw a
        # single trace zig-zagging through them.
        # Cellular files already key on `condition` (which encodes context|perturbation); genetic files
        # key on `perturbation` and need the condition appended to stay one curve per unit.
        if "condition" in df.columns and pert_id_col != "condition":
            df["pert_id"] = df[pert_id_col].astype(str) + "|" + df["condition"].astype(str)
        else:
            df["pert_id"] = df[pert_id_col]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


fits = pd.read_csv(str(FITS_DIR / 'fits_all_perturbations.csv'))
gen_curves = load_curves(str(RELIABILITY_GEN_DIR), "reliability_",
                         "n_ko_cells", "genetic", "perturbation")
cel_curves = load_curves(str(RELIABILITY_CEL_DIR), "reliability_cc_",
                         "n_stim_cells", "cellular", "condition")

RHO_COL = 'sb_from_median'


def panel_one_dataset(ax, ds, ctx, color, show_ylabel=True, show_xlabel=True):
    src = gen_curves if ctx == 'genetic' else cel_curves
    fixed = src[(src['dataset'] == ds) &
                (src['n_mode'] == 'fixed') &
                (src['gene_mode'] == 'all_genes')]
    # Restrict to units that also have a parametric fit, so every element of the panel (traces,
    # median, model line, n) describes the same set of perturbations.
    _fitted = set(fits[(fits['dataset'] == ds) & (fits['context_type'] == ctx)]['pert_id'])
    fixed = fixed[fixed['pert_id'].isin(_fitted)]
    if len(fixed) == 0:
        ax.text(0.5, 0.5, '(no curves)', ha='center', va='center',
                transform=ax.transAxes, fontsize=NM_TINY, color='#999')
        ax.set_title(ds, fontsize=NM_TINY)
        return

    rng = np.random.default_rng(42)
    perts = fixed['pert_id'].unique()
    sample = rng.choice(perts, min(30, len(perts)), replace=False)
    for p in sample:
        pdata = fixed[fixed['pert_id'] == p].sort_values('n_half')
        ax.plot(2 * pdata['n_half'], pdata[RHO_COL], '-',
                color=color, alpha=0.18, lw=0.5, rasterized=True)

    agg = fixed.groupby('n_half').agg(
        med=(RHO_COL, 'median')).reset_index()
    N = 2 * agg['n_half'].values
    # Same encoding as Figure 3a: observed median as open white-filled markers, model as the curve.
    # Every sampled depth is drawn (47 per dataset), not a subsample.
    # Observed median as a solid dark blue line (not the dataset color, which is also the faint
    # traces behind it); the model is dashed black on top so the dashes read where they coincide.
    ax.plot(N, agg['med'].values, '-', color='#1B4F9C', lw=1.1, zorder=4)

    ds_fits = fits[(fits['dataset'] == ds) &
                   (fits['context_type'] == ctx)]
    n_perts = ds_fits['pert_id'].nunique() if len(ds_fits) > 0 else 0
    if n_perts > 0:
        tau2 = ds_fits['tau_sq'].median()
        N_fine = np.linspace(max(2, N.min() * 0.4), 100, 200)
        # Drawn above the median (zorder 4) with a white halo. The model tracks the median closely
        # by construction, so at the default zorder it is painted over by the median line and becomes
        # invisible exactly where the fit is good -- the same issue corrected in Figure 3a. Dashed
        # black above the solid blue median, so the dashes read even where the two coincide.
        ax.plot(N_fine, rho_model(N_fine, tau2), color='#E07A5F', lw=1.3, zorder=6,
                dashes=(4.5, 3.0), solid_capstyle='butt',
                label=rf'$\tau^2$={tau2:.3g}')

    ax.axhline(0.5, color='grey', ls=':', alpha=0.3, lw=0.5)
    ax.set_xlim(2, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([2, 25, 50, 75, 100])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(f'{ds}\n(n={n_perts})', fontsize=NM_TINY, pad=4)
    if show_xlabel:
        ax.set_xlabel('N', fontsize=NM_TINY)
    if show_ylabel:
        ax.set_ylabel('ρ', fontsize=NM_TINY)
    ax.tick_params(labelsize=NM_TINY)
    if n_perts > 0:
        ax.legend(fontsize=NM_TINY, loc='lower right',
                  frameon=False, handlelength=1.0, borderpad=0.2)


# ── Figure ──────────────────────────────────────────────────────────
# All perturbation-condition units are treated alike, so datasets are simply ordered by name and
# drawn in one colour rather than split by which processing folder they came from.
_ctx_of = dict(zip(fits['dataset'], fits['context_type']))
all_ds = [(d, _ctx_of[d]) for d in sorted(_ctx_of)]

n_ds = len(all_ds)
# Laid out for a full A4 portrait page (8.27 x 11.69 in). Five columns x six rows gives 30 slots for
# the 29 datasets with near-square cells, which is the largest per-panel area available on A4; six
# columns leaves each panel too narrow for its tick labels.
A4_W, A4_H = 8.27, 11.69
n_cols = 5
n_rows = int(np.ceil(n_ds / n_cols))

fig = plt.figure(figsize=(A4_W, A4_H))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                       hspace=0.45, wspace=0.28,
                       left=0.065, right=0.99, top=0.975, bottom=0.04)

for k, (ds, ctx) in enumerate(all_ds):
    r, c = divmod(k, n_cols)
    ax = fig.add_subplot(gs[r, c])
    color = '#4878CF'
    panel_one_dataset(ax, ds, ctx, color,
                      show_ylabel=(c == 0),
                      show_xlabel=(r == n_rows - 1 or k >= n_ds - n_cols))

save_fig(fig, 'figS7_parametric_fits')
plt.close(fig)
print(f"✓ figS7 complete (n_datasets = {n_ds}).")

