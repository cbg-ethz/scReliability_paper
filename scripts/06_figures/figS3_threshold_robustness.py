#!/usr/bin/env python3
"""Supplementary Figure 3 — Threshold robustness of quality classification.

"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *
from _paths import FIG_DIR, FIG_PANEL_DIR, save_fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

setup_style()

# ── DATA ─────────────────────────────────────────────────────────────
wei_gen = rename_triage(pd.read_csv(intermediate_path('wei_genetic_merged_2d.csv')))
SKIP_RANK = {'baseControl', 'trainMean', 'scFoundation', 'baseMLP', 'baseReg'}


def derive_2d(df, rho_t, cos_t):
    """Re-classify based on alternative thresholds."""
    df = df.copy()
    df['triage_alt'] = 'Specific'
    df.loc[df['sb_from_median_all_genes'] < rho_t, 'triage_alt'] = 'Unreliable'
    df.loc[(df['sb_from_median_all_genes'] >= rho_t) &
           (df['cos_sim'] >= cos_t), 'triage_alt'] = 'Shared'
    return df


# Main-text thresholds: ρ = 0.5 (signal-to-noise variance ratio of 1:1) and
# φ = 0.5 (perturbation-specific to systematic variance ratio of 1:1).
# φ ≡ cos²θ is the systematic variance fraction.
PHI_THRESH_MAIN = 0.5
RHO_THRESH_MAIN = 0.5

# φ-threshold grid in [0.2, 0.8] in steps of 0.1, symmetric around the
# main-text threshold φ = 0.5. The same numerical layout as the ρ grid.
phi_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
phi_labels = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
rho_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ── PANEL A: scGPT vs CPA rank diff across (ρ, φ) thresholds ──────────
def panel_a_rank_diff(ax):
    matrix = np.full((len(rho_grid), len(phi_grid)), np.nan)
    for i, rt in enumerate(rho_grid):
        for j, phi_t in enumerate(phi_grid):
            # φ ≡ cos²θ ≥ phi_t is equivalent to cos θ ≥ sqrt(phi_t)
            ct = np.sqrt(phi_t)
            df = derive_2d(wei_gen, rt, ct)
            sub = df[df['triage_alt'] == 'Specific']
            if len(sub) < 20:
                continue
            exc = ds_weighted_excess(
                sub.assign(triage_2d='Specific'),
                'pcc', 'trainMean', 'perturbation', SKIP_RANK,
                'Specific')
            if 'scGPT' not in exc or 'CPA' not in exc:
                continue
            ranks = pd.Series(exc).rank(ascending=False)
            matrix[i, j] = ranks['scGPT'] - ranks['CPA']
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=-8, vmax=8, aspect='equal')
    ax.set_xticks(range(len(phi_grid)))
    ax.set_xticklabels(phi_labels)
    ax.set_yticks(range(len(rho_grid)))
    ax.set_yticklabels(rho_grid)
    ax.set_xlabel(r'Specificity threshold $\varphi$ ($=\cos^{2}\theta$)',
                  fontsize=NM_LABEL)
    ax.set_ylabel(r'Reliability threshold $\rho$', fontsize=NM_LABEL)
    ax.set_title('scGPT rank − CPA rank\n(positive = CPA better)',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    ax.invert_yaxis()
    for i in range(len(rho_grid)):
        for j in range(len(phi_grid)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=NM_TINY, color='#888')
            else:
                ax.text(j, i, f'{v:+.0f}', ha='center', va='center',
                        fontsize=NM_TINY,
                        color='white' if abs(v) >= 5 else 'black')
    # Main-text thresholds: φ = 0.5 and ρ = 0.5
    j_main = phi_grid.index(PHI_THRESH_MAIN)
    i_main = rho_grid.index(RHO_THRESH_MAIN)
    ax.axvline(j_main, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(i_main, color='black', ls='--', lw=0.8, alpha=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Rank difference\n(positive = CPA better)',
                 fontsize=NM_LEGEND)
    cb.ax.tick_params(labelsize=NM_TINY)


# ── PANEL B: Specific fraction across thresholds ──────────────────────
def panel_b_genuine_frac(ax):
    gen_2d = rename_triage(pd.read_csv(intermediate_path('combined_genetic_2d.csv')))
    cel_2d = rename_triage(pd.read_csv(intermediate_path('combined_cellular_2d.csv')))
    all_2d = pd.concat([gen_2d, cel_2d], ignore_index=True)

    # Compute φ = cos²θ once; classify across the grid
    all_2d = all_2d.copy()
    all_2d['phi'] = all_2d['cos_sim'] ** 2

    matrix = np.zeros((len(rho_grid), len(phi_grid)))
    for i, rt in enumerate(rho_grid):
        for j, phi_t in enumerate(phi_grid):
            reliable = (all_2d['sb_from_median_all_genes'] >= rt)
            high_sys = (all_2d['phi'] >= phi_t)
            genuine = (reliable & ~high_sys).sum()
            matrix[i, j] = 100 * genuine / len(all_2d)

    im = ax.imshow(matrix, cmap='Greens', vmin=0, vmax=50, aspect='equal')
    ax.set_xticks(range(len(phi_grid)))
    ax.set_xticklabels(phi_labels)
    ax.set_yticks(range(len(rho_grid)))
    ax.set_yticklabels(rho_grid)
    ax.set_xlabel(r'Specificity threshold $\varphi$ ($=\cos^{2}\theta$)',
                  fontsize=NM_LABEL)
    ax.set_ylabel(r'Reliability threshold $\rho$', fontsize=NM_LABEL)
    ax.set_title('% perturbations classified as Specific',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    ax.invert_yaxis()
    for i in range(len(rho_grid)):
        for j in range(len(phi_grid)):
            v = matrix[i, j]
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=NM_TINY,
                    color='white' if v >= 25 else 'black')
    j_main = phi_grid.index(PHI_THRESH_MAIN)
    i_main = rho_grid.index(RHO_THRESH_MAIN)
    ax.axvline(j_main, color='black', ls='--', lw=0.8, alpha=0.6)
    ax.axhline(i_main, color='black', ls='--', lw=0.8, alpha=0.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Specific (%)', fontsize=NM_LEGEND)
    cb.ax.tick_params(labelsize=NM_TINY)


# ── COMBINED ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(NM_FULL_W * 1.2, NM_COL_W * 1.4))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                       left=0.07, right=0.96, top=0.90, bottom=0.12)
ax_a = fig.add_subplot(gs[0, 0]); panel_a_rank_diff(ax_a)
ax_b = fig.add_subplot(gs[0, 1]); panel_b_genuine_frac(ax_b)

fig.text(0.02, 0.96, 'a', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.50, 0.96, 'b', fontsize=NM_PANEL, fontweight='bold', va='top')

save_fig(fig, 'figS3_threshold_robustness')
plt.close(fig)
print("✓ figS3 complete.")
