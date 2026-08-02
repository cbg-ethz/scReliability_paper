#!/usr/bin/env python3
"""Supplementary Figure S5 — Sensitivity to evaluation choices.

  a  Multi-metric heatmap (specific, top 100 DEGs across 6 metrics)
  b  Gene set robustness (top 100 DEGs vs top 5000 genes)
  c  PCC vs multi-metric mean rank (per-condition)
  d  E-distance win rate vs trainMean baseline (specific)
"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *
from _paths import save_fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

setup_style()

# ── DATA ─────────────────────────────────────────────────────────────
wei_gen    = rename_triage(pd.read_csv(intermediate_path('wei_genetic_merged_2d.csv')))
wei_gen_5k = rename_triage(pd.read_csv(intermediate_path('wei_genetic_5k_merged_2d.csv')))

SKIP_RANK = {'baseControl', 'trainMean', 'scFoundation', 'baseMLP', 'baseReg'}
SKIP_WEI  = {'baseControl', 'trainMean', 'scFoundation', 'baseMLP', 'baseReg'}
gen_wei = wei_gen[wei_gen['triage_2d'] == 'Specific']


# ════════════ Panel a — multi-metric heatmap ═════════════════════════
def panel_a_multi_metric(ax):
    metrics = ['pcc', 'mse_score', 'edistance_score', 'was_score',
               'sym_score', 'ac_score']
    mlabels = ['PCC', 'MSE', 'E-dist.', 'Wass.', 'Symm.', 'Acc.']
    methods_sorted = ['scouter', 'linearModel', 'CPA', 'GenePert', 'scGPT',
                      'scELMo', 'GEARS', 'biolord', 'AttentionPert',
                      'GeneCompass']
    matrix = np.full((len(methods_sorted), len(metrics)), np.nan)
    for j, met in enumerate(metrics):
        if met not in wei_gen.columns:
            continue
        exc = ds_weighted_excess(wei_gen, met, 'trainMean', 'perturbation',
                                  SKIP_GEN, 'Specific')
        for i, m in enumerate(methods_sorted):
            matrix[i, j] = exc.get(m, 0)
    vmax = np.nanmax(np.abs(matrix)) * 0.85
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mlabels)))
    ax.set_xticklabels(mlabels, rotation=35, ha='right', fontsize=NM_LABEL)
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels(methods_sorted, fontsize=NM_LABEL)
    for i in range(len(methods_sorted)):
        for j in range(len(metrics)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                        fontsize=NM_TINY,
                        color='white' if abs(v) > vmax * 0.55 else 'black')
    ax.set_title('Excess by metric (specific, top 100 DEGs)',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Excess', fontsize=NM_LEGEND)
    cb.ax.tick_params(labelsize=NM_TINY)


# ════════════ Panel b — gene set robustness ══════════════════════════
def panel_b_gene_set(ax):
    exc_100 = ds_weighted_excess(wei_gen, 'pcc', 'trainMean', 'perturbation',
                                  SKIP_GEN, 'Specific')
    exc_5k = ds_weighted_excess(wei_gen_5k, 'pcc', 'trainMean',
                                 'perturbation', SKIP_GEN, 'Specific')
    methods = sorted(exc_100.keys(), key=lambda m: exc_100[m], reverse=True)
    x = np.arange(len(methods))
    w = 0.35
    v100 = [exc_100[m] for m in methods]
    v5k = [exc_5k.get(m, 0) for m in methods]
    ax.bar(x - w / 2, v100, w, color='#2980b9', alpha=0.8,
           edgecolor='white', linewidth=0.3, label='Top 100 DEGs')
    ax.bar(x + w / 2, v5k, w, color='#e67e22', alpha=0.8,
           edgecolor='white', linewidth=0.3, label='Top 5000 genes')
    ax.axhline(0, color='#555', ls='-', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=NM_LABEL)
    ax.set_ylabel('Excess PCC', fontsize=NM_LABEL)
    ax.set_title('Gene set robustness (specific)',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    ax.legend(fontsize=NM_LEGEND, loc='upper right', frameon=False)
    r, p = spearmanr(v100, v5k)
    ax.text(0.02, 0.05, f'Spearman r = {r:.2f}',
            transform=ax.transAxes, fontsize=NM_LEGEND,
            fontstyle='italic', color='#555')


# ════════════ Panel c — PCC rank vs multi-metric rank ════════════════
def panel_c_pcc_vs_multirank(ax):
    gen_m = gen_wei[~gen_wei['method'].isin(SKIP_RANK)]
    pcc_rank_mean = gen_m.groupby('method')['Rank_pcc'].mean().sort_values()
    multi_rank_mean = gen_m.groupby('method')['Rank'].mean().sort_values()
    pcc_order = list(pcc_rank_mean.index)
    multi_order = list(multi_rank_mean.index)
    n_m = len(pcc_order)
    for m in pcc_order:
        rp = pcc_order.index(m) + 1
        rm = multi_order.index(m) + 1
        delta = rp - rm
        if m == 'scGPT':
            col = '#3498db'; lw = 2.5; alp = 1.0
        elif m == 'scouter':
            col = '#e67e22'; lw = 2.0; alp = 0.9
        elif m == 'CPA':
            col = TRIAGE_COLORS['Specific']; lw = 2.0; alp = 0.9
        else:
            col = '#bbb'; lw = 0.8; alp = 0.5
        ax.plot([0, 1], [rp, rm], color=col, lw=lw, alpha=alp,
                solid_capstyle='round')
        ax.scatter([0], [rp], color=col, s=20, zorder=5,
                   edgecolors='white', linewidth=0.3)
        ax.scatter([1], [rm], color=col, s=20, zorder=5,
                   edgecolors='white', linewidth=0.3)
        fs = NM_LEGEND if m in ('scGPT', 'CPA', 'scouter') else NM_TINY
        fw = 'bold' if m in ('scGPT', 'CPA', 'scouter') else 'normal'
        ax.text(-0.03, rp, f'{m} ({rp})', ha='right', va='center',
                fontsize=fs, color=col, fontweight=fw)
        shift_str = (f' ({"+" if delta > 0 else ""}{delta})'
                     if delta != 0 else '')
        ax.text(1.03, rm, f'({rm}) {m}{shift_str}', ha='left', va='center',
                fontsize=fs, color=col, fontweight=fw)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(n_m + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PCC rank', 'Multi-metric\nmean rank'],
                       fontweight='bold', fontsize=NM_LABEL)
    ax.set_title('Per-condition rank: PCC vs.\nmean of 7 evaluation metrics',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.axvline(0, color='#eee', lw=0.5, zorder=0)
    ax.axvline(1, color='#eee', lw=0.5, zorder=0)


# ════════════ Panel d — E-distance win rate ══════════════════════════
def panel_d_edist_winrate(ax):
    bl_vals = gen_wei[gen_wei['method'] == 'trainMean'].set_index(
        ['dataset', 'perturbation'])['edistance_score']
    methods = sorted([m for m in gen_wei['method'].unique()
                      if m not in SKIP_RANK])
    win_rates = {}
    for m in methods:
        msub = gen_wei[gen_wei['method'] == m].set_index(
            ['dataset', 'perturbation'])['edistance_score']
        shared = msub.index.intersection(bl_vals.index)
        if len(shared) > 0:
            win_rates[m] = (msub.loc[shared] > bl_vals.loc[shared]).mean() * 100
    sorted_m = sorted(win_rates, key=lambda k: win_rates[k], reverse=True)
    y = np.arange(len(sorted_m))
    vals = [win_rates[m] for m in sorted_m]
    colors = [method_color(m) for m in sorted_m]
    ax.barh(y, vals, color=colors, edgecolor='white', linewidth=0.3,
            alpha=0.85)
    ax.axvline(50, color='#555', ls='--', lw=0.6, alpha=0.5)
    for i, (m, v) in enumerate(zip(sorted_m, vals)):
        ax.text(v + 1.0, i, f'{v:.1f}%', va='center',
                fontsize=NM_TINY, color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_m, fontsize=NM_LABEL)
    ax.set_xlabel('Win rate vs. trainMean (%)', fontsize=NM_LABEL)
    ax.set_title('E-distance win rate\n(specific conditions)',
                 fontsize=NM_TITLE, pad=8, fontweight='bold')
    ax.invert_yaxis()


# ── COMBINED ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(NM_FULL_W * 1.7, NM_FULL_W * 1.10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.50,
                       left=0.07, right=0.95, top=0.95, bottom=0.08)

ax_a = fig.add_subplot(gs[0, 0]); panel_a_multi_metric(ax_a)
ax_b = fig.add_subplot(gs[0, 1]); panel_b_gene_set(ax_b)
ax_c = fig.add_subplot(gs[1, 0]); panel_c_pcc_vs_multirank(ax_c)
ax_d = fig.add_subplot(gs[1, 1]); panel_d_edist_winrate(ax_d)

fig.text(0.02, 0.96, 'a', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.50, 0.96, 'b', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.02, 0.49, 'c', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.50, 0.49, 'd', fontsize=NM_PANEL, fontweight='bold', va='top')

save_fig(fig, 'figS_sensitivity')
plt.close(fig)
print("✓ figS5 complete.")
