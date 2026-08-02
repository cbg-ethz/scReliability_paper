#!/usr/bin/env python3
"""Supplementary Figure 1 — Reliability distributions and sample-size
sensitivity.

  a  Quality fractions by cells per condition (50-cell bins from 0 to
     500, plus >500; pooled across 29 datasets).
  b  Per-dataset histograms of split-half reliability (ρ), each with a
     vertical threshold at ρ = 0.5. Genetic + cellular pooled, sorted by
     median ρ.
"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *
from _paths import FIG_DIR, save_fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

setup_style()

# ── DATA ─────────────────────────────────────────────────────────────
gen_2d = rename_triage(pd.read_csv(intermediate_path('combined_genetic_2d.csv')))
cel_2d = rename_triage(pd.read_csv(intermediate_path('combined_cellular_2d.csv')))
all_2d = pd.concat([gen_2d, cel_2d], ignore_index=True)
RHO_COL = 'sb_from_median_all_genes'

# Sort datasets by median ρ
ds_order = (all_2d.groupby('dataset')[RHO_COL]
              .median().sort_values(ascending=False).index.tolist())


# ══════════════════════════════════════════════════════════════════════
# Panel a — Quality fractions by cells per condition (pooled)
# ══════════════════════════════════════════════════════════════════════
def panel_a_sample_size(ax):
    pooled = all_2d.copy()
    pooled['n_cells'] = pooled['n_ko_cells'].fillna(pooled.get('n_stim_cells'))
    df = pooled.dropna(subset=['n_cells']).copy()

    bins   = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1e9]
    labels = ['0–50', '50–100', '100–150', '150–200', '200–250',
              '250–300', '300–350', '350–400', '400–450', '450–500', '>500']
    df['bin'] = pd.cut(df['n_cells'], bins=bins, labels=labels,
                       include_lowest=True)

    counts = (df.groupby(['bin', 'triage_2d'], observed=True)
                .size().unstack('triage_2d').fillna(0))
    counts = counts.reindex(labels)
    for c in TRIAGE_ORDER:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[TRIAGE_ORDER]
    totals = counts.sum(axis=1).astype(int)
    fracs = counts.div(counts.sum(axis=1), axis=0) * 100

    x = np.arange(len(labels))
    bar_w = 0.58
    bottom = np.zeros(len(labels))
    for tri in TRIAGE_ORDER:
        v = fracs[tri].values
        ax.bar(x, v, bar_w, bottom=bottom, color=TRIAGE_COLORS[tri],
               edgecolor='white', linewidth=0.4, label=tri)
        for i, val in enumerate(v):
            if val >= 12:
                ax.text(x[i], bottom[i] + val / 2, f'{int(round(val))}%',
                        ha='center', va='center',
                        fontsize=NM_LABEL, color='white', fontweight='bold')
        bottom += v

    for i, n in enumerate(totals.values):
        ax.text(x[i], 102, f'n={int(n):,}', ha='center', va='bottom',
                fontsize=NM_LEGEND, color='#444', rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=NM_TINY, rotation=40, ha='right')
    ax.set_xlabel('Cells per condition', fontsize=NM_LABEL)
    ax.set_ylabel('Fraction of perturbations (%)', fontsize=NM_LABEL)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=NM_LABEL)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('Quality fractions by cells per condition (pooled)',
                 pad=8, fontsize=NM_TITLE, fontweight='bold')

    ax.legend(fontsize=NM_LEGEND, loc='upper center',
              bbox_to_anchor=(0.5, -0.32), ncol=3,
              frameon=False, handlelength=1.0, columnspacing=1.2)


# ══════════════════════════════════════════════════════════════════════
# Panel b — Per-dataset reliability histograms
# ══════════════════════════════════════════════════════════════════════
def render_panel_b(fig, parent_gs, n_cols=6):
    """Per-dataset reliability histograms in a grid of subplots."""
    n_ds = len(ds_order)
    n_rows = int(np.ceil(n_ds / n_cols))

    inner = parent_gs.subgridspec(n_rows, n_cols, hspace=0.95, wspace=0.5)
    bins = np.linspace(0, 1.0, 21)

    for k, ds in enumerate(ds_order):
        r, c = divmod(k, n_cols)
        ax = fig.add_subplot(inner[r, c])
        sub = all_2d[all_2d['dataset'] == ds][RHO_COL].dropna()
        if len(sub) == 0:
            continue
        color = '#4878CF'
        counts, _ = np.histogram(sub, bins=bins)
        ax.hist(sub, bins=bins, color=color, alpha=0.7,
                edgecolor='white', linewidth=0.3)
        ax.axvline(RELIABILITY_THRESH, color='black', ls='--',
                   lw=0.8, alpha=0.7)
        # headroom so the annotation never overlaps the bars
        ax.set_ylim(0, counts.max() * 1.62 if counts.max() > 0 else 1)
        med = sub.median()
        ax.text(0.96, 0.96,
                f"n={len(sub):,}\nmed ρ={med:.2f}\n"
                f"{(sub >= 0.5).sum() / len(sub) * 100:.0f}% reliable",
                transform=ax.transAxes, fontsize=NM_TINY,
                ha='right', va='top', color='#333',
                linespacing=1.25)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(ds, fontsize=NM_TINY, pad=4)
        if c == 0:
            ax.set_ylabel('Perts', fontsize=NM_TINY)
        if r == n_rows - 1 or k >= n_ds - n_cols:
            ax.set_xlabel('ρ', fontsize=NM_TINY)
        ax.tick_params(labelsize=NM_TINY)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    return n_rows


# ══════════════════════════════════════════════════════════════════════
# COMBINED FIGURE
#   Row 1: panel a (cells per condition × quality, full width)
#   Rows 2–N: panel b (per-dataset reliability histograms, multi-subplot grid)
# ══════════════════════════════════════════════════════════════════════
n_ds = len(ds_order)
n_cols_b = 6
n_rows_b = int(np.ceil(n_ds / n_cols_b))

fig_h = 3.0 + n_rows_b * 1.5
fig = plt.figure(figsize=(n_cols_b * 1.7, fig_h))

outer = gridspec.GridSpec(2, 1, figure=fig,
                          height_ratios=[2.6, n_rows_b * 1.5],
                          hspace=0.55,
                          left=0.06, right=0.98, top=0.95, bottom=0.05)

# Panel a (top, full width)
ax_a = fig.add_subplot(outer[0])
panel_a_sample_size(ax_a)

# Panel b (multi-subplot grid)
render_panel_b(fig, outer[1], n_cols=n_cols_b)

# Panel labels
fig.text(0.02, 0.98, 'a', fontsize=NM_PANEL, fontweight='bold',
         va='top', fontfamily='sans-serif')
y_b = 1.0 - (2.6 / fig_h) - 0.04
fig.text(0.02, y_b, 'b', fontsize=NM_PANEL, fontweight='bold',
         va='top', fontfamily='sans-serif')

# Legend caption at bottom
fig.text(0.5, 0.01,
         
         'Dashed line: ρ = 0.5 reliability threshold.',
         ha='center', fontsize=NM_LEGEND, color='#444')

save_fig(fig, 'figS1_reliability_distributions')
plt.close(fig)
print(f"✓ figS1 complete (n_datasets = {n_ds}).")
