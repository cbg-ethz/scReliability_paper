#!/usr/bin/env python3
"""Supplementary Figure 8 — Scarcity holds, more conservative under
energy-distance reliability.

Two panels:
  a — Side-by-side pie charts: pooled triage composition under per-gene-mean
       Pearson reliability vs distributional E-distance reliability.
  b — Per-dataset stacked composition bars under the E-distance triage,
       sorted by Specific fraction.
"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *  # noqa
from _paths import FIG_DIR, FIG_PANEL_DIR, save_fig  # noqa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

setup_style()


ROOT = str(BASE_DIR)

# Pearson reliability + cos_sim, with the main triage labels
gen_pearson = pd.read_csv(f'{ROOT}/intermediate/combined_genetic_2d.csv')
cel_pearson = pd.read_csv(f'{ROOT}/intermediate/combined_cellular_2d.csv')
all_pearson = pd.concat([gen_pearson, cel_pearson], ignore_index=True)
all_pearson = rename_triage(all_pearson, 'triage_2d')   # → 'Unreliable','Shared','Specific'

# E-distance reliability + cos_sim, labels in triage_2d_edist
gen_edist = pd.read_csv(f'{ROOT}/intermediate/edist/combined_genetic_2d_edist.csv')
cel_edist = pd.read_csv(f'{ROOT}/intermediate/edist/combined_cellular_2d_edist.csv')
all_edist = pd.concat([gen_edist, cel_edist], ignore_index=True)

CATS = ['Unreliable', 'Shared', 'Specific']


def pie_panel(ax, counts, title):
    vals = [int(counts.get(c, 0)) for c in CATS]
    total = sum(vals)
    fracs = [v / total for v in vals]
    colors = [TRIAGE_COLORS[c] for c in CATS]
    wedges, texts, autotexts = ax.pie(
        fracs, colors=colors, autopct='%1.0f%%',
        startangle=90, counterclock=False, pctdistance=0.6,
        wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
        textprops={'fontsize': NM_TICK},
    )
    for t in autotexts:
        t.set_fontsize(NM_TICK)
        t.set_fontweight('bold')
        t.set_color('white')
    ax.set_title(title, fontsize=NM_TITLE - 1, pad=8)
    ax.text(0, -1.35, f'n = {total:,}', ha='center',
            fontsize=NM_TICK, color='#555')


def panel_a_pies(ax_pearson, ax_edist):
    c_pearson = all_pearson['triage_2d'].value_counts()
    c_edist = all_edist['triage_2d_edist'].value_counts()
    pie_panel(ax_pearson, c_pearson,
              r'Per-gene mean reliability ($\rho \geq 0.5$)')
    pie_panel(ax_edist, c_edist,
              r'E-distance reliability ($\rho_E \geq 0$)')


def panel_b_per_dataset(ax):
    counts = (all_edist.groupby(['dataset', 'triage_2d_edist'])
                    .size().unstack(fill_value=0))
    for c in CATS:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[CATS]
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100
    pct = pct.sort_values('Specific', ascending=False)
    counts = counts.loc[pct.index]
    totals = totals.loc[pct.index]

    n_ds = len(pct)
    x = np.arange(n_ds)
    bottom = np.zeros(n_ds)
    for cat in CATS:
        ax.bar(x, pct[cat].values, bottom=bottom,
               color=TRIAGE_COLORS[cat],
               width=0.72, edgecolor='white', linewidth=0.3)
        bottom += pct[cat].values
    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, rotation=50, ha='right',
                       fontsize=NM_TINY)
    ax.set_ylabel('Fraction (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title('Per-dataset quality under E-distance reliability '
                 '(29 datasets, sorted by Specific fraction)',
                 pad=10, fontsize=NM_TITLE - 1)
    ax.set_xlim(-0.7, n_ds - 0.3)
    for i, ds in enumerate(pct.index):
        n_gen = int(counts.loc[ds, 'Specific'])
        n_tot = int(totals.loc[ds])
        color = TRIAGE_COLORS['Unreliable'] if n_gen == 0 else '#333'
        ax.text(i, 102, f'{n_gen}/{n_tot}', ha='center', va='bottom',
                fontsize=NM_TINY, color=color, rotation=45)


# ── COMBINED FIGURE ─────────────────────────────────────────────────
if __name__ == '__main__':
    fig = plt.figure(figsize=(NM_FULL_W * 1.5, 5.2))
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[0.85, 1.0], hspace=0.55,
                              left=0.07, right=0.97, top=0.93, bottom=0.16)

    # Row 1: two pies + legend
    row1 = outer[0].subgridspec(1, 2, wspace=0.45)
    ax_pearson = fig.add_subplot(row1[0]); ax_pearson.set_aspect('equal')
    ax_edist = fig.add_subplot(row1[1]); ax_edist.set_aspect('equal')
    panel_a_pies(ax_pearson, ax_edist)

    leg_h = [Patch(facecolor=TRIAGE_COLORS[c], label=c) for c in CATS]
    fig.legend(handles=leg_h, loc='center', bbox_to_anchor=(0.5, 0.54),
               ncol=3, frameon=False, fontsize=NM_LEGEND,
               handlelength=1.2, columnspacing=2.0)

    # Row 2: per-dataset bars under the E-distance triage
    ax_bars = fig.add_subplot(outer[1])
    panel_b_per_dataset(ax_bars)

    fig.text(0.04, 0.96, 'a', fontsize=NM_PANEL, fontweight='bold',
             va='top')
    fig.text(0.04, 0.51, 'b', fontsize=NM_PANEL, fontweight='bold',
             va='top')

    save_fig(fig, 'figS8_edist')
    plt.close(fig)
    print("✓ figS8_edist complete.")
