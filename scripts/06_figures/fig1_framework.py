#!/usr/bin/env python3
"""Figure 1 — Quality framework reveals most pseudo-bulk perturbation
targets are unreliable or shared-response-dominated.

Panels: a1, a2 (two-step filter on Replogle K562), b (pooled summary pie),
c (per-dataset breakdown across all 29 datasets).

Outputs to figures/.
"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *  # noqa: F401,F403
from _paths import FIG_DIR, FIG_PANEL_DIR, save_fig, save_fig_panel  # noqa: E402

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

setup_style()


# ── DATA ─────────────────────────────────────────────────────────────
gen_2d = rename_triage(pd.read_csv(intermediate_path('combined_genetic_2d.csv')))
cel_2d = rename_triage(pd.read_csv(intermediate_path('combined_cellular_2d.csv')))
all_2d = pd.concat([gen_2d, cel_2d], ignore_index=True)

# Pooled summary
pooled = pd.read_csv(intermediate_path('pooled_quality_summary.csv'))
pooled_all = pooled[pooled['scope'] == 'all'].iloc[0]

# K562 example
k562 = gen_2d[gen_2d['dataset'] == 'Replogle_K562essential']

# Maps display labels → raw CSV column suffix (file-internal names).
_COL_KEY = {'Unreliable': 'unreliable', 'Shared': 'shared',
            'Specific': 'specific'}


# ── LAYOUT ───────────────────────────────────────────────────────────
LAYOUT = dict(
    figsize       = (NM_FULL_W * 1.55, 5.6),
    row_heights   = [1.0, 1.15],
    row1_widths   = [1.0, 1.0, 0.45],
    hspace        = 0.65,
    row1_wspace   = 0.45,
    margins       = dict(top=0.92, bottom=0.10, left=0.05, right=0.90),
)

PANEL_LABEL_POS = [
    ('a', (0.02, 0.97)), ('b', (0.75, 0.97)),
    ('c', (0.02, 0.47)),
]


# ══════════════════════════════════════════════════════════════════════
# Panel functions (a1, a2, b, c carried over verbatim from main fig1)
# ══════════════════════════════════════════════════════════════════════
def panel_a1_reliability(ax):
    rho = k562['sb_from_median_all_genes'].dropna()
    n_total = len(rho)
    n_reliable = int((rho >= RELIABILITY_THRESH).sum())
    n_unreliable = n_total - n_reliable
    bins = np.linspace(0, 1.0, 41)
    ax.hist(rho[rho < RELIABILITY_THRESH], bins=bins,
            color=TRIAGE_COLORS['Unreliable'], alpha=0.85,
            edgecolor='white', linewidth=0.3,
            label=f'Unreliable ({n_unreliable})')
    ax.hist(rho[rho >= RELIABILITY_THRESH], bins=bins,
            color='#7f8c8d', alpha=0.85, edgecolor='white', linewidth=0.3,
            label=f'Reliable ({n_reliable})')
    ax.axvline(RELIABILITY_THRESH, color='black', ls='--', lw=0.8, zorder=5)
    ax.set_xlabel('Reliability (ρ)')
    ax.set_ylabel('Perturbations')
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, RELIABILITY_THRESH, 1.0])
    ax.set_title(f'Step 1: reliability filter\nReplogle K562 (n={n_total:,})',
                 pad=6, fontsize=NM_TITLE - 1)
    ax.legend(loc='upper right', fontsize=NM_LEGEND, frameon=False,
              borderpad=0.2, handletextpad=0.3)
    ax.set_box_aspect(0.85)


def panel_a2_specificity(ax):
    rel = k562[k562['sb_from_median_all_genes'] >= RELIABILITY_THRESH].copy()
    cos2 = (rel['cos_sim'] ** 2).dropna()
    n_rel = len(cos2)
    n_triv = int((cos2 >= 0.5).sum())
    n_gen = int((cos2 < 0.5).sum())
    bins = np.linspace(0, 1.0, 41)
    ax.hist(cos2[cos2 < 0.5], bins=bins,
            color=TRIAGE_COLORS['Specific'], alpha=0.85,
            edgecolor='white', linewidth=0.3,
            label=f'Specific ({n_gen})')
    ax.hist(cos2[cos2 >= 0.5], bins=bins,
            color=TRIAGE_COLORS['Shared'], alpha=0.85,
            edgecolor='white', linewidth=0.3,
            label=f'Shared ({n_triv})')
    ax.axvline(0.5, color='black', ls='--', lw=0.8, zorder=5)
    ax.set_xlabel(r'Systematic variance fraction $\varphi$ ($=\cos^{2}\theta$)')
    ax.set_ylabel('Perturbations')
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(f'Step 2: specificity filter\nReliable subset (n={n_rel:,})',
                 pad=6, fontsize=NM_TITLE - 1)
    ax.legend(loc='upper right', fontsize=NM_LEGEND, frameon=False,
              borderpad=0.2, handletextpad=0.3)
    ax.set_box_aspect(0.85)


def panel_b_summary(ax):
    fracs = [float(pooled_all[f'frac_{_COL_KEY[cat]}']) for cat in TRIAGE_ORDER]
    colors = [TRIAGE_COLORS[cat] for cat in TRIAGE_ORDER]
    n_total = int(pooled_all['n_total'])

    wedges, texts, autotexts = ax.pie(
        fracs, colors=colors, autopct='%1.0f%%',
        startangle=90, counterclock=False,
        pctdistance=0.55,
        wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
        textprops={'fontsize': NM_TICK},
    )
    for t in autotexts:
        t.set_fontsize(NM_TICK)
        t.set_fontweight('bold')
        t.set_color('white')

    ax.set_title(f'n = {n_total:,}', fontsize=NM_TICK, pad=4, color='#555')

    leg_h = [Patch(facecolor=TRIAGE_COLORS[c], label=c) for c in TRIAGE_ORDER]
    ax.legend(handles=leg_h, fontsize=NM_ANNOT, frameon=False,
              loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=1, handletextpad=0.4, labelspacing=0.3)


def panel_c_all_datasets(ax):
    counts = (all_2d.groupby(['dataset', 'triage_2d'])
                    .size().unstack(fill_value=0))
    for cat in TRIAGE_ORDER:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[TRIAGE_ORDER]
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100
    pct = pct.sort_values('Specific', ascending=False)
    counts = counts.loc[pct.index]
    totals = totals.loc[pct.index]

    n_ds = len(pct)
    x = np.arange(n_ds)
    bottom = np.zeros(n_ds)
    for cat in TRIAGE_ORDER:
        ax.bar(x, pct[cat].values, bottom=bottom,
               color=TRIAGE_COLORS[cat],
               width=0.72, edgecolor='white', linewidth=0.3)
        bottom += pct[cat].values
    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, rotation=50, ha='right', fontsize=NM_TINY)
    ax.set_ylabel('Fraction (%)')
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(f'Per-dataset quality ({n_ds} datasets)', pad=10)
    ax.set_xlim(-0.7, n_ds - 0.3)
    for i, ds in enumerate(pct.index):
        n_gen = int(counts.loc[ds, 'Specific'])
        n_tot = int(totals.loc[ds])
        color = TRIAGE_COLORS['Unreliable'] if n_gen == 0 else '#333'
        ax.text(i, 102, f'{n_gen}/{n_tot}', ha='center', va='bottom',
                fontsize=NM_TINY, color=color, rotation=45)


# ══════════════════════════════════════════════════════════════════════
# NEW Panel d — Sample size × quality fractions
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# COMBINED FIGURE — 2 rows
#   Row 1: a1 | a2 | b
#   Row 2: c (full width)
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=LAYOUT['figsize'])
outer = gridspec.GridSpec(2, 1, figure=fig,
                          height_ratios=LAYOUT['row_heights'],
                          hspace=LAYOUT['hspace'],
                          **LAYOUT['margins'])

# Row 1
row1 = outer[0].subgridspec(1, 3, width_ratios=LAYOUT['row1_widths'],
                            wspace=LAYOUT['row1_wspace'])
ax = fig.add_subplot(row1[0]); panel_a1_reliability(ax)
ax = fig.add_subplot(row1[1]); panel_a2_specificity(ax)
ax = fig.add_subplot(row1[2]); panel_b_summary(ax)

# Row 2
ax = fig.add_subplot(outer[1]); panel_c_all_datasets(ax)

# Panel labels
for lbl, pos in PANEL_LABEL_POS:
    fig.text(*pos, lbl, fontsize=NM_PANEL, fontweight='bold',
             va='top', fontfamily='sans-serif')

save_fig(fig, 'fig1_combined')
plt.close(fig)


# Individual panels
save_fig_panel(panel_a1_reliability, 'fig1a1_reliability_hist',
              figsize=(2.6, 2.4))
save_fig_panel(panel_a2_specificity, 'fig1a2_specificity_hist',
              figsize=(2.6, 2.4))
save_fig_panel(panel_b_summary, 'fig1b_pooled_summary',
              figsize=(2.0, 2.4))
save_fig_panel(panel_c_all_datasets, 'fig1c_all_datasets',
              figsize=(5.5, 2.4))

print("✓ Figure 1 complete.")
