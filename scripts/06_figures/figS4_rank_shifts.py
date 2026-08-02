#!/usr/bin/env python3
"""Supplementary Figure 4: All/Specific rank-shift slopes for the seven benchmark settings other than
Wei genetic single (which is the main-text Fig 2a). Each panel uses the EXACT same rank-slope format as
Fig 2a (panel_a_rank_wei): same colours, fonts, markers, labels, and legend (imported from config), so the
"magnitude of the rank change varied" claim is shown in a consistent style.

Benchmark definitions, baselines, skip sets, the excess-over-baseline ranking and the metric labels are all
taken from config, so this figure ranks methods identically to Fig 2a and 2b. One shared legend in the
empty 8th cell.

Inputs : intermediate/{wei_genetic_combo,wei_chemical_single,wei_chemical_combo,wei_cellular_iid,
         wei_cellular_ood,ae_panel_a,ae_doubles}_merged_2d.csv
Output : figures/figS4_rank_shifts.{pdf,png,svg}
"""
import os, sys
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *  # noqa: F401,F403  -> C_POS, C_NEG, NM_LABEL, NM_TITLE, setup_style, ...
from _paths import save_fig, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

setup_style()

FLAT = '#444444'

# Wei genetic single is the main-text Fig 2a; this figure shows the OTHER SEVEN settings. Definitions,
# baselines, skip sets, ranking and metric labels all come from config so this figure cannot drift from
# Fig 2a/2b: a fix applied there applies here.
EVALS = [(lbl.replace('\n', ': '), fn, metric, bl, pcol, skip)
         for lbl, fn, metric, bl, pcol, skip in BENCHMARK_SETTINGS
         if fn != 'wei_genetic_merged_2d.csv']


def panel(ax, title, fname, metric, bl, pcol, skip):
    """Identical rank-slope format to Fig 2a (_rank_slope in fig2_benchmarks.py)."""
    df = load_benchmark(fname)
    n_spec = df[df['triage_2d'] == 'Specific'].drop_duplicates(['dataset', pcol]).shape[0]
    exc_all = ds_weighted_excess(df, metric, bl, pcol, skip, category=None)
    exc_gen = ds_weighted_excess(df, metric, bl, pcol, skip, category='Specific')
    if len(exc_gen) < 2 or n_spec < 5:
        ax.text(0.5, 0.5, f"{title}\n(insufficient specific perturbations)",
                ha='center', va='center', fontsize=NM_LABEL, transform=ax.transAxes)
        ax.axis('off'); return
    rank_all = pd.Series(exc_all).rank(ascending=False).astype(int)
    rank_gen = pd.Series(exc_gen).rank(ascending=False).astype(int)
    methods = sorted(exc_gen.keys(), key=lambda m: rank_gen[m])
    n_m = len(methods)

    X_LEFT, X_RIGHT = 0.0, 0.45
    LABEL_PAD = 0.02
    for m in methods:
        ra = int(rank_all[m]); rg = int(rank_gen[m]); delta = ra - rg
        if abs(delta) >= 3:
            color = C_POS if delta > 0 else C_NEG
            lw, alpha = 1.8, 0.92
        else:
            color, lw, alpha = FLAT, 1.0, 0.85
        ax.plot([X_LEFT, X_RIGHT], [ra, rg], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=3)
        ax.scatter([X_LEFT, X_RIGHT], [ra, rg], color=color, s=24, zorder=5,
                   edgecolors='white', linewidth=0.4)
        ax.text(X_LEFT - LABEL_PAD, ra, f"{m} ({ra})", ha='right', va='center',
                fontsize=NM_LABEL, color=color, fontweight='bold')
        ax.text(X_RIGHT + LABEL_PAD, rg, f"({rg}) {m}", ha='left', va='center',
                fontsize=NM_LABEL, color=color, fontweight='bold')
    ax.set_xlim(X_LEFT - 0.40, X_RIGHT + 0.40)
    ax.set_ylim(n_m + 0.7, 0.3)
    ax.set_xticks([X_LEFT, X_RIGHT])
    ax.set_xticklabels(['All', 'Specific'], fontsize=NM_LABEL, fontweight='bold')
    ax.tick_params(left=False, labelleft=False)
    for s in ('left', 'bottom', 'top', 'right'):
        ax.spines[s].set_visible(False)
    ax.axvline(X_LEFT,  color='#e0e0e0', lw=0.5, zorder=0)
    ax.axvline(X_RIGHT, color='#e0e0e0', lw=0.5, zorder=0)
    ax.set_title(f"{title}, {METRIC_DISPLAY.get(metric, metric)}", fontsize=NM_TITLE, pad=4)


fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.2))
flat = axes.ravel()
for ax, ev in zip(flat[:len(EVALS)], EVALS):
    panel(ax, *ev)

# 8th (empty) cell: shared legend, identical to Fig 2a's.
leg_ax = flat[len(EVALS)]
leg_ax.axis('off')
handles = [
    Line2D([0], [0], color=C_POS, lw=1.8, label='Rise ≥3 ranks'),
    Line2D([0], [0], color=C_NEG, lw=1.8, label='Drop ≥3 ranks'),
    Line2D([0], [0], color=FLAT, lw=1.0, label='|Δ|<3'),
]
leg_ax.legend(handles=handles, fontsize=NM_LABEL, frameon=False, loc='center',
              handlelength=1.5, labelspacing=0.6)

fig.tight_layout()
save_fig(fig, "figS4_rank_shifts")
print(f"✓ Supp Fig 4 (rank shifts) → {FIG_DIR}/figS4_rank_shifts.*")
