#!/usr/bin/env python3
"""Supplementary Figure 6: data-scaling of the linear model by training pool, evaluated on two test pools.
Rows = test pool (reliable, specific); columns = 3 certified metrics. Each line = a training pool
(reliable / specific / all / unreliable) swept by # training perturbations; dataset-weighted mean ± s.e.m.
Qualifying genetic gene-target datasets only (≥8 perturbations in a pool to enter that arm); legend annotates
how many datasets back each arm. Descriptive only — no conclusions in-plot.

Input  : intermediate/retraining/scaling_pools.csv  (scripts/04_retraining/11_scaling_pools.py)
Output : figures/figS6_data_scaling.{pdf,png,svg}
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '03_preprocess'))
from config import RETRAIN_SINGLE_DATASETS
from _paths import save_fig, FIG_DIR

ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "intermediate" / "retraining" / "scaling_pools.csv"
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica Neue','Helvetica','DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.titlesize':8,'axes.titleweight':'bold','axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})

d = pd.read_csv(DATA)
# Genetic-single datasets only, matching Fig 2e; see config.RETRAIN_SINGLE_DATASETS for why the
# combinatorial screens are excluded from the linear-model retraining analyses.
d = d[d.dataset.isin(RETRAIN_SINGLE_DATASETS)]
METR = [("Wei PCC-Δ (100 DEG) (↑)","wei_pcc"),("AE PCC-Δ (1000 expr) (↑)","ae_pcc"),("Wei common-DEGs (100 DEG) (↑)","wei_cdeg")]
ARMS = [("reliable","#1e8449","o"),("specific","#16a085","D"),("all","#4878CF","s"),("unreliable","#e08a1e","^")]
POOLS = ["reliable","specific"]

from matplotlib.lines import Line2D
fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.0))
for ridx, POOL in enumerate(POOLS):
    dp = d[d.test_pool == POOL]
    for cidx, (lab, m) in enumerate(METR):
        ax = axes[ridx, cidx]
        for arm, col, mk in ARMS:
            s = dp[dp.arm == arm]
            if not len(s): continue
            perds = s.groupby(["dataset","frac"]).agg(v=(m,"mean"), n=("n_train","mean")).reset_index()
            agg = perds.groupby("frac").agg(v=("v","mean"), se=("v","sem"), n=("n","mean")).reset_index().sort_values("n")
            ax.errorbar(agg.n, agg.v, yerr=agg.se, marker=mk, ms=4.0, lw=1.4, color=col, capsize=2.5)
        ax.set_xscale("log"); ax.tick_params(labelsize=8)
        if ridx == 1: ax.set_xlabel("# training perturbations", fontsize=9)
        if cidx == 0: ax.set_ylabel(f"test: {POOL}\n{lab}", fontsize=9, fontweight="bold")
        else: ax.set_ylabel(lab, fontsize=9)
        if ridx == 0: ax.set_title(lab, fontsize=10, fontweight="bold")
# Single figure-level legend below the panels. The dataset count differs between the two test pools,
# so report the range rather than a single number that would be wrong for one row.
def _arm_n(arm):
    per_pool = d[d.arm == arm].groupby("test_pool").dataset.nunique()
    lo, hi = int(per_pool.min()), int(per_pool.max())
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


handles = [Line2D([0],[0], color=col, marker=mk, ms=6, lw=1.6,
                  label=f"{arm}  (n={_arm_n(arm)})") for arm, col, mk in ARMS]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
           title="training pool (n = # datasets per test pool)", title_fontsize=9,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Data-scaling by training pool, evaluated on reliable vs specific test perturbations\n"
             "(linear model, qualifying genetic datasets, dataset-weighted)", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0.07, 1, 0.93])
save_fig(fig, "figS6_data_scaling")
print(f"✓ Supplement scaling-pools → {FIG_DIR}/figS6_data_scaling.*")
