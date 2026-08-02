#!/usr/bin/env python3
"""Supplementary Figure 5: multi-model retraining grid (no p-values). train arm × test pool, 5 certified metrics ×
3 models (linearModel, GEARS, scGPT). Each cell prints the dataset-weighted mean and is colored by that
same value on a viridis ramp scaled to its own panel's min/max, so color ranks cells within a panel but is
NOT comparable between panels. The scale bar is therefore unlabeled; the printed numbers are the
quantitative content. Shows that across models, train-reliable tracks train-all while train-specific differs.
linearModel from the certified exhaustive sweep; GEARS/scGPT from the deep retraining runs (same _lib metrics).

CPA is deliberately excluded: the arms are evaluated on unseen perturbations, which CPA is not designed to
predict (Ahlmann-Eltze et al. exclude it from their single-perturbation benchmark on the same grounds), and
its near-zero perturbation-specific signal makes every training arm indistinguishable. Its scores are kept
in intermediate/retraining/archive/. CPA remains in the Wei/AE benchmark re-analyses, where it is one of the
published methods being re-ranked.

Inputs : intermediate/retraining/{exhaustive_linear_scores,deep_per_perturbation_scores}.csv
Output : figures/figS5_retraining_grid.{pdf,png,svg}
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '03_preprocess'))
from config import canonical_quality
from _paths import save_fig, FIG_DIR

ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "intermediate" / "retraining"
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica Neue','Helvetica','DejaVu Sans']})

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_preprocess"))
from config import RETRAIN_SINGLE_DATASETS
# Genetic-single datasets only, matching Fig 2e and Supp Fig 6. On the combinatorial screens the
# linear model is fitted in a way neither Ahlmann-Eltze nor Wei use; see config.RETRAIN_SINGLE_DATASETS.
INFO = RETRAIN_SINGLE_DATASETS
lin = pd.read_csv(DATA / "exhaustive_linear_scores.csv"); lin["model"] = "linearModel"
if "pert_type" not in lin.columns: lin["pert_type"] = "single"
deep = pd.read_csv(DATA / "deep_per_perturbation_scores.csv")
cols = ["dataset","split_seed","model","arm","subset_id","perturbation","quality_label","pert_type",
        "wei_pcc_delta_top100","wei_mse_top100","wei_common_degs","ae_pearson_dlt_top1000","ca_specific","ca_reliable","ca_all"]
df = pd.concat([lin[[c for c in cols if c in lin.columns]], deep[[c for c in cols if c in deep.columns]]], ignore_index=True)
df["quality_label"] = canonical_quality(df["quality_label"])
df = df[df.dataset.isin(INFO) & (df.pert_type == "single") & df.arm.isin(["all","reliable","specific"])]

MODELS = ["linearModel","gears","scgpt"]; TRAIN = ["all","reliable","specific"]; TESTP = ["reliable","specific"]
POOL = {"all":{"Specific","Shared","Unreliable"},"reliable":{"Specific","Shared"},"specific":{"Specific"}}
METR = [("PCC-Δ (100 DEG)  (↑)","wei_pcc_delta_top100",False),("MSE, 100 DEG  (↓)","wei_mse_top100",False),
        ("Common DEGs, 100 DEG  (↑)","wei_common_degs",False),("PCC-Δ (1000 expr)  (↑)","ae_pearson_dlt_top1000",False),
        ("Systema centroid acc.  (↑)",None,True)]
def meanval(model, mcol, is_cen, tr, tp):
    """Dataset-weighted mean: average within each dataset first, then across datasets, so that the two
    large Replogle screens do not dominate. Matches Figure 2e and Supplementary Figure 6."""
    s = df[(df.model == model) & (df.arm == tr)]; sub = s[s.quality_label.isin(POOL[tp])]
    col = f"ca_{tp}" if is_cen else mcol
    if col not in sub.columns or not sub[col].notna().any(): return np.nan
    return sub.groupby("dataset")[col].mean().mean()

fig, axes = plt.subplots(len(METR), len(MODELS), figsize=(11.6, 16))
im = None
for r, (mlabel, mcol, is_cen) in enumerate(METR):
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        M = np.array([[meanval(model, mcol, is_cen, tr, tp) for tp in TESTP] for tr in TRAIN])
        fin = M[np.isfinite(M)]; vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0, 1)
        if vmax <= vmin: vmax = vmin + 1e-9
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(TRAIN)):
            for j in range(len(TESTP)):
                if not np.isfinite(M[i, j]): continue
                tcol = "white" if M[i, j] < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=14, color=tcol, fontweight="bold")
        if r == 0:
            _s = df[df.model == model]
            ax.set_title(f"{model}\n{_s.dataset.nunique()} datasets, {_s.split_seed.nunique()} seed(s)",
                         fontsize=15, fontweight="bold", pad=8)
        if c == 0: ax.set_ylabel(mlabel, fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(TESTP))); ax.set_yticks(range(len(TRAIN)))
        ax.set_xticklabels([f"test\n{t}" for t in TESTP], fontsize=13) if r == len(METR) - 1 else ax.set_xticklabels([])
        ax.set_yticklabels([f"train {t}" for t in TRAIN], fontsize=13) if c == 0 else ax.set_yticklabels([])
fig.tight_layout(rect=[0, 0.008, 0.930, 0.995])

# Scale bar without numeric ticks. Each of the 15 panels is normalized to its own min/max, so no single
# set of numbers could label them; the bar states what the color does mean -- position within a panel --
# and the exact value is printed in every cell.
import matplotlib as _mpl
cax = fig.add_axes([0.945, 0.33, 0.018, 0.34])
cb = fig.colorbar(_mpl.cm.ScalarMappable(norm=_mpl.colors.Normalize(0, 1), cmap="viridis"), cax=cax)
cb.set_ticks([0, 1]); cb.set_ticklabels(["panel\nminimum", "panel\nmaximum"])
cb.ax.tick_params(labelsize=10, length=0)
save_fig(fig, "figS5_retraining_grid")
print(f"✓ Supplement retraining grid → {FIG_DIR}/figS5_retraining_grid.*")
