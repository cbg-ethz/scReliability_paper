#!/usr/bin/env python3
"""Figure 2e (retraining headline): linear-model performance vs the amount of reliable training data,
evaluated on held-out SPECIFIC perturbations.

x = % of all training perturbations used; y = performance relative to
train-all (=1). Curve = random reliable subsets swept by size (10 draws/size); squares = the specific
arm; stars = train-all. MSE plotted as train-all/subset so up is better for every metric.
Evaluated on the SPECIFIC test pool (matching panels a-d). The reliable subset (~55% of perturbations)
recovers ~train-all performance, and the specific subset (~38%) gives similar performance with fewer targets.
No conclusions in-plot. 5 certified metrics, 3 genetic-single datasets (config.RETRAIN_SINGLE_DATASETS).

Input : intermediate/retraining/reliable_quantity_sweep.csv  (scripts/04_retraining/10_reliable_quantity_sweep.py)
Output : figures/panels/fig2e_reliable_quantity.{pdf,png,svg}
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '03_preprocess'))
from config import (setup_style, canonical_quality, RETRAIN_SINGLE_DATASETS,
                    NM_TITLE, NM_LABEL, NM_TICK, NM_LEGEND, NM_ANNOT, NM_TINY)
from _paths import FIG_PANEL_DIR

ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
SWEEP = ROOT / "intermediate" / "retraining" / "reliable_quantity_sweep.csv"

setup_style()   # shared figure style from config.py

d = pd.read_csv(SWEEP)
d["quality_label"] = canonical_quality(d["quality_label"])
d = d[d.quality_label == "Specific"]          # evaluate on the SPECIFIC test pool
# Genetic-single datasets only; see config.RETRAIN_SINGLE_DATASETS. On the combinatorial screens the
# linear model is fitted in a way neither Ahlmann-Eltze nor Wei use, so those datasets cannot support
# a like-for-like claim about training on the reliable subset.
d = d[d.dataset.isin(RETRAIN_SINGLE_DATASETS)]
DS = sorted(d.dataset.unique())
# Fractions are read from the arms present in the sweep, so the panel cannot silently
# drop a point if the sweep grid changes.
FR = sorted(float(a.rsplit("_", 1)[-1]) for a in d.arm.unique() if a.startswith("reliable_sub_")) + [1.00]
arm_for = lambda f: "reliable" if f == 1.0 else f"reliable_sub_{f:.2f}"
# Metric names carry the benchmark they come from. Both Wei and Ahlmann-Eltze score the Pearson
# correlation of the delta vector, so both are written PCC-Δ and distinguished by their gene set.
METR = [("Wei PCC-Δ (100 DEG)", "wei_pcc_delta_top100", True, "#1f6fb2"),
        ("Wei MSE (100 DEG)", "wei_mse_top100", False, "#d1495b"),
        ("Wei common-DEGs (100 DEG)", "wei_common_degs", True, "#2e8b57"),
        ("AE PCC-Δ (1000 expr)", "ae_pearson_dlt_top1000", True, "#e6822e"),
        ("Systema centroid", "ca_specific", True, "#7a5195")]
sizes = lambda arm: d[d.arm == arm].groupby("dataset")["n_train"].first(); allsz = sizes("all")
perf = lambda arm, col: d[d.arm == arm].groupby("dataset")[col].mean()
def rel(col, hib, arm):
    # Dataset-weighted mean of the raw scores first, then normalise to train-all. Normalising per dataset
    # and averaging the ratios afterwards would weight each dataset by its own train-all score instead of
    # comparing like with like; the y-axis is a ratio of means, not a mean of ratios.
    a = perf("all", col).reindex(DS); v = perf(arm, col).reindex(DS)
    keep = a.notna() & v.notna()
    if not keep.any():
        return np.nan
    am, vm = a[keep].mean(), v[keep].mean()
    return (vm / am) if hib else (am / vm)
xfrac = {f: float((sizes(arm_for(f)) / allsz).mean() * 100) for f in FR}
xspec = float((sizes("specific") / allsz).mean() * 100); Xpct = xfrac[1.0]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.axhline(1.0, color="#555", lw=0.8, ls="--", zorder=2)
ax.axvline(Xpct, color="#aaa", lw=0.7, ls=":", zorder=1)
ax.axvspan(Xpct, 100, color="#bdbdbd", alpha=0.12, lw=0, zorder=0)
# Some metrics are undefined for a dataset (the Systema centroid needs >=2 specific test perturbations,
# which Wessels lacks), so a series can rest on fewer datasets. Label it rather than averaging silently.
SERIES_LABEL = {}
for label, col, hib, c in METR:
    _n = d.loc[d[col].notna(), "dataset"].nunique()
    SERIES_LABEL[label] = label if _n == len(DS) else f"{label} (n={_n})"

for label, col, hib, c in METR:
    xs = [xfrac[f] for f in FR]; ys = [rel(col, hib, arm_for(f)) for f in FR]
    ax.plot(xs, ys, color=c, lw=1.3, marker="o", ms=3, zorder=4, label=SERIES_LABEL[label])
    ax.plot([Xpct, 100], [ys[-1], 1.0], color=c, lw=0.9, ls="--", alpha=0.5, zorder=3)
    ax.scatter([100], [1.0], color=c, marker="*", s=40, zorder=5, edgecolor="white", linewidth=0.4)
    ax.scatter([xspec], [rel(col, hib, "specific")], color=c, marker="s", s=14, zorder=5, edgecolor="white", linewidth=0.3)
ax.set_xlabel("% of all training perturbations used")
ax.set_ylabel("performance on held-out specific\nperturbations (relative to train-all)")
ax.set_xlim(0, 103); ax.set_xticks(list(range(0, 101, 10)))
ax.set_ylim(ax.get_ylim()[0], max(1.13, ax.get_ylim()[1]))
yb = ax.get_ylim()[0]
ax.text(Xpct / 2, yb + 0.008, "reliable", fontsize=NM_TINY, color="black", ha="center")
ax.text((Xpct + 100) / 2, yb + 0.008, "unreliable", fontsize=NM_TINY, color="black", ha="center")
# Every element gets a legend entry, and the two that are not measurements say so: the coloured dashed
# segment spans a range where no subset exists (the reliable pool ends at Xpct), and the star sits at 1.0
# for every metric because train-all is the normalisation reference, not because it was measured.
h = [Line2D([0], [0], color=c, lw=1.3, marker="o", ms=3, label=SERIES_LABEL[l]) for l, _, _, c in METR]
h += [Line2D([0], [0], color="#777", lw=1.3, marker="o", ms=3,
             label="measured: random reliable subsets"),
      Line2D([0], [0], color="w", marker="s", mfc="#333", ms=4,
             label="measured: specific-only arm"),
      Line2D([0], [0], color="#777", lw=0.9, ls="--", alpha=0.6,
             label="guide to train-all (not measured)"),
      Line2D([0], [0], color="#555", lw=0.8, ls="--", marker="*", mfc="#333", mec="#333", ms=7,
             label="train-all reference (= 1 by definition)")]
# Lifted clear of the "reliable" / "unreliable" annotations along the bottom axis.
ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.055), fontsize=NM_LEGEND,
          frameon=False, handlelength=1.4, labelspacing=0.32, borderpad=0.2)
ax.set_title("Linear model: performance vs amount of reliable training data", fontsize=NM_TITLE)
fig.tight_layout()
base = FIG_PANEL_DIR / "fig2e_reliable_quantity"
for ext in (".pdf", ".png", ".svg"):
    fig.savefig(str(base) + ext, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", transparent=False, dpi=300 if ext == ".png" else None)
print(f"✓ Fig 2 panel e (reliable-quantity, specific test) → {base}.pdf / .png / .svg")

