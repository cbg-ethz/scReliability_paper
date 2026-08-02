"""Supplementary rigor tables for the retraining experiment.

Generates two CSVs (no figures):

Table S — per-split breakdown:
  rows: (model, metric, split_seed)
  cols: n_datasets, n_pos, median_delta, wilcoxon_p
  Computed for the 4 cells: {trainMean, linearModel} × {centroid_accuracy,
  wei_common_degs}, separately for each of the 5 split seeds.

Table S — per-dataset / per-split / per-subset stability:
  rows: (model, metric, dataset, split_seed)
  cols: n_subsets, mean_subset_effect, std_subset_effect, min, max,
        pct_subsets_genuine_winning
  Computed for the same 4 cells, with each row showing how the 50 random
  subsets behave within that (dataset, split) cell.

The two CSVs are written to:
  results/retraining_quality_filtering/figS_rigor_per_split.csv
  results/retraining_quality_filtering/figS_rigor_per_subset.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from _lib import RESULTS_DIR

INFORMATIVE = ["Replogle_K562essential", "Replogle_RPE1essential",
               "Norman", "Schmidt", "Adamson", "Replogle_exp6", "Wessels"]

CELLS = [
    ("trainMean",   "centroid_accuracy"),
    ("trainMean",   "wei_common_degs"),
    ("linearModel", "centroid_accuracy"),
    ("linearModel", "wei_common_degs"),
]

raw = pd.read_csv(RESULTS_DIR / "per_perturbation_scores.csv")
raw = raw[raw.dataset.isin(INFORMATIVE) & (raw.pert_type == "single") &
          (raw.quality_label == "genuine")]


# ── Table 1: per-split breakdown ─────────────────────────────────────────────
rows1 = []
for model, metric in CELLS:
    for sp in sorted(raw.split_seed.unique()):
        s = raw[(raw.split_seed == sp) & (raw.model == model)]
        rm = (s[s.arm == "random_matched"]
                .groupby(["dataset", "perturbation"], as_index=False)[metric].mean())
        gen = s[s.arm == "genuine"][["dataset", "perturbation", metric]]
        merged = gen.merge(rm, on=["dataset", "perturbation"], suffixes=("_g", "_r"))
        merged["d"] = merged[f"{metric}_g"] - merged[f"{metric}_r"]
        ds_eff = merged.groupby("dataset")["d"].mean()
        ds_eff = ds_eff[ds_eff.abs() > 1e-6]
        v = ds_eff.to_numpy()
        if len(v) >= 5 and not np.allclose(v, 0):
            try:
                _, p = wilcoxon(v, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        rows1.append({
            "model":           model,
            "metric":          metric,
            "split_seed":      sp,
            "n_datasets":      int(len(v)),
            "n_pos":           int((v > 0).sum()),
            "median_delta":    float(np.median(v)),
            "mean_delta":      float(np.mean(v)),
            "wilcoxon_p":      float(p),
        })

t1 = pd.DataFrame(rows1)
out1 = RESULTS_DIR / "figS_rigor_per_split.csv"
t1.to_csv(out1, index=False)
print(f"Per-split breakdown → {out1}  ({len(t1)} rows)")


# ── Table 2: per-dataset / per-split / per-subset stability ──────────────────
rows2 = []
for model, metric in CELLS:
    for ds in INFORMATIVE:
        for sp in sorted(raw.split_seed.unique()):
            sub = raw[(raw.dataset == ds) & (raw.split_seed == sp) &
                       (raw.model == model)]
            if sub.empty:
                continue
            gen_score = sub[sub.arm == "genuine"].groupby("perturbation")[metric].mean()
            rm = sub[sub.arm == "random_matched"]
            sub_eff = []
            for sid, g in rm.groupby("subset_id"):
                comp = g.groupby("perturbation")[metric].mean()
                aligned = gen_score.reindex(comp.index)
                d = (aligned - comp).dropna()
                if len(d):
                    sub_eff.append(float(d.mean()))
            sub_eff = np.array(sub_eff)
            if len(sub_eff) == 0:
                continue
            rows2.append({
                "model":          model,
                "metric":         metric,
                "dataset":        ds,
                "split_seed":     sp,
                "n_subsets":      int(len(sub_eff)),
                "mean_subset_effect": float(sub_eff.mean()),
                "std_subset_effect":  float(sub_eff.std()),
                "min_subset_effect":  float(sub_eff.min()),
                "max_subset_effect":  float(sub_eff.max()),
                "pct_subsets_genuine_winning": float(100 * (sub_eff > 0).mean()),
            })

t2 = pd.DataFrame(rows2)
out2 = RESULTS_DIR / "figS_rigor_per_subset.csv"
t2.to_csv(out2, index=False)
print(f"Per-subset stability → {out2}  ({len(t2)} rows)")


# ── Print compact summaries ──────────────────────────────────────────────────
print("\n=== Per-split summary (median Δ, p) ===")
for model, metric in CELLS:
    sub = t1[(t1.model == model) & (t1.metric == metric)]
    print(f"\n  {model} × {metric}:")
    print(f"    {'split':<6} {'n_pos/n':>9} {'median Δ':>10} {'p':>9}")
    for _, r in sub.iterrows():
        print(f"    {int(r.split_seed):<6} "
              f"{int(r.n_pos)}/{int(r.n_datasets):<7} "
              f"{r.median_delta:>+10.4f} {r.wilcoxon_p:>9.4f}")

print("\n=== Per-subset summary (% of 50 subsets with Genuine winning) ===")
print("(rows are model × metric × dataset, columns are 5 split seeds)")
for model, metric in CELLS:
    sub = t2[(t2.model == model) & (t2.metric == metric)]
    pivot = sub.pivot_table(index="dataset", columns="split_seed",
                              values="pct_subsets_genuine_winning")
    pivot = pivot.reindex(INFORMATIVE)
    print(f"\n  {model} × {metric}:")
    print(pivot.round(0).astype('Int64').to_string())
