"""
03_paired_effects.py — collapse random subsets and compute paired effects.

For each (dataset, split_seed, model, comparator, test_set_type, perturbation,
metric):
  score_comparator[p] = mean over subset_id of comparator score on p
                        (random_* arms only; for 'all', subset is single)
  paired_effect[p]    = score_genuine[p] − score_comparator[p]   for PCC
                        score_comparator[p] − score_genuine[p]   for L2
                        (positive ⇒ genuine better)

Output: results/retraining_quality_filtering/paired_effects_per_perturbation.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _lib import RESULTS_DIR

METRICS = [
    ("wei_pcc_delta_top100",   "higher_better"),
    ("wei_mse_top100",         "lower_better"),
    ("ae_l2_top1000",          "lower_better"),
    ("ae_pearson_dlt_top1000", "higher_better"),
    ("ae_r2_raw_top1000",      "higher_better"),
    ("wei_common_degs",        "higher_better"),
    ("centroid_accuracy",      "higher_better"),
]
COMPARATORS = ["all", "random_matched", "random_nongenuine"]
TEST_SETS   = {"genuine_test": "genuine_test_only",
               "all_test":     "all_held_out"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(RESULTS_DIR / "per_perturbation_scores.csv"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "paired_effects_per_perturbation.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows")

    # Genuine arm score per (ds, split, model, perturbation, metric) — one row each
    gen = df[df["arm"] == "genuine"].copy()

    rows: list[dict] = []
    for comp in COMPARATORS:
        comp_df = df[df["arm"] == comp].copy()
        if comp_df.empty:
            continue
        # Collapse subset_id by mean (for random_* arms; for 'all', mean of one)
        metric_cols = [m for m, _ in METRICS]
        comp_collapsed = (comp_df
                          .groupby(["dataset", "split_seed", "model", "perturbation",
                                    "quality_label", "pert_type"], as_index=False)
                          [metric_cols].mean())
        merged = gen.merge(
            comp_collapsed,
            on=["dataset", "split_seed", "model", "perturbation",
                "quality_label", "pert_type"],
            suffixes=("_gen", "_cmp"),
        )
        for ts_label, ts_filter in TEST_SETS.items():
            sub = merged if ts_filter == "all_held_out" \
                else merged[merged["quality_label"] == "genuine"]
            for metric, direction in METRICS:
                gcol = f"{metric}_gen"
                ccol = f"{metric}_cmp"
                eff = sub[gcol] - sub[ccol] if direction == "higher_better" \
                      else sub[ccol] - sub[gcol]
                rec = pd.DataFrame({
                    "dataset":         sub["dataset"],
                    "split_seed":      sub["split_seed"],
                    "model":           sub["model"],
                    "comparator":      comp,
                    "test_set_type":   ts_label,
                    "perturbation":    sub["perturbation"],
                    "quality_label":   sub["quality_label"],
                    "pert_type":       sub["pert_type"],
                    "metric":          metric,
                    "score_genuine":   sub[gcol],
                    "score_comparator":sub[ccol],
                    "paired_effect_positive_means_genuine_better": eff,
                })
                rows.append(rec)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["paired_effect_positive_means_genuine_better"])
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows → {args.output}")
    print("\nRows per (comparator × test_set × metric × model):")
    print(out.groupby(["comparator", "test_set_type", "metric", "model"]).size().to_string())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
