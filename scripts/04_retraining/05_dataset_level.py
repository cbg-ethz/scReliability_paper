"""
05_dataset_level.py — collapse splits within each dataset.

For each (dataset, model, comparator, test_set_type, metric), compute the
dataset-level effect as the mean of split-level mean paired effects (the
within-split bootstrap means computed in step 04). This treats split-level
estimates as repeated measures, not independent biological replicates.

Output: results/retraining_quality_filtering/dataset_level_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _lib import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(RESULTS_DIR / "bootstrap_within_dataset_split.csv"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "dataset_level_summary.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["conclusion"] != "insufficient_n"].copy()
    print(f"Loaded {len(df):,} split-level cells (after dropping insufficient_n)")

    keys = ["dataset", "model", "comparator", "test_set_type", "metric"]
    rows = []
    for k, g in df.groupby(keys):
        rows.append({
            "dataset":               k[0],
            "model":                 k[1],
            "comparator":            k[2],
            "test_set_type":         k[3],
            "metric":                k[4],
            "n_valid_splits":        len(g),
            "mean_effect_across_splits":   float(g["mean_effect"].mean()),
            "median_effect_across_splits": float(g["mean_effect"].median()),
            "n_positive_splits":     int((g["mean_effect"] > 0).sum()),
            "n_negative_splits":     int((g["mean_effect"] < 0).sum()),
            "min_split_effect":      float(g["mean_effect"].min()),
            "max_split_effect":      float(g["mean_effect"].max()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} dataset-level cells → {args.output}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
