"""
06_cross_dataset.py — cross-dataset inference using dataset-level effects.

For each (model, comparator, test_set_type, metric):
  - n_datasets, median, mean of dataset-level mean effects
  - n_positive, n_negative
  - PRIMARY: paired Wilcoxon signed-rank against 0 (uses magnitudes)
  - SECONDARY: two-sided sign test against 0.5

Degeneracy filter: datasets whose mean effect across splits falls below
--degenerate-tol are dropped. This is applied to EVERY comparator, not only
random_matched, and run_all.sh invokes it with 1e-6 rather than the 1e-12
default, so the effective cut is looser than the argument default suggests.
The dropped dataset names are reported so the filter is visible in the output.

Output: results/retraining_quality_filtering/cross_dataset_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

from _lib import RESULTS_DIR


def safe_wilcoxon(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank against zero. Returns NaN for n<5
    or when all values are zero."""
    if len(values) < 5 or np.allclose(values, 0):
        return float("nan")
    try:
        _, p = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except ValueError:
        return float("nan")


def safe_sign_test(values: np.ndarray) -> float:
    nz = values[values != 0]
    if len(nz) == 0:
        return float("nan")
    n_pos = int((nz > 0).sum())
    return float(binomtest(n_pos, len(nz), p=0.5, alternative="two-sided").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(RESULTS_DIR / "dataset_level_summary.csv"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "cross_dataset_summary.csv"))
    ap.add_argument("--degenerate-tol", type=float, default=1e-12,
                    help="datasets with |effect| < tol are dropped as degenerate")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    keys = ["model", "comparator", "test_set_type", "metric"]

    rows = []
    for k, g in df.groupby(keys):
        eff = g["mean_effect_across_splits"].to_numpy()
        ds_kept = g[np.abs(eff) >= args.degenerate_tol].copy()
        ds_dropped = len(g) - len(ds_kept)
        if ds_dropped:
            _names = sorted(set(g.loc[np.abs(eff) < args.degenerate_tol, "dataset"]))
            print(f"  [degenerate-tol {args.degenerate_tol:g}] {k}: dropped {ds_dropped} "
                  f"dataset(s): {', '.join(_names)}", flush=True)
        v = ds_kept["mean_effect_across_splits"].to_numpy()
        rows.append({
            "model":                 k[0],
            "comparator":            k[1],
            "test_set_type":         k[2],
            "metric":                k[3],
            "n_datasets":            len(v),
            "n_datasets_dropped_degenerate": ds_dropped,
            "median_dataset_effect": float(np.median(v)) if len(v) else float("nan"),
            "mean_dataset_effect":   float(np.mean  (v)) if len(v) else float("nan"),
            "n_positive_datasets":   int((v > 0).sum()),
            "n_negative_datasets":   int((v < 0).sum()),
            "wilcoxon_p_two_sided":  safe_wilcoxon(v),
            "sign_test_p_two_sided": safe_sign_test(v),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} cross-dataset cells → {args.output}")
    print("\nResults:")
    cols = ["model","comparator","test_set_type","metric",
            "n_datasets","median_dataset_effect","n_positive_datasets",
            "wilcoxon_p_two_sided","sign_test_p_two_sided"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
