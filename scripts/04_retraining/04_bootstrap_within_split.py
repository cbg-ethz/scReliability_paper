"""
04_bootstrap_within_split.py — bootstrap CIs at the perturbation level.

For each (dataset, split_seed, model, comparator, test_set_type, metric),
resample held-out test perturbations with replacement (B replicates), compute
mean paired effect each time, report CI and a CI-based decision label.

We deliberately do NOT report a "bootstrap p-value" — only CIs and conclusions
(genuine_better / comparator_better / inconclusive).

Output: results/retraining_quality_filtering/bootstrap_within_dataset_split.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _lib import RESULTS_DIR, deterministic_subseed


def bootstrap_mean(values: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Return observed mean, median, CI low (2.5%), CI high (97.5%)."""
    n = len(values)
    obs_mean = float(np.mean(values))
    obs_med  = float(np.median(values))
    if n < 2:
        return obs_mean, obs_med, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs_mean, obs_med, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(RESULTS_DIR / "paired_effects_per_perturbation.csv"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "bootstrap_within_dataset_split.csv"))
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed",        type=int, default=20260427)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} paired-effect rows")

    keys = ["dataset", "split_seed", "model", "comparator", "test_set_type", "metric"]
    rows = []
    for k, g in df.groupby(keys):
        # Distinct seed per cell for reproducibility
        seed_int = (args.seed + deterministic_subseed(*k)) % (2**31 - 1)
        rng = np.random.default_rng(seed_int)
        vals = g["paired_effect_positive_means_genuine_better"].to_numpy()
        n = len(vals)
        obs_mean, obs_med, lo, hi = bootstrap_mean(vals, args.n_bootstrap, rng)

        if np.isnan(lo) or np.isnan(hi):
            conclusion = "insufficient_n"
        elif lo > 0:
            conclusion = "genuine_better"
        elif hi < 0:
            conclusion = "comparator_better"
        else:
            conclusion = "inconclusive"

        rows.append({
            "dataset":            k[0],
            "split_seed":         k[1],
            "model":              k[2],
            "comparator":         k[3],
            "test_set_type":      k[4],
            "metric":             k[5],
            "n_test_perturbations": int(n),
            "mean_effect":        obs_mean,
            "median_effect":      obs_med,
            "ci_low":             lo,
            "ci_high":            hi,
            "conclusion":         conclusion,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} cells → {args.output}")
    print("\nConclusion distribution:")
    print(out["conclusion"].value_counts().to_string())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
