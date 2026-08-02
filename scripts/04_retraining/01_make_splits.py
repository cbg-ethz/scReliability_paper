"""
01_make_splits.py — generate per-dataset stratified train/test splits.

For each dataset and each split_seed in {0..N_SPLITS-1}, partition perturbations
75/25 stratified by (quality_label × pert_type). Written to:

    results/retraining_quality_filtering/splits/{dataset}_split{seed}.csv

Columns: perturbation, split, triage_2d, quality_label, pert_type
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    DATASETS, RESULTS_DIR, SPLITS_DIR, deterministic_subseed,
    load_quality_table, make_stratified_split, dump_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits",  type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed",      type=int, default=20260427)
    args = ap.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    qual = load_quality_table()
    qual = qual[qual["dataset"].isin(DATASETS)].copy()
    qual = qual.dropna(subset=["quality_label"])

    log_rows = []
    for ds in DATASETS:
        sub = qual[qual["dataset"] == ds][
            ["perturbation", "triage_2d", "quality_label", "pert_type"]
        ].drop_duplicates("perturbation").reset_index(drop=True)
        if sub.empty:
            print(f"  [{ds}] no quality rows — skipped")
            continue
        for k in range(args.n_splits):
            split_seed = (args.seed + 1000 * k + deterministic_subseed(ds)) % (2**31 - 1)
            df = make_stratified_split(sub, split_seed, test_frac=args.test_frac)
            n_train = (df["split"] == "train").sum()
            n_test  = (df["split"] == "test").sum()
            n_gtr   = ((df["split"] == "train") & (df["quality_label"] == "genuine")).sum()
            n_gte   = ((df["split"] == "test")  & (df["quality_label"] == "genuine")).sum()
            n_ngtr  = ((df["split"] == "train") & (df["quality_label"] != "genuine")).sum()
            out = SPLITS_DIR / f"{ds}_split{k}.csv"
            df.to_csv(out, index=False)
            log_rows.append(dict(
                dataset=ds, split=k, split_seed=split_seed,
                n_train=n_train, n_test=n_test,
                n_genuine_train=n_gtr, n_genuine_test=n_gte,
                n_nongenuine_train=n_ngtr, file=out.name,
            ))
            print(f"  [{ds}] split{k}: train={n_train} test={n_test}  "
                  f"gen_train={n_gtr} gen_test={n_gte} nongen_train={n_ngtr}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(RESULTS_DIR / "splits_summary.csv", index=False)
    dump_config(vars(args) | {"step": "01_make_splits"}, RESULTS_DIR)
    print(f"\nSplits summary → {RESULTS_DIR/'splits_summary.csv'}")


if __name__ == "__main__":
    main()
