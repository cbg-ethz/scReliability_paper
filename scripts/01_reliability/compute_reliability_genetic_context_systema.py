#!/usr/bin/env python3
"""
compute_reliability_genetic_context_systema.py
==================================
Compute Systema's systematic variation metric (Viñas Torné et al., Nature
Biotechnology 2025) for each perturbation in the Wei et al. (scPerturBench)
datasets, and merge with pre-computed split-half reliability to produce a
2D evaluation framework.

SYSTEMA'S SYSTEMATIC VARIATION METRIC
--------------------------------------
From the paper (Fig. 3a, Methods):

  "We define perturbation-specific shifts as vectors that point to the
   centroid of cells that underwent the same type of perturbation using
   the centroid of control cells as reference, while the average
   perturbation effect is the vector pointing to the centroid of all
   perturbed cells."

For each perturbation p:
  1. control_mean   = mean expression across all control cells
  2. centroid_p     = mean expression across cells with perturbation p
  3. perturbed_mean = mean expression across ALL non-control cells
                      (cell-level mean, NOT perturbation-level mean;
                       perturbations with more cells contribute more)
  4. δ_p   = centroid_p   - control_mean     (perturbation-specific shift)
  5. δ_avg = perturbed_mean - control_mean   (average perturbation effect)
  6. cos_sim_p = cosine_similarity(δ_p, δ_avg)

High cos_sim → perturbation shift is aligned with the average effect
               (dominated by systematic / shared variation)
Low cos_sim  → perturbation has a unique, specific transcriptional effect

SYSTEMA'S RE-REFERENCED EVALUATION
------------------------------------
Standard evaluation:  PCC(predicted - control_mean, observed - control_mean)
Systema evaluation:   PCC(predicted - perturbed_mean, observed - perturbed_mean)

The re-referencing removes the shared systematic component, isolating
perturbation-specific effects. We do NOT re-run method predictions here;
we only compute the cos_sim diagnostic per perturbation.

2D FRAMEWORK
-------------
Dimension 1 (reliability ρ):  Split-half reliability (from existing CSVs)
  - High ρ → ground truth is reproducible across cell samples
  - Low ρ  → ground truth is noisy / Unreliable

Dimension 2 (systematic variation cos_sim):
  - High cos_sim → perturbation effect ≈ average effect (systematic)
  - Low cos_sim  → perturbation has unique, specific effect

Combined triage:
  - Low ρ                          → Unmeasurable (noisy ground truth)
  - High ρ, high cos_sim           → Falsely solved (systematic only)
  - High ρ, low cos_sim, high CE   → Genuinely solved
  - High ρ, low cos_sim, low CE    → True frontier

USAGE
-----
# Single dataset:
python compute_reliability_genetic_context_systema.py --dataset Norman

# All 13 genetic datasets:
python compute_reliability_genetic_context_systema.py --dataset genetic

# All datasets:
python compute_reliability_genetic_context_systema.py --dataset all

OUTPUTS
-------
  systema_{dataset}.csv              Per-perturbation: cos_sim, norms, etc.
  systema_reliability_2d_{dataset}.csv  Merged: cos_sim + reliability
  systema_summary.csv                Cross-dataset summary (if multiple)

DEPENDENCIES
------------
  numpy, pandas, scipy (sparse), anndata
  (No additional packages beyond what compute_reliability_genetic_context.py uses)
"""

import os
import sys
import json
import time
import argparse
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# DATASET CONFIGURATION — same as compute_reliability_genetic_context.py
# ============================================================

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
DATA_DIR = "" + _ROOT + "/data/Wei_et_al_data/genetic_context_preprocessed_h5ad"

GENETIC_DATASETS = {
    "Norman": {},
    "Adamson": {},
    "Replogle_K562essential": {},
    "Replogle_RPE1essential": {},
    "Replogle_exp6": {},
    "Replogle_exp7": {},
    "Replogle_exp8": {},
    "TianActivation": {},
    "TianInhibition": {},
    "Frangieh": {"has_conditions": True, "condition_col": "condition"},
    "Papalexi": {},
    "Wessels": {},
    "Schmidt": {},
}

CHEMICAL_DATASETS = {
    # sciplex3 single-drug datasets: Wei evaluates at cov_drug_dose_name
    # (cell_line + drug + dose). Match exactly to enable benchmark merge.
    "sciplex3_A549": {"sparse_x": True, "pert_col": "cov_drug_dose_name", "ctrl_label": "A549_control_1.0"},
    "sciplex3_K562": {"pert_col": "cov_drug_dose_name", "ctrl_label": "K562_control_1.0"},
    "sciplex3_MCF7": {"pert_col": "cov_drug_dose_name", "ctrl_label": "MCF7_control_1.0"},
    # sciplex3_comb: same convention; comb h5ad is A549-only.
    "sciplex3_comb": {"pert_col": "cov_drug_dose_name", "ctrl_label": "A549_control_1.0"},
}

ALL_DATASETS = {}
for name, cfg in GENETIC_DATASETS.items():
    ALL_DATASETS[name] = {
        "path": os.path.join(DATA_DIR, f"{name}.h5ad"),
        "pert_col": "perturbation",
        "ctrl_label": "control",
        "category": "genetic",
        **cfg,
    }
for name, cfg in CHEMICAL_DATASETS.items():
    ALL_DATASETS[name] = {
        "path": os.path.join(DATA_DIR, f"{name}.h5ad"),
        "pert_col": "perturbation",
        "ctrl_label": "control",
        "category": "chemical",
        **cfg,
    }

DEFAULT_OUTPUT_DIR = "" + _ROOT + "/output/reliability_fig1/reliability_genetic_systema"
DEFAULT_RELIABILITY_DIR = "" + _ROOT + "/output/reliability_fig1/reliability_genetic"


# ============================================================
# COSINE SIMILARITY — vectorized for all perturbations at once
# ============================================================

def cosine_similarity(a, b):
    """
    Cosine similarity between two 1D vectors.

    Parameters
    ----------
    a, b : ndarray, shape (G,)

    Returns
    -------
    float : cosine similarity in [-1, 1]
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(deltas, avg_delta):
    """
    Cosine similarity between each row of `deltas` and a single vector.

    Parameters
    ----------
    deltas    : ndarray, shape (N_perts, G)
    avg_delta : ndarray, shape (G,)

    Returns
    -------
    cos_sims : ndarray, shape (N_perts,)
    """
    # Numerator: dot products
    dots = deltas @ avg_delta  # (N_perts,)

    # Denominators
    norms_delta = np.linalg.norm(deltas, axis=1)  # (N_perts,)
    norm_avg = np.linalg.norm(avg_delta)

    denom = norms_delta * norm_avg
    denom[denom < 1e-12] = 1.0  # avoid division by zero

    return dots / denom


# ============================================================
# CORE: COMPUTE SYSTEMA METRICS FOR ONE DATASET
# ============================================================

RELIABILITY_THRESH = 0.5  # ρ cutoff used to define "reliable" perturbations.
                          # cos_sim is computed ONLY for perts with ρ ≥ this
                          # value; others receive cos_sim = NaN. δ_avg is
                          # likewise computed from cells of reliable perts
                          # only, so the systematic axis is not contaminated
                          # by noise-dominated perturbations.

def _load_reliable_perts(reliability_dir, dataset_name, rel_col_fallbacks=(
        "sb_from_median_all_genes", "sb_from_mean_all_genes",
        "sb_from_median", "sb_from_mean")):
    """Load the per-perturbation reliability summary and return the set of
    (condition, perturbation) pairs with ρ ≥ RELIABILITY_THRESH.
    If the file is missing, returns None (caller should fail).
    Also returns the raw reliability DataFrame for later merging."""
    rel_path = Path(reliability_dir) / f"reliability_summary_{dataset_name}.csv"
    if not rel_path.exists():
        return None, None
    rel_df = pd.read_csv(rel_path)
    rel_col = next((c for c in rel_col_fallbacks if c in rel_df.columns), None)
    if rel_col is None:
        return None, rel_df
    reliable = rel_df[rel_df[rel_col] >= RELIABILITY_THRESH]
    # Index by (condition, perturbation) if both present, else by perturbation
    if "condition" in reliable.columns:
        key_set = set(zip(reliable["condition"].astype(str),
                          reliable["perturbation"].astype(str)))
    else:
        key_set = set(reliable["perturbation"].astype(str))
    return key_set, rel_df


def compute_systema_metrics(dataset_name, config, args):
    """
    Compute Systema's systematic variation metric with a reliability-first
    filter:

      1. Load the per-perturbation reliability (ρ) from `args.reliability_dir`.
      2. Identify reliable perts (ρ ≥ 0.5).
      3. δ_avg = mean(cells of RELIABLE non-control perts) − mean(control)
         (not all non-control cells — noise-dominated perts should not
          shape the systematic axis).
      4. cos_sim is computed ONLY for reliable perts; unreliable perts
         receive cos_sim = NaN (interpreting the angle of noise is a
         category error).

    Returns
    -------
    df : DataFrame with columns:
        dataset, perturbation, condition, n_ko_cells,
        cos_sim, norm_delta_p, norm_delta_avg, norm_delta_specific,
        frac_systematic_var
    """
    t_start = time.time()
    h5ad_path = config["path"]
    pert_col = config["pert_col"]
    ctrl_label = config["ctrl_label"]

    print(f"\n{'='*70}")
    print(f"  Systema analysis: {dataset_name}")
    print(f"  Path: {h5ad_path}")
    print(f"{'='*70}")

    # ---- Load reliability (used to filter which perts define δ_avg) ----
    print(f"  [0/3] Loading reliability data …")
    reliable_keys, rel_df = _load_reliable_perts(
        args.reliability_dir, dataset_name)
    if reliable_keys is None:
        raise FileNotFoundError(
            f"Reliability data missing for {dataset_name}. "
            f"The reliability-filtered systema pipeline requires it. "
            f"Looked in: {args.reliability_dir}")
    print(f"        Reliable perts: {len(reliable_keys)}")

    # ---- Load h5ad ----
    print(f"  [1/3] Loading h5ad...")
    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    print(f"        {adata.n_obs:,} cells × {adata.n_vars:,} genes, "
          f"loaded in {time.time()-t0:.1f}s")

    # ---- Prepare expression matrix ----
    print(f"  [2/3] Preparing expression matrix...")
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
        is_sparse = True
    else:
        X = np.asarray(X, dtype=np.float32)
        is_sparse = False

    # ---- Determine condition groups ----
    if config.get("has_conditions", False):
        condition_col = config["condition_col"]
        conditions = adata.obs[condition_col].unique().tolist()
        print(f"        Conditions: {conditions}")
        condition_groups = [
            (cond, adata.obs[condition_col] == cond)
            for cond in conditions
        ]
    else:
        condition_groups = [("all", np.ones(adata.n_obs, dtype=bool))]

    # ---- Compute per condition ----
    all_rows = []

    for cond_name, cond_mask in condition_groups:
        if config.get("has_conditions"):
            print(f"\n  ── Condition: {cond_name} ({cond_mask.sum():,} cells) ──")

        # Get labels within this condition
        labels = adata.obs[pert_col].astype(str).str.strip().values
        cond_indices = np.where(
            cond_mask.values if hasattr(cond_mask, 'values') else cond_mask
        )[0]
        cond_labels = labels[cond_indices]

        # Separate control and perturbed cells
        ctrl_mask = cond_labels == ctrl_label
        ctrl_idx = cond_indices[ctrl_mask]
        ko_mask = ~ctrl_mask
        ko_idx_all = cond_indices[ko_mask]
        ko_labels = cond_labels[ko_mask]

        n_ctrl = len(ctrl_idx)
        n_ko_total = len(ko_idx_all)
        unique_perts = np.unique(ko_labels)
        n_perts = len(unique_perts)

        print(f"        Controls: {n_ctrl:,} cells")
        print(f"        Perturbed: {n_ko_total:,} cells across {n_perts} perturbations")

        if n_ctrl < 4 or n_perts < 2:
            print(f"        ⚠ Skipping (too few controls or perturbations)")
            continue

        # ---- Step 1: Compute control_mean ----
        if is_sparse:
            # Use sparse mean (memory efficient)
            control_mean = np.asarray(
                X[ctrl_idx].mean(axis=0), dtype=np.float64
            ).ravel()
        else:
            control_mean = X[ctrl_idx].astype(np.float64).mean(axis=0)

        # ---- Step 2: Identify reliable perts in this condition and
        #              build a cell-mask for "reliable non-control" ----
        def _is_reliable(pert):
            if reliable_keys is None:
                return False
            if isinstance(next(iter(reliable_keys)), tuple):
                return (str(cond_name), str(pert)) in reliable_keys
            return str(pert) in reliable_keys

        reliable_cond_perts = np.array([_is_reliable(p) for p in unique_perts])
        reliable_pert_set = set(unique_perts[reliable_cond_perts])
        print(f"        Reliable perts in this condition: "
              f"{len(reliable_pert_set)}/{len(unique_perts)}")

        reliable_cell_mask_local = np.array(
            [lbl in reliable_pert_set for lbl in ko_labels], dtype=bool)
        reliable_cell_idx = ko_idx_all[reliable_cell_mask_local]

        # ---- Step 3: Compute δ_avg from cells of RELIABLE perts only ----
        if len(reliable_cell_idx) == 0:
            delta_avg = np.zeros_like(control_mean)
            norm_delta_avg = 0.0
            print(f"        ⚠ no reliable perts — δ_avg set to zero; "
                  f"all cos_sim will be NaN")
        else:
            if is_sparse:
                perturbed_mean = np.asarray(
                    X[reliable_cell_idx].mean(axis=0), dtype=np.float64
                ).ravel()
            else:
                perturbed_mean = X[reliable_cell_idx].astype(np.float64).mean(axis=0)
            delta_avg = perturbed_mean - control_mean
            norm_delta_avg = float(np.linalg.norm(delta_avg))
            print(f"        ||δ_avg|| = {norm_delta_avg:.4f}  "
                  f"(from {len(reliable_cell_idx):,} cells in "
                  f"{len(reliable_pert_set)} reliable perts)")

        # ---- Step 4: For each perturbation, compute δ_p and — if reliable —
        #              cos_sim, norm_delta_specific, frac_systematic_var.
        #              Unreliable perts get NaN for these three. ----
        print(f"  [3/3] Computing cosine similarities (reliable perts only)...")
        t0 = time.time()

        for pert in unique_perts:
            pert_mask_local = ko_labels == pert
            pert_idx = ko_idx_all[pert_mask_local]
            n_ko = len(pert_idx)
            is_rel = pert in reliable_pert_set

            # Compute centroid of this perturbation's cells
            if is_sparse:
                centroid_p = np.asarray(
                    X[pert_idx].mean(axis=0), dtype=np.float64
                ).ravel()
            else:
                centroid_p = X[pert_idx].astype(np.float64).mean(axis=0)

            # δ_p = centroid_p - control_mean  (always computed; does not
            # depend on δ_avg, so record the norm even for unreliable perts)
            delta_p = centroid_p - control_mean
            norm_delta_p = float(np.linalg.norm(delta_p))

            if is_rel and norm_delta_avg > 1e-12:
                cos_sim = cosine_similarity(delta_p, delta_avg)
                proj_scalar = np.dot(delta_p, delta_avg) / (norm_delta_avg ** 2)
                delta_specific = delta_p - proj_scalar * delta_avg
                norm_delta_specific = float(np.linalg.norm(delta_specific))
                frac_systematic = (float(cos_sim ** 2)
                                   if norm_delta_p > 1e-12 else np.nan)
            else:
                cos_sim = np.nan
                norm_delta_specific = np.nan
                frac_systematic = np.nan

            all_rows.append({
                "dataset": dataset_name,
                "perturbation": pert,
                "condition": cond_name,
                "n_ko_cells": n_ko,
                "cos_sim": cos_sim,
                "norm_delta_p": norm_delta_p,
                "norm_delta_avg": norm_delta_avg,
                "norm_delta_specific": norm_delta_specific,
                "frac_systematic_var": frac_systematic,
            })

        elapsed = time.time() - t0
        print(f"        Done in {elapsed:.1f}s")

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print(f"  ⚠ No results for {dataset_name}")
        return df

    # ---- Summary statistics ----
    elapsed_total = time.time() - t_start
    print(f"\n  ── Summary for {dataset_name} ──")
    print(f"  Perturbations: {len(df)}")
    print(f"  cos_sim — mean: {df.cos_sim.mean():.3f} ± {df.cos_sim.std():.3f}, "
          f"median: {df.cos_sim.median():.3f}")
    print(f"  cos_sim — min: {df.cos_sim.min():.3f}, "
          f"max: {df.cos_sim.max():.3f}")
    print(f"  frac with cos_sim > 0.5: "
          f"{(df.cos_sim > 0.5).sum()}/{len(df)} "
          f"({100*(df.cos_sim > 0.5).mean():.1f}%)")
    print(f"  frac with cos_sim > 0.7: "
          f"{(df.cos_sim > 0.7).sum()}/{len(df)} "
          f"({100*(df.cos_sim > 0.7).mean():.1f}%)")
    print(f"  Total time: {elapsed_total:.1f}s")

    return df


# ============================================================
# MERGE WITH RELIABILITY
# ============================================================

def merge_with_reliability(systema_df, reliability_dir, dataset_name):
    """
    Merge Systema cos_sim with pre-computed split-half reliability.

    Parameters
    ----------
    systema_df : DataFrame from compute_systema_metrics()
    reliability_dir : Path to directory with reliability_summary_*.csv
    dataset_name : str

    Returns
    -------
    merged : DataFrame with both cos_sim and reliability columns
    """
    rel_path = Path(reliability_dir) / f"reliability_summary_{dataset_name}.csv"

    if not rel_path.exists():
        print(f"  ⚠ Reliability file not found: {rel_path}")
        print(f"    Returning Systema metrics only (no merge)")
        return systema_df

    rel_df = pd.read_csv(rel_path)
    print(f"  Loaded reliability: {len(rel_df)} perturbations from {rel_path.name}")

    # Merge on perturbation (and condition if present)
    merge_cols_left = ["perturbation"]
    merge_cols_right = ["perturbation"]

    if "condition" in rel_df.columns and "condition" in systema_df.columns:
        merge_cols_left.append("condition")
        merge_cols_right.append("condition")

    merged = systema_df.merge(
        rel_df,
        on=merge_cols_left,
        how="inner",
        suffixes=("", "_rel"),
    )

    print(f"  Merged: {len(merged)} perturbations "
          f"(Systema: {len(systema_df)}, Reliability: {len(rel_df)})")

    # ---- 2D classification ----
    # Use sb_from_median_all_genes as the reliability metric
    rel_col = "sb_from_median_all_genes"
    if rel_col not in merged.columns:
        # Fallback to sb_from_mean_all_genes
        rel_col = "sb_from_mean_all_genes"

    if rel_col in merged.columns:
        rho = merged[rel_col]
        cos = merged["cos_sim"]

        # Classification thresholds. Mirror of config.py in the visualization
        # pipeline. cos θ = 1/√2 is the variance-equipartition point
        # (cos²θ = 0.5 means exactly half of the perturbation's variance lies
        # along the systematic axis).
        #
        # cos_sim values here already come from the reliability-first filter
        # implemented in compute_systema_metrics(), so cos_sim is NaN for
        # unreliable perts. "reliable >= threshold" therefore still partitions
        # the triage correctly without needing a second-pass patch.
        COSSIM_HIGH_THRESH = 1.0 / np.sqrt(2)   # ≈ 0.7071

        merged["reliable"] = rho >= RELIABILITY_THRESH
        merged["high_systematic"] = cos >= COSSIM_HIGH_THRESH

        # 2D triage categories
        conditions = [
            (~merged["reliable"]),                                      # Unmeasurable
            (merged["reliable"]) & (merged["high_systematic"]),         # Falsely solved
            (merged["reliable"]) & (~merged["high_systematic"]),        # Genuine signal
        ]
        choices = ["Unreliable", "Shared", "Specific"]
        merged["triage_2d"] = np.select(conditions, choices, default="other")

        # Print 2D triage summary
        print(f"\n  ── 2D Triage Summary ({dataset_name}) ──")
        print(f"  Reliability metric: {rel_col}")
        for cat in ["Unreliable", "Shared", "Specific"]:
            n = (merged["triage_2d"] == cat).sum()
            pct = 100 * n / len(merged)
            print(f"    {cat:20s}: {n:>5} ({pct:5.1f}%)")

        # Also compute ceiling on genuine signal
        ceiling_col = "ceiling_from_median_all_genes"
        if ceiling_col not in merged.columns:
            ceiling_col = "ceiling_from_mean_all_genes"
        if ceiling_col in merged.columns:
            genuine = merged[merged["triage_2d"] == "Specific"]
            if len(genuine) > 0:
                print(f"    Genuine signal: mean ceiling = "
                      f"{genuine[ceiling_col].mean():.3f}, "
                      f"mean cos_sim = {genuine['cos_sim'].mean():.3f}")
    else:
        print(f"  ⚠ Reliability column '{rel_col}' not found in reliability data")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Systema systematic variation + reliability 2D framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset name, 'all', 'genetic', or 'chemical'. "
             f"Options: {', '.join(ALL_DATASETS.keys())}",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reliability_dir", default=DEFAULT_RELIABILITY_DIR,
                        help="Directory with reliability_summary_*.csv files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    ds_arg = args.dataset.lower()
    if ds_arg == "all":
        datasets_to_run = list(ALL_DATASETS.keys())
    elif ds_arg == "genetic":
        datasets_to_run = list(GENETIC_DATASETS.keys())
    elif ds_arg == "chemical":
        datasets_to_run = list(CHEMICAL_DATASETS.keys())
    else:
        if args.dataset not in ALL_DATASETS:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(ALL_DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run = [args.dataset]

    print(f"Systema Systematic Variation + Reliability 2D Framework")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Output: {out_dir}")
    print(f"  Reliability dir: {args.reliability_dir}")

    cross_dataset_summaries = []

    for ds_name in datasets_to_run:
        config = ALL_DATASETS[ds_name]

        if not Path(config["path"]).exists():
            print(f"\n  ✗ SKIPPING {ds_name}: file not found at {config['path']}")
            continue

        try:
            # ---- Compute Systema metrics ----
            systema_df = compute_systema_metrics(ds_name, config, args)

            if len(systema_df) == 0:
                continue

            # Save Systema-only results
            systema_path = out_dir / f"systema_{ds_name}.csv"
            systema_df.to_csv(systema_path, index=False)
            print(f"\n  Saved: {systema_path}")

            # ---- Merge with reliability ----
            merged_df = merge_with_reliability(
                systema_df, args.reliability_dir, ds_name
            )

            merged_path = out_dir / f"systema_reliability_2d_{ds_name}.csv"
            merged_df.to_csv(merged_path, index=False)
            print(f"  Saved: {merged_path}")

            # ---- Cross-dataset summary row ----
            summary_row = {
                "dataset": ds_name,
                "category": config["category"],
                "n_perturbations": len(systema_df),
                "cos_sim_mean": systema_df.cos_sim.mean(),
                "cos_sim_std": systema_df.cos_sim.std(),
                "cos_sim_median": systema_df.cos_sim.median(),
                "frac_cos_gt_0.5": (systema_df.cos_sim > 0.5).mean(),
                "frac_cos_gt_0.7": (systema_df.cos_sim > 0.7).mean(),
            }

            # Add 2D triage counts if available
            if "triage_2d" in merged_df.columns:
                for cat in ["Unreliable", "Shared", "Specific"]:
                    n = (merged_df["triage_2d"] == cat).sum()
                    summary_row[f"n_{cat}"] = n
                    summary_row[f"frac_{cat}"] = n / len(merged_df) if len(merged_df) > 0 else np.nan

            cross_dataset_summaries.append(summary_row)

        except Exception as e:
            print(f"\n  ✗ FAILED on {ds_name}: {e}")
            traceback.print_exc()

    # ---- Cross-dataset summary ----
    if len(cross_dataset_summaries) > 1:
        summary_df = pd.DataFrame(cross_dataset_summaries)
        summary_path = out_dir / "systema_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*70}")
        print(f"  Cross-Dataset Summary")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n  Saved: {summary_path}")

        # Grand totals
        if "n_Unreliable" in summary_df.columns:
            total = summary_df["n_perturbations"].sum()
            total_unmeas = summary_df.get("n_Unreliable", pd.Series([0])).sum()
            total_falsely = summary_df.get("n_Shared", pd.Series([0])).sum()
            total_genuine = summary_df.get("n_Specific", pd.Series([0])).sum()
            total_2d = total_unmeas + total_falsely + total_genuine

            print(f"\n  ── Grand Totals (across all datasets) ──")
            print(f"  Total perturbations (Systema):  {int(total)}")
            print(f"  Total perturbations (merged):   {int(total_2d)}")
            print(f"    Unmeasurable:    {int(total_unmeas):>5} "
                  f"({100*total_unmeas/total_2d:.1f}%)" if total_2d > 0 else "")
            print(f"    Falsely solved:  {int(total_falsely):>5} "
                  f"({100*total_falsely/total_2d:.1f}%)" if total_2d > 0 else "")
            print(f"    Genuine signal:  {int(total_genuine):>5} "
                  f"({100*total_genuine/total_2d:.1f}%)" if total_2d > 0 else "")

            effective_benchmark = total_genuine
            print(f"\n  ★ Effective benchmark size: {int(effective_benchmark)} perturbations "
                  f"({100*effective_benchmark/total_2d:.1f}% of evaluated)")

    print(f"\n{'='*70}")
    print(f"All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
