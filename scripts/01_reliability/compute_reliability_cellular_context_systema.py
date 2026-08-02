#!/usr/bin/env python3
"""
compute_reliability_cellular_context_systema.py
==========================================
Compute Systema's systematic variation metric (Viñas Torné et al., Nature
Biotechnology 2025) for each (context, perturbation) condition in the Wei et al.
(scPerturBench) CELLULAR CONTEXT datasets, and merge with pre-computed
split-half reliability to produce a 2D evaluation framework.

ADAPTATION FROM GENETIC PERTURBATION VERSION
----------------------------------------------
In genetic perturbation:
  - One cell population, many gene knockouts
  - δ_avg = mean(all KO cells) - mean(control cells)
  - cos_sim measures whether a KO looks like the "average KO"

In cellular context:
  - Multiple contexts (cell types / patients / species)
  - Each context has its own control cells
  - δ_avg is computed PER CONTEXT:
      δ_avg_ctx = mean(all perturbed cells in context) - mean(ctrl cells in context)
  - cos_sim measures whether a specific drug/stimulus in this context
    looks like the average perturbation effect in this context

Computing δ_avg across contexts would be biologically meaningless (mixing
different cell types), so the per-context approach is essential.

SYSTEMA'S SYSTEMATIC VARIATION METRIC (per context)
-----------------------------------------------------
For each context C and perturbation p:
  1. control_mean_C   = mean expression across control cells in context C
  2. centroid_Cp      = mean expression across stimulated cells (C, p)
  3. perturbed_mean_C = mean expression across ALL non-control cells in C
                        (cell-level mean; perturbations with more cells
                         contribute more)
  4. δ_p   = centroid_Cp    - control_mean_C     (perturbation-specific shift)
  5. δ_avg = perturbed_mean_C - control_mean_C   (avg perturbation effect in C)
  6. cos_sim_p = cosine_similarity(δ_p, δ_avg)

2D FRAMEWORK (same interpretation as genetic version)
------------------------------------------------------
  - Low ρ                          → Unmeasurable (noisy ground truth)
  - High ρ, high cos_sim           → Falsely solved (systematic only)
  - High ρ, low cos_sim            → Genuine signal (true frontier)

USAGE
-----
# Single dataset:
python compute_reliability_cellular_context_systema.py --dataset kangCrossCell

# All 12 cellular context datasets:
python compute_reliability_cellular_context_systema.py --dataset all

# With custom paths:
python compute_reliability_cellular_context_systema.py --dataset all \
    --data_dir /path/to/cellular_h5ad \
    --reliability_dir /path/to/reliability_output \
    --output_dir /path/to/output

OUTPUTS
-------
  systema_cc_{dataset}.csv                  Per-condition: cos_sim, norms, etc.
  systema_cc_reliability_2d_{dataset}.csv   Merged: cos_sim + reliability
  systema_cc_summary.csv                    Cross-dataset summary (if multiple)

DEPENDENCIES
------------
  numpy, pandas, scipy (sparse), anndata
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
# DATASET CONFIGURATION — same as compute_reliability_cellular_context.py
# ============================================================
# Updated with corrected column names from discovery/run validation.

CELLULAR_DATASETS = {
    # ---- Kang et al. (IFN-beta stimulation of PBMCs) ----
    "kangCrossCell": {
        "context_col": "cell_type",
        "condition_col": "condition",
        "ctrl_label": "control",
    },
    "kangCrossPatient": {
        "context_col": "sample_id",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Haber et al. (intestinal epithelium, infections) ----
    "Haber": {
        "context_col": "cell_type",
        "condition_col": "condition",
        "ctrl_label": "Control",
    },

    # ---- Afriat et al. (liver infection) ----
    # Use condition1 (Pericentral, Periportal — 2 zones) to match Wei et al.'s
    # benchmark aggregation. Finer cell-state subdivisions (cluster_names) were
    # used in earlier iterations but yielded triage at higher granularity than
    # Wei's evaluation, blocking direct re-evaluation merge.
    "Afriat": {
        "context_col": "condition1",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- McFarland et al. (cancer cell lines + drugs) ----
    "McFarland": {
        "context_col": "cell_line",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Parekh et al. (TF overexpression) ----
    "Parekh": {
        "context_col": "cell_type",
        "condition_col": "perturbation",
        "ctrl_label": "CTRL",
    },

    # ---- TCDD (dioxin exposure, liver cell types) ----
    "TCDD": {
        "context_col": "celltype",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Cross-patient (cancer patients + drugs) ----
    "crossPatient": {
        "context_col": "patient",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Cross-species (LPS stimulation) ----
    # Use pre-pooled condition1 (mouse, pig, rabbit, rat — 4 species, pooled
    # across individuals) and condition2 (LPS vs control, pooled across LPS
    # time-points lps2/lps4/lps6) to match Wei et al.'s benchmark aggregation.
    "crossSpecies": {
        "context_col": "condition1",
        "condition_col": "condition2",
        "ctrl_label": "control",
    },

    # ---- Kaggle/NeurIPS Cross-Cell (drugs x cell types) ----
    "KaggleCrossCell": {
        "context_col": "cell_type",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Kaggle/NeurIPS Cross-Patient (drugs x patients) ----
    "KaggleCrossPatient": {
        "context_col": "donor_id",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Sciplex3 (drugs x cell lines) ----
    # Use condition2 (drug name without dose) to match Wei et al.'s benchmark
    # aggregation. Per-dose reliability would discard Wei's pooling and block
    # re-evaluation merge; pooled reliability gives a single label per
    # (cell_line, drug) consistent with Wei's evaluation unit.
    "sciplex3": {
        "context_col": "cell_line",
        "condition_col": "condition2",
        "ctrl_label": "control",
    },
}

# Fallback ctrl labels (same as reliability script)
CTRL_LABEL_CANDIDATES = [
    "control", "Control", "ctrl", "CTRL",
    "DMSO", "dmso", "Vehicle", "vehicle",
    "Uninfected", "uninfected", "untreated", "Untreated",
    "normal", "Normal", "baseline", "Baseline",
]

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
DEFAULT_DATA_DIR = "" + _ROOT + "/data/Wei_et_al_data/cellular_context_preprocessed_h5ad"
DEFAULT_OUTPUT_DIR = "" + _ROOT + "/output/reliability_fig1/reliability_cellular_systema"
DEFAULT_RELIABILITY_DIR = "" + _ROOT + "/output/reliability_fig1/reliability_cellular"


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity between two 1D vectors."""
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
    dots = deltas @ avg_delta
    norms_delta = np.linalg.norm(deltas, axis=1)
    norm_avg = np.linalg.norm(avg_delta)
    denom = norms_delta * norm_avg
    denom[denom < 1e-12] = 1.0
    return dots / denom


# ============================================================
# CORE: COMPUTE SYSTEMA METRICS FOR ONE CELLULAR CONTEXT DATASET
# ============================================================

RELIABILITY_THRESH = 0.5  # ρ cutoff — cos_sim is NaN for perts with ρ below this.


def _load_reliable_perts_cellular(reliability_dir, dataset_name,
                                  rel_col_fallbacks=(
                                      "sb_from_median",
                                      "sb_from_mean",
                                      "sb_from_median_all_genes",
                                      "sb_from_mean_all_genes")):
    """Load the per-(context,perturbation) reliability summary and return
    the set of 'condition' keys (format 'ctx|pert') with ρ ≥ 0.5.
    The cellular reliability CSVs are named `reliability_cc_{dataset}.csv`.
    """
    for fname in (f"reliability_cc_{dataset_name}.csv",
                  f"reliability_summary_{dataset_name}.csv"):
        rel_path = Path(reliability_dir) / fname
        if rel_path.exists():
            break
    else:
        return None, None

    rel_df = pd.read_csv(rel_path)
    rel_col = next((c for c in rel_col_fallbacks if c in rel_df.columns), None)
    if rel_col is None:
        return None, rel_df
    reliable = rel_df[rel_df[rel_col] >= RELIABILITY_THRESH]
    if "condition" in reliable.columns:
        return set(reliable["condition"].astype(str)), rel_df
    return set(reliable["perturbation"].astype(str)), rel_df


def compute_systema_metrics(dataset_name, data_dir, args):
    """
    Compute Systema's systematic variation metric for all (context, perturbation)
    conditions in a cellular context dataset.

    Key properties of this implementation:
      • δ_avg is computed PER CONTEXT (not global)
      • δ_avg uses cells from RELIABLE perts in that context only
      • cos_sim is computed only for reliable perts; unreliable perts get NaN.

    Returns
    -------
    df : DataFrame with columns:
        dataset, condition, context, perturbation, n_stim_cells,
        cos_sim, norm_delta_p, norm_delta_avg, norm_delta_specific,
        frac_systematic_var
    """
    t_start = time.time()

    if dataset_name not in CELLULAR_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {', '.join(CELLULAR_DATASETS.keys())}")

    cfg = CELLULAR_DATASETS[dataset_name]
    context_col = cfg["context_col"]
    condition_col = cfg["condition_col"]
    ctrl_label = cfg["ctrl_label"]

    h5ad_path = Path(data_dir) / f"{dataset_name}.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"File not found: {h5ad_path}")

    print(f"\n{'='*70}")
    print(f"  Systema analysis: {dataset_name} (Cellular Context)")
    print(f"  Path: {h5ad_path}")
    print(f"  context_col='{context_col}' | condition_col='{condition_col}' "
          f"| ctrl='{ctrl_label}'")
    print(f"{'='*70}")

    # ---- Load reliability (used to filter which perts define δ_avg) ----
    print(f"  [0/3] Loading reliability data …")
    reliable_conds, _rel_df = _load_reliable_perts_cellular(
        args.reliability_dir, dataset_name)
    if reliable_conds is None:
        raise FileNotFoundError(
            f"Reliability data missing for {dataset_name}. "
            f"The reliability-filtered systema pipeline requires it. "
            f"Looked in: {args.reliability_dir}")
    print(f"        Reliable (context, pert) pairs: {len(reliable_conds)}")

    # ---- Load ----
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

    # ---- Validate columns ----
    if context_col not in adata.obs.columns:
        raise ValueError(f"context_col '{context_col}' not in obs columns: "
                         f"{list(adata.obs.columns)}")
    if condition_col not in adata.obs.columns:
        raise ValueError(f"condition_col '{condition_col}' not in obs columns: "
                         f"{list(adata.obs.columns)}")

    contexts = adata.obs[context_col].astype(str).str.strip().values
    conditions = adata.obs[condition_col].astype(str).str.strip().values

    unique_contexts = sorted(np.unique(contexts))
    unique_conditions = sorted(np.unique(conditions))

    # Validate control label (with fuzzy fallback)
    if ctrl_label not in unique_conditions:
        ctrl_found = None
        for cand in CTRL_LABEL_CANDIDATES:
            if cand in unique_conditions:
                ctrl_found = cand
                break
        if ctrl_found:
            print(f"        Note: ctrl_label '{ctrl_label}' not found, "
                  f"using '{ctrl_found}'")
            ctrl_label = ctrl_found
        else:
            raise ValueError(f"ctrl_label '{ctrl_label}' not in condition values: "
                             f"{unique_conditions}")

    stim_conditions = [c for c in unique_conditions if c != ctrl_label]
    print(f"        Contexts ({len(unique_contexts)}): {unique_contexts}")
    print(f"        Perturbations ({len(stim_conditions)}): "
          f"ctrl='{ctrl_label}', stim ({len(stim_conditions)})")

    # ---- Compute per context ----
    print(f"  [3/3] Computing cosine similarities per context...")
    all_rows = []
    total_conditions = 0
    total_skipped = 0

    for ctx in unique_contexts:
        ctx_mask = contexts == ctx

        # Control cells for this context
        ctrl_mask = ctx_mask & (conditions == ctrl_label)
        ctrl_idx = np.where(ctrl_mask)[0]
        n_ctrl = len(ctrl_idx)

        if n_ctrl < 4:
            print(f"        Context '{ctx}': only {n_ctrl} controls, skipping")
            continue

        # All perturbed (non-control) cells in this context
        ko_mask = ctx_mask & (conditions != ctrl_label)
        ko_idx_all = np.where(ko_mask)[0]
        ko_labels = conditions[ko_idx_all]
        n_ko_total = len(ko_idx_all)

        unique_perts_in_ctx = sorted(np.unique(ko_labels))
        n_perts = len(unique_perts_in_ctx)

        if n_perts < 2:
            # Need at least 2 perturbations to define a meaningful δ_avg
            # With only 1, cos_sim = 1.0 trivially
            print(f"        Context '{ctx}': only {n_perts} perturbation(s), "
                  f"cos_sim=1.0 trivially")
            # Still compute but flag it
            pass

        # ---- Step 1: control_mean for this context ----
        if is_sparse:
            control_mean = np.asarray(
                X[ctrl_idx].mean(axis=0), dtype=np.float64
            ).ravel()
        else:
            control_mean = X[ctrl_idx].astype(np.float64).mean(axis=0)

        if n_ko_total == 0:
            print(f"        Context '{ctx}': no perturbed cells, skipping")
            continue

        # ---- Step 2: identify reliable perts in this context ----
        def _is_reliable_cell(ctx_str, pert_str):
            cond_name = f"{ctx_str}|{pert_str}"
            return cond_name in reliable_conds

        reliable_pert_set = {p for p in unique_perts_in_ctx
                             if _is_reliable_cell(ctx, p)}
        reliable_cell_mask_local = np.array(
            [lbl in reliable_pert_set for lbl in ko_labels], dtype=bool)
        reliable_cell_idx = ko_idx_all[reliable_cell_mask_local]

        # ---- Step 3: δ_avg from cells of reliable perts only ----
        if len(reliable_cell_idx) == 0:
            delta_avg = np.zeros_like(control_mean)
            norm_delta_avg = 0.0
            print(f"        Context '{ctx}': no reliable perts; "
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
            print(f"        Context '{ctx}': {n_ctrl:,} ctrl, "
                  f"{len(reliable_cell_idx):,} reliable-pert cells "
                  f"({len(reliable_pert_set)}/{n_perts} perts), "
                  f"||δ_avg||={norm_delta_avg:.4f}")

        # ---- Step 4: For each perturbation, compute δ_p; cos_sim only if reliable ----
        for pert in unique_perts_in_ctx:
            pert_mask_local = ko_labels == pert
            pert_idx = ko_idx_all[pert_mask_local]
            n_stim = len(pert_idx)

            if n_stim < 4:
                total_skipped += 1
                continue

            total_conditions += 1
            is_rel = pert in reliable_pert_set

            if is_sparse:
                centroid_p = np.asarray(
                    X[pert_idx].mean(axis=0), dtype=np.float64
                ).ravel()
            else:
                centroid_p = X[pert_idx].astype(np.float64).mean(axis=0)

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

            cond_name = f"{ctx}|{pert}"
            all_rows.append({
                "dataset": dataset_name,
                "condition": cond_name,
                "context": ctx,
                "perturbation": pert,
                "n_stim_cells": n_stim,
                "n_perts_in_context": n_perts,
                "cos_sim": cos_sim,
                "norm_delta_p": norm_delta_p,
                "norm_delta_avg": norm_delta_avg,
                "norm_delta_specific": norm_delta_specific,
                "frac_systematic_var": frac_systematic,
            })

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print(f"  ⚠ No results for {dataset_name}")
        return df

    # ---- Summary statistics ----
    elapsed_total = time.time() - t_start
    print(f"\n  ── Summary for {dataset_name} ──")
    print(f"  Conditions computed: {total_conditions}")
    print(f"  Skipped (few cells): {total_skipped}")
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

    # Per-context breakdown
    print(f"\n  Per-context cos_sim (median):")
    for ctx, gdf in df.groupby("context"):
        print(f"    {ctx:20s}: {gdf.cos_sim.median():.3f} "
              f"(n={len(gdf)}, n_perts_in_ctx={gdf.n_perts_in_context.iloc[0]})")

    print(f"  Total time: {elapsed_total:.1f}s")

    del adata
    return df


# ============================================================
# MERGE WITH RELIABILITY
# ============================================================

def merge_with_reliability(systema_df, reliability_dir, dataset_name):
    """
    Merge Systema cos_sim with pre-computed split-half reliability
    from the cellular context reliability script.

    Merges on the 'condition' column (format: "context|perturbation").
    """
    rel_path = Path(reliability_dir) / f"reliability_cc_summary_{dataset_name}.csv"

    if not rel_path.exists():
        print(f"  ⚠ Reliability file not found: {rel_path}")
        print(f"    Returning Systema metrics only (no merge)")
        return systema_df

    rel_df = pd.read_csv(rel_path)
    print(f"  Loaded reliability: {len(rel_df)} conditions from {rel_path.name}")

    # Merge on condition (= "context|perturbation")
    merged = systema_df.merge(
        rel_df,
        on="condition",
        how="inner",
        suffixes=("", "_rel"),
    )

    print(f"  Merged: {len(merged)} conditions "
          f"(Systema: {len(systema_df)}, Reliability: {len(rel_df)})")

    if len(merged) < len(systema_df):
        missing = set(systema_df["condition"]) - set(merged["condition"])
        if len(missing) <= 10:
            print(f"  Unmatched conditions: {sorted(missing)}")
        else:
            print(f"  Unmatched conditions: {len(missing)} "
                  f"(first 5: {sorted(missing)[:5]})")

    # ---- 2D classification ----
    # Use sb_from_median_all_genes as the reliability metric
    rel_col = "sb_from_median_all_genes"
    if rel_col not in merged.columns:
        rel_col = "sb_from_mean_all_genes"

    if rel_col in merged.columns:
        rho = merged[rel_col]
        cos = merged["cos_sim"]

        # Classification thresholds (mirror of config.py in the visualization
        # pipeline). cos_sim values already come from the reliability-first
        # filter implemented in compute_systema_metrics() above, so no
        # post-hoc patch is needed.
        COSSIM_HIGH_THRESH = 1.0 / np.sqrt(2)   # ≈ 0.7071

        merged["reliable"] = rho >= RELIABILITY_THRESH
        merged["high_systematic"] = cos >= COSSIM_HIGH_THRESH

        # 2D triage categories
        conditions = [
            (~merged["reliable"]),
            (merged["reliable"]) & (merged["high_systematic"]),
            (merged["reliable"]) & (~merged["high_systematic"]),
        ]
        choices = ["Unreliable", "Shared", "Specific"]
        merged["triage_2d"] = np.select(conditions, choices, default="other")

        # Print 2D triage summary
        print(f"\n  ── 2D Triage Summary ({dataset_name}) ──")
        print(f"  Reliability metric: {rel_col}")
        for cat in ["Unreliable", "Shared", "Specific"]:
            n = (merged["triage_2d"] == cat).sum()
            pct = 100 * n / len(merged) if len(merged) > 0 else 0
            print(f"    {cat:20s}: {n:>5} ({pct:5.1f}%)")

        # Ceiling on genuine signal
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
        description="Compute Systema systematic variation + reliability 2D "
                    "framework for cellular context datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset name or 'all'. "
             f"Options: {', '.join(CELLULAR_DATASETS.keys())}",
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to directory with cellular context h5ad files")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reliability_dir", default=DEFAULT_RELIABILITY_DIR,
                        help="Directory with reliability_cc_summary_*.csv files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.dataset.lower() == "all":
        datasets_to_run = list(CELLULAR_DATASETS.keys())
    else:
        if args.dataset not in CELLULAR_DATASETS:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(CELLULAR_DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run = [args.dataset]

    print(f"Systema Systematic Variation + Reliability 2D (Cellular Context)")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Reliability dir: {args.reliability_dir}")

    cross_dataset_summaries = []

    for ds_name in datasets_to_run:
        h5ad_path = data_dir / f"{ds_name}.h5ad"

        if not h5ad_path.exists():
            print(f"\n  ✗ SKIPPING {ds_name}: file not found at {h5ad_path}")
            continue

        try:
            # ---- Compute Systema metrics ----
            systema_df = compute_systema_metrics(ds_name, data_dir, args)

            if len(systema_df) == 0:
                continue

            # Save Systema-only results
            systema_path = out_dir / f"systema_cc_{ds_name}.csv"
            systema_df.to_csv(systema_path, index=False)
            print(f"\n  Saved: {systema_path}")

            # ---- Merge with reliability ----
            merged_df = merge_with_reliability(
                systema_df, args.reliability_dir, ds_name
            )

            merged_path = out_dir / f"systema_cc_reliability_2d_{ds_name}.csv"
            merged_df.to_csv(merged_path, index=False)
            print(f"  Saved: {merged_path}")

            # ---- Cross-dataset summary row ----
            summary_row = {
                "dataset": ds_name,
                "n_conditions": len(systema_df),
                "n_contexts": systema_df["context"].nunique(),
                "n_perturbations": systema_df["perturbation"].nunique(),
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
                    summary_row[f"frac_{cat}"] = (
                        n / len(merged_df) if len(merged_df) > 0 else np.nan
                    )

            cross_dataset_summaries.append(summary_row)

        except Exception as e:
            print(f"\n  ✗ FAILED on {ds_name}: {e}")
            traceback.print_exc()

    # ---- Cross-dataset summary ----
    if len(cross_dataset_summaries) > 1:
        summary_df = pd.DataFrame(cross_dataset_summaries)
        summary_path = out_dir / "systema_cc_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*70}")
        print(f"  Cross-Dataset Summary (Cellular Context)")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n  Saved: {summary_path}")

        # Grand totals
        if "n_Unreliable" in summary_df.columns:
            total = summary_df["n_conditions"].sum()
            total_unmeas = summary_df.get("n_Unreliable", pd.Series([0])).sum()
            total_falsely = summary_df.get("n_Shared", pd.Series([0])).sum()
            total_genuine = summary_df.get("n_Specific", pd.Series([0])).sum()
            total_2d = total_unmeas + total_falsely + total_genuine

            if total_2d > 0:
                print(f"\n  ── Grand Totals (across all datasets) ──")
                print(f"  Total conditions (Systema):  {int(total)}")
                print(f"  Total conditions (merged):   {int(total_2d)}")
                print(f"    Unmeasurable:    {int(total_unmeas):>5} "
                      f"({100*total_unmeas/total_2d:.1f}%)")
                print(f"    Falsely solved:  {int(total_falsely):>5} "
                      f"({100*total_falsely/total_2d:.1f}%)")
                print(f"    Genuine signal:  {int(total_genuine):>5} "
                      f"({100*total_genuine/total_2d:.1f}%)")

                print(f"\n  ★ Effective benchmark size: {int(total_genuine)} conditions "
                      f"({100*total_genuine/total_2d:.1f}% of evaluated)")

    print(f"\n{'='*70}")
    print(f"All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
