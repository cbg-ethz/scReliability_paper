#!/usr/bin/env python3
"""
compute_reliability_cellular_context.py
========================================
Compute split-half Δ-reliability for ALL 12 Wei et al. (scPerturBench)
CELLULAR CONTEXT generalization datasets.

Adapted from compute_reliability_genetic_context.py (genetic perturbation version v2).

KEY DIFFERENCE FROM GENETIC PERTURBATION:
  In genetic perturbation:
    - Each "condition" = one gene knockout in one cell line
    - KO cells vs control cells from the same cell line
    - Δ = mean(KO cells) - mean(control cells)

  In cellular context (OOD):
    - Each "condition" = (context × perturbation) e.g. (B cells, IFN-beta)
    - context = cell type / patient / species (the held-out entity)
    - perturbation = treatment applied (stimulation, drug, infection)
    - Stimulated cells of context X under perturbation P vs control cells of context X
    - Δ = mean(stimulated cells in context X) - mean(control cells in context X)

  The reliability question is identical:
    "How reproducible is the measured perturbation response Δ for this condition?"

Two analysis modes (same as genetic v2):
  1. PRIMARY — "max" split: uses ALL available cells per condition,
     split into two equal halves. This estimates reliability of the actual
     ground truth used in benchmark scoring. The Spearman-Brown corrected
     value directly answers: "how reliable is this condition's ground
     truth?" → used for ceiling efficiency computation.

  2. SECONDARY — fixed n_half curve: computes reliability at fixed cell
     counts [8, 16, 32, 64, 128, 256, 512] to show how reliability scales
     with sample size. Produces the "how many cells do you need?" figure.

Two gene modes for each (same as genetic v2):
  - all_genes:   Pearson correlation across ALL HVGs
  - top1k_expr:  Pearson correlation across top-1000 most highly expressed
                 genes in control cells (Ahlmann-Eltze et al. 2025 approach).
                 Non-circular: gene selection is based on control expression
                 level, computed once per context, independent of perturbation
                 effects and split-half sampling.

USAGE
-----
# Step 1: ALWAYS run discovery first to validate column names!
python compute_reliability_cellular_context.py --dataset discover --data_dir /path/to/figshare

# Step 2: Single dataset:
python compute_reliability_cellular_context.py --dataset kangCrossCell --data_dir /path/to/figshare

# Step 3: All datasets:
python compute_reliability_cellular_context.py --dataset all --data_dir /path/to/figshare

# SLURM batch (recommended):
for ds in kangCrossCell kangCrossPatient Haber Afriat McFarland Parekh \\
          TCDD crossPatient crossSpecies KaggleCrossCell KaggleCrossPatient sciplex3; do
    sbatch -J "rel_cc_${ds}" -c 8 --mem=64G -t 6:00:00 \\
        --wrap "python compute_reliability_cellular_context.py --dataset ${ds} \\
                --data_dir /path/to/figshare --workers 8 --n_repeats 100"
done

OUTPUTS (per dataset)
-------
  reliability_cc_{dataset}.csv            Long-format: per (condition, n_mode, gene_mode)
  reliability_cc_summary_{dataset}.csv    Wide: per condition, max reliability & ceiling
  reliability_cc_{dataset}_metadata.json  Run metadata, parameters, timing

METHODOLOGY
-----------
  For each (context, perturbation) condition with N_stim stimulated cells
  and N_ctrl control cells in the same context:

  [PRIMARY — max mode]
    1. n = min(floor(N_stim/2), floor(N_ctrl/2))
    2. Repeat R times:
       - Permute stim cells → half_A (first n), half_B (next n)
       - Permute ctrl cells → half_C (first n), half_D (next n)
       - Δ_1 = mean(half_A) - mean(half_C)
       - Δ_2 = mean(half_B) - mean(half_D)
       - r = Pearson(Δ_1, Δ_2)
    3. Spearman-Brown: r_SB = 2 * mean(r) / (1 + mean(r))
       This estimates reliability of the FULL N_stim + N_ctrl measurement
    4. Ceiling = sqrt(max(0, r_SB))

  [SECONDARY — fixed n curve]
    Same as above but at each n in [8, 16, 32, 64, 128, 256, 512],
    skipping infeasible values. Shows reliability scaling.

  Gene modes:
    - all_genes: correlation across all HVGs (~5000)
    - topK_expr: correlation across top-K genes by mean expression in
      control cells (Ahlmann-Eltze et al. 2025). Gene selection is computed
      ONCE per context from control cells only → fully independent of
      perturbation effects and split-half sampling.
"""

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
import os
import sys
import json
import time
import argparse
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# DATASET CONFIGURATION — Wei et al. Cellular Context
# ============================================================
# Each dataset needs:
#   context_col:  obs column identifying the cellular context
#   condition_col: obs column identifying control vs stimulated
#   ctrl_label:   value in condition_col that marks control cells
#
# Column names verified against each dataset's obs; `--dataset discover`
# re-validates them against a downloaded copy.

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
    # benchmark aggregation level (the unit on which they report performance).
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
    # Use pre-pooled condition1 (4 species pooled across individuals) and
    # condition2 (LPS vs control, pooled across LPS time-points) to match Wei.
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
    # Use condition2 (drug name without dose) to match Wei's evaluation unit;
    # per-dose reliability would block re-evaluation merge with their results.
    "sciplex3": {
        "context_col": "cell_line",
        "condition_col": "condition2",
        "ctrl_label": "control",
    },
}


# Common alternative names to try during auto-discovery
CONTEXT_COL_CANDIDATES = [
    "cell_type", "celltype", "cell_line", "cellLine", "cell_line_id",
    "patient", "donor", "species", "sample", "batch",
    "outSample", "outsample",
]
CONDITION_COL_CANDIDATES = [
    "condition", "perturbation", "treatment", "stim", "stimulation",
    "drug", "compound", "perturb",
]
CTRL_LABEL_CANDIDATES = [
    "control", "Control", "ctrl", "CTRL",
    "DMSO", "dmso", "Vehicle", "vehicle",
    "Uninfected", "uninfected", "untreated", "Untreated",
    "normal", "Normal", "baseline", "Baseline",
]

# Fixed n_half values for the scaling curve.
# Original 7-point grid (powers of 2). n_total = 2 * n_half spans 16-1024.
N_HALF_FIXED = [8, 16, 32, 64, 128, 256, 512]


# ============================================================
# VECTORIZED CORRELATION
# ============================================================

def batch_pearsonr(X, Y):
    """
    Row-wise Pearson correlation between corresponding rows of X and Y.

    Parameters
    ----------
    X, Y : ndarray, shape (R, G)

    Returns
    -------
    r : ndarray, shape (R,)
    """
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    num = (X * Y).sum(axis=1)
    denom = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    denom[denom == 0] = 1.0
    return np.clip(num / denom, -1.0, 1.0)


# ============================================================
# CORE: SINGLE CONDITION RELIABILITY
# ============================================================

def compute_one_condition(args):
    """
    Compute split-half reliability for one (context, perturbation) condition.

    Logic is IDENTICAL to compute_one_perturbation in the genetic v2 script:
      - "KO cells"     --> stimulated cells in this (context, perturbation)
      - "control cells" --> control cells in the SAME context
      - Δ = mean(stim_half) - mean(ctrl_half)

    Parameters (packed tuple)
    ----------
    cond_name, X_stim_dense, n_stim, n_half_max, n_half_fixed_feasible,
    n_repeats, expr_idx, seed

    Returns
    -------
    rows : list of dicts
    """
    (cond_name, X_stim_dense, n_stim, n_half_max, n_half_fixed_feasible,
     n_repeats, expr_idx, seed) = args

    ctrl_dense = _G_CTRL_DENSE      # global: precomputed dense control matrix

    n_ctrl = ctrl_dense.shape[0]
    n_genes = ctrl_dense.shape[1]

    rng = np.random.default_rng(seed)
    rows = []

    try:
        X_stim = X_stim_dense   # already dense float32, extracted in parent
        X_ctrl = ctrl_dense

        # expr_idx is precomputed per-context (top-k genes by mean ctrl expression)
        # — no per-condition gene selection needed

        # ---- Collect all n values to compute ----
        # Primary: max split
        # Secondary: fixed values that are feasible
        all_n_configs = []

        # Primary (max) mode
        all_n_configs.append((n_half_max, "max"))

        # Secondary (fixed) modes
        for n_fixed in n_half_fixed_feasible:
            if n_fixed != n_half_max:  # avoid duplicate if max happens to equal a fixed value
                all_n_configs.append((n_fixed, "fixed"))
            else:
                # Still compute it, but tag as both
                all_n_configs.append((n_fixed, "fixed"))

        # ---- Compute reliability for each n ----
        for n, n_mode in all_n_configs:
            if n < 4:  # minimum for meaningful correlation
                continue

            # Pre-allocate
            deltas_1 = np.empty((n_repeats, n_genes), dtype=np.float32)
            deltas_2 = np.empty((n_repeats, n_genes), dtype=np.float32)

            for r in range(n_repeats):
                perm_stim = rng.permutation(n_stim)
                half_A = perm_stim[:n]
                half_B = perm_stim[n:2 * n]

                perm_ctrl = rng.permutation(n_ctrl)
                half_C = perm_ctrl[:n]
                half_D = perm_ctrl[n:2 * n]

                deltas_1[r] = X_stim[half_A].mean(axis=0) - X_ctrl[half_C].mean(axis=0)
                deltas_2[r] = X_stim[half_B].mean(axis=0) - X_ctrl[half_D].mean(axis=0)

            # ---- Correlations ----
            r_all = batch_pearsonr(deltas_1, deltas_2)
            r_expr = batch_pearsonr(deltas_1[:, expr_idx], deltas_2[:, expr_idx])

            # ---- Store results for both gene modes ----
            expr_k_label = f"top{len(expr_idx)}_expr"
            for gene_mode, r_values in [("all_genes", r_all),
                                        (expr_k_label, r_expr)]:
                median_r = float(np.median(r_values))
                mean_r = float(np.mean(r_values))

                # Both Spearman-Brown variants are stored: SB(mean_r), the classical
                # estimator of E[r], and SB(median_r), robust to outlier splits.
                # The reported reliability is SB(median_r). Per-repeat SB is also
                # kept to characterise the full distribution.
                def _sb(r):
                    # Spearman-Brown prophecy with rho clipped to [0, 1].
                    # Negative values indicate no reliable signal (field convention:
                    # reliability = signal_var / total_var ∈ [0, 1] by definition).
                    d = 1.0 + r
                    rho = np.where(np.abs(d) > 1e-10, 2.0 * r / d, 0.0)
                    return np.maximum(rho, 0.0)

                sb_from_mean = float(_sb(mean_r))
                sb_from_median = float(_sb(median_r))

                # Per-repeat SB for distribution diagnostics
                sb_per_repeat = _sb(r_values)
                sb_mean_of_per_repeat = float(np.mean(sb_per_repeat))
                sb_median_of_per_repeat = float(np.median(sb_per_repeat))

                # Skewness of raw r distribution (diagnostic)
                r_std = float(np.std(r_values))
                if r_std > 1e-10:
                    skewness = float(np.mean(((r_values - mean_r) / r_std) ** 3))
                else:
                    skewness = 0.0

                rows.append({
                    "condition": cond_name,
                    "n_stim_cells": n_stim,
                    "n_half": n,
                    "n_mode": n_mode,       # "max" or "fixed"
                    "gene_mode": gene_mode,  # "all_genes" or "topK_expr"
                    "n_repeats": int(n_repeats),
                    # Raw split-half r distribution
                    "mean_r": mean_r,
                    "std_r": r_std,
                    "median_r": median_r,
                    "q25_r": float(np.percentile(r_values, 25)),
                    "q75_r": float(np.percentile(r_values, 75)),
                    "min_r": float(np.min(r_values)),
                    "max_r": float(np.max(r_values)),
                    "skewness_r": skewness,
                    # Spearman-Brown — multiple approaches stored
                    "sb_from_mean": sb_from_mean,         # SB(mean(r_i))
                    "sb_from_median": sb_from_median,     # SB(median(r_i))
                    "sb_mean_per_repeat": sb_mean_of_per_repeat,   # mean(SB(r_i))
                    "sb_median_per_repeat": sb_median_of_per_repeat, # median(SB(r_i))
                    # Ceilings from each approach
                    "ceiling_from_mean": float(np.sqrt(max(0.0, sb_from_mean))),
                    "ceiling_from_median": float(np.sqrt(max(0.0, sb_from_median))),
                })

    except Exception as e:
        rows.append({
            "condition": cond_name,
            "n_stim_cells": n_stim,
            "n_half": -1,
            "n_mode": "ERROR",
            "gene_mode": "ERROR",
            "n_repeats": 0,
            "mean_r": np.nan, "std_r": np.nan, "median_r": np.nan,
            "q25_r": np.nan, "q75_r": np.nan,
            "min_r": np.nan, "max_r": np.nan,
            "skewness_r": np.nan,
            "sb_from_mean": np.nan, "sb_from_median": np.nan,
            "sb_mean_per_repeat": np.nan, "sb_median_per_repeat": np.nan,
            "ceiling_from_mean": np.nan, "ceiling_from_median": np.nan,
            "error": str(e),
        })

    return rows


# ============================================================
# GLOBALS FOR POOL WORKERS
# ============================================================
# Only ctrl_dense is shared via Pool initializer.
# Stim submatrices are extracted in the parent and passed per-task.

_G_CTRL_DENSE = None


def _init_worker(ctrl_dense):
    """Pool initializer: set ctrl_dense in each worker process."""
    global _G_CTRL_DENSE
    _G_CTRL_DENSE = ctrl_dense


# ============================================================
# DISCOVER MODE
# ============================================================

def discover_datasets(data_dir):
    """
    Scan all h5ad files and print obs columns + unique values.
    Use this FIRST to validate/update the CELLULAR_DATASETS config.
    """
    data_dir = Path(data_dir)
    h5ad_files = sorted(data_dir.glob("*.h5ad"))

    if not h5ad_files:
        print(f"  ERROR: No .h5ad files found in {data_dir}")
        print(f"  Download from: https://doi.org/10.6084/m9.figshare.28143422")
        return

    print(f"\n{'='*70}")
    print(f"  DISCOVERY MODE: Scanning {len(h5ad_files)} h5ad files")
    print(f"  Directory: {data_dir}")
    print(f"{'='*70}")

    for h5ad_path in h5ad_files:
        ds_name = h5ad_path.stem
        print(f"\n{'_'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"  File: {h5ad_path.name} ({h5ad_path.stat().st_size / 1e9:.2f} GB)")

        try:
            # Read just obs (backed mode to save memory)
            adata = ad.read_h5ad(h5ad_path, backed='r')
            obs = adata.obs
            print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
            print(f"  Obs columns: {list(obs.columns)}")

            # Show unique values for each column (if manageable)
            for col in obs.columns:
                try:
                    vals = obs[col].unique()
                    n_unique = len(vals)
                    if n_unique <= 30:
                        sorted_vals = sorted([str(v) for v in vals.tolist()])
                        print(f"    {col} ({n_unique} unique): {sorted_vals}")
                    else:
                        sample_vals = sorted([str(v) for v in vals.tolist()])[:10]
                        print(f"    {col} ({n_unique} unique): [first 10] {sample_vals}...")
                except Exception:
                    print(f"    {col}: [could not read values]")

            # Validate against our config
            if ds_name in CELLULAR_DATASETS:
                cfg = CELLULAR_DATASETS[ds_name]
                status = "OK"

                if cfg['context_col'] not in obs.columns:
                    status = "NEEDS FIX"
                    print(f"  !! context_col '{cfg['context_col']}' NOT in obs!")
                    for cand in CONTEXT_COL_CANDIDATES:
                        if cand in obs.columns:
                            print(f"     --> Suggest: '{cand}'")

                if cfg['condition_col'] not in obs.columns:
                    status = "NEEDS FIX"
                    print(f"  !! condition_col '{cfg['condition_col']}' NOT in obs!")
                    for cand in CONDITION_COL_CANDIDATES:
                        if cand in obs.columns:
                            print(f"     --> Suggest: '{cand}'")

                elif cfg['ctrl_label'] not in [str(v) for v in obs[cfg['condition_col']].unique()]:
                    status = "NEEDS FIX"
                    all_vals = [str(v) for v in obs[cfg['condition_col']].unique()]
                    print(f"  !! ctrl_label '{cfg['ctrl_label']}' NOT in {cfg['condition_col']} values!")
                    print(f"     Available: {sorted(all_vals)}")
                    for cand in CTRL_LABEL_CANDIDATES:
                        if cand in all_vals:
                            print(f"     --> Suggest ctrl_label='{cand}'")

                if status == "OK":
                    ctx_vals = sorted([str(v) for v in obs[cfg['context_col']].unique()])
                    cond_vals = sorted([str(v) for v in obs[cfg['condition_col']].unique()])
                    n_ctrl = (obs[cfg['condition_col']].astype(str) == cfg['ctrl_label']).sum()
                    print(f"  [OK] Config validated.")
                    print(f"    Contexts: {ctx_vals}")
                    print(f"    Conditions: {cond_vals}")
                    print(f"    Control cells: {n_ctrl:,}")
                else:
                    print(f"  [{status}] Please update CELLULAR_DATASETS['{ds_name}'] in the script!")
            else:
                print(f"  [UNKNOWN] Not in CELLULAR_DATASETS config.")
                print(f"  You may need to add an entry for '{ds_name}'.")

            adata.file.close()
            del adata

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"  Discovery complete. Fix any [NEEDS FIX] entries before running.")
    print(f"{'='*70}")


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(dataset_name, data_dir, args):
    """
    Process one cellular context dataset.

    Structure mirrors the genetic v2 script's process_dataset, but with
    an outer loop over contexts (cell types / patients / species) instead
    of the Frangieh condition loop. For each context:
      1. Get control cells for this context
      2. Compute top-K expressed genes from these controls (non-circular)
      3. For each perturbation applied to this context:
           - Get stimulated cells
           - Compute split-half Δ-reliability
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
    print(f"  Processing: {dataset_name} (Cellular Context)")
    print(f"  Path: {h5ad_path}")
    print(f"  context_col='{context_col}' | condition_col='{condition_col}' | ctrl='{ctrl_label}'")
    print(f"{'='*70}")

    # ---- Step 1: Load ----
    print(f"  [1/5] Loading h5ad...")
    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    print(f"        {adata.n_obs:,} cells × {adata.n_vars:,} genes, loaded in {time.time()-t0:.1f}s")

    # ---- Step 2: Extract expression matrix ----
    print(f"  [2/5] Preparing expression matrix...")
    t0 = time.time()

    X = adata.X
    if sp.issparse(X):
        X_norm = X.tocsr().astype(np.float32)
        is_sparse = True
        density = X_norm.nnz / (X_norm.shape[0] * X_norm.shape[1]) * 100
        print(f"        Sparse CSR: {X_norm.nnz:,} nnz ({density:.1f}% density)")
    else:
        X_norm = np.asarray(X, dtype=np.float32)
        is_sparse = False
        print(f"        Dense: {X_norm.nbytes / 1e9:.2f} GB")

    # Sanity check
    sample = X_norm[:100].toarray() if is_sparse else X_norm[:100]
    xmax = sample.max()
    print(f"        Value range check (100 cells): max={xmax:.2f}")
    if xmax > 30:
        print(f"        ⚠ WARNING: max={xmax:.2f} seems high for log-normalized data!")
    print(f"        Prepared in {time.time()-t0:.1f}s")

    # ---- Step 3: Validate columns ----
    print(f"  [3/5] Validating obs columns...")

    if context_col not in adata.obs.columns:
        raise ValueError(f"context_col '{context_col}' not in obs columns: "
                         f"{list(adata.obs.columns)}. Run --dataset discover first!")
    if condition_col not in adata.obs.columns:
        raise ValueError(f"condition_col '{condition_col}' not in obs columns: "
                         f"{list(adata.obs.columns)}. Run --dataset discover first!")

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
            print(f"        Note: ctrl_label '{ctrl_label}' not found, using '{ctrl_found}'")
            ctrl_label = ctrl_found
        else:
            raise ValueError(f"ctrl_label '{ctrl_label}' not in condition values: "
                             f"{unique_conditions}. Run --dataset discover!")

    stim_conditions = [c for c in unique_conditions if c != ctrl_label]

    print(f"        Contexts ({len(unique_contexts)}): {unique_contexts}")
    print(f"        Conditions: ctrl='{ctrl_label}', stim ({len(stim_conditions)}): {stim_conditions}")

    # ---- Step 4: Build tasks per context ----
    # This is analogous to the condition_groups loop in the genetic v2 script
    # (used for Frangieh), but here EVERY dataset has context stratification.
    print(f"  [4/5] Building tasks...")

    all_rows = []
    total_conditions = 0
    total_skipped_ctrl = 0
    total_skipped_stim = 0

    for ctx in unique_contexts:
        ctx_mask = contexts == ctx

        # ---- Controls for this context ----
        ctrl_mask = ctx_mask & (conditions == ctrl_label)
        ctrl_idx = np.where(ctrl_mask)[0]
        n_ctrl = len(ctrl_idx)

        if n_ctrl < 16:
            n_stim_here = sum(1 for sc in stim_conditions
                              if (ctx_mask & (conditions == sc)).any())
            print(f"        Context '{ctx}': only {n_ctrl} controls, skipping "
                  f"({n_stim_here} conditions lost)")
            total_skipped_ctrl += n_stim_here
            continue

        # Precompute dense control matrix
        # Use all controls (field convention: Virtual Cell Challenge, scPerturBench,
        # Ahlmann-Eltze 2025, PerturBench, Replogle 2022 all use full control pools
        # for pseudo-bulk mean Delta — no subsampling).
        if is_sparse:
            ctrl_dense = X_norm[ctrl_idx].toarray().astype(np.float32)
        else:
            ctrl_dense = X_norm[ctrl_idx].astype(np.float32)

        ctrl_n_effective = ctrl_dense.shape[0]

        # ---- Compute top-k gene index by mean control expression ----
        # (Ahlmann-Eltze et al. 2025 approach: rank by expression, not effect size)
        # Computed ONCE per context from control cells → non-circular
        ctrl_mean_expr = ctrl_dense.mean(axis=0)
        expr_k = min(args.expr_k, ctrl_dense.shape[1])
        expr_idx = np.argpartition(ctrl_mean_expr, -expr_k)[-expr_k:]
        expr_idx = expr_idx[np.argsort(ctrl_mean_expr[expr_idx])[::-1]]

        # ---- Build tasks for all perturbations in this context ----
        tasks = []
        n_skipped = 0
        cell_count_dist = []

        for i, stim_cond in enumerate(stim_conditions):
            stim_mask = ctx_mask & (conditions == stim_cond)
            stim_idx = np.where(stim_mask)[0]
            n_stim = len(stim_idx)

            if n_stim == 0:
                continue  # this (context, perturbation) combo doesn't exist

            total_conditions += 1
            cell_count_dist.append(n_stim)

            # Max n_half: min of half-stim and half-ctrl (same as v2)
            n_half_max = min(n_stim // 2, ctrl_n_effective // 2)

            if n_half_max < 4:
                n_skipped += 1
                total_skipped_stim += 1
                continue

            # Fixed n values that are feasible
            n_half_fixed_feasible = [
                n for n in N_HALF_FIXED
                if n <= n_half_max
            ]

            cond_name = f"{ctx}|{stim_cond}"
            task_seed = args.seed + i

            # Extract dense stim submatrix HERE in parent (avoids sending
            # full X_norm to workers — critical for macOS spawn)
            X_sub = X_norm[stim_idx]
            if sp.issparse(X_sub):
                X_stim_dense = X_sub.toarray().astype(np.float32)
            elif not isinstance(X_sub, np.ndarray):
                X_stim_dense = np.array(X_sub, dtype=np.float32)
            else:
                X_stim_dense = (X_sub.astype(np.float32)
                                if X_sub.dtype != np.float32 else X_sub)

            tasks.append((
                cond_name,
                X_stim_dense,       # dense stim submatrix (small per condition)
                n_stim,
                n_half_max,
                n_half_fixed_feasible,
                args.n_repeats,
                expr_idx,           # precomputed per-context (top-k by ctrl expression)
                task_seed,
            ))

        if not tasks:
            print(f"        Context '{ctx}': {n_ctrl:,} ctrl (eff {ctrl_n_effective:,}), "
                  f"0 tasks" +
                  (f", {n_skipped} skipped" if n_skipped else ""))
            continue

        print(f"        Context '{ctx}': {n_ctrl:,} ctrl (eff {ctrl_n_effective:,}), "
              f"{len(tasks)} tasks" +
              (f", {n_skipped} skipped" if n_skipped else ""))
        print(f"          Top-{expr_k} expressed genes: "
              f"expr range [{ctrl_mean_expr[expr_idx[-1]]:.3f}, {ctrl_mean_expr[expr_idx[0]]:.3f}]")

        # Cell count distribution
        cell_counts = np.array(cell_count_dist)
        print(f"          Stim cells — min: {cell_counts.min()}, "
              f"median: {np.median(cell_counts):.0f}, max: {cell_counts.max()}")

        # ---- Step 5: Compute ----
        n_workers = min(args.workers, len(tasks))

        # Set globals for worker access (sequential mode uses these directly)
        global _G_CTRL_DENSE
        _G_CTRL_DENSE = ctrl_dense

        t0 = time.time()

        if n_workers <= 1:
            ctx_rows = []
            for i, task in enumerate(tasks):
                if (i + 1) % 20 == 0 or i == 0 or i == len(tasks) - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    print(f"          [{i+1:>4}/{len(tasks)}] "
                          f"({rate:.1f}/s, ETA {eta/60:.1f}m)")
                result = compute_one_condition(task)
                ctx_rows.extend(result)
        else:
            ctx_rows = []
            chunk_size = max(1, len(tasks) // (n_workers * 10))
            # Only ctrl_dense goes via initializer (small per context).
            # Stim submatrices are in each task tuple (small per condition).
            # This avoids pickling the full X_norm matrix to each worker.
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(ctrl_dense,),
            ) as pool:
                results_iter = pool.imap_unordered(
                    compute_one_condition,
                    tasks,
                    chunksize=chunk_size,
                )
                done = 0
                for result in results_iter:
                    ctx_rows.extend(result)
                    done += 1
                    if done % 20 == 0 or done == len(tasks):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (len(tasks) - done) / rate if rate > 0 else 0
                        print(f"          [{done:>4}/{len(tasks)}] "
                              f"({rate:.1f}/s, ETA {eta/60:.1f}m)")

        elapsed_ctx = time.time() - t0
        print(f"          Done in {elapsed_ctx/60:.1f} min")

        # Parse condition name into context + perturbation
        for row in ctx_rows:
            parts = row["condition"].split("|", 1)
            row["context"] = parts[0]
            row["perturbation"] = parts[1] if len(parts) > 1 else ""
        all_rows.extend(ctx_rows)

        # Reset cell_count_dist for next context
        cell_count_dist = []

    # Clean up
    _G_CTRL_DENSE = None
    del adata

    # ---- Build DataFrames ----
    df_long = pd.DataFrame(all_rows)
    df_long.insert(0, "dataset", dataset_name)

    # Check errors
    errors = df_long[df_long["n_mode"] == "ERROR"]
    if len(errors) > 0:
        print(f"\n  ⚠ {len(errors)} condition(s) had errors:")
        for _, row in errors.head(5).iterrows():
            print(f"    {row['condition']}: {row.get('error', 'unknown')}")

    # ---- Summary per condition ----
    df_valid = df_long[df_long["n_mode"] != "ERROR"].copy()
    summary_rows = []

    for cond, cdata in df_valid.groupby("condition"):
        n_stim = cdata["n_stim_cells"].iloc[0]
        ctx = cdata["context"].iloc[0]
        pert = cdata["perturbation"].iloc[0]

        row = {
            "dataset": dataset_name,
            "condition": cond,
            "context": ctx,
            "perturbation": pert,
            "n_stim_cells": n_stim,
        }

        # Primary (max) result
        max_data = cdata[cdata["n_mode"] == "max"]
        for gene_mode in max_data["gene_mode"].unique():
            gm_data = max_data[max_data["gene_mode"] == gene_mode]
            if len(gm_data) == 0:
                continue
            gm_row = gm_data.iloc[0]
            suffix = f"_{gene_mode}"
            row[f"n_half_max{suffix}"] = int(gm_row["n_half"])
            row[f"median_r_max{suffix}"] = gm_row["median_r"]
            row[f"mean_r_max{suffix}"] = gm_row["mean_r"]
            row[f"std_r_max{suffix}"] = gm_row["std_r"]
            row[f"skewness_r_max{suffix}"] = gm_row["skewness_r"]
            # All 4 SB approaches
            row[f"sb_from_mean{suffix}"] = gm_row["sb_from_mean"]
            row[f"sb_from_median{suffix}"] = gm_row["sb_from_median"]
            row[f"sb_mean_per_repeat{suffix}"] = gm_row["sb_mean_per_repeat"]
            row[f"sb_median_per_repeat{suffix}"] = gm_row["sb_median_per_repeat"]
            row[f"ceiling_from_mean{suffix}"] = gm_row["ceiling_from_mean"]
            row[f"ceiling_from_median{suffix}"] = gm_row["ceiling_from_median"]

        # Fixed curve: smallest n where median_r >= 0.5 (n*)
        fixed_data = cdata[cdata["n_mode"] == "fixed"]
        for gene_mode in fixed_data["gene_mode"].unique():
            gm_data = (fixed_data[fixed_data["gene_mode"] == gene_mode]
                       .sort_values("n_half"))
            if len(gm_data) == 0:
                continue
            suffix = f"_{gene_mode}"
            reliable_rows = gm_data[gm_data["median_r"] >= 0.5]
            if len(reliable_rows) > 0:
                row[f"n_star{suffix}"] = int(reliable_rows.iloc[0]["n_half"])
            else:
                row[f"n_star{suffix}"] = np.nan

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # ---- Print summary stats ----
    elapsed_total = time.time() - t_start
    print(f"\n  ── Summary for {dataset_name} ──")
    print(f"  Total conditions found: {total_conditions}")
    print(f"  Conditions processed: {len(df_summary)}")
    print(f"  Skipped (few ctrl): {total_skipped_ctrl}")
    print(f"  Skipped (few stim): {total_skipped_stim}")

    for gene_mode in ["all_genes", f"top{args.expr_k}_expr"]:
        col_sb_mean = f"sb_from_mean_{gene_mode}"
        col_sb_median = f"sb_from_median_{gene_mode}"
        col_ceil = f"ceiling_from_mean_{gene_mode}"
        col_skew = f"skewness_r_max_{gene_mode}"
        if col_sb_mean in df_summary.columns:
            vals_mean = df_summary[col_sb_mean].dropna()
            vals_median = df_summary[col_sb_median].dropna()
            skew_vals = df_summary[col_skew].dropna()
            n_reliable = (vals_mean >= 0.5).sum()
            print(f"  [{gene_mode}] Reliable (SB≥0.5 from mean): {n_reliable}/{len(vals_mean)} "
                  f"({100 * n_reliable / len(vals_mean):.1f}%)")
            print(f"    SB(mean_r) median: {vals_mean.median():.3f}, "
                  f"SB(median_r) median: {vals_median.median():.3f}")
            if len(skew_vals) > 0:
                print(f"    Skewness of r dist — median: {skew_vals.median():.3f}, "
                      f"mean: {skew_vals.mean():.3f}")
            if col_ceil in df_summary.columns:
                ceil_vals = df_summary[col_ceil].dropna()
                if len(ceil_vals) > 0:
                    print(f"    Ceiling median: {ceil_vals.median():.3f}")

    print(f"  Total time: {elapsed_total / 60:.1f} min")

    # ---- Metadata ----
    metadata = {
        "dataset": dataset_name,
        "task": "cellular_context",
        "config": {
            "context_col": context_col,
            "condition_col": condition_col,
            "ctrl_label": ctrl_label,
            "data_dir": str(data_dir),
        },
        "parameters": {
            "n_repeats": args.n_repeats,
            "expr_k": args.expr_k,
            "n_half_fixed": N_HALF_FIXED,
            "workers": args.workers,
            "seed": args.seed,
        },
        "results": {
            "n_conditions_total": total_conditions,
            "n_conditions_processed": len(df_summary),
            "n_contexts": len(unique_contexts),
            "n_perturbations": len(stim_conditions),
            "total_time_min": round(elapsed_total / 60, 2),
        },
        "timestamp": datetime.now().isoformat(),
    }

    return df_long, df_summary, metadata


# ============================================================
# MAIN
# ============================================================

def main():
    global N_HALF_FIXED
    parser = argparse.ArgumentParser(
        description="Compute split-half Δ-reliability for Wei et al. "
                    "Cellular Context generalization datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset name, 'all', or 'discover'. "
             f"Options: {', '.join(CELLULAR_DATASETS.keys())}",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to directory with cellular context h5ad files "
             "(from https://doi.org/10.6084/m9.figshare.28143422)",
    )
    parser.add_argument("--output_dir",
                        default="" + _ROOT + "/output/reliability_fig1/reliability_cellular",
                        help="Output dir (default: output/reliability_fig1/reliability_cellular)")
    parser.add_argument("--n_repeats", type=int, default=100,
                        help="Split-half repetitions (default: 100)")
    parser.add_argument("--expr_k", type=int, default=1000,
                        help="Number of top expressed genes for restricted mode "
                             "(Ahlmann-Eltze approach; default: 1000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_half_grid", type=str, default=None,
                        help=("Comma-separated n_half values for the fixed-n curve. "
                              "Default: %s. n_total per arm = 2 * n_half."
                              % ",".join(map(str, N_HALF_FIXED))))
    args = parser.parse_args()

    if args.n_half_grid:
        N_HALF_FIXED = sorted({int(x.strip()) for x in args.n_half_grid.split(",")})

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir does not exist: {data_dir}")
        sys.exit(1)

    # ---- Discovery mode ----
    if args.dataset.lower() == "discover":
        discover_datasets(data_dir)
        return

    # ---- Output dir ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = data_dir / "reliability_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determine datasets ----
    if args.dataset.lower() == "all":
        datasets_to_run = list(CELLULAR_DATASETS.keys())
    else:
        if args.dataset not in CELLULAR_DATASETS:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(CELLULAR_DATASETS.keys())}")
            print(f"Run --dataset discover to inspect files")
            sys.exit(1)
        datasets_to_run = [args.dataset]

    print(f"Wei et al. Cellular Context — Reliability Computation")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Parameters: n_repeats={args.n_repeats}, expr_k={args.expr_k}, "
          f"workers={args.workers}, seed={args.seed}")
    print(f"  Fixed n_half values: {N_HALF_FIXED}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {out_dir}")

    for ds_name in datasets_to_run:
        h5ad_path = data_dir / f"{ds_name}.h5ad"

        if not h5ad_path.exists():
            print(f"\n  ✗ SKIPPING {ds_name}: not found at {h5ad_path}")
            continue

        try:
            df_long, df_summary, metadata = process_dataset(
                ds_name, data_dir, args)

            # Save
            long_path = out_dir / f"reliability_cc_{ds_name}.csv"
            summary_path = out_dir / f"reliability_cc_summary_{ds_name}.csv"
            meta_path = out_dir / f"reliability_cc_{ds_name}_metadata.json"

            df_long.to_csv(long_path, index=False)
            df_summary.to_csv(summary_path, index=False)
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"\n  Saved:")
            print(f"    {long_path}")
            print(f"    {summary_path}")
            print(f"    {meta_path}")

            del df_long, df_summary

        except Exception as e:
            print(f"\n  ✗ FAILED on {ds_name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
