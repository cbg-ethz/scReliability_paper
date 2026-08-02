#!/usr/bin/env python3
"""
compute_reliability_genetic_context.py
===========================
Compute split-half Δ-reliability for ALL 17 Wei et al. (scPerturBench)
perturbation generalization datasets.

Two analysis modes:
  1. PRIMARY — "max" split: uses ALL available cells per perturbation,
     split into two equal halves. This estimates reliability of the actual
     ground truth used in benchmark scoring. The Spearman-Brown corrected
     value directly answers: "how reliable is this perturbation's ground
     truth?" → used for ceiling efficiency computation.

  2. SECONDARY — fixed n_half curve: computes reliability at fixed cell
     counts [8, 16, 32, 64, 128, 256, 512] to show how reliability scales
     with sample size. Produces the "how many cells do you need?" figure.

Two gene modes for each:
  - all_genes:   Pearson correlation across ALL ~5,000 HVGs
  - top1k_expr:  Pearson correlation across top-1000 most highly expressed
                 genes in control cells (Ahlmann-Eltze et al. 2025 approach).
                 Non-circular: gene selection is based on control expression
                 level, computed once per dataset, independent of perturbation
                 effects and split-half sampling.

Wei et al. standardized all 17 datasets:
  - perturbation column: 'perturbation'
  - control label: 'control'
  - X matrix: already log1p(CP10K) normalized → no normalization needed
  - ~5,000 HVGs pre-selected per dataset

Special handling:
  - Frangieh: has 3 experimental conditions (Control, IFNγ, Co-culture).
    Reliability is computed WITHIN each condition separately, since Δ
    expression differs across conditions.
  - sciplex3_A549: sparse X matrix → needs .toarray()
  - sciplex3 chemical datasets: included as supplementary analysis

USAGE
-----
# Single dataset:
python compute_reliability_genetic_context.py --dataset Norman

# All 17 datasets:
python compute_reliability_genetic_context.py --dataset all

# Custom parameters:
python compute_reliability_genetic_context.py --dataset Norman --n_repeats 100 --expr_k 1000 --workers 8

# SLURM batch (recommended — one job per dataset):
for ds in Norman Adamson Replogle_K562essential Replogle_RPE1essential \\
          Replogle_exp6 Replogle_exp7 Replogle_exp8 \\
          TianActivation TianInhibition Frangieh Papalexi Wessels Schmidt \\
          sciplex3_A549 sciplex3_K562 sciplex3_MCF7 sciplex3_comb; do
    sbatch -J "rel_${ds}" -c 8 --mem=64G -t 6:00:00 \
        --wrap "python compute_reliability_genetic_context.py --dataset ${ds} --workers 8 --n_repeats 100"
done

OUTPUTS (per dataset)
-------
  reliability_{dataset}.csv            Long-format: per (perturbation, n_mode, gene_mode)
  reliability_summary_{dataset}.csv    Wide: per perturbation, max reliability & ceiling
  reliability_{dataset}_metadata.json  Run metadata, parameters, timing

METHODOLOGY
-----------
  For each perturbation with N_ko KO cells and N_ctrl control cells:

  [PRIMARY — max mode]
    1. n = min(floor(N_ko/2), floor(N_ctrl/2))
    2. Repeat R times:
       - Permute KO cells → half_A (first n), half_B (next n)
       - Permute ctrl cells → half_C (first n), half_D (next n)
       - Δ_1 = mean(half_A) - mean(half_C)
       - Δ_2 = mean(half_B) - mean(half_D)
       - r = Pearson(Δ_1, Δ_2)
    3. Spearman-Brown: r_SB = 2 * median(r) / (1 + median(r))
       This estimates reliability of the FULL N_ko + N_ctrl measurement
    4. Ceiling = sqrt(max(0, r_SB))

  [SECONDARY — fixed n curve]
    Same as above but at each n in [8, 16, 32, 64, 128, 256, 512],
    skipping infeasible values. Shows reliability scaling.

  Gene modes:
    - all_genes: correlation across all ~5000 HVGs
    - top1k_expr: correlation across top-1000 genes by mean expression in
      control cells (Ahlmann-Eltze et al. 2025). Gene selection is computed
      ONCE per dataset from control cells only → fully independent of
      perturbation effects and split-half sampling.
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
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# DATASET CONFIGURATION — Wei et al. (scPerturBench)
# ============================================================
# All 17 datasets share: pert_col='perturbation', ctrl_label='control',
# X is log1p(CP10K), ~5000 HVGs pre-selected.

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
DATA_DIR = "" + _ROOT + "/data/Wei_et_al_data/genetic_context_preprocessed_h5ad"

# Genetic perturbation datasets (13)
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

# Chemical perturbation datasets (4)
CHEMICAL_DATASETS = {
    # Match Wei aggregation exactly: cov_drug_dose_name (cell_line + drug + dose).
    "sciplex3_A549": {"sparse_x": True, "pert_col": "cov_drug_dose_name", "ctrl_label": "A549_control_1.0"},
    "sciplex3_K562": {"pert_col": "cov_drug_dose_name", "ctrl_label": "K562_control_1.0"},
    "sciplex3_MCF7": {"pert_col": "cov_drug_dose_name", "ctrl_label": "MCF7_control_1.0"},
    "sciplex3_comb": {"pert_col": "cov_drug_dose_name", "ctrl_label": "A549_control_1.0"},
}

# Merge all
ALL_DATASETS = {}
for name, cfg in GENETIC_DATASETS.items():
    ALL_DATASETS[name] = {
        "path": os.path.join(DATA_DIR, f"{name}.h5ad"),
        "pert_col": "perturbation",
        "ctrl_label": "control",
        "skip_norm": True,
        "category": "genetic",
        **cfg,
    }
for name, cfg in CHEMICAL_DATASETS.items():
    ALL_DATASETS[name] = {
        "path": os.path.join(DATA_DIR, f"{name}.h5ad"),
        "pert_col": "perturbation",
        "ctrl_label": "control",
        "skip_norm": True,
        "category": "chemical",
        **cfg,
    }

DEFAULT_OUTPUT_DIR = "" + _ROOT + "/output/reliability_fig1/reliability_genetic"

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
# CORE: SINGLE PERTURBATION RELIABILITY
# ============================================================

def compute_one_perturbation(args):
    """
    Compute split-half reliability for a single perturbation.

    Parameters (packed tuple)
    ----------
    pert_name, X_ko_dense, n_half_max, n_half_fixed_feasible,
    n_repeats, expr_idx, seed

    Returns
    -------
    rows : list of dicts
    """
    (pert_name, X_ko_dense, n_ko, n_half_max, n_half_fixed_feasible,
     n_repeats, expr_idx, seed) = args

    ctrl_dense = _G_CTRL_DENSE      # global: precomputed dense control matrix

    n_ctrl = ctrl_dense.shape[0]
    n_genes = ctrl_dense.shape[1]

    rng = np.random.default_rng(seed)
    rows = []

    try:
        X_ko = X_ko_dense   # already dense float32, extracted in parent
        X_ctrl = ctrl_dense

        # expr_idx is precomputed per-dataset (top-k genes by mean ctrl expression)
        # — no per-perturbation gene selection needed

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
                perm_ko = rng.permutation(n_ko)
                half_A = perm_ko[:n]
                half_B = perm_ko[n:2 * n]

                perm_ctrl = rng.permutation(n_ctrl)
                half_C = perm_ctrl[:n]
                half_D = perm_ctrl[n:2 * n]

                deltas_1[r] = X_ko[half_A].mean(axis=0) - X_ctrl[half_C].mean(axis=0)
                deltas_2[r] = X_ko[half_B].mean(axis=0) - X_ctrl[half_D].mean(axis=0)

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
                    "perturbation": pert_name,
                    "n_ko_cells": n_ko,
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
            "perturbation": pert_name,
            "n_ko_cells": n_ko,
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
# KO submatrices are extracted in the parent and passed per-task.

_G_CTRL_DENSE = None


def _init_worker(ctrl_dense):
    """Pool initializer: set ctrl_dense in each worker process."""
    global _G_CTRL_DENSE
    _G_CTRL_DENSE = ctrl_dense


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(dataset_name, config, args):
    """
    Process one dataset: load → build tasks → compute reliability.

    For Frangieh, iterates over experimental conditions and computes
    reliability within each condition separately.
    """
    t_start = time.time()
    h5ad_path = config["path"]
    pert_col = config["pert_col"]
    ctrl_label = config["ctrl_label"]

    print(f"\n{'='*70}")
    print(f"  Processing: {dataset_name}")
    print(f"  Path: {h5ad_path}")
    print(f"  Category: {config['category']}")
    print(f"{'='*70}")

    # ---- Step 1: Load ----
    print(f"  [1/5] Loading h5ad...")
    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    print(f"        {adata.n_obs:,} cells × {adata.n_vars:,} genes, loaded in {time.time()-t0:.1f}s")

    # ---- Step 2: Extract expression matrix ----
    print(f"  [2/5] Preparing expression matrix (skip_norm={config['skip_norm']})...")
    t0 = time.time()

    X = adata.X
    if sp.issparse(X):
        # Keep sparse for memory, will .toarray() per perturbation
        X_norm = X.tocsr().astype(np.float32)
        is_sparse = True
        print(f"        Sparse CSR: {X_norm.nnz:,} nnz ({X_norm.nnz/(X_norm.shape[0]*X_norm.shape[1])*100:.1f}% density)")
    else:
        X_norm = np.asarray(X, dtype=np.float32)
        is_sparse = False
        print(f"        Dense: {X_norm.nbytes/1e9:.2f} GB")

    # Sanity check
    sample = X_norm[:100].toarray() if is_sparse else X_norm[:100]
    xmax = sample.max()
    print(f"        Value range check (100 cells): max={xmax:.2f}")
    if xmax > 30 and config["skip_norm"]:
        print(f"        ⚠ WARNING: max={xmax:.2f} seems high for log-normalized data!")
    print(f"        Prepared in {time.time()-t0:.1f}s")

    # ---- Determine if we need condition-stratified processing ----
    if config.get("has_conditions", False):
        condition_col = config["condition_col"]
        conditions = adata.obs[condition_col].unique().tolist()
        # Remove the experimental "Control" condition from the list if it refers
        # to the untreated condition — we still compute reliability for it
        print(f"        Frangieh conditions: {conditions}")
        condition_groups = [(cond, adata.obs[condition_col] == cond) for cond in conditions]
    else:
        condition_groups = [("all", np.ones(adata.n_obs, dtype=bool))]

    # ---- Process each condition group ----
    all_rows = []

    for cond_name, cond_mask in condition_groups:
        if config.get("has_conditions"):
            print(f"\n  ── Condition: {cond_name} ({cond_mask.sum():,} cells) ──")

        # ---- Step 3: Identify perturbations and controls within condition ----
        labels = adata.obs[pert_col].astype(str).str.strip().values

        # Apply condition mask
        cond_indices = np.where(cond_mask.values if hasattr(cond_mask, 'values') else cond_mask)[0]
        cond_labels = labels[cond_indices]

        # Controls
        ctrl_mask_local = cond_labels == ctrl_label
        ctrl_idx = cond_indices[ctrl_mask_local]
        n_ctrl = len(ctrl_idx)
        print(f"  [3/5] Controls: {n_ctrl:,} cells")

        if n_ctrl < 16:
            print(f"        ⚠ Too few controls ({n_ctrl}), skipping condition '{cond_name}'")
            continue

        # KO perturbations
        ko_mask_local = ~ctrl_mask_local
        ko_labels = cond_labels[ko_mask_local]
        ko_global_idx = cond_indices[ko_mask_local]

        unique_perts = np.unique(ko_labels)
        n_perts = len(unique_perts)
        print(f"        Perturbations: {n_perts:,}")

        # ---- Step 4: Build task list ----
        print(f"  [4/5] Building tasks...")

        # Precompute dense control matrix
        # Use all controls (field convention: Virtual Cell Challenge, scPerturBench,
        # Ahlmann-Eltze 2025, PerturBench, Replogle 2022 all use full control pools
        # for pseudo-bulk mean Delta — no subsampling).
        if is_sparse:
            ctrl_dense = X_norm[ctrl_idx].toarray().astype(np.float32)
        else:
            ctrl_dense = X_norm[ctrl_idx].astype(np.float32)

        ctrl_n_effective = ctrl_dense.shape[0]
        print(f"        Control pool: {ctrl_n_effective:,} cells, "
              f"{ctrl_dense.nbytes/1e9:.2f} GB")

        # ---- Compute top-k gene index by mean control expression ----
        # (Ahlmann-Eltze et al. 2025 approach: rank by expression, not effect size)
        ctrl_mean_expr = ctrl_dense.mean(axis=0)
        expr_k = min(args.expr_k, ctrl_dense.shape[1])
        expr_idx = np.argpartition(ctrl_mean_expr, -expr_k)[-expr_k:]
        expr_idx = expr_idx[np.argsort(ctrl_mean_expr[expr_idx])[::-1]]
        print(f"        Top-{expr_k} expressed genes: "
              f"expr range [{ctrl_mean_expr[expr_idx[-1]]:.3f}, {ctrl_mean_expr[expr_idx[0]]:.3f}]")

        tasks = []
        n_skipped = 0
        cell_count_dist = []

        for i, pert in enumerate(unique_perts):
            pert_mask = ko_labels == pert
            ko_idx = ko_global_idx[pert_mask]
            n_ko = len(ko_idx)
            cell_count_dist.append(n_ko)

            # Max n_half: min of half-KO and half-ctrl
            n_half_max = min(n_ko // 2, ctrl_n_effective // 2)

            if n_half_max < 4:
                n_skipped += 1
                continue

            # Fixed n values that are feasible
            n_half_fixed_feasible = [
                n for n in N_HALF_FIXED
                if n <= n_half_max
            ]

            pert_seed = args.seed + i

            # Extract dense KO submatrix HERE in parent (avoids sending
            # full X_norm to workers — critical for macOS spawn)
            X_sub = X_norm[ko_idx]
            if sp.issparse(X_sub):
                X_ko_dense = X_sub.toarray().astype(np.float32)
            elif not isinstance(X_sub, np.ndarray):
                X_ko_dense = np.array(X_sub, dtype=np.float32)
            else:
                X_ko_dense = X_sub.astype(np.float32) if X_sub.dtype != np.float32 else X_sub

            tasks.append((
                pert,
                X_ko_dense,       # dense KO submatrix (small per perturbation)
                n_ko,
                n_half_max,
                n_half_fixed_feasible,
                args.n_repeats,
                expr_idx,         # precomputed per-dataset (top-k by ctrl expression)
                pert_seed,
            ))

        print(f"        Tasks: {len(tasks):,} perturbations")
        print(f"        Skipped: {n_skipped:,} (too few cells)")

        # Cell count distribution
        cell_counts = np.array(cell_count_dist)
        print(f"        Cell counts — min: {cell_counts.min()}, "
              f"p25: {np.percentile(cell_counts,25):.0f}, "
              f"median: {np.median(cell_counts):.0f}, "
              f"p75: {np.percentile(cell_counts,75):.0f}, "
              f"max: {cell_counts.max()}")

        # Feasibility summary
        for n_fixed in N_HALF_FIXED:
            n_feasible = sum(1 for t in tasks if n_fixed <= t[2])
            pct = 100 * n_feasible / len(tasks) if tasks else 0
            print(f"        n_half={n_fixed:>4}: {n_feasible:>5}/{len(tasks)} feasible ({pct:.1f}%)")

        # Max n_half distribution
        max_ns = [t[2] for t in tasks]
        print(f"        Max n_half — min: {min(max_ns)}, median: {int(np.median(max_ns))}, max: {max(max_ns)}")

        # ---- Step 5: Compute ----
        n_workers = min(args.workers, len(tasks))
        print(f"  [5/5] Computing ({n_workers} workers, {args.n_repeats} repeats)...")
        t0 = time.time()

        # Set globals for worker access (sequential mode uses these directly)
        global _G_CTRL_DENSE
        _G_CTRL_DENSE = ctrl_dense

        if n_workers <= 1:
            cond_rows = []
            for i, task in enumerate(tasks):
                if (i + 1) % 50 == 0 or i == 0 or i == len(tasks) - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    print(f"        [{i+1:>5}/{len(tasks)}] "
                          f"({rate:.1f} perts/s, ETA {eta/60:.1f} min)")
                result = compute_one_perturbation(task)
                cond_rows.extend(result)
        else:
            cond_rows = []
            chunk_size = max(1, len(tasks) // (n_workers * 10))
            # Only ctrl_dense goes via initializer (small: ~0.01-0.15 GB).
            # KO submatrices are in each task tuple (small per perturbation).
            # This avoids pickling the full X_norm matrix to each worker.
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(ctrl_dense,),
            ) as pool:
                results_iter = pool.imap_unordered(
                    compute_one_perturbation,
                    tasks,
                    chunksize=chunk_size,
                )
                done = 0
                for result in results_iter:
                    cond_rows.extend(result)
                    done += 1
                    if done % 100 == 0 or done == len(tasks):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (len(tasks) - done) / rate if rate > 0 else 0
                        print(f"        [{done:>5}/{len(tasks)}] "
                              f"({rate:.1f} perts/s, ETA {eta/60:.1f} min)")

        elapsed_compute = time.time() - t0
        print(f"        Done in {elapsed_compute/60:.1f} min")

        # Add condition info
        for row in cond_rows:
            row["condition"] = cond_name
        all_rows.extend(cond_rows)

    # Clean up
    _G_CTRL_DENSE = None
    del adata

    # ---- Build DataFrames ----
    df_long = pd.DataFrame(all_rows)
    df_long.insert(0, "dataset", dataset_name)

    # Check errors
    errors = df_long[df_long["n_mode"] == "ERROR"]
    if len(errors) > 0:
        print(f"\n  ⚠ {len(errors)} perturbation(s) had errors:")
        for _, row in errors.head(5).iterrows():
            print(f"    {row['perturbation']}: {row.get('error', 'unknown')}")

    # ---- Summary per perturbation ----
    df_valid = df_long[df_long["n_mode"] != "ERROR"].copy()
    summary_rows = []

    for (pert, cond), pdata in df_valid.groupby(["perturbation", "condition"]):
        n_ko = pdata["n_ko_cells"].iloc[0]

        row = {
            "dataset": dataset_name,
            "perturbation": pert,
            "condition": cond,
            "n_ko_cells": n_ko,
        }

        # Primary (max) result
        max_data = pdata[pdata["n_mode"] == "max"]
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
        fixed_data = pdata[pdata["n_mode"] == "fixed"]
        for gene_mode in fixed_data["gene_mode"].unique():
            gm_data = fixed_data[fixed_data["gene_mode"] == gene_mode].sort_values("n_half")
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
    print(f"  Perturbations processed: {len(df_summary)}")

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
                  f"({100*n_reliable/len(vals_mean):.1f}%)")
            print(f"    SB(mean_r) median: {vals_mean.median():.3f}, "
                  f"SB(median_r) median: {vals_median.median():.3f}")
            print(f"    Skewness of r dist — median: {skew_vals.median():.3f}, "
                  f"mean: {skew_vals.mean():.3f}")

    print(f"  Total time: {elapsed_total/60:.1f} min")

    # ---- Metadata ----
    metadata = {
        "dataset": dataset_name,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in config.items()},
        "parameters": {
            "n_repeats": args.n_repeats,
            "expr_k": args.expr_k,
            "n_half_fixed": N_HALF_FIXED,
            "workers": args.workers,
            "seed": args.seed,
        },
        "results": {
            "n_perturbations_processed": len(df_summary),
            "n_conditions": len(condition_groups),
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
        description="Compute split-half Δ-reliability for Wei et al. scPerturBench datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset name, 'all', 'genetic', or 'chemical'. "
             f"Options: {', '.join(ALL_DATASETS.keys())}",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
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

    print(f"Wei et al. Reliability Computation")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Parameters: n_repeats={args.n_repeats}, expr_k={args.expr_k}, "
          f"workers={args.workers}, seed={args.seed}")
    print(f"  Fixed n_half values: {N_HALF_FIXED}")
    print(f"  Output: {out_dir}")

    for ds_name in datasets_to_run:
        config = ALL_DATASETS[ds_name]

        if not Path(config["path"]).exists():
            print(f"\n  ✗ SKIPPING {ds_name}: file not found at {config['path']}")
            continue

        try:
            df_long, df_summary, metadata = process_dataset(ds_name, config, args)

            # Save
            long_path = out_dir / f"reliability_{ds_name}.csv"
            summary_path = out_dir / f"reliability_summary_{ds_name}.csv"
            meta_path = out_dir / f"reliability_{ds_name}_metadata.json"

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
