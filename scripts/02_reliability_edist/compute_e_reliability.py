#!/usr/bin/env python3
"""
compute_e_reliability.py
========================
Distributional split-half reliability for all 29 perturbation datasets,
using energy (E-) distance instead of Pearson correlation on per-gene
mean effect vectors. The specificity axis is unchanged: it remains the
per-gene-mean cos^2(theta).

For each perturbation:

  rho_E(N) = 1 - median_over_seeds( E_within(N) / E_between )

where
  E_within(N) = E-distance between two random half-distributions of N cells
                each (within the perturbed pool)
  E_between   = E-distance between the full perturbed pool and the full
                control pool (the perturbation effect magnitude)

The grid of N values, the per-perturbation inclusion criterion (n_ko >= 8
and n_ctrl >= 8), the per-dataset (perturbation, context, condition) units,
and the use of ALL cells (no subsampling caps) match the Pearson pipeline,
so the only difference is the metric: Pearson on per-gene means becomes
E-distance on cell distributions.

Compute strategy
-----------------------------------
- Precompute D_cc_sum (control-vs-control pairwise distance sum) ONCE per
  (dataset, condition).
- Per perturbation: compute D_pp (within perturbation) and D_pc (perturbation
  vs control) once, reuse across all seeds and all N values.
- Parallelize across perturbations using joblib.

Outputs
-------
intermediate/edist/e_reliability_{dataset}.csv
  per-perturbation rho_E(N) at each N in N_GRID plus the full pass (n_half=-1).

intermediate/edist/e_reliability_all.csv
  concatenated across all 29 datasets.
"""

import os
import sys
import hashlib
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
GENETIC_DIR = Path("" + _ROOT + "/data/Wei_et_al_data/genetic_context_preprocessed_h5ad")
CELLULAR_DIR = Path("" + _ROOT + "/data/Wei_et_al_data/cellular_context_preprocessed_h5ad")
OUT_DIR = Path("" + _ROOT + "/intermediate/edist")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All 29 datasets
GENETIC_DATASETS = [
    "Norman", "Adamson",
    "Replogle_K562essential", "Replogle_RPE1essential",
    "Replogle_exp6", "Replogle_exp7", "Replogle_exp8",
    "TianActivation", "TianInhibition",
    "Frangieh", "Papalexi", "Wessels", "Schmidt",
    "sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7", "sciplex3_comb",
]

CELLULAR_DATASETS = [
    "Afriat", "crossPatient", "crossSpecies", "Haber",
    "KaggleCrossCell", "KaggleCrossPatient",
    "kangCrossCell", "kangCrossPatient",
    "McFarland", "Parekh", "sciplex3", "TCDD",
]

# Cell-count grid for rho_E(N) curves (n_half values).
N_GRID = [8, 16, 32, 64, 128, 256, 512]

# Number of split-half resamples per (perturbation, N).
N_SEEDS = 100

# Parallelism
N_WORKERS = int(os.environ.get("N_WORKERS", 14))


def _stable_seed(*parts):
    """Deterministic 32-bit seed. Python's hash() is salted per process, so it cannot be
    used for anything whose output must reproduce across runs."""
    s = "::".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:4], "big")


# ============================================================
# E-DISTANCE FROM A PRECOMPUTED PAIRWISE DISTANCE MATRIX
# ============================================================

def edistance_from_blocks(d_xy_sum, d_xx_sum, d_yy_sum, n_x, n_y):
    """
    Given precomputed sums of pairwise L2 distances between blocks X and Y
    and within each block, return the energy distance.

      E = 2 * mean_xy - mean_xx - mean_yy
    where mean_xy = d_xy_sum / (n_x * n_y), etc.
    """
    mean_xy = d_xy_sum / (n_x * n_y)
    mean_xx = d_xx_sum / (n_x * n_x)
    mean_yy = d_yy_sum / (n_y * n_y)
    return 2.0 * mean_xy - mean_xx - mean_yy


def _sum_block(D, rows, cols):
    """Sum of D[rows][:, cols]. Both rows, cols are 1d index arrays."""
    return float(D[np.ix_(rows, cols)].sum())


def _sum_within(D, idx):
    """Sum of D[idx][:, idx], including diagonal (which is 0 for L2)."""
    return float(D[np.ix_(idx, idx)].sum())


# ============================================================
# PER-PERTURBATION COMPUTATION
# ============================================================

def compute_one_perturbation(
    pert_name,
    X_pert,             # (n_pert, G) float32 dense — ALL perturbed cells, no cap
    X_ctrl,             # (n_ctrl, G) float32 dense — ALL control cells, no cap
    D_cc_sum,           # scalar, precomputed sum of pairwise dists within X_ctrl
    n_ctrl,             # int, X_ctrl.shape[0]
    n_grid,
    n_seeds,
    seed_offset,
):
    """
    Compute ρ_E(N) at each N in n_grid + {full} for one perturbation.

    Returns a list of dict rows (one per N), with the full pass tagged
    n_half = -1.
    """
    n_pert = X_pert.shape[0]

    # Precompute within-perturbation and perturbation-vs-control pairwise distances.
    # D_cc_sum is passed in already (one-shot per dataset condition).
    D_pp = pairwise_distances(X_pert, X_pert, metric="euclidean").astype(np.float64)
    D_pc = pairwise_distances(X_pert, X_ctrl, metric="euclidean").astype(np.float64)

    # E_between(p, ctrl): perturbation vs control on full pools.
    e_between = edistance_from_blocks(
        D_pc.sum(), D_pp.sum(), D_cc_sum,
        n_pert, n_ctrl,
    )

    rng = np.random.default_rng(_stable_seed(pert_name) ^ seed_offset)
    rows = []

    # N values to evaluate: each grid N at most once, plus the full pass.
    N_full = n_pert // 2
    if N_full < 4:
        return rows
    grid_Ns = [N for N in n_grid if 4 <= N < N_full]
    work_Ns = grid_Ns + [N_full]

    # Degenerate effect (E_between <= 0): emit rho_E = 0 rows.
    if not np.isfinite(e_between) or e_between <= 0:
        for N in work_Ns:
            rows.append(dict(
                perturbation=pert_name,
                n_half=N if N != N_full else -1,
                e_within_med=np.nan, e_between=e_between,
                rho_E=0.0, n_cells=n_pert,
            ))
        return rows

    for N in work_Ns:
        ratios = np.empty(n_seeds, dtype=np.float64)
        for s in range(n_seeds):
            perm = rng.permutation(n_pert)
            h1 = perm[:N]
            h2 = perm[N:2 * N]
            d11 = _sum_within(D_pp, h1)
            d22 = _sum_within(D_pp, h2)
            d12 = _sum_block(D_pp, h1, h2)
            e_within = edistance_from_blocks(d12, d11, d22, N, N)
            ratios[s] = e_within / e_between
        e_within_med = float(np.median(ratios) * e_between)
        rho = float(np.clip(1.0 - np.median(ratios), -1.0, 1.0))
        rows.append(dict(
            perturbation=pert_name,
            n_half=N if N != N_full else -1,
            e_within_med=e_within_med, e_between=e_between,
            rho_E=rho, n_cells=n_pert,
        ))

    return rows


# ============================================================
# DATASET-LEVEL ORCHESTRATION
# ============================================================

# Cellular dataset configs.
# context_col: stratifies cells (e.g., cell_type, cell_line, patient). Each context value
#              is treated as a separate condition, with its own control set.
# condition_col: the perturbation label within each context.
CELLULAR_CFG = {
    "Haber":             dict(context_col="cell_type",  condition_col="condition",    ctrl_label="Control"),
    "Afriat":            dict(context_col="condition1", condition_col="perturbation", ctrl_label="control"),
    "McFarland":         dict(context_col="cell_line",  condition_col="perturbation", ctrl_label="control"),
    "Parekh":            dict(context_col="cell_type",  condition_col="perturbation", ctrl_label="CTRL"),
    "TCDD":              dict(context_col="celltype",   condition_col="perturbation", ctrl_label="control"),
    "crossPatient":      dict(context_col="patient",    condition_col="perturbation", ctrl_label="control"),
    "crossSpecies":      dict(context_col="condition1", condition_col="condition2",   ctrl_label="control"),
    "KaggleCrossCell":   dict(context_col="cell_type",  condition_col="perturbation", ctrl_label="control"),
    "KaggleCrossPatient":dict(context_col="donor_id",   condition_col="perturbation", ctrl_label="control"),
    "kangCrossCell":     dict(context_col="cell_type",  condition_col="perturbation", ctrl_label="control"),
    "kangCrossPatient":  dict(context_col="sample_id",  condition_col="perturbation", ctrl_label="control"),
    # condition2 = drug name without dose, the evaluation unit used throughout the pipeline.
    "sciplex3":          dict(context_col="cell_line",  condition_col="condition2",   ctrl_label="control"),
}


def get_dataset_config(name):
    """Resolve dataset path and column names."""
    if name in GENETIC_DATASETS:
        path = GENETIC_DIR / f"{name}.h5ad"
        pert_col = "perturbation"
        ctrl_label = "control"
        if name.startswith("sciplex3"):
            pert_col = "cov_drug_dose_name"
            if "A549" in name or "comb" in name:
                ctrl_label = "A549_control_1.0"
            elif "K562" in name:
                ctrl_label = "K562_control_1.0"
            elif "MCF7" in name:
                ctrl_label = "MCF7_control_1.0"
        has_condition = (name == "Frangieh")
        is_cellular = False
        context_col = None
    elif name in CELLULAR_DATASETS:
        path = CELLULAR_DIR / f"{name}.h5ad"
        cfg = CELLULAR_CFG.get(name, dict(
            context_col=None, condition_col="perturbation", ctrl_label="control"))
        pert_col = cfg["condition_col"]
        ctrl_label = cfg["ctrl_label"]
        context_col = cfg["context_col"]
        has_condition = False
        is_cellular = True
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dict(path=path, pert_col=pert_col, ctrl_label=ctrl_label,
                has_condition=has_condition, is_cellular=is_cellular,
                context_col=context_col)


def process_dataset(name, n_grid=N_GRID, n_seeds=N_SEEDS, n_workers=N_WORKERS):
    cfg = get_dataset_config(name)
    print(f"\n========== {name} ==========", flush=True)
    t0 = time.time()
    if not cfg["path"].exists():
        print(f"  [skip] file not found: {cfg['path']}", flush=True)
        return None

    adata = ad.read_h5ad(cfg["path"])
    print(f"  loaded: {adata.shape}, pert_col={cfg['pert_col']}", flush=True)

    pert_col = cfg["pert_col"]
    ctrl_label = cfg["ctrl_label"]

    # Build (condition, sub_adata, ctrl_label_for_sub) tuples.
    # For genetic: single 'all' condition unless Frangieh (has biological condition).
    # For cellular: one condition per context value (e.g., cell_type, cell_line).
    work_units = []
    if cfg["is_cellular"] and cfg["context_col"] is not None:
        if cfg["context_col"] not in adata.obs.columns:
            print(f"  [warn] context_col {cfg['context_col']!r} not in obs; "
                  f"falling back to 'all'", flush=True)
            work_units.append(("all", adata, ctrl_label))
        else:
            for ctx_val in adata.obs[cfg["context_col"]].astype(str).unique():
                sub = adata[adata.obs[cfg["context_col"]].astype(str) == ctx_val]
                # sciplex3 cellular: ctrl label depends on cell_line context value
                if ctrl_label is None:
                    cl = ctx_val
                    cl_ctrl = f"{cl}_control_1.0"
                    work_units.append((ctx_val, sub, cl_ctrl))
                else:
                    work_units.append((ctx_val, sub, ctrl_label))
    elif cfg["has_condition"] and "condition" in adata.obs.columns:
        for c in adata.obs["condition"].astype(str).unique():
            sub = adata[adata.obs["condition"].astype(str) == c]
            work_units.append((c, sub, ctrl_label))
    else:
        work_units.append(("all", adata, ctrl_label))

    all_rows = []
    for cond, sub, cond_ctrl in work_units:
        ctrl_label_eff = cond_ctrl

        labels = sub.obs[pert_col].astype(str).values
        ctrl_mask_sub = (labels == ctrl_label_eff)
        if ctrl_mask_sub.sum() < 8:
            # No matched control in this condition; fall back to global ctrl
            ctrl_mask_global = (adata.obs[pert_col].astype(str) == ctrl_label_eff).values
            ctrl_source = adata
            ctrl_idx_all = np.where(ctrl_mask_global)[0]
        else:
            ctrl_source = sub
            ctrl_idx_all = np.where(ctrl_mask_sub)[0]

        if len(ctrl_idx_all) < 8:
            print(f"  cond={cond!r}: <8 control cells, skip", flush=True)
            continue

        # Use ALL control cells (no cap).
        X_ctrl = ctrl_source.X[ctrl_idx_all]
        if sp.issparse(X_ctrl):
            X_ctrl = X_ctrl.toarray()
        X_ctrl = np.asarray(X_ctrl, dtype=np.float32)
        n_ctrl = X_ctrl.shape[0]
        # Precompute ctrl-vs-ctrl pairwise distance sum ONCE per condition.
        D_cc_sum = float(pairwise_distances(X_ctrl, X_ctrl,
                                            metric="euclidean")
                         .astype(np.float64).sum())
        print(f"  cond={cond!r}: n_ctrl={n_ctrl}", flush=True)

        # List of perturbations within this work unit.
        nonctrl_mask_sub = (labels != ctrl_label_eff) & (labels != "nan")
        if not nonctrl_mask_sub.any():
            print(f"    [skip cond] no perturbed cells", flush=True)
            continue
        unique_perts = pd.Series(labels[nonctrl_mask_sub]).value_counts()
        # Inclusion criterion: skip if n_half_max = min(n_ko, n_ctrl) // 2 < 4,
        # which requires n_ko >= 8 (we already require n_ctrl >= 8 above).
        perts = [p for p in unique_perts.index if unique_perts[p] >= 8]
        print(f"  {len(perts)} perturbations to process", flush=True)

        def build_args(p):
            mask = (labels == p)
            idx_all = np.where(mask)[0]
            X_p = sub.X[idx_all]
            if sp.issparse(X_p):
                X_p = X_p.toarray()
            X_p = np.asarray(X_p, dtype=np.float32)
            return X_p

        def per_pert(p):
            try:
                X_p = build_args(p)
                rows = compute_one_perturbation(
                    p, X_p, X_ctrl, D_cc_sum, n_ctrl,
                    n_grid, n_seeds,
                    seed_offset=_stable_seed(name),
                )
                for r in rows:
                    r["dataset"] = name
                    r["condition"] = cond
                return rows
            except Exception as e:
                print(f"    [fail {p}] {e}", flush=True)
                return []

        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(per_pert)(p) for p in perts
        )
        for rs in results:
            all_rows.extend(rs)

    if not all_rows:
        print(f"  [empty] {name}", flush=True)
        return None

    df = pd.DataFrame(all_rows)
    out = OUT_DIR / f"e_reliability_{name}.csv"
    df.to_csv(out, index=False)
    dt = time.time() - t0
    print(f"  -> {out}  ({len(df)} rows, {dt:.1f}s, "
          f"{len(df)/max(dt,1):.1f} rows/s)", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="One dataset name or 'all'")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    if args.dataset == "all":
        targets = GENETIC_DATASETS + CELLULAR_DATASETS
    else:
        targets = [args.dataset]

    all_dfs = []
    for name in targets:
        df = process_dataset(name, n_seeds=args.seeds, n_workers=args.workers)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        # Rebuild the combined table from every per-dataset file on disk, so that a single-dataset
        # run still leaves a complete e_reliability_all.csv.
        parts = []
        for name in GENETIC_DATASETS + CELLULAR_DATASETS:
            f = OUT_DIR / f"e_reliability_{name}.csv"
            if f.exists():
                parts.append(pd.read_csv(f))
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(OUT_DIR / "e_reliability_all.csv", index=False)
        print(f"\n>>> Combined: {len(combined)} rows from {len(parts)} datasets "
              f"-> {OUT_DIR / 'e_reliability_all.csv'}")


if __name__ == "__main__":
    main()
