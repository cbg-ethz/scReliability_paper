#!/usr/bin/env python3
"""Preprocess 04 — Fit reliability model & simulate pilot experiments.

Reads raw split-half reliability curves from the Fig 3 grid (dense step-2:
n_total = 8..100 cells, the wet-lab pilot regime) and fits the parametric
saturation model ρ(N) = Nτ²/(Nτ²+1) to each perturbation.

Cohort filter
-------------
Only perturbations with `n_cells ≥ N_FULL_MIN` are retained. Below that
threshold the perturbation is itself at pilot scale and would not give a
reliable ground-truth τ² to validate against.

Pilot grid
----------
For each retained pert, additional fits are produced from leave-future-out
subsets (only points with `n_total ≤ pilot_N`). Used by Fig 3 to compare
pilot-derived predictions against the full-data ground truth. The compute-
side script skips n_half < 4 ("minimum for meaningful correlation"), so the
data grid is n_half ∈ {4..50} → smallest valid pilot at MIN_POINTS=2 is
pilot_N=10.

Outputs (to FITS_DIR; see config.py)
------------------------------------
- fits_all_perturbations.csv     : per-pert τ², R², N* at 0.5/0.7/0.9
- pilot_validation_extended.csv     : per-(pert, pilot_N) full vs. pilot τ²
"""
import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import (RELIABILITY_GEN_DIR, RELIABILITY_CEL_DIR,
                    GENETIC_SYSTEMA_DIR, CELLULAR_SYSTEMA_DIR,
                    FITS_DIR, rho_model, n_star, derive_triage_2d)

os.makedirs(str(FITS_DIR), exist_ok=True)

# ── Cohort filter and pilot grid ─────────────────────────────────────
N_FULL_MIN = 100        # only validate against perts with ≥100 cells (full exp)
MIN_POINTS = 2          # 2 distinct N values are enough to fit τ² (R²=1 by
                        # construction; out-of-sample validation is via the
                        # within-2-fold metric)
# Note: don't filter pilot-uninformative perts (those with all-noise ρ in the
# subset). They produce a horizontal cluster in the pilot-vs-full scatter at
# log10(N*_pilot) ≈ 5 because the optimizer collapses to a near-bound default
# τ², but they ARE real failure cases and removing them would inflate the
# within-2-fold metric. Keep them; report the failure rate honestly.

PILOT_NHALF_VALUES = list(range(5, 50))  # [5, 6, …, 49] → pilot_N 10..98 step 2


# ══════════════════════════════════════════════════════════════════
# FITTING
# ══════════════════════════════════════════════════════════════════

def fit_perturbation(n_total_arr, rho_obs_arr, min_points=MIN_POINTS):
    """Fit τ² for a single perturbation from its reliability curve.

    Returns both unweighted and N-weighted R² for fit-quality assessment.
    The N-weighted version accounts for the fact that residual variance at
    each cell-count point scales roughly as 1/N — larger-N points are more
    precise estimates of the true reliability and should carry more weight
    when assessing fit quality.

    `min_points` is enforced on the count of UNIQUE N values (not raw rows),
    because multi-condition datasets contribute multiple rows per N (one per
    condition) which would otherwise inflate the row count and admit fits
    with too few distinct N values.
    """
    mask = np.isfinite(rho_obs_arr) & (rho_obs_arr > -1.0)
    n_total = n_total_arr[mask].astype(float)
    rho_obs = np.clip(rho_obs_arr[mask].astype(float), 0.0, 0.999)

    if len(rho_obs) < min_points:
        return None
    if len(np.unique(n_total)) < min_points:
        return None

    try:
        popt, _ = curve_fit(rho_model, n_total, rho_obs,
                            p0=[0.001], bounds=(1e-8, 10.0), maxfev=5000)
    except Exception:
        return None
    tau_sq = float(popt[0])
    if not np.isfinite(tau_sq) or tau_sq <= 0:
        return None

    rho_pred = rho_model(n_total, tau_sq)
    ss_res = float(np.sum((rho_obs - rho_pred) ** 2))
    ss_tot = float(np.sum((rho_obs - rho_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    weights = n_total / n_total.sum()
    ss_res_w = float(np.sum(weights * (rho_obs - rho_pred) ** 2))
    rho_mean_w = float(np.sum(weights * rho_obs))
    ss_tot_w = float(np.sum(weights * (rho_obs - rho_mean_w) ** 2))
    r2_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0

    return {
        "tau_sq": tau_sq,
        "fit_r2": r2,
        "fit_r2_weighted": r2_w,
        "n_star_50": float(n_star(tau_sq, 0.5)),
        "n_star_70": float(n_star(tau_sq, 0.7)),
        "n_star_90": float(n_star(tau_sq, 0.9)),
        "n_points_used": int(len(rho_obs)),
        "max_N_used": float(np.max(n_total)),
        "min_N_used": float(np.min(n_total)),
    }


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_curves(rdir, prefix, n_col, ctx, pert_id_col):
    """Load raw split-half reliability curves and standardize columns.

    For genetic curves, each (perturbation, condition) is a distinct
    biological experiment (e.g., Frangieh has Control / IFNγ / Co-culture
    for the same gene KO) and gets its own pert_id so it is fit separately.
    """
    dfs = []
    for f in sorted(Path(rdir).glob(f"{prefix}*.csv")):
        if "summary" in f.name or "metadata" in f.name:
            continue
        ds = f.stem.replace(prefix, "")
        df = pd.read_csv(f)
        df["dataset"] = ds
        dfs.append(df)
    if not dfs:
        print(f"  WARNING: no curves in {rdir} with prefix '{prefix}'")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["n_cells"] = combined[n_col]
    if ctx == "genetic" and "condition" in combined.columns:
        cond = combined["condition"].fillna("").astype(str)
        has_cond = cond != ""
        combined["pert_id"] = np.where(
            has_cond,
            combined[pert_id_col].astype(str) + "|" + cond,
            combined[pert_id_col].astype(str)
        )
    else:
        combined["pert_id"] = combined[pert_id_col]
    combined["context_type"] = ctx
    return combined


def load_systema_2d(sdir, context_type):
    """Load Systema triage categories. (Triage is computed from the canonical
    Fig 1 7-point grid in `output/reliability_fig1/`; it does not depend on
    the dense Fig 3 grid.)"""
    sdir = Path(sdir)
    prefix = ("systema_cc_reliability_2d_" if context_type == "cellular"
              else "systema_reliability_2d_")
    dfs = []
    for f in sorted(sdir.glob(f"{prefix}*.csv")):
        dfs.append(pd.read_csv(f))
    if not dfs:
        print(f"  WARNING: no systema files in {sdir}")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if context_type == "genetic":
        if "condition" in combined.columns:
            cond = combined["condition"].fillna("").astype(str)
            has_cond = cond != ""
            combined["pert_id"] = np.where(
                has_cond,
                combined["perturbation"].astype(str) + "|" + cond,
                combined["perturbation"].astype(str)
            )
        else:
            combined["pert_id"] = combined["perturbation"]
    else:
        combined["pert_id"] = combined["condition"]
    # The systema source files still carry the legacy label vocabulary. Re-derive the canonical
    # Unreliable/Shared/Specific labels from rho and cos_sim so every table ships the same vocabulary.
    _rho = next((c for c in ("sb_from_median_all_genes", "sb_from_median") if c in combined.columns), None)
    if _rho is not None and "cos_sim" in combined.columns:
        combined["triage_2d"] = derive_triage_2d(combined, rho_col=_rho)
    return combined


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════

print(f"Loading curves (Fig 3 grid; cohort filter n_cells ≥ {N_FULL_MIN})...")
gen_curves = load_curves(str(RELIABILITY_GEN_DIR), "reliability_",
                         "n_ko_cells", "genetic", "perturbation")
cel_curves = load_curves(str(RELIABILITY_CEL_DIR), "reliability_cc_",
                         "n_stim_cells", "cellular", "condition")

n_gen_pre = gen_curves["pert_id"].nunique()
n_cel_pre = cel_curves["pert_id"].nunique()
gen_curves = gen_curves[gen_curves["n_cells"] >= N_FULL_MIN].copy()
cel_curves = cel_curves[cel_curves["n_cells"] >= N_FULL_MIN].copy()
print(f"  Genetic: {gen_curves['dataset'].nunique()} datasets, "
      f"{gen_curves['pert_id'].nunique()}/{n_gen_pre} (pert,cond) units after filter")
print(f"  Cellular: {cel_curves['dataset'].nunique()} datasets, "
      f"{cel_curves['pert_id'].nunique()}/{n_cel_pre} conditions after filter")

print("\nLoading systema triage (canonical 7-pt Fig 1 pipeline)...")
gen_systema = load_systema_2d(str(GENETIC_SYSTEMA_DIR), "genetic")
cel_systema = load_systema_2d(str(CELLULAR_SYSTEMA_DIR), "cellular")


# ══════════════════════════════════════════════════════════════════
# FIT τ² (curve_fit on fixed-mode rows)
# ══════════════════════════════════════════════════════════════════

def run_fitting(curves_df, gene_mode="all_genes", min_points=MIN_POINTS):
    """Fit τ² for every (dataset, pert_id). Uses fixed-n rows only."""
    fixed = curves_df[
        (curves_df["n_mode"] == "fixed") &
        (curves_df["gene_mode"] == gene_mode)
    ].copy()
    rho_col = ("sb_from_median" if "sb_from_median" in fixed.columns
               else "sb_from_mean")
    print(f"  Fitting with reliability column: {rho_col}")
    ctx = fixed["context_type"].iloc[0]
    results = []
    for (ds, pert), grp in fixed.groupby(["dataset", "pert_id"]):
        n_total = 2.0 * grp["n_half"].values.astype(float)
        rho_obs = grp[rho_col].values.astype(float)
        fit = fit_perturbation(n_total, rho_obs, min_points=min_points)
        if fit is not None:
            fit["dataset"] = ds
            fit["pert_id"] = pert
            fit["n_cells"] = int(grp["n_cells"].iloc[0])
            fit["context_type"] = ctx
            results.append(fit)
    return pd.DataFrame(results)


print("\nFitting parametric model...")
gen_fits = run_fitting(gen_curves)
cel_fits = run_fitting(cel_curves)
print(f"  Genetic: {len(gen_fits)} fitted")
print(f"  Cellular: {len(cel_fits)} fitted")


# ── Merge systema triage ──
def merge_systema(fits_df, systema_df):
    if systema_df.empty:
        fits_df["triage_2d"] = "unknown"
        fits_df["cos_sim"] = np.nan
        fits_df["frac_systematic_var"] = np.nan
        return fits_df
    cols = [c for c in ["dataset", "pert_id", "triage_2d", "cos_sim",
                        "frac_systematic_var"] if c in systema_df.columns]
    slim = systema_df[cols].drop_duplicates(subset=["dataset", "pert_id"])
    merged = fits_df.merge(slim, on=["dataset", "pert_id"], how="left")
    merged["triage_2d"] = merged["triage_2d"].fillna("unknown")
    return merged


gen_fits = merge_systema(gen_fits, gen_systema)
cel_fits = merge_systema(cel_fits, cel_systema)
print(f"  Triage (genetic): {gen_fits['triage_2d'].value_counts().to_dict()}")
print(f"  Triage (cellular): {cel_fits['triage_2d'].value_counts().to_dict()}")


# ── Power analysis ──
def compute_power(fits_df):
    fits_df = fits_df.copy()
    fits_df["actual_N"] = fits_df["n_cells"]
    fits_df["predicted_rho_at_actual_N"] = rho_model(
        fits_df["actual_N"].values.astype(float),
        fits_df["tau_sq"].values.astype(float))
    fits_df["is_powered_50"] = fits_df["actual_N"] >= fits_df["n_star_50"]
    fits_df["is_powered_70"] = fits_df["actual_N"] >= fits_df["n_star_70"]
    fits_df["is_powered_90"] = fits_df["actual_N"] >= fits_df["n_star_90"]
    return fits_df


gen_fits = compute_power(gen_fits)
cel_fits = compute_power(cel_fits)

fits_all = pd.concat([gen_fits, cel_fits], ignore_index=True)
fits_path = str(FITS_DIR / "fits_all_perturbations.csv")
fits_all.to_csv(fits_path, index=False)
print(f"\n  Total fits: {len(fits_all)} "
      f"({len(gen_fits)} genetic + {len(cel_fits)} cellular)")
print(f"  Median R2: {fits_all['fit_r2'].median():.4f}")
print(f"  Saved: {fits_path}")


# ══════════════════════════════════════════════════════════════════
# PILOT VALIDATION (leave-future-out)
# ══════════════════════════════════════════════════════════════════

print(f"\nSimulating pilot experiments (PILOT_NHALF_VALUES = "
      f"{PILOT_NHALF_VALUES[:3]}…{PILOT_NHALF_VALUES[-3:]})...")

fixed_all = pd.concat([gen_curves, cel_curves], ignore_index=True)
fixed_all = fixed_all[
    (fixed_all["n_mode"] == "fixed") &
    (fixed_all["gene_mode"] == "all_genes")
]
rho_col_pilot = ("sb_from_median" if "sb_from_median" in fixed_all.columns
                 else "sb_from_mean")
print(f"  Pilot using reliability column: {rho_col_pilot}")
for _ct, _g in fixed_all.groupby("context_type"):
    print(f"    {_ct}: {len(_g.drop_duplicates(['dataset', 'pert_id']))} units")

pilot_rows = []
for (ds, pert, ctx), grp in fixed_all.groupby(["dataset", "pert_id", "context_type"]):
    n_total = 2.0 * grp["n_half"].values.astype(float)
    rho_obs = grp[rho_col_pilot].values.astype(float)

    fit_full = fit_perturbation(n_total, rho_obs, min_points=MIN_POINTS)
    if fit_full is None:
        continue
    tau_full = fit_full["tau_sq"]

    for pilot_max_nhalf in PILOT_NHALF_VALUES:
        max_N = 2 * pilot_max_nhalf
        subset_mask = n_total <= max_N

        if subset_mask.sum() < MIN_POINTS:
            continue
        if subset_mask.sum() >= len(grp):
            continue  # need a future point for leave-future-out

        fit_pilot = fit_perturbation(n_total[subset_mask],
                                     rho_obs[subset_mask],
                                     min_points=MIN_POINTS)
        if fit_pilot is None or fit_pilot["tau_sq"] <= 0:
            continue

        pilot_rows.append({
            "dataset": ds, "pert": pert, "context_type": ctx,
            "pilot_max_nhalf": pilot_max_nhalf,
            "pilot_N": int(max_N),
            "tau_full": tau_full,
            "tau_pilot": fit_pilot["tau_sq"],
            "nstar_full": float(n_star(tau_full, 0.5)),
            "nstar_pilot": float(n_star(fit_pilot["tau_sq"], 0.5)),
        })

pilot_df = pd.DataFrame(pilot_rows)
print(f"  {len(pilot_df)} pilot validation rows")

for pN in sorted(pilot_df["pilot_N"].unique()):
    sub = pilot_df[(pilot_df["pilot_N"] == pN) &
                   (pilot_df["tau_full"] > 0) & (pilot_df["tau_pilot"] > 0)]
    if len(sub) == 0:
        continue
    r = np.corrcoef(np.log10(sub["tau_full"]),
                    np.log10(sub["tau_pilot"]))[0, 1]
    ns = sub[(sub["nstar_full"] > 0) & (sub["nstar_full"] < 1e6) &
             (sub["nstar_pilot"] > 0) & (sub["nstar_pilot"] < 1e6)]
    if len(ns) > 0:
        fold = ns["nstar_pilot"] / ns["nstar_full"]
        w2x = 100 * ((fold >= 0.5) & (fold <= 2)).mean()
    else:
        w2x = 0
    print(f"    pilot_N={pN:>3d}: n={len(sub):>5d}, "
          f"tau2 corr={r:.3f}, within 2-fold={w2x:.1f}%")

pilot_path = str(FITS_DIR / "pilot_validation_extended.csv")
pilot_df.to_csv(pilot_path, index=False)
print(f"  Saved: {pilot_path}")

print("\n✓ Preprocessing 04 complete.")
