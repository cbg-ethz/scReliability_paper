#!/usr/bin/env python3
"""
build_edist_triage.py
=====================
Build the energy-distance triage tables, combining E-distance reliability (rho_E)
with the per-perturbation specificity (phi = cos^2(theta)) from the main pipeline.

Triage rule (thresholds are set by RHO_E_THRESH and PHI_THRESH below):

  rho_E <  RHO_E_THRESH               -> Unreliable
  rho_E >= RHO_E_THRESH, phi >= 0.5   -> Shared
  rho_E >= RHO_E_THRESH, phi <  0.5   -> Specific

RHO_E_THRESH is 0, the point at which a perturbation's effect equals its
within-perturbation dispersion, and is the E-distance analogue of Pearson rho = 0.5.

Outputs in intermediate/edist/:
  combined_genetic_2d_edist.csv       triage for genetic perturbations
  combined_cellular_2d_edist.csv      triage for cellular contexts
  pooled_quality_summary_edist.csv    pooled across 29 datasets
  edist_vs_pearson_validation.csv     per-dataset agreement with the Pearson triage
  wei_genetic_merged_2d_edist.csv     triage merged with Wei rankings
  ae_panel_a_merged_2d_edist.csv      triage merged with Ahlmann-Eltze panel A
  ae_panel_c_merged_2d_edist.csv      triage merged with Ahlmann-Eltze panel C
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

import os as _os, pathlib as _pathlib  # portability: repo-root anchor
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
ROOT = Path("" + _ROOT + "")
EDIST = ROOT / "intermediate" / "edist"
MAIN = ROOT / "intermediate"

PHI_THRESH = 0.5
# rho_E threshold = 0: principled analog of Pearson rho = 0.5 (signal:noise = 1:1).
# rho_E = 1 - E_within/E_between, so rho_E = 0 <=> E_within = E_between (signal == noise).
RHO_E_THRESH = 0.0


def load_e_reliability():
    """Load e_reliability_all.csv and keep only full-N rows (n_half == -1)."""
    df = pd.read_csv(EDIST / "e_reliability_all.csv")
    full = df[df["n_half"] == -1].copy()
    full = full.rename(columns={"e_within_med": "e_within_full",
                                "e_between": "e_between"})
    cols = ["dataset", "condition", "perturbation",
            "n_cells", "e_within_full", "e_between", "rho_E"]
    return full[cols]


def build_edist_combined(pearson_path: Path, edist_full: pd.DataFrame,
                         label: str) -> pd.DataFrame:
    """
    Merge the Pearson combined_(genetic|cellular)_2d table with E-distance rho_E,
    apply triage rule.
    """
    pearson = pd.read_csv(pearson_path)
    # For cellular tables `condition` means "<context>|<perturbation>" on the Pearson side but the
    # context alone on the E-distance side, so the cellular branch joins on `context`; genetic joins
    # on `condition`.
    right = edist_full
    merge_on = ["dataset", "perturbation"]
    if "context" in pearson.columns and pearson["context"].notna().any():
        right = edist_full.rename(columns={"condition": "context"})
        merge_on.append("context")
    elif "condition" in pearson.columns and "condition" in edist_full.columns:
        merge_on.append("condition")
    merged = pearson.merge(right, on=merge_on, how="left", suffixes=("", "_edist"))

    # Report unmatched rows explicitly: `NaN >= RHO_E_THRESH` is False, so an unmatched row would
    # otherwise be labelled a confident "Unreliable".
    n_miss = int(merged["rho_E"].isna().sum())
    if n_miss:
        missing = merged.loc[merged["rho_E"].isna(), "dataset"].value_counts().to_dict()
        print(f"[{label}] WARNING {n_miss}/{len(merged)} rows have no rho_E match "
              f"(join key {merge_on}); by dataset: {missing}", flush=True)
        if n_miss == len(merged):
            raise RuntimeError(
                f"[{label}] the E-distance join matched 0 rows on {merge_on}; refusing to write a "
                f"table in which every perturbation would be labelled Unreliable.")

    # triage
    rel = merged["rho_E"] >= RHO_E_THRESH
    phi = merged["cos_sim"] ** 2  # standard cos^2(theta) specificity
    merged["phi"] = phi
    merged["rho_E_thresh"] = RHO_E_THRESH
    merged["phi_thresh"] = PHI_THRESH

    triage = np.where(
        ~rel,
        "Unreliable",
        np.where(phi >= PHI_THRESH, "Shared", "Specific"),
    )
    merged["triage_2d_edist"] = triage
    merged["triage_2d"] = triage   # overwrite for downstream figure scripts

    # Convenience flags
    merged["reliable_edist"] = rel
    merged["shared_edist"] = (rel & (phi >= PHI_THRESH))
    merged["specific_edist"] = (rel & (phi < PHI_THRESH))

    out = EDIST / f"combined_{label}_2d_edist.csv"
    merged.to_csv(out, index=False)
    print(f"[{label}] {len(merged)} rows -> {out}", flush=True)
    return merged


def build_pooled_summary(edist_gen, edist_cel):
    """Build the pooled per-dataset summary."""
    combo = pd.concat([edist_gen.assign(category="genetic"),
                       edist_cel.assign(category="cellular")],
                      ignore_index=True)

    def summary(g):
        return pd.Series({
            "n_total": len(g),
            "n_unreliable": int((g["triage_2d_edist"] == "Unreliable").sum()),
            "n_shared":     int((g["triage_2d_edist"] == "Shared").sum()),
            "n_specific":   int((g["triage_2d_edist"] == "Specific").sum()),
            "frac_unreliable": float((g["triage_2d_edist"] == "Unreliable").mean()),
            "frac_shared":     float((g["triage_2d_edist"] == "Shared").mean()),
            "frac_specific":   float((g["triage_2d_edist"] == "Specific").mean()),
            "category": g["category"].iloc[0],
        })

    pooled = combo.groupby("dataset").apply(summary).reset_index()
    out = EDIST / "pooled_quality_summary_edist.csv"
    pooled.to_csv(out, index=False)
    print(f"[pooled] {len(pooled)} datasets -> {out}", flush=True)

    overall = {
        "n_total": int(pooled["n_total"].sum()),
        "n_unreliable": int(pooled["n_unreliable"].sum()),
        "n_shared":     int(pooled["n_shared"].sum()),
        "n_specific":   int(pooled["n_specific"].sum()),
    }
    overall["frac_unreliable"] = overall["n_unreliable"] / overall["n_total"]
    overall["frac_shared"]     = overall["n_shared"]     / overall["n_total"]
    overall["frac_specific"]   = overall["n_specific"]   / overall["n_total"]
    print(f"[pooled overall E-distance]  n={overall['n_total']:,}")
    print(f"   unreliable {overall['frac_unreliable']:.3f} | "
          f"shared {overall['frac_shared']:.3f} | "
          f"specific {overall['frac_specific']:.3f}")
    return pooled, overall


def edist_vs_pearson_validation(edist_gen, edist_cel):
    """Per-dataset Spearman correlation and triage agreement against the Pearson triage."""
    combo = pd.concat([edist_gen, edist_cel], ignore_index=True)
    has = combo[["dataset", "perturbation",
                  "sb_from_median_all_genes",
                  "rho_E", "cos_sim",
                  "triage_2d_edist"]].copy()
    has = has.dropna(subset=["sb_from_median_all_genes", "rho_E"])

    # Pearson triage
    pearson_rel = has["sb_from_median_all_genes"] >= 0.5
    pearson_phi = has["cos_sim"] ** 2
    pearson_triage = np.where(
        ~pearson_rel, "Unreliable",
        np.where(pearson_phi >= 0.5, "Shared", "Specific"),
    )
    has["triage_2d_pearson"] = pearson_triage

    rows = []
    for ds, sub in has.groupby("dataset"):
        sp_rho, _ = spearmanr(sub["sb_from_median_all_genes"], sub["rho_E"])
        try:
            kappa = cohen_kappa_score(sub["triage_2d_pearson"], sub["triage_2d_edist"])
        except Exception:
            kappa = np.nan
        agree = float((sub["triage_2d_pearson"] == sub["triage_2d_edist"]).mean())
        rows.append(dict(dataset=ds, n=len(sub),
                          spearman_rho=sp_rho,
                          triage_agreement=agree, cohen_kappa=kappa))

    val = pd.DataFrame(rows)
    out = EDIST / "edist_vs_pearson_validation.csv"
    val.to_csv(out, index=False)
    print(f"[validation] -> {out}", flush=True)
    print(f"  median Spearman(rho) = {val.spearman_rho.median():.3f}")
    print(f"  median triage agreement = {val.triage_agreement.median():.3f}")
    print(f"  median Cohen kappa = {val.cohen_kappa.median():.3f}")

    # Pooled stats
    sp_rho_pool, _ = spearmanr(has["sb_from_median_all_genes"], has["rho_E"])
    kappa_pool = cohen_kappa_score(has["triage_2d_pearson"], has["triage_2d_edist"])
    print(f"  POOLED Spearman(rho) = {sp_rho_pool:.3f}")
    print(f"  POOLED Cohen kappa = {kappa_pool:.3f}")
    return val, dict(
        spearman_rho_pooled=float(sp_rho_pool),
        kappa_pooled=float(kappa_pool),
    )


def merge_wei_ae(edist_gen):
    """Re-merge Wei and AE rank tables with triage.

    Use a deduplicated (dataset, perturbation) -> triage_2d_edist map so that
    Pandas indexing does not complain about duplicate index rows.
    """
    edist_map_df = (edist_gen[["dataset", "perturbation", "triage_2d_edist"]]
                 .drop_duplicates(subset=["dataset", "perturbation"]))

    def _swap_triage(src_csv, out_name):
        if not (MAIN / src_csv).exists():
            return
        df = pd.read_csv(MAIN / src_csv)
        df["triage_2d_pearson"] = df["triage_2d"]
        df = df.merge(edist_map_df, on=["dataset", "perturbation"], how="left")
        df["triage_2d"] = df["triage_2d_edist"].fillna(df["triage_2d_pearson"])
        df = df.drop(columns=["triage_2d_edist"])
        out = EDIST / out_name
        df.to_csv(out, index=False)
        print(f"[{out_name}] {len(df)} rows -> {out}", flush=True)

    _swap_triage("wei_genetic_merged_2d.csv",        "wei_genetic_merged_2d_edist.csv")
    _swap_triage("wei_genetic_combo_merged_2d.csv",  "wei_genetic_combo_merged_2d_edist.csv")
    _swap_triage("ae_panel_a_merged_2d.csv",         "ae_panel_a_merged_2d_edist.csv")
    _swap_triage("ae_panel_c_merged_2d.csv",         "ae_panel_c_merged_2d_edist.csv")
    _swap_triage("ae_doubles_merged_2d.csv",         "ae_doubles_merged_2d_edist.csv")


def main():
    print("Loading E-distance e_reliability_all.csv...", flush=True)
    edist_full = load_e_reliability()
    print(f"  {len(edist_full)} (pert, condition) rows at full N", flush=True)

    print("\nBuilding E-distance combined_genetic_2d...", flush=True)
    edist_gen = build_edist_combined(MAIN / "combined_genetic_2d.csv", edist_full, "genetic")

    print("\nBuilding E-distance combined_cellular_2d...", flush=True)
    edist_cel = build_edist_combined(MAIN / "combined_cellular_2d.csv", edist_full, "cellular")

    print("\nBuilding pooled summary...", flush=True)
    pooled, overall = build_pooled_summary(edist_gen, edist_cel)

    print("\nValidation E-distance vs Pearson...", flush=True)
    val, pooled_stats = edist_vs_pearson_validation(edist_gen, edist_cel)

    print("\nMerging into Wei/AE rank tables...", flush=True)
    merge_wei_ae(edist_gen)

    # Save pooled-stats JSON for use in figures/manuscript
    import json
    overall["pooled_stats"] = pooled_stats
    with open(EDIST / "summary_stats_edist.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n[done] summary -> {EDIST / 'summary_stats_edist.json'}")


if __name__ == "__main__":
    main()
