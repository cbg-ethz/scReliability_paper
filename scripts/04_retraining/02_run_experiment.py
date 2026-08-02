"""
02_run_experiment.py — run all arms × models × splits × subsets, score every
test perturbation. Multiprocessed across (dataset, split_seed) jobs.

For each (dataset, split):
  1. Load split, h5ad, compute pseudobulks once.
  2. For each arm in {all, genuine} and each model in {trainMean, linearModel}:
       fit once, score every test perturbation.
  3. For each random_matched and random_nongenuine subset_id in [0, S):
       fit once per model, score every test perturbation.

Outputs:
  results/retraining_quality_filtering/
    per_perturbation_scores.csv
    skipped_cases.csv
    config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    DATASETS, H5AD_FILE, H5AD_DIR, RESULTS_DIR, SPLITS_DIR,
    MIN_CELLS_PER_PERT, N_TOP_DEGS_WEI, N_TOP_EXPR_AE,
    load_quality_table, parse_pert_genes, classify_pert_type,
    compute_pseudobulks, compute_wei_deg_idx, compute_ae_top_expr_idx,
    fit_trainmean, fit_linear_model, predict_linear_model,
    fit_additive_model, predict_additive_model,
    wei_pcc_delta_top100, wei_mse_top100, wei_common_degs,
    ae_l2_top1000, ae_pearson_dlt_top1000, ae_r2_raw_top1000,
    centroid_accuracy_for_set,
    stratified_subset, deterministic_subseed, dump_config, dense_mean,
    load_rho_cos_lookup, derive_quality_label_from_thresh,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-(dataset, split) job
# ──────────────────────────────────────────────────────────────────────────────
def run_one(args_tuple):
    (dataset, split_seed_idx, n_random_matched,
     min_genuine_train, min_genuine_test, min_nongenuine_train,
     rho_thresh, cos_thresh) = args_tuple

    rows: list[dict] = []
    skip_rows: list[dict] = []

    # Load split
    split_path = SPLITS_DIR / f"{dataset}_split{split_seed_idx}.csv"
    if not split_path.exists():
        return rows, skip_rows + [dict(dataset=dataset, split_seed=split_seed_idx,
                                       reason="split_csv_missing")]
    sdf = pd.read_csv(split_path)

    # Threshold-sensitivity override: re-derive `quality_label` from rho/cos
    # using user-supplied thresholds. Splits stratification stays at default
    # threshold (apples-to-apples comparison across thresholds).
    if rho_thresh is not None and cos_thresh is not None:
        lookup = load_rho_cos_lookup()
        lookup = lookup[lookup.dataset == dataset][["perturbation", "rho", "cos"]]
        sdf = sdf.drop(columns=[c for c in ["quality_label"] if c in sdf.columns])
        sdf = sdf.merge(lookup, on="perturbation", how="left")
        sdf["quality_label"] = sdf.apply(
            lambda r: derive_quality_label_from_thresh(
                r.rho, r.cos, rho_thresh, cos_thresh),
            axis=1,
        )

    train_df = sdf[sdf["split"] == "train"].copy()
    test_df  = sdf[sdf["split"] == "test"].copy()

    n_train_total = len(train_df)
    n_test_total  = len(test_df)
    n_gen_tr  = (train_df["quality_label"] == "genuine").sum()
    n_gen_te  = (test_df ["quality_label"] == "genuine").sum()
    n_ng_tr   = (train_df["quality_label"] != "genuine").sum()

    base_skip = dict(dataset=dataset, split_seed=split_seed_idx,
                     n_train_total=n_train_total, n_test_total=n_test_total,
                     n_genuine_train=int(n_gen_tr), n_genuine_test=int(n_gen_te),
                     n_nongenuine_train=int(n_ng_tr))

    if n_gen_tr < min_genuine_train:
        skip_rows.append(dict(reason=f"n_genuine_train={n_gen_tr}<{min_genuine_train}",
                              **base_skip))
        return rows, skip_rows
    if n_gen_te < min_genuine_test:
        skip_rows.append(dict(reason=f"n_genuine_test={n_gen_te}<{min_genuine_test}",
                              **base_skip))
        return rows, skip_rows

    # Load h5ad
    h5_path = H5AD_DIR / H5AD_FILE[dataset]
    if not h5_path.exists():
        skip_rows.append(dict(reason="h5ad_missing", **base_skip))
        return rows, skip_rows

    t0 = time.time()
    adata = ad.read_h5ad(h5_path)
    gene_names = adata.var_names.tolist()
    gene_to_col = {g: i for i, g in enumerate(gene_names)}

    ctrl_mask = (adata.obs["perturbation"] == "control").to_numpy()
    if ctrl_mask.sum() == 0:
        skip_rows.append(dict(reason="no_control_cells", **base_skip))
        return rows, skip_rows
    ctrl_mean = dense_mean(adata[ctrl_mask].X)
    top_expr_idx = compute_ae_top_expr_idx(ctrl_mean)

    # Pseudobulks for every perturbation in this split
    all_perts = train_df["perturbation"].tolist() + test_df["perturbation"].tolist()
    delta_dict, expr_dict = compute_pseudobulks(adata, all_perts, ctrl_mean)

    # Drop perts that didn't pass the cell-count threshold
    train_df = train_df[train_df["perturbation"].isin(delta_dict)].reset_index(drop=True)
    test_df  = test_df [test_df ["perturbation"].isin(delta_dict)].reset_index(drop=True)

    if (test_df["quality_label"] == "genuine").sum() < min_genuine_test:
        skip_rows.append(dict(reason="n_genuine_test_after_cellfilter_too_low",
                              **base_skip))
        return rows, skip_rows

    # Wei DEG indices for every test pert (one rank_genes_groups call)
    wei_deg_idx = compute_wei_deg_idx(adata, test_df["perturbation"].tolist(),
                                      gene_to_col)

    # Free the cell-level matrix; we only need the pseudobulks now
    del adata

    # Genuine-train pool & non-genuine-train pool (for random subsets)
    genuine_train_df    = train_df[train_df["quality_label"] == "genuine"]
    nongenuine_train_df = train_df[train_df["quality_label"] != "genuine"]

    n_genuine_train = len(genuine_train_df)

    # Define arm specs: list of (arm_name, subset_id, train_perts_list)
    # arm_name ∈ {all, genuine, random_matched, random_nongenuine}; subset_id is
    # an int for random_* arms (else NA-coded as -1).
    arm_specs: list[tuple[str, int, list[str]]] = [
        ("all",     -1, train_df["perturbation"].tolist()),
        ("genuine", -1, genuine_train_df["perturbation"].tolist()),
    ]

    # random_matched: stratified by pert_type from the FULL train pool
    for s in range(n_random_matched):
        seed = deterministic_subseed(dataset, split_seed_idx, "random_matched", s)
        chosen = stratified_subset(train_df[["perturbation", "pert_type"]],
                                   genuine_train_df[["perturbation", "pert_type"]],
                                   seed=seed)
        if chosen is None:
            skip_rows.append(dict(reason=f"random_matched_subset_{s}_pool_too_small",
                                  **base_skip))
            continue
        arm_specs.append(("random_matched", s, chosen))

    # random_nongenuine: stratified by pert_type from the NON-genuine train pool only.
    # If non-genuine pool is empty (exp7/exp8) or insufficient, skip the whole arm.
    if len(nongenuine_train_df) >= min_nongenuine_train:
        # Quick feasibility check: per pert_type, must have at least as many
        # non-genuine as genuine_train requires.
        feasible = True
        for ptype, n in genuine_train_df["pert_type"].value_counts().items():
            if (nongenuine_train_df["pert_type"] == ptype).sum() < n:
                feasible = False
                break
        if feasible:
            for s in range(n_random_matched):
                seed = deterministic_subseed(dataset, split_seed_idx,
                                             "random_nongenuine", s)
                chosen = stratified_subset(
                    nongenuine_train_df[["perturbation", "pert_type"]],
                    genuine_train_df[["perturbation", "pert_type"]],
                    seed=seed,
                )
                if chosen is None:
                    skip_rows.append(dict(
                        reason=f"random_nongenuine_subset_{s}_pool_too_small",
                        **base_skip))
                    continue
                arm_specs.append(("random_nongenuine", s, chosen))
        else:
            skip_rows.append(dict(reason="random_nongenuine_pool_strata_insufficient",
                                  **base_skip))
    else:
        skip_rows.append(dict(reason="no_nongenuine_train_pool", **base_skip))

    # ── Score every arm × model ───────────────────────────────────────────────
    test_perts = test_df["perturbation"].tolist()
    test_quality = dict(zip(test_df["perturbation"], test_df["quality_label"]))
    test_ptype   = dict(zip(test_df["perturbation"], test_df["pert_type"]))

    def _row(model_name, arm_name, subset_id, n_arm, p, pred, gt, deg, cacc):
        return dict(
            dataset=dataset, split_seed=split_seed_idx, model=model_name,
            arm=arm_name, subset_id=subset_id,
            perturbation=p, quality_label=test_quality[p],
            pert_type=test_ptype[p],
            n_train_perturbations=n_arm,
            n_test_perturbations=len(test_perts),
            wei_pcc_delta_top100=wei_pcc_delta_top100(pred, gt, deg),
            wei_mse_top100=wei_mse_top100(pred, gt, deg),
            wei_common_degs=wei_common_degs(pred, deg),
            centroid_accuracy=cacc,
            ae_l2_top1000=ae_l2_top1000(pred, gt, top_expr_idx),
            ae_pearson_dlt_top1000=ae_pearson_dlt_top1000(pred, gt, top_expr_idx),
            ae_r2_raw_top1000=ae_r2_raw_top1000(pred, gt, ctrl_mean, top_expr_idx),
        )

    for arm_name, subset_id, arm_train in arm_specs:
        n_arm = len(arm_train)
        if n_arm < 2:
            continue
        train_deltas = np.array([delta_dict[p] for p in arm_train])
        train_exprs  = np.array([expr_dict[p]  for p in arm_train])

        # Models: trainMean (any cardinality), linearModel (singles only,
        # AE-strict), additive_model (doubles only, requires component singles).
        tm_pred = fit_trainmean(train_deltas)
        lm = fit_linear_model(train_exprs, train_deltas, arm_train, gene_names)
        am = fit_additive_model(arm_train, delta_dict)

        # Collect each model's predictions over its applicable test perturbations
        # (trainMean: all; linearModel: singles; additive_model: doubles).
        preds_by_model: dict[str, dict[str, np.ndarray]] = {
            "trainMean": {}, "linearModel": {}, "additive_model": {}}
        for p in test_perts:
            preds_by_model["trainMean"][p] = tm_pred
            if test_ptype[p] == "single" and lm is not None:
                pr = predict_linear_model(lm, p)
                if pr is not None:
                    preds_by_model["linearModel"][p] = pr
            elif test_ptype[p] == "double":
                pr = predict_additive_model(am, p)
                if pr is not None:
                    preds_by_model["additive_model"][p] = pr

        # Score each model. Centroid accuracy (Systema) is a retrieval metric over
        # the full set of test perturbations the model predicts, so it is computed
        # once per (arm, model) and then attached per perturbation.
        for model_name, preds in preds_by_model.items():
            plist = list(preds.keys())
            if not plist:
                continue
            if len(plist) >= 2:
                pred_mat = np.array([preds[p] for p in plist])
                gt_mat   = np.array([delta_dict[p] for p in plist])
                cacc_map = dict(zip(plist, centroid_accuracy_for_set(pred_mat, gt_mat)))
            else:
                cacc_map = {plist[0]: float("nan")}
            for p in plist:
                rows.append(_row(model_name, arm_name, subset_id, n_arm, p,
                                 preds[p], delta_dict[p], wei_deg_idx.get(p),
                                 float(cacc_map[p])))

    elapsed = time.time() - t0
    print(f"  ✓ {dataset} split{split_seed_idx}: {len(rows):,} rows in {elapsed:.1f}s",
          flush=True)
    return rows, skip_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits",             type=int, default=5)
    ap.add_argument("--n-random-matched",     type=int, default=50)
    ap.add_argument("--min-genuine-train",    type=int, default=5)
    ap.add_argument("--min-genuine-test",     type=int, default=5)
    ap.add_argument("--min-nongenuine-train", type=int, default=5)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of datasets (default: all 9)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--output-dir", default=str(RESULTS_DIR))
    ap.add_argument("--rho-thresh", type=float, default=None,
                    help="If set with --cos-thresh, override quality_label "
                         "using these thresholds (sensitivity sweep)")
    ap.add_argument("--cos-thresh", type=float, default=None,
                    help="If set with --rho-thresh, override quality_label")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DATASETS

    jobs = [
        (ds, k, args.n_random_matched, args.min_genuine_train,
         args.min_genuine_test, args.min_nongenuine_train,
         args.rho_thresh, args.cos_thresh)
        for ds in datasets for k in range(args.n_splits)
    ]
    print(f"Total jobs: {len(jobs)}  (workers={args.workers})")

    all_rows, all_skip = [], []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            rows, skip_rows = fut.result()
            all_rows.extend(rows)
            all_skip.extend(skip_rows)

    elapsed = time.time() - t_start
    print(f"\nAll jobs done in {elapsed/60:.1f} min  ({len(all_rows):,} rows)")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_perturbation_scores.csv", index=False)
    print(f"  → {out_dir/'per_perturbation_scores.csv'}")

    sk = pd.DataFrame(all_skip)
    sk.to_csv(out_dir / "skipped_cases.csv", index=False)
    print(f"  → {out_dir/'skipped_cases.csv'}  ({len(sk)} rows)")

    cfg = vars(args) | {"step": "02_run_experiment", "elapsed_sec": elapsed}
    dump_config(cfg, out_dir)


if __name__ == "__main__":
    main()
