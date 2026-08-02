#!/usr/bin/env python3
"""10 — Quantity-within-RELIABLE sweep for the LINEAR model (certified pipeline). Produces the data behind
Figure 2e (scripts/06_figures/fig2e_reliable_quantity.py).

Reuses the EXACT certified linear fit (fit_linear_model_weighted from run_exhaustive_linear) and a VERBATIM
copy of its run_one scoring loop (same data load, pseudobulk, DEG, _lib metrics). ONLY the arm list differs:
random subsets of the RELIABLE training pool swept by size (10 draws each), plus full reliable / full all /
specific as references.

Paths are anchored via env vars (defaults to the local working tree):
  SCRELIABILITY_ROOT — repo root for the output CSV (default = repo root, two levels up)
Output : <PAPER>/intermediate/retraining/reliable_quantity_sweep.csv
"""
import os, sys, time
from pathlib import Path

PAPER = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))                   # _lib (this folder)

import numpy as np, pandas as pd, anndata as ad
from concurrent.futures import ProcessPoolExecutor, as_completed
from _lib import (parse_pert_genes, compute_pseudobulks, compute_wei_deg_idx, compute_ae_top_expr_idx,
    predict_linear_model, dense_mean, wei_pcc_delta_top100, wei_mse_top100, wei_common_degs,
    ae_pearson_dlt_top1000, centroid_accuracy_for_set, H5AD_DIR, H5AD_FILE, SPLITS_DIR)
from _lib import fit_linear_model_weighted, deterministic_subseed

INFO = ["Replogle_K562essential","Replogle_RPE1essential","Norman","Schmidt","Adamson","Replogle_exp6","Wessels"]
FRACS = [0.1,0.2,0.3,0.4,0.5,0.65,0.8]    # fraction of |reliable| for random reliable subsets
N_DRAW = 10
OUTDIR = PAPER / "intermediate" / "retraining"; OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUTDIR / "reliable_quantity_sweep.csv")

def sweep_build_arms(train_df, seed):
    perts = train_df.perturbation.tolist()
    ql = dict(zip(train_df.perturbation, train_df.quality_label))
    genuine = [p for p in perts if ql.get(p) == "genuine"]
    reliable = [p for p in perts if ql.get(p) in ("genuine","trivial")]
    arms = [("all",-1,perts,None),("specific",-1,genuine,None),("reliable",-1,reliable,None)]
    rng = np.random.default_rng(seed); nR = len(reliable)
    for f in FRACS:
        s = max(2, int(round(f*nR)))
        if s >= nR: continue
        for d in range(N_DRAW):
            arms.append((f"reliable_sub_{f:.2f}", d, list(rng.choice(reliable, size=s, replace=False)), None))
    return arms

def run_one(job):   # VERBATIM run_exhaustive_linear.run_one, only build_arms -> sweep_build_arms
    dataset, k = job; rows = []
    sp = SPLITS_DIR/f"{dataset}_split{k}.csv"
    if not sp.exists(): return rows
    sdf = pd.read_csv(sp)
    train_df = sdf[sdf.split == "train"]; test_df = sdf[sdf.split == "test"]
    if (train_df.quality_label == "genuine").sum() < 5 or (test_df.quality_label == "genuine").sum() < 5: return rows
    adata = ad.read_h5ad(H5AD_DIR/H5AD_FILE[dataset])
    gene_names = adata.var_names.tolist(); gene_to_col = {g:i for i,g in enumerate(gene_names)}
    ctrl = (adata.obs.perturbation == "control").to_numpy()
    if ctrl.sum() == 0: return rows
    ctrl_mean = dense_mean(adata[ctrl].X); top_expr = compute_ae_top_expr_idx(ctrl_mean)
    all_perts = train_df.perturbation.tolist() + test_df.perturbation.tolist()
    delta_dict, expr_dict = compute_pseudobulks(adata, all_perts, ctrl_mean)
    train_df = train_df[train_df.perturbation.isin(delta_dict)]; test_df = test_df[test_df.perturbation.isin(delta_dict)]
    test_single = test_df[test_df.perturbation.apply(lambda p: len(parse_pert_genes(p)) == 1)]
    tq = dict(zip(test_single.perturbation, test_single.quality_label))
    deg = compute_wei_deg_idx(adata, test_single.perturbation.tolist(), gene_to_col); del adata
    test_perts = test_single.perturbation.tolist()
    pool = {"specific":[p for p in test_perts if tq[p] == "genuine"],
            "reliable":[p for p in test_perts if tq[p] in ("genuine","trivial")],
            "all":list(test_perts)}
    for arm, sub, arm_train, weights in sweep_build_arms(train_df, deterministic_subseed(dataset, k)):
        if len(arm_train) < 2: continue
        td = np.array([delta_dict[p] for p in arm_train]); te = np.array([expr_dict[p] for p in arm_train])
        lm = fit_linear_model_weighted(te, td, arm_train, gene_names, weights)
        if lm is None: continue
        preds = {p:predict_linear_model(lm,p) for p in test_perts}; preds = {p:v for p,v in preds.items() if v is not None}
        if len(preds) < 2: continue
        ca = {es:{} for es in pool}
        for es, plist in pool.items():
            pl = [p for p in plist if p in preds]
            if len(pl) >= 2:
                pm = np.array([preds[p] for p in pl]); gm = np.array([delta_dict[p] for p in pl])
                ca[es] = dict(zip(pl, centroid_accuracy_for_set(pm,gm)))
        for p, pr in preds.items():
            gt = delta_dict[p]; dg = deg.get(p)
            rows.append(dict(dataset=dataset, split_seed=k, arm=arm, subset_id=sub, perturbation=p,
                quality_label=tq[p], n_train=len(arm_train),
                wei_pcc_delta_top100=wei_pcc_delta_top100(pr,gt,dg),
                wei_mse_top100=wei_mse_top100(pr,gt,dg),
                wei_common_degs=wei_common_degs(pr,dg),
                ae_pearson_dlt_top1000=ae_pearson_dlt_top1000(pr,gt,top_expr),
                ca_specific=float(ca["specific"].get(p,np.nan)),
                ca_reliable=float(ca["reliable"].get(p,np.nan)),
                ca_all=float(ca["all"].get(p,np.nan))))
    print(f"  done {dataset} split{k}: {len(rows)} rows", flush=True)
    return rows

def main():
    jobs = [(ds,k) for ds in INFO for k in range(5)]
    print(f"{len(jobs)} jobs (linear, certified fit); FRACS={FRACS} x {N_DRAW} draws", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one,j) for j in jobs]):
            rows.extend(fut.result())
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n{len(rows)} rows in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
