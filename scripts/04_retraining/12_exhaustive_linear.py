#!/usr/bin/env python3
"""Exhaustive linear-model retraining strategies (filter vs weight) — singles eval.

Arms (all on the AE reduced-rank linear model; trained on each arm's perturbations,
weighted variants weight the perturbation dimension in the ridge solve):
  all                         train on all train perts, uniform   (= original 'all')
  specific                    filter to specific (genuine)
  reliable                    filter to reliable (genuine + trivial/shared)
  random_matched              random subset size-matched to specific (control)
  weight_rho                  train on all, weight by reliability rho
  weight_spec                 train on all, weight by specificity (1 - phi); unreliable -> 0
  weight_rho_spec             train on all, weight by rho * (1 - phi)
  reliable_then_weight_spec   filter to reliable, then weight by (1 - phi)

Evaluated on SINGLE held-out test perturbations, 4 metrics (PCC-delta, MSE,
Common-DEGs, centroid accuracy), centroid accuracy computed per eval set
(specific / reliable / all) since it depends on the candidate pool.
"""
import os, sys, time
from pathlib import Path

# Anchored on the repository, like every other script here; override with SCRELIABILITY_ROOT.
os.environ.setdefault("SCRELIABILITY_ROOT", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import anndata as ad

from _lib import (
    K_PCA, RIDGE_LAMBDA, N_TOP_DEGS_WEI,
    DATASETS, H5AD_DIR, H5AD_FILE, SPLITS_DIR, QUALITY_GEN,
    parse_pert_genes, compute_pseudobulks, compute_wei_deg_idx,
    compute_ae_top_expr_idx, predict_linear_model, dense_mean,
    wei_pcc_delta_top100, wei_mse_top100, wei_common_degs,
    ae_pearson_dlt_top1000, centroid_accuracy_for_set, deterministic_subseed,
)

# Supplementary Figure 5 reads this table from intermediate/retraining/.
OUT_DIR = os.path.join(os.environ["SCRELIABILITY_ROOT"], "intermediate", "retraining")
N_RANDOM = 10


def fit_linear_model_weighted(train_exprs, train_deltas, train_perts, gene_names, weights=None):
    """AE reduced-rank linear model with optional per-perturbation weights on the
    ridge solve. weights=None -> identical to the unweighted _lib.fit_linear_model."""
    n_perts, n_genes = train_deltas.shape
    if n_perts < 2:
        return None
    w = np.ones(n_perts) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    if w.sum() <= 0:
        return None
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    X_train = train_exprs.T
    X_c = X_train - X_train.mean(axis=0)[None, :]
    K = min(K_PCA, n_perts - 1, n_genes - 1)
    if K < 1:
        return None
    try:
        U, S, _ = np.linalg.svd(X_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    G = U[:, :K] * S[:K][None, :]
    b = (w[:, None] * train_deltas).sum(0) / w.sum()          # weighted per-gene center
    pert_to_idx = {p: i for i, p in enumerate(train_perts)}
    P_rows, valid = [], []
    for p in train_perts:
        gidx = [gene_to_idx[g] for g in parse_pert_genes(p) if g in gene_to_idx]
        if not gidx:
            continue
        P_rows.append(G[gidx, :].mean(0)); valid.append(pert_to_idx[p])
    if len(P_rows) < K + 1:
        return None
    P = np.array(P_rows)
    wv = w[valid]
    if wv.sum() <= 0:
        return None
    wv = wv * (len(wv) / wv.sum())                            # normalize mean 1
    Y_c = train_deltas.T[:, valid] - b[:, None]
    GtG  = G.T @ G + RIDGE_LAMBDA * np.eye(K)
    PtP  = P.T @ (P * wv[:, None]) + RIDGE_LAMBDA * np.eye(K)  # P' diag(w) P + λI
    GtYP = (G.T @ Y_c) @ (P * wv[:, None])                     # G' Y_c diag(w) P
    try:
        W = np.linalg.solve(GtG, GtYP) @ np.linalg.inv(PtP)
    except np.linalg.LinAlgError:
        return None
    return {"G": G, "W": W, "b": b, "gene_to_idx": gene_to_idx, "K": K}


def build_arms(train_df, qinfo, seed):
    perts = train_df.perturbation.tolist()
    ql = dict(zip(train_df.perturbation, train_df.quality_label))
    genuine  = [p for p in perts if ql.get(p) == "genuine"]
    reliable = [p for p in perts if ql.get(p) in ("genuine", "trivial")]
    def rho(p):
        r = qinfo.get(p, (np.nan, np.nan))[0]
        return float(np.clip(r, 0, 1)) if np.isfinite(r) else 0.0
    def spec(p):
        ph = qinfo.get(p, (np.nan, np.nan))[1]
        return float(1 - ph) if np.isfinite(ph) else 0.0
    arms = [
        ("all", -1, perts, None),
        ("specific", -1, genuine, None),
        ("reliable", -1, reliable, None),
        ("weight_rho", -1, perts, [rho(p) for p in perts]),
        ("weight_spec", -1, perts, [spec(p) for p in perts]),
        ("weight_rho_spec", -1, perts, [rho(p) * spec(p) for p in perts]),
        ("reliable_then_weight_spec", -1, reliable, [spec(p) for p in reliable]),
    ]
    rng = np.random.default_rng(seed)
    if len(genuine) >= 2 and len(perts) >= len(genuine):
        for s in range(N_RANDOM):
            arms.append(("random_matched", s,
                         rng.choice(perts, size=len(genuine), replace=False).tolist(), None))
    return arms


def run_one(job):
    dataset, k, qinfo = job
    rows = []
    sp = SPLITS_DIR / f"{dataset}_split{k}.csv"
    if not sp.exists():
        return rows
    sdf = pd.read_csv(sp)
    train_df = sdf[sdf.split == "train"]; test_df = sdf[sdf.split == "test"]
    if (train_df.quality_label == "genuine").sum() < 5 or (test_df.quality_label == "genuine").sum() < 5:
        return rows
    adata = ad.read_h5ad(H5AD_DIR / H5AD_FILE[dataset])
    gene_names = adata.var_names.tolist()
    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    ctrl = (adata.obs.perturbation == "control").to_numpy()
    if ctrl.sum() == 0:
        return rows
    ctrl_mean = dense_mean(adata[ctrl].X)
    top_expr = compute_ae_top_expr_idx(ctrl_mean)
    all_perts = train_df.perturbation.tolist() + test_df.perturbation.tolist()
    delta_dict, expr_dict = compute_pseudobulks(adata, all_perts, ctrl_mean)
    train_df = train_df[train_df.perturbation.isin(delta_dict)]
    test_df  = test_df[test_df.perturbation.isin(delta_dict)]
    # single test perts only (linear model target), with their quality
    test_single = test_df[test_df.perturbation.apply(lambda p: len(parse_pert_genes(p)) == 1)]
    tq = dict(zip(test_single.perturbation, test_single.quality_label))
    deg = compute_wei_deg_idx(adata, test_single.perturbation.tolist(), gene_to_col)
    del adata
    test_perts = test_single.perturbation.tolist()
    # eval-set membership
    pool = {
        "specific": [p for p in test_perts if tq[p] == "genuine"],
        "reliable": [p for p in test_perts if tq[p] in ("genuine", "trivial")],
        "all":      list(test_perts),
    }
    for arm, sub, arm_train, weights in build_arms(train_df, qinfo, deterministic_subseed(dataset, k, "exhaustive_linear")):
        if len(arm_train) < 2:
            continue
        td = np.array([delta_dict[p] for p in arm_train])
        te = np.array([expr_dict[p] for p in arm_train])
        lm = fit_linear_model_weighted(te, td, arm_train, gene_names, weights)
        if lm is None:
            continue
        preds = {}
        for p in test_perts:
            pr = predict_linear_model(lm, p)
            if pr is not None:
                preds[p] = pr
        if len(preds) < 2:
            continue
        # centroid accuracy per eval-set pool
        ca = {es: {} for es in pool}
        for es, plist in pool.items():
            pl = [p for p in plist if p in preds]
            if len(pl) >= 2:
                pm = np.array([preds[p] for p in pl]); gm = np.array([delta_dict[p] for p in pl])
                ca[es] = dict(zip(pl, centroid_accuracy_for_set(pm, gm)))
        for p, pr in preds.items():
            gt = delta_dict[p]; dg = deg.get(p)
            rows.append(dict(
                dataset=dataset, split_seed=k, arm=arm, subset_id=sub,
                perturbation=p, quality_label=tq[p],
                n_train=len(arm_train),
                wei_pcc_delta_top100=wei_pcc_delta_top100(pr, gt, dg),
                wei_mse_top100=wei_mse_top100(pr, gt, dg),
                wei_common_degs=wei_common_degs(pr, dg),
                ae_pearson_dlt_top1000=ae_pearson_dlt_top1000(pr, gt, top_expr),
                ca_specific=float(ca["specific"].get(p, np.nan)),
                ca_reliable=float(ca["reliable"].get(p, np.nan)),
                ca_all=float(ca["all"].get(p, np.nan)),
            ))
    print(f"  done {dataset} split{k}: {len(rows)} rows", flush=True)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    q = pd.read_csv(QUALITY_GEN)
    qinfo_by_ds = {}
    for ds, g in q.groupby("dataset"):
        qinfo_by_ds[ds] = {r.perturbation: (r.sb_from_median_all_genes, r.frac_systematic_var)
                           for r in g.itertuples()}
    jobs = [(ds, k, qinfo_by_ds.get(ds, {})) for ds in DATASETS for k in range(5)]
    print(f"{len(jobs)} jobs")
    t0 = time.time(); all_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            all_rows.extend(fut.result())
    df = pd.DataFrame(all_rows)
    out = os.path.join(OUT_DIR, "exhaustive_linear_scores.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows in {(time.time()-t0)/60:.1f} min -> {out}")
    print("arms:", sorted(df.arm.unique()))


if __name__ == "__main__":
    main()
