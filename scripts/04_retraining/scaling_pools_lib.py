#!/usr/bin/env python3
"""Parameterized runner for the scaling-law experiments. Shared linear model + _lib metrics.
Pseudobulks are cached once per dataset (seed-independent); experiments are then cheap linear fits.
Pools follow the paper's quality classes: reliable rho>=0.5; unreliable rho<0.5; specific rho>=0.5 and
phi<0.5; shared rho>=0.5 and phi>=0.5, where rho = sb_from_median_all_genes and phi = cos^2(theta).
The cached `cos_sim` column holds cos(theta), so the specificity cut is cos(theta) < 1/sqrt(2)."""
import os, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as spx
# Paths anchored via env (defaults to local working tree). MAIN = working tree (h5ads + certified fit + _lib
# data dirs); PAPER = repo root (cache + output under intermediate/), from SCRELIABILITY_ROOT.
PAPER = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))            # _lib (this folder)
from _lib import (parse_pert_genes, compute_pseudobulks, compute_wei_deg_idx, compute_ae_top_expr_idx,
    predict_linear_model, dense_mean, wei_pcc_delta_top100, wei_mse_top100, wei_common_degs,
    ae_pearson_dlt_top1000, centroid_accuracy_for_set, H5AD_DIR, H5AD_FILE, SPLITS_DIR)
from _lib import fit_linear_model_weighted, deterministic_subseed

DATASETS=["Replogle_K562essential","Replogle_RPE1essential","Norman","Schmidt","Adamson","Replogle_exp6","Wessels"]
ROOT=str(PAPER / "intermediate" / "retraining")
CACHE=f"{ROOT}/scaling_cache"; os.makedirs(CACHE, exist_ok=True)
RP=pd.read_csv(f"{PAPER}/intermediate/combined_genetic_2d.csv")

def _splithalf(adata, perts, ctrl_mean, seed=0):
    rng=np.random.default_rng(seed); X=adata.X; obs=adata.obs.perturbation.values; d1={}; d2={}
    for p in set(perts):
        idx=np.where(obs==p)[0]
        if len(idx)<4: continue
        Xb=X[idx]; Xb=Xb.toarray() if spx.issparse(Xb) else np.asarray(Xb,float)
        perm=rng.permutation(len(idx)); h=len(idx)//2
        d1[p]=Xb[perm[:h]].mean(0)-ctrl_mean; d2[p]=Xb[perm[h:]].mean(0)-ctrl_mean
    return d1,d2

def build_cache(dataset, splithalf=False):
    out=f"{CACHE}/{dataset}{'_sh' if splithalf else ''}.pkl"
    if os.path.exists(out): return out
    adata=ad.read_h5ad(H5AD_DIR/f"{dataset}.h5ad"); gene_names=list(adata.var_names); g2c={g:i for i,g in enumerate(gene_names)}
    ctrl=(adata.obs.perturbation=="control").to_numpy(); ctrl_mean=dense_mean(adata[ctrl].X); top_expr=compute_ae_top_expr_idx(ctrl_mean)
    allp=[p for p in adata.obs.perturbation.unique() if p!="control"]
    delta,expr=compute_pseudobulks(adata, allp, ctrl_mean)
    singles=[p for p in delta if len(parse_pert_genes(p))==1]
    deg=compute_wei_deg_idx(adata, singles, g2c)
    d1=d2=None
    if splithalf: d1,d2=_splithalf(adata, list(delta), ctrl_mean)
    del adata
    sub=RP[RP.dataset==dataset].drop_duplicates("perturbation").set_index("perturbation")
    rho={p:(float(sub.loc[p,"sb_from_median_all_genes"]) if p in sub.index else np.nan) for p in delta}
    phi={p:(float(sub.loc[p,"cos_sim"]) if p in sub.index else np.nan) for p in delta}
    pickle.dump(dict(delta=delta,expr=expr,deg=deg,gene_names=gene_names,top_expr=top_expr,rho=rho,phi=phi,d1=d1,d2=d2),open(out,"wb"))
    return out

def load_cache(dataset, splithalf=False):
    return pickle.load(open(f"{CACHE}/{dataset}{'_sh' if splithalf else ''}.pkl","rb"))

def get_split(dataset, seed):
    """Load the train/test split for one dataset and seed.

    The splits are inputs to the analysis, not something to regenerate: silently falling back to a fresh
    random split would change every downstream pool while still producing plausible-looking output.
    """
    f=SPLITS_DIR/f"{dataset}_split{seed}.csv"
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"split file not found: {f}\n"
            f"Expected the published splits under {SPLITS_DIR}. Set SCRELIABILITY_ROOT to the repository "
            f"root, or regenerate them with scripts/04_retraining/01_make_splits.py.")
    sdf=pd.read_csv(f)
    return sdf[sdf.split=="train"].perturbation.tolist(), sdf[sdf.split=="test"].perturbation.tolist()

# phi = cos^2(theta) < 0.5 is equivalent to cos(theta) < 1/sqrt(2); the cache stores cos(theta).
COS_SPECIFIC = 1.0/np.sqrt(2)

def in_pool(rho, cos_sim, pool):
    """Membership of a perturbation in a training/test pool, using the paper's quality thresholds."""
    if pool=="all": return True
    if pool=="reliable": return rho>=0.5
    if pool=="unreliable": return rho<0.5
    if pool=="specific": return (rho>=0.5) and (cos_sim<COS_SPECIFIC)
    if pool=="shared": return (rho>=0.5) and (cos_sim>=COS_SPECIFIC)
    return False

def pool_members(perts, rho, cos_sim, pool):
    return [p for p in perts if in_pool(rho.get(p,np.nan), cos_sim.get(p,np.nan), pool)]

def fit_score(cache, train_perts, test_perts, weights=None, targets=None):
    """targets: optional dict pert->response to TRAIN on (default = cache delta). Always SCORE vs cache delta."""
    delta=cache["delta"]; expr=cache["expr"]; deg=cache["deg"]; gn=cache["gene_names"]; top_expr=cache["top_expr"]
    rho=cache["rho"]; phi=cache["phi"]; tgt=targets if targets is not None else delta
    tr=[p for p in train_perts if p in tgt and p in expr]
    if len(tr)<2: return []
    td=np.array([tgt[p] for p in tr]); te=np.array([expr[p] for p in tr])
    w=None if weights is None else np.array([max(0.0,float(weights.get(p,0))) for p in tr])
    lm=fit_linear_model_weighted(te,td,tr,gn,w)
    if lm is None: return []
    preds={p:predict_linear_model(lm,p) for p in test_perts if p in delta}
    preds={p:v for p,v in preds.items() if v is not None}; tp=list(preds)
    ca={}
    for pool in ["all","reliable","unreliable","specific","shared"]:
        pl=pool_members(tp,rho,phi,pool)
        if len(pl)>=2:
            pm=np.array([preds[p] for p in pl]); gm=np.array([delta[p] for p in pl]); ca[pool]=dict(zip(pl,centroid_accuracy_for_set(pm,gm)))
        else: ca[pool]={}
    rows=[]
    for p in tp:
        pr=preds[p]; gt=delta[p]; dg=deg.get(p)
        rows.append(dict(perturbation=p,test_rho=rho.get(p,np.nan),test_phi=phi.get(p,np.nan),n_train=len(tr),
            wei_pcc=wei_pcc_delta_top100(pr,gt,dg),wei_mse=wei_mse_top100(pr,gt,dg),wei_cdeg=wei_common_degs(pr,dg),
            ae_pcc=ae_pearson_dlt_top1000(pr,gt,top_expr),
            ca_specific=float(ca["specific"].get(p,np.nan)),ca_reliable=float(ca["reliable"].get(p,np.nan)),ca_all=float(ca["all"].get(p,np.nan))))
    return rows

if __name__=="__main__":
    import time, sys
    sh = "--sh" in sys.argv
    for ds in DATASETS:
        t=time.time(); build_cache(ds, splithalf=sh); print(f"cached {ds}{' (sh)' if sh else ''} {round(time.time()-t,1)}s",flush=True)
    print("cache built")
