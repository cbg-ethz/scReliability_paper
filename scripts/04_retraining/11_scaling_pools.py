#!/usr/bin/env python3
"""11 — Data-scaling by training pool. Produces the data behind the supplementary scaling figure
(scripts/06_figures/figS6_data_scaling.py).

Scaling on the 7 informative genetic gene-target datasets, 4 training arms
(reliable / unreliable / all / specific), evaluated on BOTH reliable and specific test pools. Each arm sweeps
FRACS of its own pool. Per-test-pool aggregation done in-run (small CSV). A dataset contributes to an arm only
if that pool has >=MIN_POOL train perts, and to a test pool only if it has >=2 test perts there.

Certified linear fit + cached pseudobulks via scaling_pools_lib. Builds the cache on first run (one-time,
seed-independent). Output: <PAPER>/intermediate/retraining/scaling_pools.csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import scaling_pools_lib as L
# The 7 informative genetic datasets (same as Fig 2e): genetic, all three quality classes present
# (so all four training arms are populated), and >=15 specific perturbations for a test pool.
DATASETS=["Replogle_K562essential","Replogle_RPE1essential","Norman","Schmidt","Adamson","Replogle_exp6","Wessels"]
FRACS=[0.1,0.2,0.3,0.4,0.55,0.7,0.85,1.0]; N_DRAW=6; MIN_POOL=8
METRICS=["wei_pcc","wei_mse","wei_cdeg","ae_pcc"]
def summarize(fr):
    df=pd.DataFrame(fr); out=[]
    for pool in ["reliable","specific"]:
        # test_phi holds cos(theta); the specific class is phi = cos^2(theta) < 0.5.
        sub=df[df.test_rho>=0.5] if pool=="reliable" else df[(df.test_rho>=0.5)&(df.test_phi<L.COS_SPECIFIC)]
        if len(sub)<2: continue
        rec={"test_pool":pool,"n_test":len(sub),"n_train":int(df.n_train.iloc[0])}
        for m in METRICS: rec[m]=float(sub[m].mean())
        rec["ca"]=float(sub["ca_"+pool].mean()); out.append(rec)
    return out
def run_ds(dataset):
    cache=L.load_cache(dataset); rho,phi=cache["rho"],cache["phi"]; rows=[]
    for seed in range(5):
        try: train,test=L.get_split(dataset,seed)
        except Exception: continue
        train=[p for p in train if p in cache["delta"]]
        test_s=[p for p in test if p in cache["delta"] and len(L.parse_pert_genes(str(p)))==1]
        if len(test_s)<2: continue
        pools={"reliable":L.pool_members(train,rho,phi,"reliable"),"unreliable":L.pool_members(train,rho,phi,"unreliable"),
               "specific":L.pool_members(train,rho,phi,"specific"),"all":train}
        rng=np.random.default_rng(L.deterministic_subseed(dataset, seed))
        for arm,pool in pools.items():
            if len(pool)<MIN_POOL: continue
            for f in FRACS:
                s=max(2,min(len(pool),int(round(f*len(pool)))))
                for d in range(N_DRAW):
                    tr=list(rng.choice(pool,s,replace=False))
                    fr=L.fit_score(cache,tr,test_s)
                    if not fr: continue
                    for rec in summarize(fr):
                        rec.update(dataset=dataset,seed=seed,arm=arm,frac=f,draw=d); rows.append(rec)
    print(f"done {dataset}: {len(rows)} rows",flush=True); return rows
def main():
    for ds in DATASETS: L.build_cache(ds)        # one-time pseudobulk cache (seed-independent)
    rows=[]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for fut in as_completed([ex.submit(run_ds,ds) for ds in DATASETS]): rows.extend(fut.result())
    out=f"{L.ROOT}/scaling_pools.csv"
    pd.DataFrame(rows).to_csv(out,index=False); print(len(rows),"rows ->",out)
if __name__=="__main__": main()
