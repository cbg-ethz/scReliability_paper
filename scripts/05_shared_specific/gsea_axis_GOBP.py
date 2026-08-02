#!/usr/bin/env python3
"""Shared-axis vs specific-residual program SETS on GO Biological Process (5407 processes, no Hallmark ceiling).
Geometric decomposition in the framework's raw centroid space (delta_avg shared; residual specific), GSEA via
blitzgsea (fast, Enrichr libraries). Save FDR<0.05 GO terms for each shared axis (per context) and each GENUINE
perturbation residual. Then we compare the two SETS: overlap, specific-only, breadth — per modality."""
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger().setLevel(logging.ERROR)

import os, json, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, blitzgsea as blitz
import sys
import os as _os, pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "03_preprocess"))
from config import intermediate_path, LEGACY_QUALITY_MAP
from contexts import is_guide, GEN_CONF, CEL_CONF, GEN_DIR, CEL_DIR
_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
ROOT = _ROOT
GENESETS = f"{_ROOT}/data/genesets"
OUT = f"{_ROOT}/intermediate/shared_specific"
_os.makedirs(OUT, exist_ok=True)
CHUNK = 20000
CHEM={"sciplex3_A549","sciplex3_K562","sciplex3_MCF7","sciplex3_comb"}
GO={t:[g.upper() for g in v] for t,v in json.load(open(f"{GENESETS}/genesets_GO_Biological_Process_2023.json")).items()}
PERM=300

def gsea_terms(vals, genesU):
    s=pd.Series(vals,index=genesU); s=s[~s.index.duplicated()]
    if s.std()<1e-9 or (s.value_counts().iloc[0]/s.shape[0])>0.5: return []
    sig=pd.DataFrame({0:s.index.values,1:s.values})
    try:
        r=blitz.gsea(sig,GO,permutations=PERM,verbose=False)
        return [str(t) for t,f in zip(r.index,r['fdr'].values) if f<0.05]
    except BaseException: return []

def run_dataset(name,kind):
    t0=time.time()
    if kind=="genetic":
        cfg=GEN_CONF[name]; path=f"{GEN_DIR}/{name}.h5ad"; pcol,ctrl,ccol=cfg["pert"],cfg["ctrl"],cfg["cond"]
        tab=pd.read_csv(intermediate_path("combined_genetic_2d.csv"))
    else:
        ctx_col,pcol,ctrl=CEL_CONF[name]; path=f"{CEL_DIR}/{name}.h5ad"; ccol=ctx_col
        tab=pd.read_csv(intermediate_path("combined_cellular_2d.csv"))
    tab=tab[(tab.dataset==name)&tab.reliable]; mod="chemical" if name in CHEM else kind
    A=ad.read_h5ad(path,backed="r"); genes=np.array(A.var_names); ng=A.n_vars; ntot=A.n_obs
    keep=np.array([not is_guide(g) for g in genes]); genesU=[g.upper() for g in genes[keep]]
    pcv=A.obs[pcol].astype(str).str.strip().values
    if kind=="genetic" and ccol is None:
        ctxv=np.array(["all"]*ntot); ctrl_mask=pcv==ctrl; key=pcv; rel={str(r.perturbation):r.triage_2d for r in tab.itertuples()}
    elif kind=="genetic":
        ctxv=A.obs[ccol].astype(str).str.strip().values; ctrl_mask=pcv==ctrl
        key=np.array([f"{c}|||{p}" for c,p in zip(ctxv,pcv)]); rel={f"{r.condition}|||{r.perturbation}":r.triage_2d for r in tab.itertuples()}
    else:
        ctxv=A.obs[ctx_col].astype(str).str.strip().values; condv=A.obs[pcol].astype(str).str.strip().values
        ctrl_mask=condv==ctrl; key=np.array([f"{c}|||{p}" for c,p in zip(ctxv,condv)]); rel={f"{r.context}|||{r.perturbation}":r.triage_2d for r in tab.itertuples()}
    is_rel=np.array([k in rel for k in key])&(~ctrl_mask)
    uctx=sorted(set(ctxv)); cidx={c:i for i,c in enumerate(uctx)}; cc=np.array([cidx[c] for c in ctxv])
    perts=sorted(set(key[is_rel])); pidx={p:i for i,p in enumerate(perts)}
    pc=np.array([pidx[k] if r else -1 for k,r in zip(key,is_rel)]); P=len(perts)
    if P==0: return [],[]
    csum=np.zeros((len(uctx),ng)); cn=np.zeros(len(uctx),int); psum=np.zeros((P,ng)); pn=np.zeros(P,int); X=A.X
    for s in range(0,ntot,CHUNK):
        e=min(s+CHUNK,ntot); Xb=X[s:e]; Xb=Xb.toarray() if sp.issparse(Xb) else np.asarray(Xb,float)
        b_cc=cc[s:e]; b_cm=ctrl_mask[s:e]; b_pc=pc[s:e]
        for k in np.unique(b_cc[b_cm]):
            sel=b_cm&(b_cc==k); csum[k]+=Xb[sel].sum(0); cn[k]+=int(sel.sum())
        v=b_pc>=0
        if v.any(): np.add.at(psum,b_pc[v],Xb[v]); np.add.at(pn,b_pc[v],1)
    cmean=np.divide(csum,cn[:,None],out=np.zeros_like(csum),where=cn[:,None]>0)
    ctx_of={p:(p.split("|||")[0] if "|||" in p else "all") for p in perts}
    davg=np.zeros((len(uctx),ng)); dnorm=np.zeros(len(uctx))
    for c,k in cidx.items():
        mem=[pidx[p] for p in perts if ctx_of[p]==c]
        if not mem or cn[k]==0: continue
        davg[k]=psum[mem].sum(0)/max(pn[mem].sum(),1)-cmean[k]; dnorm[k]=np.linalg.norm(davg[k])
    shared=[]
    for c,k in cidx.items():
        if dnorm[k]<1e-9 or cn[k]<4: continue
        shared.append(dict(name=name,modality=mod,context=c,terms=gsea_terms(davg[k][keep],genesU)))
    spec=[]
    for p,i in pidx.items():
        if pn[i]==0 or LEGACY_QUALITY_MAP.get(rel[p], rel[p]) != "Specific": continue   # specific perturbations only
        k=cidx[ctx_of[p]]
        if dnorm[k]<1e-9: continue
        dp=psum[i]/pn[i]-cmean[k]; resid=dp-(np.dot(dp,davg[k])/dnorm[k]**2)*davg[k]
        spec.append(dict(name=name,modality=mod,context=ctx_of[p],perturbation=p.split("|||")[-1],terms=gsea_terms(resid[keep],genesU)))
    print(f"  {name:24s} shared={len(shared)} specific={len(spec)}  {round(time.time()-t0,1)}s",flush=True)
    return shared,spec

if __name__=="__main__":
    print(f"GO-BP {len(GO)} terms; blitzgsea perm={PERM}; shared axis + GENUINE residuals")
    jobs=[(n,"genetic") for n in GEN_CONF if n!="sciplex3_comb"]+[(n,"cellular") for n in CEL_CONF if n!="sciplex3"]
    SH,SP=[],[]
    for n,k in jobs:
        try: sh,spp=run_dataset(n,k); SH+=sh; SP+=spp
        except Exception as ex:
            import traceback; print(f"  ERR {n}: {ex} :: {traceback.format_exc()[-160:]}",flush=True)
    json.dump({"shared":SH,"specific":SP},open(f"{OUT}/gsea_axis_GOBP.json","w"))
    print(f"\n{len(SH)} shared axes, {len(SP)} genuine residuals -> {OUT}/gsea_axis_GOBP.json")
