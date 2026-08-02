#!/usr/bin/env python3
"""Add sciplex3_comb (Wei chemical-COMBINATION dataset; A549; ENSEMBL gene IDs) to the decomposition, with a
complete ENSEMBL->symbol map (mygene, 4975/5000). Same GSEA methods as the rest: Hallmark via gseapy.prerank
(perm=1000) for shared_nes/resid_nes; GO-BP via blitzgsea for term lists. Appends to gsea_axis_raw.json and
gsea_axis_GOBP.json (dedup any prior sciplex3_comb)."""
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger().setLevel(logging.ERROR)

import json, numpy as np, pandas as pd, scipy.sparse as spx, anndata as ad, gseapy as gp, blitzgsea as blitz, sys
import os as _os, pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "03_preprocess"))
from config import intermediate_path, LEGACY_QUALITY_MAP
from contexts import is_guide, GEN_CONF, GEN_DIR, GENESETS_DIR, load_hallmark

_ROOT = _os.environ.get("SCRELIABILITY_ROOT", str(_pathlib.Path(__file__).resolve().parents[2]))
# Read and append to the shared_specific artifacts in THIS repository. The gene-set libraries and the
# ENSEMBL->symbol map live under data/ (see data/README.md).
OUT = f"{_ROOT}/intermediate/shared_specific"
HALL={t:[x.upper() for x in v] for t,v in load_hallmark().items()}
GO={t:[g.upper() for g in v] for t,v in json.load(open(f"{GENESETS_DIR}/genesets_GO_Biological_Process_2023.json")).items()}
e2s=json.load(open(f"{GENESETS_DIR}/ens2sym_sciplex3_comb.json"))

def gsea_nes(vec, keep, genesU):
    s=pd.Series(np.asarray(vec,float)[keep],index=genesU); s=s[~s.index.duplicated()].sort_values(ascending=False)
    if s.shape[0]<200 or s.std()<1e-12 or (s.value_counts().iloc[0]/s.shape[0])>0.5: return {}
    try:
        r=gp.prerank(rnk=s,gene_sets=HALL,min_size=5,max_size=500,permutation_num=1000,threads=4,seed=0,no_plot=True,outdir=None,verbose=False).res2d
        return {str(row.Term).replace("HALLMARK_",""):(float(row.NES),float(row["FDR q-val"])) for _,row in r.iterrows()}
    except BaseException: return {}

def gsea_terms(vec, keep, genesU):
    s=pd.Series(np.asarray(vec,float)[keep],index=genesU); s=s[~s.index.duplicated()]
    if s.std()<1e-9 or (s.value_counts().iloc[0]/s.shape[0])>0.5: return []
    sig=pd.DataFrame({0:s.index.values,1:s.values})
    try:
        r=blitz.gsea(sig,GO,permutations=300,verbose=False)
        return [str(t) for t,f in zip(r.index,r["fdr"].values) if f<0.05]
    except BaseException: return []

cfg=GEN_CONF["sciplex3_comb"]; pcol,ctrl=cfg["pert"],cfg["ctrl"]
tab=pd.read_csv(intermediate_path("combined_genetic_2d.csv")); tab=tab[(tab.dataset=="sciplex3_comb")&tab.reliable]
rel={str(r.perturbation):r.triage_2d for r in tab.itertuples()}
A=ad.read_h5ad(f"{GEN_DIR}/sciplex3_comb.h5ad",backed="r"); genes=np.array(A.var_names); ng=A.n_vars; ntot=A.n_obs
syms=np.array([e2s.get(g,"") for g in genes]); keep=(syms!="")&np.array([not is_guide(s) for s in syms])
genesU=[s.upper() for s in syms[keep]]
pcv=A.obs[pcol].astype(str).str.strip().values; ctrl_mask=pcv==ctrl
perts=sorted(set(p for p in pcv if p in rel)); pidx={p:i for i,p in enumerate(perts)}; P=len(perts)
pc=np.array([pidx[p] if p in pidx else -1 for p in pcv])
csum=np.zeros(ng); cn=0; psum=np.zeros((P,ng)); pn=np.zeros(P,int); X=A.X
for s in range(0,ntot,20000):
    e=min(s+20000,ntot); Xb=X[s:e]; Xb=Xb.toarray() if spx.issparse(Xb) else np.asarray(Xb,float)
    cm=ctrl_mask[s:e]; csum+=Xb[cm].sum(0); cn+=int(cm.sum())
    bp=pc[s:e]; v=bp>=0
    if v.any(): np.add.at(psum,bp[v],Xb[v]); np.add.at(pn,bp[v],1)
cmean=csum/max(cn,1); mem=[i for i in range(P) if pn[i]>0]
davg=psum[mem].sum(0)/max(pn[mem].sum(),1)-cmean; dnorm=np.linalg.norm(davg)
print(f"matched {P} reliable perts, control n={cn}, davg||={dnorm:.3f}")
sh_raw={"name":"sciplex3_comb","modality":"chemical","context":"all","shared_nes":gsea_nes(davg,keep,genesU)}
sh_go ={"name":"sciplex3_comb","modality":"chemical","context":"all","terms":gsea_terms(davg,keep,genesU)}
nsig_h=sum(1 for t,v in sh_raw["shared_nes"].items() if v[1]<0.05)
print(f"shared axis: Hallmark sig={nsig_h} {[t for t,v in sh_raw['shared_nes'].items() if v[1]<0.05][:6]} | GO sig={len(sh_go['terms'])}")
res_raw=[]; res_go=[]
for p,i in pidx.items():
    if pn[i]==0 or LEGACY_QUALITY_MAP.get(rel[p], rel[p]) != "Specific": continue
    dp=psum[i]/pn[i]-cmean; resid=dp-(np.dot(dp,davg)/dnorm**2)*davg
    res_raw.append({"name":"sciplex3_comb","modality":"chemical","context":"all","perturbation":p,"triage":rel[p],"resid_nes":gsea_nes(resid,keep,genesU)})
    res_go.append({"name":"sciplex3_comb","modality":"chemical","context":"all","perturbation":p,"terms":gsea_terms(resid,keep,genesU)})
print(f"{len(res_raw)} genuine residuals")
for path,sh,res,shk,rk in [(f"{OUT}/gsea_axis_raw.json",sh_raw,res_raw,"shared","residuals"),
                            (f"{OUT}/gsea_axis_GOBP.json",sh_go,res_go,"shared","specific")]:
    D=json.load(open(path))
    D[shk]=[r for r in D[shk] if r["name"]!="sciplex3_comb"]+[sh]
    D[rk] =[r for r in D[rk]  if r["name"]!="sciplex3_comb"]+res
    json.dump(D,open(path,"w"))
print("appended sciplex3_comb to gsea_axis_raw.json + gsea_axis_GOBP.json")
