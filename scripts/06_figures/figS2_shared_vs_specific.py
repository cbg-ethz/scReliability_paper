#!/usr/bin/env python3
"""Supplementary Figure 2: shared vs specific responses of RELIABLE perturbations.
(A) Shared-axis Hallmark program enrichment (preranked GSEA dotplot, category-grouped).
(B,C) Distinct significant programs (shared vs specific) and specific-only programs, in BOTH Hallmark and GO-BP.
Qualifying set computed from criteria (scripts/05_shared_specific/qualifying.py): >=10 reliable perts/context
+ coherent shared axis (>=1 Hallmark program at FDR<0.05).

Inputs : intermediate/shared_specific/{gsea_axis_raw,gsea_axis_GOBP}.json  (scripts/05_shared_specific/)
Output : figures/figS2_shared_vs_specific.{pdf,png,svg}
"""
import os, sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm, matplotlib.colors as mcolors
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_shared_specific"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_preprocess"))
from qualifying import qualifying_pairs
from config import LEGACY_QUALITY_MAP
from _paths import save_fig, FIG_DIR

ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
GSEA = ROOT / "intermediate" / "shared_specific"
NM_T, NM_L, NM_TK, NM_TINY = 9, 8, 7, 5.5
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica Neue','Helvetica','DejaVu Sans'],'font.size':NM_TK,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.titlesize':NM_T,'axes.titleweight':'bold','axes.labelsize':NM_L,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.6,'figure.facecolor':'white','savefig.facecolor':'white'})
CATEGORY = [("Proliferation /\ncell cycle",["Myc Targets V1","Myc Targets V2","E2F Targets","G2-M Checkpoint","Mitotic Spindle","DNA Repair"]),
 ("Stress /\nproteostasis",["Unfolded Protein Response","mTORC1 Signaling","PI3K/AKT/mTOR  Signaling","Reactive Oxygen Species Pathway","Hypoxia"]),
 ("Cell\ndeath",["p53 Pathway","Apoptosis"]),
 ("Immune /\ninflammatory",["Interferon Gamma Response","Interferon Alpha Response","Inflammatory Response","TNF-alpha Signaling via NF-kB","Complement","Allograft Rejection","IL-2/STAT5 Signaling","IL-6/JAK/STAT3 Signaling"]),
 ("Metabolism",["Oxidative Phosphorylation","Fatty Acid Metabolism","heme Metabolism","Adipogenesis","Cholesterol Homeostasis","Xenobiotic Metabolism","Glycolysis","Peroxisome","Pperoxisome","Bile Acid Metabolism"]),
 ("Signaling /\ndifferentiation",["Epithelial Mesenchymal Transition","Apical Junction","Estrogen Response Early","Estrogen Response Late","Androgen Response","KRAS Signaling Up","KRAS Signaling Dn","Coagulation","Angiogenesis","Myogenesis","Notch Signaling","Wnt-beta Catenin Signaling","TGF-beta Signaling"])]
P = qualifying_pairs(); QP = [(n,c) for n,c,m in P]; NGEN = sum(1 for n,c,m in P if m == "genetic")
RAW = {(r["name"],r["context"]):r["shared_nes"] for r in json.load(open(GSEA/"gsea_axis_raw.json"))["shared"]}
SHlist = [RAW[p] for p in QP if p in RAW]; labs = [(n if c == "all" else f"{n}·{c}") for (n,c) in QP if (n,c) in RAW]
present = set(t for nes in SHlist for t,v in nes.items() if v[1] < 0.05)
cols = []; cat_span = []
for cat, ps in CATEGORY:
    sub = [p for p in ps if p in present]
    if sub: cat_span.append((cat, len(cols), len(cols)+len(sub))); cols += sub
# Any significant program not covered by the curated categories still gets plotted, so the column count
# always equals the number of significant programs reported in panel b.
_other = sorted(present - set(cols))
if _other: cat_span.append(("Other", len(cols), len(cols)+len(_other))); cols += _other
assert len(cols) == len(present), f"{len(present)-len(cols)} significant programs would not be plotted"
# Order rows by their shared-axis pathway-enrichment profile (hierarchical clustering on the signed-NES
# matrix) so datasets with similar programs sit together — modality (genetic/cellular/chemical) is ignored.
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    Mrows = np.array([[nes[c][0] if (c in nes and nes[c][1] < 0.05) else 0.0 for c in cols] for nes in SHlist])
    if len(Mrows) > 2:
        _ord = list(leaves_list(linkage(Mrows, method="average", metric="euclidean")))
        SHlist = [SHlist[i] for i in _ord]; labs = [labs[i] for i in _ord]
except Exception:
    pass

def load(lib):
    if lib == "Hallmark":
        D = json.load(open(GSEA/"gsea_axis_raw.json"))
        sh = [set(t for t,v in r["shared_nes"].items() if v[1] < 0.05) for r in D["shared"] if (r["name"],r["context"]) in set(QP)]
        # The GSEA JSONs still store the older label vocabulary, so normalise at the read boundary
        # (config.LEGACY_QUALITY_MAP) rather than testing for legacy strings in the logic here.
        sp = [set(t for t,v in r["resid_nes"].items() if v[1] < 0.05) for r in D["residuals"]
              if (r["name"],r["context"]) in set(QP)
              and LEGACY_QUALITY_MAP.get(r["triage"], r["triage"]) == "Specific"]
    else:
        D = json.load(open(GSEA/"gsea_axis_GOBP.json"))
        sh = [set(r["terms"]) for r in D["shared"] if (r["name"],r["context"]) in set(QP)]
        sp = [set(r["terms"]) for r in D["specific"] if (r["name"],r["context"]) in set(QP)]
    return sh, sp

fig = plt.figure(figsize=(12.5,10.8))
gs = fig.add_gridspec(3,2,height_ratios=[2.5,1.0,1.0],width_ratios=[0.26,1.0],hspace=0.78,wspace=0.62,top=0.95,bottom=0.06,left=0.14,right=0.9)
PANEL_FS = 13
# ---------- a: heatmap ----------
ax = fig.add_subplot(gs[0,:]); norm = mcolors.TwoSlopeNorm(vmin=-3,vcenter=0,vmax=3); cmap = cm.RdBu_r
ax.text(-0.085,1.05,"a",transform=ax.transAxes,fontsize=PANEL_FS,fontweight="bold",va="bottom",ha="left")
for i, nes in enumerate(SHlist):
    for j, c in enumerate(cols):
        v = nes.get(c)
        if v and v[1] < 0.05:
            size = 18+42*min(-np.log10(max(v[1],1e-10))/4,1.5)
            ax.scatter(j,i,s=size,c=[cmap(norm(v[0]))],edgecolor="#333",linewidth=0.3,zorder=3)
ax.set_xlim(-0.6,len(cols)-0.4); ax.set_ylim(-0.6,len(labs)-0.4); ax.invert_yaxis()
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=90,fontsize=NM_TINY)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=NM_TINY)
ax.set_axisbelow(True); ax.grid(True,color="#eee",lw=0.4)
for cat, a, b in cat_span:
    ax.axvline(a-0.5,color="#999",lw=0.6); ax.text((a+b-1)/2,-0.62,cat.replace("\n"," "),ha="center",va="bottom",fontsize=NM_TINY,fontweight="bold")
ax.axvline(len(cols)-0.5,color="#999",lw=0.6)
cax = ax.inset_axes([1.015,0.45,0.012,0.5]); sm = cm.ScalarMappable(norm=norm,cmap=cmap)
cb = fig.colorbar(sm,cax=cax); cb.set_label("NES",fontsize=NM_TINY); cb.ax.tick_params(labelsize=NM_TINY)
# dot-size legend BELOW the colorbar (no overlap)
for k, (fdr, lab) in enumerate([(0.05,"0.05"),(1e-4,"1e-4")]):
    ax.scatter(1.02,0.20-k*0.07,s=18+42*min(-np.log10(fdr)/4,1.5),transform=ax.transAxes,c="#bbb",edgecolor="#333",lw=0.3,clip_on=False)
    ax.text(1.05,0.20-k*0.07,f"FDR {lab}",transform=ax.transAxes,fontsize=NM_TINY,va="center")
ax.set_title("Shared axis Hallmark program enrichment — preranked GSEA, FDR < 0.05",fontsize=NM_T,pad=26)
# ---------- b–e: bars ----------
LETTERS = [("b","c"),("d","e")]
for ri, lib in enumerate(["Hallmark","GO-BP"]):
    sh, sp = load(lib); SHs = set().union(*sh) if sh else set(); SPs = set().union(*sp) if sp else set()
    a0 = fig.add_subplot(gs[ri+1,0]); a1 = fig.add_subplot(gs[ri+1,1])
    ymax = max(len(SHs),len(SPs))
    a0.bar([0,1],[len(SHs),len(SPs)],0.62,color=["#f39c12","#1e8449"],edgecolor="#222",lw=0.5)
    for xi, v in zip([0,1],[len(SHs),len(SPs)]): a0.text(xi,v+ymax*0.04,str(v),ha="center",fontsize=NM_T-1,fontweight="bold")
    a0.set_ylim(0,ymax*1.22); a0.set_xlim(-0.7,1.7); a0.set_xticks([0,1]); a0.set_xticklabels(["shared\naxis","specific"],fontsize=NM_TK)
    a0.set_ylabel(f"# significant\n{lib} programs",fontsize=NM_L,labelpad=2); a0.set_title(f"{lib}:\n# programs",fontsize=NM_T-0.5)
    a0.text(-0.62,1.06,LETTERS[ri][0],transform=a0.transAxes,fontsize=PANEL_FS,fontweight="bold",va="bottom",ha="left")
    spc = Counter(t for s in sp for t in s); so = [(t,c) for t,c in spc.most_common(300) if t not in SHs][:11][::-1]
    a1.barh(range(len(so)),[c for _,c in so],0.74,color="#1e8449",edgecolor="#222",lw=0.4)
    a1.set_yticks(range(len(so))); a1.set_yticklabels([t.split(' (GO')[0][:42] for t,_ in so],fontsize=NM_TINY+0.5)
    a1.set_ylim(-0.7,len(so)-0.3); a1.set_xlim(0,max(c for _,c in so)*1.04)
    a1.set_xlabel(f"# specific perturbations significant (of {len(sp)})",fontsize=NM_L)
    a1.set_title(f"{lib}: programs significant in specific residuals but not the shared axis",fontsize=NM_T-0.5)
    a1.text(-0.30,1.06,LETTERS[ri][1],transform=a1.transAxes,fontsize=PANEL_FS,fontweight="bold",va="bottom",ha="left")
save_fig(fig, "figS2_shared_vs_specific")
print(f"✓ Supp Fig 2 (shared vs specific) → {FIG_DIR}/figS2_shared_vs_specific.* | {len(labs)} rows, {len(cols)} cols, {len(set(n for n,_,_ in P))} datasets")
