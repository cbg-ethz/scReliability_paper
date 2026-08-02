#!/usr/bin/env python3
"""Compute the qualifying (dataset, context, modality) triples for the shared/specific decomposition,
from the criteria — NOT hardcoded:
  A) >= min_rel reliable perturbations in that context (so the shared axis is a stable, non-self-dominated average)
  B) the shared axis enriches >=1 Hallmark program at FDR<0.05 (a coherent common response exists).
Reliable counts come from the shared/specific tables; coherence from the shared-axis GSEA (gsea_axis_raw.json)."""
import os, sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_preprocess"))   # config.intermediate_path
from config import intermediate_path

ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
RAW = str(ROOT / "intermediate" / "shared_specific" / "gsea_axis_raw.json")

def qualifying_pairs(min_rel=10, raw_json=RAW):
    genR = pd.read_csv(intermediate_path("combined_genetic_2d.csv")); genR = genR[genR.reliable]
    celR = pd.read_csv(intermediate_path("combined_cellular_2d.csv")); celR = celR[celR.reliable]
    def nrel(name, ctx, mod):
        if mod == "cellular": return int(((celR.dataset == name) & (celR.context.astype(str) == ctx)).sum())
        s = genR[genR.dataset == name]
        return len(s) if (ctx == "all" or "condition" not in s.columns) else int((s.condition.astype(str) == ctx).sum())
    out = []
    for r in json.load(open(raw_json))["shared"]:
        nsig = sum(1 for t, v in r["shared_nes"].items() if v[1] < 0.05)
        if nrel(r["name"], r["context"], r["modality"]) >= min_rel and nsig >= 1:
            out.append((r["name"], r["context"], r["modality"]))
    # genetic first, then others; stable order
    return sorted(out, key=lambda x: (x[2] != "genetic", x[0], x[1]))

if __name__ == "__main__":
    P = qualifying_pairs(); ng = sum(1 for _, _, m in P if m == "genetic")
    print(f"{len(P)} qualifying contexts ({ng} genetic, {len(P)-ng} other), {len(set(n for n,_,_ in P))} datasets:")
    for n, c, m in P: print(f"  {m:9s} {n:24s} {c}")
