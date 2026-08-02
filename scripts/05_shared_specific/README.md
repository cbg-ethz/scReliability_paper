# shared-vs-specific GSEA analysis

Produces **Supplementary Figure 2** (`scripts/06_figures/figS2_shared_vs_specific.py`): what biological
programs the *shared* axis of reliable perturbations represents (Hallmark dotplot), and how the *specific*
residual programs differ in number and breadth (Hallmark + GO-BP bars).

## Files
- `qualifying.py` — qualifying (dataset, context, modality) triples from criteria (≥10 reliable perts/context
  + ≥1 Hallmark program at FDR<0.05). Self-contained: reads `intermediate/combined_{genetic,cellular}_2d.csv`
  and `intermediate/shared_specific/gsea_axis_raw.json`. **Imported by the figure script.**
- `gsea_axis_raw.py` — shared-axis δ_avg decomposition + per-gene specific residual, preranked GSEA (gseapy,
  Hallmark, perm=1000) → `gsea_axis_raw.json` (shared_nes / resid_nes per context).
- `gsea_axis_GOBP.py` — same decomposition, GO-BP via blitzgsea (perm=300) → `gsea_axis_GOBP.json`.
- `add_sciplex3_comb.py` — maps sciplex3_comb ENSEMBL→symbol and appends its Hallmark+GO rows to both JSONs.
- `contexts.py` — per-dataset column names, h5ad directories, `is_guide` and the Hallmark loader used by the
  three scripts above.

## Run order

The three scripts are **order-dependent**:

```bash
python gsea_axis_raw.py        # Hallmark NES/FDR  -> intermediate/shared_specific/gsea_axis_raw.json
python gsea_axis_GOBP.py       # GO-BP term lists  -> intermediate/shared_specific/gsea_axis_GOBP.json
python add_sciplex3_comb.py    # appends sciplex3_comb rows to BOTH files, de-duplicating any prior ones
```

`gsea_axis_GOBP.py` deliberately skips `sciplex3_comb` (that dataset's `var_names` are ENSEMBL IDs);
`add_sciplex3_comb.py` maps them to symbols and appends the rows, so it must run last. It **rewrites both
JSONs in place** — back them up before re-running.

## Inputs / reproducibility

The figure's actual inputs are the two consolidated JSONs in `intermediate/shared_specific/`, git-ignored
like everything else under `intermediate/`. Regenerating them needs the per-dataset h5ads (~68 GB, see
`data/README.md`), the gene-set libraries and ENSEMBL→symbol map under `data/genesets/`, and the two GSEA
packages declared in `environment.yml` (`gseapy`, `blitzgsea`). Both GSEA calls are seeded, so results are
reproducible at those versions.

A regenerated `gsea_axis_raw.json` will carry the current `Specific`/`Shared`/`Unreliable` vocabulary where
the shipped file carries the older `genuine_signal`/`falsely_solved` labels. The figure script normalises
both through `config.LEGACY_QUALITY_MAP`, so either is fine.

`qualifying.py` is the only script here the figure imports directly, and it runs from the repo with no
extra dependencies.
