# scReliability — paper analysis code

Analysis and figure code for the manuscript
**"Reliable single-cell perturbations explain and improve model performance."**

The framework assigns every perturbation–context observation a two-axis quality
label from the raw single-cell data:

- **reliability** ρ — split-half reproducibility of the pseudo-bulk effect vector;
- **specificity** φ = cos²θ — how much of the reliable effect is unique to the
  perturbation versus a response shared across perturbations.

This gives three classes — **Unreliable** (ρ < 0.5), **Shared** (ρ ≥ 0.5, φ ≥ 0.5),
and **Specific** (ρ ≥ 0.5, φ < 0.5) — which are used to re-examine perturbation
benchmarks and to plan reliable screens.

## Repository layout

```
scReliability_paper/
├── README.md            environment.yml      run_all.sh
├── data/                # raw inputs — not tracked; see data/README.md for downloads
└── scripts/
    ├── 01_reliability/        # split-half reliability ρ + specificity φ → triage tables
    ├── 02_reliability_edist/  # energy-distance reliability variant (Supplementary Figure 8)
    ├── 03_preprocess/         # config + style, and preprocessing → intermediate/ tables
    ├── 04_retraining/         # the reliable-vs-all retraining experiment (Figure 2 panel e)
    ├── 05_shared_specific/    # shared-axis vs specific-residual GSEA (Supplementary Figure 2)
    └── 06_figures/            # the published figure scripts (Figures 1-3, Supplementary 1-8)
```

## Setup

```bash
conda env create -f environment.yml
conda activate screliability
```

Then obtain the raw data and place it under `data/` (see `data/README.md`).

### Where paths point
All scripts resolve inputs/outputs relative to a **repo-root anchor**. By default
this is the repository directory itself. To keep the (large) data and outputs
elsewhere, set:

```bash
export SCRELIABILITY_ROOT=/path/to/a/dir/that/contains/data
```

`data/`, `output/`, `intermediate/`, `results/`, and `figures/` all live under that
root and are git-ignored.

## Running

End to end:

```bash
bash run_all.sh                 # uses ./ as the root and `python` as the interpreter
PYTHON=$(which python) SCRELIABILITY_ROOT=/data/screl bash run_all.sh
```

Stage by stage (see `run_all.sh` for the exact order):

1. **Reliability** — `scripts/01_reliability/run_all_parallel.sh` computes split-half
   ρ for all 29 datasets, then the two `*_systema.py` scripts merge in the cos θ
   diagnostic and write the triage tables.
2. **E-distance reliability** — `scripts/02_reliability_edist/` produces the
   energy-distance triage used in Supplementary Figure 8.
3. **Preprocess** — `scripts/03_preprocess/preprocess_01..05` build the combined
   triage + benchmark tables in `intermediate/`.
4. **Retraining** — `scripts/04_retraining/run_all.sh` runs the reliable-vs-all
   training experiment; its scores feed Figure 2 panel e.
5. **Shared vs specific** — `scripts/05_shared_specific/` runs the GSEA behind
   Supplementary Figure 2 (see that directory's README for the run order).
6. **Figures** — `scripts/06_figures/*.py` render the panels into `figures/`.

The directories are numbered in dependency order: figures come last because every
figure reads tables written by the stages above it.

Figure 3 and Supplementary Figure 7 use a denser reliability grid, built by stage 1b of `run_all.sh`:

```bash
N_HALF_GRID_OVERRIDE="$(seq -s, 1 50)" OUT_BASE=output/reliability_fig3 \
  bash scripts/01_reliability/run_all_parallel.sh
```

## Figures

| Figure | Script |
|---|---|
| Fig 1 — quality framework | `fig1_framework.py` |
| Fig 2 — filtering reshapes benchmarks | `fig2_benchmarks.py` (panel e also stands alone as `fig2e_reliable_quantity.py`) |
| Fig 3 — prospective design | `fig3_prospective.py` |
| Supp 1 — reliability distributions | `figS1_reliability_distributions.py` |
| Supp 2 — shared vs specific programs | `figS2_shared_vs_specific.py` |
| Supp 3 — threshold robustness | `figS3_threshold_robustness.py` |
| Supp 4 — rank shifts, remaining settings | `figS4_rank_shifts.py` |
| Supp 5 — retraining grid | `figS5_retraining_grid.py` |
| Supp 6 — data scaling by training pool | `figS6_data_scaling.py` |
| Supp 7 — parametric fits | `figS7_parametric_fits.py` |
| Supp 8 — energy-distance triage | `figS8_edist.py` |

All figure scripts live in `scripts/06_figures/`.

Each script saves a combined draft plus individual panels (`.pdf`/`.png`/`.svg`,
with editable SVG text). The **main figures (1–3) are assembled and cropped in
PowerPoint** for final layout; `scripts/06_figures/crop_polished_figs.py` performs
the crop from a PowerPoint export — point it at your export with
`SCRELIABILITY_PPT_PDF=/path/to/figures.pdf`.

## Notes

- The interpreter is overridable everywhere via `PYTHON=…`; the data/output root via
  `SCRELIABILITY_ROOT=…`. No absolute paths are baked into the code.
- This repository is the **analysis code**; the `scReliability` Python toolkit is a
  separate package.
- GEARS and scGPT were retrained with [Systema](https://github.com/mlbio-epfl/systema)
  (`run_gears_arm.py`, `run_scgpt.py`) on the splits in
  `results/retraining_quality_filtering/splits/`. Those runs need a GPU and are not part of
  `run_all.sh`. The linear model follows Ahlmann-Eltze et al. (`scripts/04_retraining/_lib.py`).

## License

MIT — see [LICENSE](LICENSE).
