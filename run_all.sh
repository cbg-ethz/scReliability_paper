#!/usr/bin/env bash
# End-to-end pipeline for the scReliability paper analysis.
#
# Prerequisites:
#   1. Environment:  conda env create -f environment.yml && conda activate screliability
#   2. Raw data under ./data/  (see data/README.md), OR set SCRELIABILITY_ROOT to a
#      directory that contains data/ and will receive output/ intermediate/ results/ figures/.
#
# Overrides:  PYTHON=/path/to/python   SCRELIABILITY_ROOT=/path/to/data-root
#
# NOTE: this is the intended run order. A few stages need the notes in README.md:
#   - Figure 3 and Supplementary Figure 7 need the denser reliability grid built by stage 1b.
#   - The main figures (Fig 1-3) are assembled in PowerPoint; the scripts here
#     produce the panels / *_combined drafts (see scripts/06_figures/crop_polished_figs.py).
set -e
# Pinned: some figure code iterates sets of gene-set names, whose order is hash-salted.
export PYTHONHASHSEED=0
ROOT="$(cd "$(dirname "$0")" && pwd)"
export SCRELIABILITY_ROOT="${SCRELIABILITY_ROOT:-$ROOT}"
PY="${PYTHON:-python}"
echo "SCRELIABILITY_ROOT = $SCRELIABILITY_ROOT"
echo "interpreter        = $($PY -c 'import sys; print(sys.executable)')"

echo "=== 1. Split-half reliability (per dataset, parallel) ==="
bash "$ROOT/scripts/01_reliability/run_all_parallel.sh"
echo "--- Systema merge stage (cos_sim + reliability -> triage tables) ---"
$PY "$ROOT/scripts/01_reliability/compute_reliability_genetic_context_systema.py"
$PY "$ROOT/scripts/01_reliability/compute_reliability_cellular_context_systema.py"

echo "=== 1b. Dense reliability grid (Figure 3 / Supplementary Figure 7 parametric fits) ==="
# Dense grid (n_half = 1..50) for the tau^2 fits. Costs about as much as stage 1.
N_HALF_GRID_OVERRIDE="$(seq -s, 1 50)" OUT_BASE="$SCRELIABILITY_ROOT/output/reliability_fig3" \
    bash "$ROOT/scripts/01_reliability/run_all_parallel.sh"

echo "=== 2. Energy-distance reliability (Supplementary Figure 8) ==="
$PY "$ROOT/scripts/02_reliability_edist/compute_e_reliability.py"

echo "=== 3. Preprocess -> intermediate/ benchmark + triage tables ==="
$PY "$ROOT/scripts/03_preprocess/preprocess_01_combine_triage.py"
# Needs the combined_*_2d.csv tables from preprocess_01.
$PY "$ROOT/scripts/02_reliability_edist/build_edist_triage.py"
$PY "$ROOT/scripts/03_preprocess/preprocess_02_merge_benchmarks.py"
$PY "$ROOT/scripts/03_preprocess/preprocess_03_derived_tables.py"
$PY "$ROOT/scripts/03_preprocess/preprocess_04_reliability_model.py"
$PY "$ROOT/scripts/03_preprocess/preprocess_05_bootstrap_ranks.py" --n_boot 1000 --seed 42

echo "=== 4. Retraining experiment (Figure 2 panel e) ==="
bash "$ROOT/scripts/04_retraining/run_all.sh"

# Stage 5 (scripts/05_shared_specific/) regenerates the GSEA tables behind Supplementary Figure 2. It is
# not run here: it needs the ~68 GB of per-dataset h5ads and many hours of GSEA, and the figure reads the
# consolidated JSONs it produces. See scripts/05_shared_specific/README.md for the run order.

echo "=== 6. Figures -> figures/ ==="
cd "$ROOT/scripts/06_figures"
$PY fig1_framework.py
$PY fig2_benchmarks.py
$PY fig2e_reliable_quantity.py            # Fig 2 panel e (reliable-quantity); -> figures/panels/
$PY fig3_prospective.py
# Supplementary figures, named in manuscript appearance order.
$PY figS1_reliability_distributions.py
$PY figS2_shared_vs_specific.py           # shared-vs-specific GSEA (compute: scripts/05_shared_specific/)
$PY figS3_threshold_robustness.py         # manuscript Supplementary Figure 3
$PY figS4_rank_shifts.py
$PY figS5_retraining_grid.py
$PY figS6_data_scaling.py
$PY figS7_parametric_fits.py
$PY figS8_edist.py

echo "✓ Done. Figure panels are in figures/."
