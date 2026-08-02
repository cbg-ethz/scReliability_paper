#!/usr/bin/env bash
# Run the reliable-vs-all retraining experiment end-to-end.
# Outputs to results/retraining_quality_filtering/. The per-perturbation scores
# feed Figure 2 panel e, rendered by scripts/06_figures/fig2_benchmarks.py.
#
# Override the interpreter with PYTHON=/path/to/python; the root with SCRELIABILITY_ROOT.
set -e
ROOT="${SCRELIABILITY_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
SCR="scripts/04_retraining"

echo "── 01: make stratified splits ──"
$PY $SCR/01_make_splits.py --n-splits 5

echo "── 02: run experiment (parallel across datasets × splits) ──"
$PY $SCR/02_run_experiment.py --n-splits 5 --n-random-matched 50 --workers 10

echo "── 03: paired effects ──"
$PY $SCR/03_paired_effects.py

echo "── 04: bootstrap within (dataset, split) ──"
$PY $SCR/04_bootstrap_within_split.py --n-bootstrap 10000

echo "── 05: dataset-level summary ──"
$PY $SCR/05_dataset_level.py

echo "── 06: cross-dataset inference ──"
$PY $SCR/06_cross_dataset.py --degenerate-tol 1e-6

# Steps 10 and 11 write to intermediate/retraining/ and are the only producers of the two tables the
# quantity figures read: 10 -> reliable_quantity_sweep.csv (Figure 2 panel e, and the fig2e standalone),
# 11 -> scaling_pools.csv (Supplementary Figure 6). Both build a pseudobulk cache on first run.
echo "── 10: quantity-within-reliable sweep (Figure 2 panel e) ──"
$PY $SCR/10_reliable_quantity_sweep.py

echo "── 11: data-scaling by training pool (Supplementary Figure 6) ──"
$PY $SCR/11_scaling_pools.py

echo "── 12: exhaustive linear retraining strategies (Supplementary Figure 5) ──"
$PY $SCR/12_exhaustive_linear.py

# 08 (threshold sensitivity) and 09 (rigor tables) write standalone CSVs that no figure reads; run them
# directly if those tables are wanted.
#
# Supplementary Figure 5 also needs intermediate/retraining/deep_per_perturbation_scores.csv, the
# GEARS / scGPT retraining scores. Those runs need a GPU and are not part of this pipeline; see
# scripts/04_retraining/README.md.

echo "✓ retraining results in results/retraining_quality_filtering/ and intermediate/retraining/"
echo "  Figure 2 panel e is rendered by scripts/06_figures/fig2_benchmarks.py."
