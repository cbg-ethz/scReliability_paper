#!/bin/bash
# Launch split-half reliability compute for all 29 datasets in parallel.
#
# Usage:
#   bash run_all_parallel.sh                                   # -> output/reliability_fig1/
#   OUT_BASE=output/reliability_fig3 bash run_all_parallel.sh  # Fig 3's denser grid
#
# Override the interpreter with PYTHON=/path/to/python; override the data/output
# root with SCRELIABILITY_ROOT (defaults to the repo root inferred from $0).
#
# AFTER this finishes, run the Systema merge stage to produce the triage tables
# that the figure pipeline reads (these are NOT launched here; run once each):
#   $PYTHON "$SCRIPT_DIR/compute_reliability_genetic_context_systema.py"
#   $PYTHON "$SCRIPT_DIR/compute_reliability_cellular_context_systema.py"
# They write systema_reliability_2d_*.csv / systema_cc_reliability_2d_*.csv into
# $OUT_BASE/reliability_{genetic,cellular}_systema/, which preprocess_01 consumes.

set -e

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="${SCRELIABILITY_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_BASE="${OUT_BASE:-$PROJECT_ROOT/output/reliability_fig1}"
LOG_DIR="$OUT_BASE/logs"
CELLULAR_DATA_DIR="$PROJECT_ROOT/data/Wei_et_al_data/cellular_context_preprocessed_h5ad"
mkdir -p "$LOG_DIR" "$OUT_BASE/reliability_genetic" "$OUT_BASE/reliability_cellular"

# Concurrency cap (number of datasets running at once); tune to your machine.
# Total worker processes ≈ MAX_PARALLEL * WORKERS. All three are env-overridable,
# e.g.  MAX_PARALLEL=8 WORKERS=2 bash run_all_parallel.sh
MAX_PARALLEL="${MAX_PARALLEL:-6}"
WORKERS="${WORKERS:-2}"
N_REPEATS="${N_REPEATS:-100}"

# Optional override: pass an n_half grid via env var to compute on a non-default
# grid (e.g. denser sampling for Fig 3). Empty = use script default.
N_HALF_GRID_OVERRIDE="${N_HALF_GRID_OVERRIDE:-}"
GRID_FLAG=""
[ -n "$N_HALF_GRID_OVERRIDE" ] && GRID_FLAG="--n_half_grid $N_HALF_GRID_OVERRIDE"

# Genetic datasets
GENETIC_DATASETS=(
    Norman Adamson
    Replogle_K562essential Replogle_RPE1essential
    Replogle_exp6 Replogle_exp7 Replogle_exp8
    TianActivation TianInhibition
    Frangieh Papalexi Wessels Schmidt
    sciplex3_A549 sciplex3_K562 sciplex3_MCF7 sciplex3_comb
)

# Cellular datasets
CELLULAR_DATASETS=(
    kangCrossCell kangCrossPatient
    Haber Afriat McFarland Parekh TCDD
    crossPatient crossSpecies
    KaggleCrossCell KaggleCrossPatient
    sciplex3
)

# Throttle launcher: enqueue jobs and wait when MAX_PARALLEL reached
launch() {
    local cmd="$1"
    local logfile="$2"
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
    done
    echo "  → $logfile"
    eval "$cmd > '$logfile' 2>&1" &
}

echo "== Genetic context (${#GENETIC_DATASETS[@]} datasets) =="
for ds in "${GENETIC_DATASETS[@]}"; do
    launch "$PYTHON $SCRIPT_DIR/compute_reliability_genetic_context.py \
            --dataset $ds --output_dir $OUT_BASE/reliability_genetic \
            --workers $WORKERS --n_repeats $N_REPEATS --seed 42 $GRID_FLAG" \
           "$LOG_DIR/genetic_${ds}.log"
done

echo "== Cellular context (${#CELLULAR_DATASETS[@]} datasets) =="
for ds in "${CELLULAR_DATASETS[@]}"; do
    launch "$PYTHON $SCRIPT_DIR/compute_reliability_cellular_context.py \
            --dataset $ds --data_dir $CELLULAR_DATA_DIR \
            --output_dir $OUT_BASE/reliability_cellular \
            --workers $WORKERS --n_repeats $N_REPEATS --seed 42 $GRID_FLAG" \
           "$LOG_DIR/cellular_${ds}.log"
done

echo "All jobs launched. Waiting for completion..."
wait
echo "✓ All datasets finished. Next: run the two *_systema.py merge scripts (see header)."
