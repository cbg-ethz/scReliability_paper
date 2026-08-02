# Retraining experiments

Tests whether filtering training targets by reliability class improves a perturbation-prediction model.
Run everything with `bash run_all.sh` from this directory's parent chain (or via the top-level `run_all.sh`).

## Steps

| step | writes | feeds |
|---|---|---|
| `01_make_splits.py` | `results/retraining_quality_filtering/splits/` | everything below (**tracked in git** — these define the published experiment) |
| `02_run_experiment.py` | `results/retraining_quality_filtering/per_perturbation_scores.csv` | 03-06 |
| `03_paired_effects.py` | paired-effect tables | 04-06 |
| `04_bootstrap_within_split.py` | bootstrap CIs | 05, 06 |
| `05_dataset_level.py` | per-dataset summary | 06 |
| `06_cross_dataset.py` | cross-dataset inference | — |
| `10_reliable_quantity_sweep.py` | `intermediate/retraining/reliable_quantity_sweep.csv` | **Figure 2 panel e** |
| `11_scaling_pools.py` | `intermediate/retraining/scaling_pools.csv` | **Supplementary Figure 6** |
| `12_exhaustive_linear.py` | `intermediate/retraining/exhaustive_linear_scores.csv` | **Supplementary Figure 5** (linear rows) |

`08_threshold_sensitivity.py` and `09_rigor_tables.py` write standalone tables that no figure reads; run
them directly if wanted.

## The one input this pipeline does not produce

Supplementary Figure 5 also reads:

```
intermediate/retraining/deep_per_perturbation_scores.csv
```

These are per-perturbation scores for the two deep models (**GEARS, scGPT**) retrained under the same
quality-filtering arms as the linear model. Those runs use Systema (`run_gears_arm.py`, `run_scgpt.py`) on the splits in
`results/retraining_quality_filtering/splits/`. They require a GPU and are **not** part of `run_all.sh`; the linear arms in the same figure come from `12_exhaustive_linear.py`,
which does run here. The table covers 2 models x 7 genetic datasets (12,246 rows).

CPA was retrained too but is excluded from the figure, because the arms are evaluated on unseen
perturbations, which CPA is not designed to predict. Its scores are under
`intermediate/retraining/archive/`; it is still included in the Wei and Ahlmann-Eltze re-analyses.

Everything else in Supplementary Figure 5, and every other figure in the paper, is reproducible from this
repository given the raw data (see `data/README.md`).

The split files use the pipeline's internal label vocabulary. It maps to the three classes in the
paper as: `genuine` / `genuine_signal` -> Specific, `trivial` / `falsely_solved` -> Shared,
`unreliable` / `unmeasurable` -> Unreliable (`canonical_quality` in `scripts/03_preprocess/config.py`).
