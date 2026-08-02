"""Threshold-sensitivity sweep for the binary retraining (filtering) experiment.

For each (rho_t, cos_t) on a 5×5 grid, re-derive `quality_label` from per-pert
(ρ, cos) values, run the full pipeline (02 → 03 → 04 → 05 → 06), and aggregate
the cross-dataset Wilcoxon results into a single sensitivity table.

Splits stratification stays at the default threshold (the 5-seed splits are
fixed); only the genuine/random_matched arm definitions and the test-set
quality stratification change with the threshold.

Output:
  results/retraining_quality_filtering/sensitivity/
    rho{R}_cos{C}/  — per-threshold pipeline outputs
    sensitivity_summary.csv  — aggregated cross-dataset stats per cell × threshold
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable  # subprocess the pipeline with the same interpreter
SCR_DIR = Path(__file__).parent
RV2 = SCR_DIR.parents[1] / "results" / "retraining_quality_filtering"


def fmt(v: float) -> str:
    """Filename-safe threshold encoding."""
    return f"{v:.3f}".replace(".", "p")


def run_threshold(rho_t: float, cos_t: float, out_dir: Path,
                   n_splits: int, n_random_matched: int, workers: int,
                   degenerate_tol: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 02_run_experiment with overridden thresholds
    cmd = [PY, str(SCR_DIR / "02_run_experiment.py"),
           "--n-splits", str(n_splits),
           "--n-random-matched", str(n_random_matched),
           "--workers", str(workers),
           "--rho-thresh", str(rho_t),
           "--cos-thresh", str(cos_t),
           "--output-dir", str(out_dir)]
    subprocess.run(cmd, check=True, capture_output=True)

    # 03 → 06 analysis pipeline (each step reads/writes CSVs in out_dir)
    for step, extra in [
        ("03_paired_effects.py",        []),
        ("04_bootstrap_within_split.py", []),
        ("05_dataset_level.py",          []),
        ("06_cross_dataset.py",          ["--degenerate-tol", str(degenerate_tol)]),
    ]:
        cmd = [PY, str(SCR_DIR / step),
               "--input",  str(out_dir / _input_for(step)),
               "--output", str(out_dir / _output_for(step))]
        cmd += extra
        subprocess.run(cmd, check=True, capture_output=True)


def _input_for(step):
    return {
        "03_paired_effects.py":        "per_perturbation_scores.csv",
        "04_bootstrap_within_split.py": "paired_effects_per_perturbation.csv",
        "05_dataset_level.py":          "bootstrap_within_dataset_split.csv",
        "06_cross_dataset.py":          "dataset_level_summary.csv",
    }[step]


def _output_for(step):
    return {
        "03_paired_effects.py":        "paired_effects_per_perturbation.csv",
        "04_bootstrap_within_split.py": "bootstrap_within_dataset_split.csv",
        "05_dataset_level.py":          "dataset_level_summary.csv",
        "06_cross_dataset.py":          "cross_dataset_summary.csv",
    }[step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-grid", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--cos-grid", nargs="+", type=float,
                    default=[0.4, 0.5, 1.0/np.sqrt(2), 0.8, 0.9])
    ap.add_argument("--n-splits",         type=int, default=5)
    ap.add_argument("--n-random-matched", type=int, default=50)
    ap.add_argument("--workers",          type=int, default=10)
    ap.add_argument("--degenerate-tol",   type=float, default=1e-6)
    ap.add_argument("--output-dir",
                    default=str(RV2 / "sensitivity"))
    args = ap.parse_args()

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    print(f"Sweep: {len(args.rho_grid)}×{len(args.cos_grid)} = "
          f"{len(args.rho_grid)*len(args.cos_grid)} threshold combinations")

    summary_rows = []
    t_start = time.time()
    for ri, rho_t in enumerate(args.rho_grid):
        for ci, cos_t in enumerate(args.cos_grid):
            cell_dir = out_root / f"rho_{fmt(rho_t)}_cos_{fmt(cos_t)}"
            t0 = time.time()
            try:
                run_threshold(rho_t, cos_t, cell_dir,
                              args.n_splits, args.n_random_matched,
                              args.workers, args.degenerate_tol)
                cd_path = cell_dir / "cross_dataset_summary.csv"
                if cd_path.exists():
                    cd = pd.read_csv(cd_path)
                    cd["rho_thresh"] = rho_t
                    cd["cos_thresh"] = cos_t
                    summary_rows.append(cd)
                # Also count availability: n_unique_perts in genuine_test
                pe_path = cell_dir / "paired_effects_per_perturbation.csv"
                pe = pd.read_csv(pe_path) if pe_path.exists() else pd.DataFrame()
                if len(pe):
                    avail = (pe[(pe.test_set_type == "genuine_test") &
                                 (pe.comparator == "random_matched") &
                                 (pe.metric == "wei_pcc_delta_top100") &
                                 (pe.model == "linearModel")]
                              .drop_duplicates(["dataset", "perturbation"])
                              .groupby("dataset").size().to_dict())
                else:
                    avail = {}
                (cell_dir / "availability.json").write_text(json.dumps(avail))
                print(f"  ✓ ρ={rho_t:.3f} cos={cos_t:.3f}  "
                      f"({time.time()-t0:.1f}s)  avail={sum(avail.values())} perts "
                      f"across {len(avail)} datasets")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ ρ={rho_t:.3f} cos={cos_t:.3f}  FAILED: "
                      f"{e.stderr.decode()[-300:]}")

    if summary_rows:
        sm = pd.concat(summary_rows, ignore_index=True)
        sm.to_csv(out_root / "sensitivity_summary.csv", index=False)
        print(f"\nSensitivity summary → {out_root/'sensitivity_summary.csv'}  "
              f"({len(sm)} rows)")
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
