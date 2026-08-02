#!/usr/bin/env python3
"""Dataset configuration for the shared-axis vs specific-residual GSEA scripts.

The three `gsea_axis_*` scripts previously imported these names from a helper module that lived outside the
repository, which made them unrunnable from a clean checkout. The definitions are reproduced here verbatim
so this directory is self-contained; the per-dataset column names match those used by
`scripts/01_reliability/compute_reliability_{genetic,cellular}_context.py`.

Paths are anchored on SCRELIABILITY_ROOT, like every other script in the repository.
"""
import os
from pathlib import Path

_ROOT = Path(os.environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))

GEN_DIR = f"{_ROOT}/data/Wei_et_al_data/genetic_context_preprocessed_h5ad"
CEL_DIR = f"{_ROOT}/data/Wei_et_al_data/cellular_context_preprocessed_h5ad"
GENESETS_DIR = f"{_ROOT}/data/genesets"


def is_guide(g):
    """True for guide-level labels that are not real gene targets."""
    return ("posA" in g or "posB" in g or g.startswith("NegCtrl") or "_+_" in g or "_-_" in g)


# genetic / chemical: how to find the perturbation label, the control label, and any condition axis
GEN_STD = dict(pert="perturbation", ctrl="control", cond=None)
GEN_CONF = {
    "Adamson": GEN_STD, "Norman": GEN_STD, "Papalexi": GEN_STD,
    "Replogle_K562essential": GEN_STD, "Replogle_RPE1essential": GEN_STD,
    "Replogle_exp6": GEN_STD, "Replogle_exp7": GEN_STD, "Replogle_exp8": GEN_STD,
    "Schmidt": GEN_STD, "TianActivation": GEN_STD, "TianInhibition": GEN_STD, "Wessels": GEN_STD,
    "Frangieh": dict(pert="perturbation", ctrl="control", cond="condition"),
    "sciplex3_A549": dict(pert="cov_drug_dose_name", ctrl="A549_control_1.0", cond=None),
    "sciplex3_K562": dict(pert="cov_drug_dose_name", ctrl="K562_control_1.0", cond=None),
    "sciplex3_MCF7": dict(pert="cov_drug_dose_name", ctrl="MCF7_control_1.0", cond=None),
    "sciplex3_comb": dict(pert="cov_drug_dose_name", ctrl="A549_control_1.0", cond=None),
}

# cellular: (context_col, condition_col, ctrl_label)
CEL_CONF = {
    "kangCrossCell": ("cell_type", "condition", "control"),
    "kangCrossPatient": ("sample_id", "perturbation", "control"),
    "Haber": ("cell_type", "condition", "Control"),
    "Afriat": ("condition1", "perturbation", "control"),
    "McFarland": ("cell_line", "perturbation", "control"),
    "Parekh": ("cell_type", "perturbation", "CTRL"),
    "TCDD": ("celltype", "perturbation", "control"),
    "crossPatient": ("patient", "perturbation", "control"),
    "crossSpecies": ("condition1", "condition2", "control"),
    "KaggleCrossCell": ("cell_type", "perturbation", "control"),
    "KaggleCrossPatient": ("donor_id", "perturbation", "control"),
    "sciplex3": ("cell_line", "condition2", "control"),
}


def load_hallmark():
    """Hallmark gene sets, upper-cased, from the tracked Enrichr export (see data/README.md)."""
    import json
    with open(f"{GENESETS_DIR}/genesets.json") as fh:
        return {t: [g.upper() for g in gs] for t, gs in json.load(fh)["Hallmark"].items()}
