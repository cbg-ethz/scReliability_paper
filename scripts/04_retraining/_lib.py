"""
_lib.py — shared utilities for the v2 retraining pipeline.

Model implementations used by the retraining experiments.
(Wei + AE exact). New additions: perturbation-type annotation, stratified
sampling (triage_2d × pert_type), AE L2 on top-1000 expressed and Wei PCC-Δ
on top-100 DEGs as the two primary metrics.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__import__("os").environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
H5AD_DIR     = BASE_DIR / "data" / "Wei_et_al_data" / "genetic_context_preprocessed_h5ad"
QUALITY_GEN  = BASE_DIR / "intermediate" / "combined_genetic_2d.csv"
QUALITY_CEL  = BASE_DIR / "intermediate" / "combined_cellular_2d.csv"
RESULTS_DIR  = BASE_DIR / "results" / "retraining_quality_filtering"
SPLITS_DIR   = RESULTS_DIR / "splits"

# ── Hyperparameters (AE defaults; promoted to constants for clarity) ──────────
K_PCA               = 10
RIDGE_LAMBDA        = 0.1
MIN_CELLS_PER_PERT  = 3
N_TOP_DEGS_WEI      = 100
N_TOP_EXPR_AE       = 1000

# ── Triage label mapping (file convention → user-facing) ──────────────────────
TRIAGE_FILE_TO_USER = {
    "Specific": "genuine",
    "Shared": "trivial",
    "Unreliable":   "unreliable",
}

DATASETS = [
    "Adamson", "Norman",
    "Replogle_K562essential", "Replogle_RPE1essential",
    "Replogle_exp6", "Replogle_exp7", "Replogle_exp8",
    "Schmidt", "Wessels",
]

H5AD_FILE = {ds: f"{ds}.h5ad" for ds in DATASETS}


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation parsing & typing
# ──────────────────────────────────────────────────────────────────────────────
def parse_pert_genes(pert_name: str) -> list[str]:
    """'BRCA1' → ['BRCA1']; 'AHR+CEBPE' → ['AHR','CEBPE']; 'control' → [] ."""
    if pert_name == "control":
        return []
    return [g.strip() for g in str(pert_name).split("+") if g.strip() and g.strip() != "control"]


def classify_pert_type(pert_name: str) -> str:
    """'single' if 1 gene; 'double' if 2+; 'control' if no genes."""
    g = parse_pert_genes(pert_name)
    if not g:
        return "control"
    return "single" if len(g) == 1 else "double"


# ──────────────────────────────────────────────────────────────────────────────
# Quality labels
# ──────────────────────────────────────────────────────────────────────────────
def load_quality_table() -> pd.DataFrame:
    """Per-(dataset, perturbation) triage label table.
    Columns: dataset, perturbation, triage_2d (file convention), quality_label
    (user-facing: genuine/trivial/unreliable), pert_type (single/double).
    """
    parts = []
    for f in [QUALITY_GEN, QUALITY_CEL]:
        if f.exists():
            d = pd.read_csv(f)[["dataset", "perturbation", "triage_2d"]]
            parts.append(d)
    q = pd.concat(parts, ignore_index=True)
    q["quality_label"] = q["triage_2d"].map(TRIAGE_FILE_TO_USER)
    q["pert_type"]     = q["perturbation"].apply(classify_pert_type)
    return q


# ──────────────────────────────────────────────────────────────────────────────
# Stratified train/test split by (quality_label × pert_type)
# ──────────────────────────────────────────────────────────────────────────────
def make_stratified_split(perts_df: pd.DataFrame,
                           split_seed: int,
                           test_frac: float = 0.25) -> pd.DataFrame:
    """Stratified split by (quality_label × pert_type). Returns perts_df with
    a 'split' column ('train' or 'test'). Each (label, type) stratum is
    randomly partitioned to keep proportions balanced across the split.
    """
    rng = np.random.default_rng(split_seed)
    out = perts_df.copy()
    out["_stratum"] = out["quality_label"].astype(str) + "::" + out["pert_type"].astype(str)
    out["split"] = ""
    for stratum, grp in out.groupby("_stratum"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) >= 2 else 0
        out.loc[idx[:n_test], "split"] = "test"
        out.loc[idx[n_test:], "split"] = "train"
    return out.drop(columns=["_stratum"])


# ──────────────────────────────────────────────────────────────────────────────
# Stratified random subset sampling (matched on pert_type)
# ──────────────────────────────────────────────────────────────────────────────
def stratified_subset(pool: pd.DataFrame,
                       target: pd.DataFrame,
                       seed: int) -> list[str] | None:
    """Sample from `pool` (DataFrame with 'perturbation','pert_type') a
    subset whose pert_type counts match those of `target`. Returns a list
    of perturbation names, or None if the pool can't supply enough perts
    of any required type.

    `target`: the genuine_train DataFrame (same columns).
    """
    rng = np.random.default_rng(seed)
    target_counts = target["pert_type"].value_counts().to_dict()
    chosen: list[str] = []
    for ptype, n in target_counts.items():
        candidates = pool[pool["pert_type"] == ptype]["perturbation"].to_numpy()
        # Exclude any perts already in target (must be drawn from disjoint pool
        # for random_nongenuine; for random_matched, pool is full train and may
        # overlap genuine — that overlap is fine and expected).
        if len(candidates) < n:
            return None
        chosen.extend(rng.choice(candidates, size=n, replace=False).tolist())
    return chosen


LEGACY_QUALITY_MAP = {
    "genuine": "Specific", "genuine_signal": "Specific",
    "trivial": "Shared",   "falsely_solved": "Shared",
    "unreliable": "Unreliable", "unmeasurable": "Unreliable",
}


def canonical_quality(s):
    """Map any stored quality/triage label onto the canonical Unreliable/Shared/Specific vocabulary."""
    return s.map(lambda v: LEGACY_QUALITY_MAP.get(v, v))


def deterministic_subseed(*parts) -> int:
    """64-bit deterministic seed from any tuple of strings/ints."""
    s = "::".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Pseudobulk
# ──────────────────────────────────────────────────────────────────────────────
def dense_mean(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X).mean(axis=0)


def compute_pseudobulks(adata: ad.AnnData,
                         perts: Iterable[str],
                         ctrl_mean: np.ndarray) -> tuple[dict, dict]:
    """Per-perturbation pseudobulk delta (mean−ctrl) and raw expression mean.
    Skips perturbations with fewer than MIN_CELLS_PER_PERT cells.
    Returns (delta_dict, expr_dict)."""
    deltas, exprs = {}, {}
    pert_col = adata.obs["perturbation"]
    for p in perts:
        mask = (pert_col == p).to_numpy()
        if mask.sum() < MIN_CELLS_PER_PERT:
            continue
        m = dense_mean(adata[mask].X)
        exprs[p]  = m
        deltas[p] = m - ctrl_mean
    return deltas, exprs


# ──────────────────────────────────────────────────────────────────────────────
# Models (Ahlmann-Eltze formulation)
# ──────────────────────────────────────────────────────────────────────────────
def fit_trainmean(train_deltas: np.ndarray) -> np.ndarray:
    """trainMean baseline: mean of training deltas. Same prediction for all test."""
    return train_deltas.mean(axis=0)


def fit_linear_model(train_exprs: np.ndarray,
                      train_deltas: np.ndarray,
                      train_perts: list[str],
                      gene_names: list[str]) -> dict | None:
    """Ahlmann-Eltze bilinear linear model — exact AE implementation.

    G  = U[:, :K] * S[:K]  from SVD of column-centered expression matrix.
    b  = mean of training deltas.
    P  = G rows for perturbation target genes (mean over genes for combos).
    W  = (G^T G + λI)^{-1} G^T (Y_delta − b) P (P^T P + λI)^{-1}.
    pred = G W p_tilde + b   (delta space).
    """
    assert train_exprs.shape == train_deltas.shape
    n_perts, n_genes = train_deltas.shape
    if n_perts < 2:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    X_train   = train_exprs.T
    col_means = X_train.mean(axis=0)
    X_c       = X_train - col_means[None, :]

    K = min(K_PCA, n_perts - 1, n_genes - 1)
    if K < 1:
        return None
    try:
        U, S, _ = np.linalg.svd(X_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    G = U[:, :K] * S[:K][None, :]
    b = train_deltas.mean(axis=0)

    pert_to_idx   = {p: i for i, p in enumerate(train_perts)}
    P_rows, valid_col_idx = [], []
    for p in train_perts:
        gidx = [gene_to_idx[g] for g in parse_pert_genes(p) if g in gene_to_idx]
        if not gidx:
            continue
        P_rows.append(G[gidx, :].mean(axis=0))
        valid_col_idx.append(pert_to_idx[p])

    if len(P_rows) < K + 1:
        return None
    P = np.array(P_rows)

    Y       = train_deltas.T[:, valid_col_idx]
    Y_c     = Y - b[:, None]
    GtG     = G.T @ G + RIDGE_LAMBDA * np.eye(K)
    PtP     = P.T @ P + RIDGE_LAMBDA * np.eye(K)
    GtYP    = (G.T @ Y_c) @ P
    try:
        W = np.linalg.solve(GtG, GtYP) @ np.linalg.inv(PtP)
    except np.linalg.LinAlgError:
        return None

    return {"G": G, "W": W, "b": b, "gene_to_idx": gene_to_idx, "K": K}


# ----------------------------------------------------------------------------
# Reduced-rank linear model with optional per-perturbation weights on the ridge solve.
# weights=None reproduces the unweighted fit_linear_model above.
# ----------------------------------------------------------------------------
def fit_linear_model_weighted(train_exprs, train_deltas, train_perts, gene_names, weights=None):
    """AE reduced-rank linear model with optional per-perturbation weights on the
    ridge solve. weights=None -> identical to the unweighted _lib.fit_linear_model."""
    n_perts, n_genes = train_deltas.shape
    if n_perts < 2:
        return None
    w = np.ones(n_perts) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    if w.sum() <= 0:
        return None
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    X_train = train_exprs.T
    X_c = X_train - X_train.mean(axis=0)[None, :]
    K = min(K_PCA, n_perts - 1, n_genes - 1)
    if K < 1:
        return None
    try:
        U, S, _ = np.linalg.svd(X_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    G = U[:, :K] * S[:K][None, :]
    b = (w[:, None] * train_deltas).sum(0) / w.sum()          # weighted per-gene center
    pert_to_idx = {p: i for i, p in enumerate(train_perts)}
    P_rows, valid = [], []
    for p in train_perts:
        gidx = [gene_to_idx[g] for g in parse_pert_genes(p) if g in gene_to_idx]
        if not gidx:
            continue
        P_rows.append(G[gidx, :].mean(0)); valid.append(pert_to_idx[p])
    if len(P_rows) < K + 1:
        return None
    P = np.array(P_rows)
    wv = w[valid]
    if wv.sum() <= 0:
        return None
    wv = wv * (len(wv) / wv.sum())                            # normalize mean 1
    Y_c = train_deltas.T[:, valid] - b[:, None]
    GtG  = G.T @ G + RIDGE_LAMBDA * np.eye(K)
    PtP  = P.T @ (P * wv[:, None]) + RIDGE_LAMBDA * np.eye(K)  # P' diag(w) P + λI
    GtYP = (G.T @ Y_c) @ (P * wv[:, None])                     # G' Y_c diag(w) P
    try:
        W = np.linalg.solve(GtG, GtYP) @ np.linalg.inv(PtP)
    except np.linalg.LinAlgError:
        return None
    return {"G": G, "W": W, "b": b, "gene_to_idx": gene_to_idx, "K": K}


def predict_linear_model(lm: dict, pert_name: str) -> np.ndarray | None:
    """Predict delta for one SINGLE test perturbation.

    Strict AE behavior: returns None for combinatorial perturbations
    (run_linear_pretrained_model.R line 137 only matches single-gene
    `clean_condition` against pert_emb columns; combos yield NA).
    """
    genes = parse_pert_genes(pert_name)
    if len(genes) != 1:
        return None
    g = genes[0]
    if g not in lm["gene_to_idx"]:
        return None
    p_tilde = lm["G"][lm["gene_to_idx"][g], :]
    return lm["G"] @ lm["W"] @ p_tilde + lm["b"]


# ──────────────────────────────────────────────────────────────────────────────
# Additive model — combinatorial baseline (AE: run_additive_model.py)
#   pred(A+B) in expression = baseline + (single_A_obs − baseline) + (single_B_obs − baseline)
#   in delta space:           single_A_delta + single_B_delta
#   Requires both single A and single B to exist in the training set.
# ──────────────────────────────────────────────────────────────────────────────
def fit_additive_model(train_perts: list[str],
                        delta_dict: dict) -> dict:
    """Build {gene_name: single_pert_delta} lookup from training perturbations.
    Only single-gene training perts are eligible to populate the lookup.
    """
    lookup: dict[str, np.ndarray] = {}
    for p in train_perts:
        genes = parse_pert_genes(p)
        if len(genes) == 1 and p in delta_dict:
            lookup[genes[0]] = delta_dict[p]
    return {"single_lookup": lookup}


def predict_additive_model(am: dict, pert_name: str) -> np.ndarray | None:
    """Predict delta for a combinatorial perturbation as the sum of its
    component single-perturbation deltas. Returns None if any component's
    single is absent from training (strict pairing).
    """
    genes = parse_pert_genes(pert_name)
    if len(genes) < 2:
        return None
    deltas = []
    for g in genes:
        if g not in am["single_lookup"]:
            return None
        deltas.append(am["single_lookup"][g])
    return np.sum(deltas, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def wei_pcc_delta_top100(y_pred_delta, y_true_delta, deg_idx):
    if deg_idx is None or len(deg_idx) < 2:
        return float("nan")
    return _corr(y_pred_delta[deg_idx], y_true_delta[deg_idx])


def wei_mse_top100(y_pred_delta, y_true_delta, deg_idx):
    """Wei MSE on top 100 DEGs (same DEG set as PCC-delta)."""
    if deg_idx is None or len(deg_idx) < 2:
        return float("nan")
    d = y_pred_delta[deg_idx] - y_true_delta[deg_idx]
    return float(np.mean(d * d))


def ae_l2_top1000(y_pred_delta, y_true_delta, top_expr_idx):
    d = y_pred_delta[top_expr_idx] - y_true_delta[top_expr_idx]
    return float(np.sqrt(np.sum(d * d)))


def ae_pearson_dlt_top1000(y_pred_delta, y_true_delta, top_expr_idx):
    """AE Pearson-delta = cor(pred_delta, true_delta) on top 1000 expressed."""
    return _corr(y_pred_delta[top_expr_idx], y_true_delta[top_expr_idx])


def ae_r2_raw_top1000(y_pred_delta, y_true_delta, ctrl_mean, top_expr_idx):
    """AE r² raw = cor(pred_expr, obs_expr) on top 1000 expressed,
    where pred_expr = ctrl_mean + pred_delta and obs_expr = ctrl_mean + true_delta."""
    pred_expr = ctrl_mean[top_expr_idx] + y_pred_delta[top_expr_idx]
    obs_expr  = ctrl_mean[top_expr_idx] + y_true_delta[top_expr_idx]
    return _corr(pred_expr, obs_expr)


# ──────────────────────────────────────────────────────────────────────────────
# Common-DEGs (Wei et al.) — pseudobulk adaptation. True DEGs are Wei's t-test
# top-N set (deg_idx, from rank_genes_groups on the real cells, calDEG/getDEG).
# Predicted DEGs are the model's top-N genes by |predicted delta| (the natural
# ranking for a pseudobulk mean predictor, which produces no per-cell variance).
# Score = recovery fraction |pred_top ∩ true_top| / N, in [0, 1].
# ──────────────────────────────────────────────────────────────────────────────
def wei_common_degs(y_pred_delta, deg_idx, n_top: int = N_TOP_DEGS_WEI):
    if deg_idx is None or len(deg_idx) < 2:
        return float("nan")
    n = min(n_top, len(deg_idx))
    pred_top = set(np.argsort(-np.abs(y_pred_delta))[:n].tolist())
    true_top = set(np.asarray(deg_idx)[:n].tolist())
    return len(pred_top & true_top) / n


# ──────────────────────────────────────────────────────────────────────────────
# Centroid accuracy (Vinas Torné et al., Nat Biotechnol 2025) — their PROPOSED
# metric. For a set of test perturbations with predicted and ground-truth effect
# (delta) vectors, the accuracy for perturbation i is the fraction of OTHER
# perturbations whose ground-truth centroid is farther from prediction i than i's
# own centroid. Euclidean distance on deltas equals distance on post-perturbation
# profiles (the constant control mean cancels).
# Source: github.com/mlbio-epfl/systema, evaluation/centroid_accuracy.py
# ──────────────────────────────────────────────────────────────────────────────
def centroid_accuracy_for_set(pred_mat: np.ndarray,
                              gt_mat: np.ndarray) -> np.ndarray:
    """pred_mat, gt_mat: (n_perts, n_genes), SAME perturbation order.
    Returns an array of per-perturbation centroid accuracies in [0, 1].
    Matches Systema's centroid_accuracy.py: score = (#{j: d(pred_i, gt_j) >
    d(pred_i, gt_i)}) / (n - 1)."""
    n = pred_mat.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    D = cdist(pred_mat, gt_mat, metric="euclidean")   # D[i, j] = dist(pred_i, gt_j)
    self_d = np.diag(D)
    return (D > self_d[:, None]).sum(axis=1) / (n - 1)


def compute_wei_deg_idx(adata: ad.AnnData,
                         test_perts: list[str],
                         gene_to_col: dict,
                         k_top: int = N_TOP_DEGS_WEI) -> dict:
    """For each test perturbation, return the gene indices of the top-K DEGs by
    |t-stat| from sc.tl.rank_genes_groups vs control (Wei's exact method)."""
    if not test_perts:
        return {}
    pert_mask = adata.obs["perturbation"].isin(test_perts + ["control"])
    sub = adata[pert_mask].copy()
    sc.tl.rank_genes_groups(sub, "perturbation", groups=test_perts,
                            reference="control", method="t-test", use_raw=False)
    rgg = sub.uns["rank_genes_groups"]
    out = {}
    for p in test_perts:
        names  = np.asarray(rgg["names"][p])
        scores = np.asarray(rgg["scores"][p])
        order  = np.argsort(-np.abs(scores))
        k = min(k_top, len(names))
        out[p] = np.array([gene_to_col[g] for g in names[order[:k]]
                           if g in gene_to_col])
    return out


def compute_ae_top_expr_idx(ctrl_mean: np.ndarray,
                             k_top: int = N_TOP_EXPR_AE) -> np.ndarray:
    """Top-K most highly expressed genes in control. AE convention."""
    n = min(k_top, ctrl_mean.shape[0])
    return np.argsort(ctrl_mean)[-n:]


# ──────────────────────────────────────────────────────────────────────────────
# Config dump
# ──────────────────────────────────────────────────────────────────────────────
def dump_config(cfg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "config.json"
    p.write_text(json.dumps(cfg, indent=2, default=str))
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-sensitivity helpers (used by the sensitivity sweep; not part of the
# main pipeline's default behavior). Allow re-deriving `quality_label` from
# (ρ, cos) values at runtime, given user-supplied thresholds.
# ──────────────────────────────────────────────────────────────────────────────
def load_rho_cos_lookup() -> pd.DataFrame:
    """Per (dataset, perturbation) → ρ (sb_from_median_all_genes) and
    cos_sim (with-p; LOO bias is small per the tmp-weighted study and not
    relevant to threshold-sensitivity sweep design)."""
    parts = []
    for f in [QUALITY_GEN, QUALITY_CEL]:
        if f.exists():
            d = pd.read_csv(f)
            cols = ["dataset", "perturbation"]
            if "sb_from_median_all_genes" in d.columns:
                d = d.rename(columns={"sb_from_median_all_genes": "rho"})
                cols.append("rho")
            if "cos_sim" in d.columns:
                d = d.rename(columns={"cos_sim": "cos"})
                cols.append("cos")
            parts.append(d[cols])
    return pd.concat(parts, ignore_index=True)


def derive_quality_label_from_thresh(rho: float, cos: float,
                                       rho_thresh: float,
                                       cos_thresh: float) -> str:
    """Re-derive quality_label given user-supplied thresholds."""
    if pd.isna(rho) or rho < rho_thresh:
        return "unreliable"
    if pd.isna(cos) or cos >= cos_thresh:
        return "trivial"
    return "genuine"
