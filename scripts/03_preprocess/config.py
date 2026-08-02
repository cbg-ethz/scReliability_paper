"""
Central configuration — publication figure style.

Font sizes calibrated for a 7.09" full-width figure.
Every figure is saved in three formats by _paths.save_fig / save_fig_panel:
  - PDF  : journal submission asset (TrueType fonts, pdf.fonttype=42).
  - SVG  : editable master for PowerPoint / Illustrator / Inkscape
           (svg.fonttype='none' keeps labels as <text>, not outlined paths).
  - PNG  : 300 dpi raster preview for Slack / Google Docs.
"""
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ══════════════════ PATHS ══════════════════
BASE_DIR = Path(__import__("os").environ.get("SCRELIABILITY_ROOT", Path(__file__).resolve().parents[2]))
WEI_DIR              = BASE_DIR / 'data' / 'Wei_et_al_data' / 'Results_from_original_study'
AE_XLSX              = BASE_DIR / 'data' / 'Ahlmann-Eltze_data' / 'Source Data Fig.2.xlsx'
# Fig 1 inputs (canonical 7-point split-half reliability + Systema 2D triage).
GENETIC_SYSTEMA_DIR  = BASE_DIR / 'output' / 'reliability_fig1' / 'reliability_genetic_systema'
CELLULAR_SYSTEMA_DIR = BASE_DIR / 'output' / 'reliability_fig1' / 'reliability_cellular_systema'

# Shared intermediates (preprocess derived tables for fig1/fig2/figS*).
INTERMEDIATE_DIR     = BASE_DIR / 'intermediate'

# Figure outputs.
OUT_DIR              = BASE_DIR / 'figures'
PANEL_DIR            = BASE_DIR / 'figures' / 'panels'

# Fig 3 inputs (raw reliability curves at fig 3's grid + parametric fits).
# Fig 3's grid may be denser than Fig 1's so the parametric ρ(N) fit has
# enough leverage; Fig 1 only needs ρ at max feasible N for triage.
RELIABILITY_GEN_DIR  = BASE_DIR / 'output' / 'reliability_fig3' / 'reliability_genetic'
RELIABILITY_CEL_DIR  = BASE_DIR / 'output' / 'reliability_fig3' / 'reliability_cellular'
FITS_DIR             = BASE_DIR / 'output' / 'reliability_fig3' / 'parametric_fits'

def intermediate_path(name): return str(INTERMEDIATE_DIR / name)

# ══════════════════ DATASET MAPPINGS ══════════════════
AE_DATASET_MAP = {
    'adamson': 'Adamson',
    'replogle_k562_essential': 'Replogle_K562essential',
    'replogle_rpe1_essential': 'Replogle_RPE1essential',
    'norman_from_scfoundation': 'Norman',  # AE doubles benchmark (perturbation_prediction.xlsx)
}
AE_XLSX_DOUBLES = BASE_DIR / 'data' / 'Ahlmann-Eltze_data' / 'Source Data Fig.4.xlsx'
def extract_perturbation_from_wei_op(op):
    return re.sub(r'_\d+$', '', str(op))
def clean_ae_perturbation(pert):
    return re.sub(r'\+ctrl$', '', str(pert))

# ══════════════════ TRIAGE ══════════════════
# ρ = 0.5 (Spearman-Brown): "at least 50% of variance is reproducible across splits"
# cos θ = 1/√2 (variance equipartition): "at most 50% of variance aligns with the
#   systematic axis" — internally consistent with the ρ = 0.5 variance-fraction framework,
#   since cos²θ is the fraction of perturbation variance explained by δ_avg.
RELIABILITY_THRESH = 0.5
COSSIM_HIGH_THRESH = 1.0 / np.sqrt(2)   # ≈ 0.7071

TRIAGE_ORDER = ['Unreliable', 'Shared', 'Specific']
TRIAGE_COLORS = {'Unreliable':'#d64541','Shared':'#f39c12','Specific':'#1e8449'}

def derive_triage_2d(df, rho_col, cos_col='cos_sim',
                     rho_thresh=None, cos_thresh=None):
    """Assign triage_2d labels using project-wide thresholds.
    Returns a Series of {'Unreliable','Shared','Specific'}."""
    import numpy as np
    rho_t = RELIABILITY_THRESH if rho_thresh is None else rho_thresh
    cos_t = COSSIM_HIGH_THRESH if cos_thresh is None else cos_thresh
    reliable = df[rho_col] >= rho_t
    high_sys = df[cos_col] >= cos_t
    return np.select(
        [~reliable, reliable & high_sys, reliable & ~high_sys],
        ['Unreliable', 'Shared', 'Specific'],
        default='other'
    )

# ══════════════════ METHOD CATEGORIES ══════════════════
METHOD_CATEGORY = {
    'scGPT':'Foundation','GeneCompass':'Foundation','scELMo':'Foundation',
    'scgpt':'Foundation','geneformer':'Foundation','uce':'Foundation',
    'scbert':'Foundation','uce33':'Foundation',
    'GEARS':'Task-specific','CPA':'Task-specific','scouter':'Task-specific',
    'GenePert':'Task-specific','AttentionPert':'Task-specific',
    'gears':'Task-specific','cpa':'Task-specific',
    'linearModel':'Simple','biolord':'Simple','lpm_selftrained':'Simple',
    'lpm_k562PertEmb':'LPM','lpm_rpe1PertEmb':'LPM',
    'lpm_gearsPertEmb':'LPM','lpm_scgptGeneEmb':'LPM',
    'lpm_scFoundationGeneEmb':'LPM','lpm_randomGeneEmb':'LPM',
    'lpm_randomPertEmb':'LPM',
    'CellOT':'Task-specific','trVAE':'VAE','scPreGAN':'Task-specific',
    'inVAE':'VAE','scPRAM':'Task-specific','scGen':'VAE',
    'scVIDR':'VAE','SCREEN':'Task-specific','scDisInFact':'VAE',
    'mean':'Simple',
}
AE_ARCH_GROUP = {
    'scgpt':'Foundation model','geneformer':'Foundation model',
    'uce':'Foundation model','uce33':'Foundation model','scbert':'Foundation model',
    'gears':'Perturbation-specific','cpa':'Perturbation-specific',
    'lpm_selftrained':'Perturbation-specific','lpm_randomPertEmb':'Perturbation-specific',
    'lpm_randomGeneEmb':'Perturbation-specific','lpm_scgptGeneEmb':'Foundation model',
    'lpm_scFoundationGeneEmb':'Foundation model','lpm_gearsPertEmb':'Perturbation-specific',
    'lpm_k562PertEmb':'Perturbation-specific','lpm_rpe1PertEmb':'Perturbation-specific',
    'mean':'Perturbation-specific',
}
CATEGORY_COLORS = {'Foundation':'#3498db','Task-specific':'#2ecc71','Simple':'#95a5a6','LPM':'#9b59b6','VAE':'#e67e22'}
def method_color(m):
    return CATEGORY_COLORS.get(METHOD_CATEGORY.get(m,'Simple'),'#95a5a6')

# ══════════════════ ACCENT COLORS ══════════════════
C_POS='#1e8449'; C_NEG='#d64541'; C_NEUTRAL='#7f8c8d'
CTX_COLORS = {'genetic': '#4878CF', 'cellular': '#D65F5F'}
SKIP_GEN = {'baseControl','trainMean','baseMLP','baseReg','scFoundation'}

# ---- Figure style ----
NM_FULL_W = 7.09
NM_COL_W  = 3.46

# Font sizes (pt) — calibrated for a 7" figure width
NM_PANEL  = 12     # panel letter
NM_TITLE  = 9      # subplot titles
NM_LABEL  = 8      # axis labels
NM_TICK   = 7      # tick labels
NM_LEGEND = 6.5    # legend entries
NM_ANNOT  = 6      # bar annotations
NM_TINY   = 5      # heatmap cells

def setup_style():
    plt.rcParams.update({
        'figure.dpi':150, 'savefig.dpi':300,
        'font.family':'sans-serif',
        'font.sans-serif':['Arial','Helvetica Neue','Helvetica','DejaVu Sans'],
        'font.size':NM_TICK,
        'mathtext.default':'regular',
        'text.usetex':False, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'axes.titlesize':NM_TITLE, 'axes.titleweight':'bold',
        'axes.labelsize':NM_LABEL,
        'xtick.labelsize':NM_TICK, 'ytick.labelsize':NM_TICK,
        'legend.fontsize':NM_LEGEND,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':0.6,
        'xtick.major.width':0.6, 'ytick.major.width':0.6,
        'xtick.major.size':3.5, 'ytick.major.size':3.5,
        'xtick.direction':'out', 'ytick.direction':'out',
        'legend.frameon':False, 'legend.borderpad':0.3,
        'legend.handletextpad':0.4, 'legend.handlelength':1.2,
        'legend.labelspacing':0.35,
        'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
    })

# ══════════════════ HELPERS ══════════════════
# The retraining pipeline's stored files use an older label vocabulary. Translate on read so that all
# downstream code compares against the canonical Unreliable/Shared/Specific names.
LEGACY_QUALITY_MAP = {
    'genuine': 'Specific', 'genuine_signal': 'Specific',
    'trivial': 'Shared',   'falsely_solved': 'Shared',
    'unreliable': 'Unreliable', 'unmeasurable': 'Unreliable',
}


def canonical_quality(s):
    """Map any stored quality/triage label onto the canonical vocabulary."""
    return s.map(lambda v: LEGACY_QUALITY_MAP.get(v, v))


def rename_triage(df, col='triage_2d'):
    """No-op kept for backward compatibility: triage_2d already stores the
    canonical Unreliable/Shared/Specific labels."""
    return df


def get_baseline_dict(df, metric, bl_method, pert_col):
    # Wei's `op` encodes perturbation_fold, so a perturbation can carry a baseline score per fold while
    # the method rows span several folds. Average them rather than keeping whichever fold sorted first.
    bl = (df[df['method'] == bl_method]
          .groupby(['dataset', pert_col], as_index=False)[metric].mean())
    return dict(zip(zip(bl['dataset'], bl[pert_col]), bl[metric].astype(float)))

# ══════════════════ RELIABILITY MODEL ══════════════════
def rho_model(N, tau_sq):
    """Spearman-Brown: ρ = Nτ²/(Nτ²+1)"""
    return N * tau_sq / (N * tau_sq + 1.0)

def n_star(tau_sq, rho=0.5):
    """Required sample size for target reliability."""
    if tau_sq <= 0 or rho >= 1:
        return np.inf
    return rho / (tau_sq * (1.0 - rho))


def ds_weighted_excess(df, metric, bl_method, pert_col, skip, category=None):
    """Mean across datasets of a method's mean excess over the benchmark's baseline.

    Wei's `op` encodes perturbation_fold, so one perturbation can be scored in more than one fold. Each
    method row is differenced against the baseline from its OWN fold, which is the like-for-like comparison
    and keeps Wei's own unit of observation: their per-perturbation ranks are likewise computed per `op`.
    Collapsing folds first would instead give every perturbation equal weight regardless of how many folds
    it was scored in, which changes the published Figure 2a ranks and departs from Wei's aggregation.
    """
    key = 'op' if 'op' in df.columns else pert_col
    b = df[df['method'] == bl_method].groupby(['dataset', key], as_index=False)[metric].mean()
    bl = dict(zip(zip(b['dataset'], b[key]), b[metric].astype(float)))
    sub = df if category is None else df[df['triage_2d'] == category]
    sub = sub[~sub['method'].isin(set(skip) | {bl_method})]
    methods = sorted(sub['method'].unique())
    if not methods:
        return {}
    per = sub[['method', 'dataset', key, metric]].copy()
    per['_bl'] = [bl.get(k, np.nan) for k in zip(per['dataset'], per[key])]
    per = per.dropna(subset=['_bl'])
    per['_exc'] = per[metric].astype(float) - per['_bl']
    got = per.groupby(['method', 'dataset'])['_exc'].mean().groupby('method').mean()
    return {m: (float(got[m]) if m in got.index else 0.0) for m in methods}


# ══════════════════ RETRAINING DATASETS ══════════════════
# The retraining analyses (Figure 2 panel e, Supplementary Figure 6) use only Wei's genetic-SINGLE
# datasets. Their four genetic-COMBO screens (Norman, Schmidt, Replogle_exp6, Wessels) are excluded
# because neither reference fits a linear model the way pooling them would require:
#   - Ahlmann-Eltze exclude combination perturbations from the fit entirely (their perturbation-embedding
#     columns are gene names, so a "A+B" condition matches nothing) and use a separate additive model for
#     doubles; `linearModel` does not appear in their doubles analysis at all.
#   - Wei keep the two settings in separate benchmarks. For combination datasets they train on singles plus
#     half the doubles and TEST ON DOUBLES, and their `linearModel` substitutes the trainMean baseline's
#     predictions for every combination (Perturbation_generalization/Genetic/linearModel.py, predComb).
# Training pools for these three contain no combinations, so the fit here is identical to the published
# linear model, and the held-out sets are single-gene as in both references.
RETRAIN_SINGLE_DATASETS = ['Adamson', 'Replogle_K562essential', 'Replogle_RPE1essential']

# ══════════════════ BENCHMARK SETTINGS ══════════════════
# The eight published benchmark settings, defined once so every figure ranks methods the same way.
# Each entry is (label, merged table, headline metric, baseline method, perturbation key, methods to skip).
# The baseline is excluded from the ranking by ds_weighted_excess, since every score is an excess over it.
BENCHMARK_SETTINGS = [
    ("Wei\nGenetic single",  'wei_genetic_merged_2d.csv',         'pcc',      'trainMean',      'perturbation', {'baseControl', 'trainMean', 'scFoundation'}),
    ("Wei\nGenetic combo",   'wei_genetic_combo_merged_2d.csv',   'pcc',      'trainMean',      'perturbation', {'baseControl', 'trainMean', 'scFoundation'}),
    ("Wei\nChemical single", 'wei_chemical_single_merged_2d.csv', 'pcc',      'trainMean',      'perturbation', {'baseControl', 'trainMean'}),
    ("Wei\nChemical combo",  'wei_chemical_combo_merged_2d.csv',  'pcc',      'trainMean',      'perturbation', {'baseControl', 'trainMean'}),
    ("Wei\nCellular IID",    'wei_cellular_iid_merged_2d.csv',    'pcc',      'trainMean',      'condition',    {'baseControl', 'trainMean'}),
    ("Wei\nCellular OOD",    'wei_cellular_ood_merged_2d.csv',    'pcc',      'trainMean',      'condition',    {'baseControl', 'trainMean'}),
    ("AE\nGenetic single",   'ae_panel_a_merged_2d.csv',          'r2_delta', 'mean',           'perturbation', set()),
    ("AE\nGenetic combo",    'ae_doubles_merged_2d.csv',          'r2_delta', 'additive_model', 'perturbation', {'no_change'}),
]

# Wei et al. score every setting with these six metrics and rank methods by the mean of the six ranks.
# All six are normalised to [0, 1] with HIGHER being better, including the four named after distances:
# `mse_score`, `edistance_score`, `was_score` and `sym_score` are goodness scores derived from those
# distances, not the distances themselves. Verified against their published per-perturbation tables, where
# a score of 0.0 carries the worst rank (e.g. baseMLP, edistance_score = 0.0, Rank_edistance = 15 of 15).
# The Ahlmann-Eltze benchmark reports only its single delta correlation, so it has no metric axis.
WEI_METRICS = [('pcc', True, 'PCC-Δ'), ('mse_score', True, 'MSE score'), ('edistance_score', True, 'E-distance score'),
               ('was_score', True, 'Wasserstein score'), ('sym_score', True, 'KL score'), ('DEG_score', True, 'common-DEGs')]

# Both benchmarks score the Pearson correlation of the delta vector and differ only in the gene set.
# Ahlmann-Eltze's column is named `r2_delta` but their code computes a correlation, not a squared quantity,
# so it must not be labelled r2 anywhere.
METRIC_DISPLAY = {'pcc': 'PCC-Δ (100 DEG)', 'r2_delta': 'PCC-Δ (1000 expr)'}


def load_benchmark(fname):
    """Read one merged benchmark table with canonical triage labels, collapsing AE folds to one row."""
    df = rename_triage(pd.read_csv(intermediate_path(fname)))
    if fname.startswith('ae_'):
        df = df[df['train'] == 'test'].copy()
        df = df.groupby(['method', 'dataset', 'perturbation', 'triage_2d'],
                        as_index=False).agg(r2_delta=('r2_delta', 'mean'))
    return df


def top_method(df, metric, higher_is_better, bl_method, pert_col, skip, category=None):
    """Best method by dataset-weighted excess over baseline, honouring the metric's direction."""
    exc = ds_weighted_excess(df, metric, bl_method, pert_col, skip, category)
    if not exc:
        return None, {}
    return (max(exc, key=exc.get) if higher_is_better else min(exc, key=exc.get)), exc
